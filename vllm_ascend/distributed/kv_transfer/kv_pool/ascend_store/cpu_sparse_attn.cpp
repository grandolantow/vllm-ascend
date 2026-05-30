#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>
#include <ATen/Parallel.h>
#include <torch/script.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <omp.h>
#include <assert.h>
#include <torch/torch.h>
#include <pthread.h>


void bindToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}


at::Tensor restore_tensor(uintptr_t ptr_val, const std::vector<int64_t>& shape, torch::ScalarType dtype = torch::kBFloat16) {
    if (ptr_val == 0) return at::Tensor(); // 处理空指针情况
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
}

constexpr int32_t OLD_FLAG = 1;
constexpr int32_t NEW_FLAG = 2;
constexpr int32_t BOTH_FLAG = 3;
constexpr int32_t EPOCH_RESET_THRESHOLD = 2'147'483'000;
constexpr int CACHE_MISS_TOPK_AUTO_THREADS = 64;

#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#define HOT_FUNCTION __attribute__((hot))
#define RESTRICT __restrict__
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FORCE_INLINE inline
#define HOT_FUNCTION
#define RESTRICT
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

FORCE_INLINE bool has_old(const int32_t* RESTRICT mark, int32_t token, int32_t base) {
    const int32_t marker = mark[token];
    return marker == base + OLD_FLAG || marker == base + BOTH_FLAG;
}

FORCE_INLINE bool has_new(const int32_t* RESTRICT mark, int32_t token, int32_t base) {
    const int32_t marker = mark[token];
    return marker == base + NEW_FLAG || marker == base + BOTH_FLAG;
}

FORCE_INLINE void set_old(int32_t* RESTRICT mark, int32_t token, int32_t base) {
    const int32_t marker = mark[token];
    if (UNLIKELY(marker == base + NEW_FLAG)) {
        mark[token] = base + BOTH_FLAG;
    } else if (LIKELY(marker != base + OLD_FLAG && marker != base + BOTH_FLAG)) {
        mark[token] = base + OLD_FLAG;
    }
}

FORCE_INLINE void set_new(int32_t* RESTRICT mark, int32_t token, int32_t base) {
    const int32_t marker = mark[token];
    if (UNLIKELY(marker == base + OLD_FLAG)) {
        mark[token] = base + BOTH_FLAG;
    } else if (LIKELY(marker != base + NEW_FLAG && marker != base + BOTH_FLAG)) {
        mark[token] = base + NEW_FLAG;
    }
}

FORCE_INLINE int choose_cache_miss_topk_threads(
    int num_reqs,
    int workspace_threads,
    int requested_threads
) {
    if (num_reqs <= 1 || workspace_threads <= 1) {
        return 1;
    }

    int threads = requested_threads;
    if (threads <= 0) {
        threads = std::min(num_reqs, CACHE_MISS_TOPK_AUTO_THREADS);
    }

    threads = std::max(1, threads);
    threads = std::min(threads, num_reqs);
    threads = std::min(threads, workspace_threads);
    return threads;
}

FORCE_INLINE void process_one_cache_miss_topk_row(
    int row,
    const int topk,
    const int64_t max_token,
    const int64_t* RESTRICT req_ids,
    int64_t* RESTRICT last_req_ids,
    int64_t* RESTRICT topk_indices_old,
    int32_t* RESTRICT topk_indices_new,
    int32_t* RESTRICT mark,
    int32_t* RESTRICT miss,
    int32_t* RESTRICT epoch
) {
    int32_t base = epoch[0] + 4;
    if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
        std::fill(mark, mark + max_token, 0);
        base = 4;
    }
    epoch[0] = base;

    int64_t* RESTRICT old_row = topk_indices_old + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT new_row = topk_indices_new + static_cast<int64_t>(row) * topk;
    const int64_t req_id = req_ids[row];

    if (UNLIKELY(last_req_ids[row] != req_id)) {
        std::fill(old_row, old_row + topk, -1);
        int write_pos = 0;
        for (int slot = 0; slot < topk; ++slot) {
            const int32_t value = new_row[slot];
            new_row[slot] = -1;
            if (LIKELY(value >= 0 && value < max_token) &&
                !has_old(mark, value, base)) {
                set_old(mark, value, base);
                old_row[write_pos] = static_cast<int64_t>(value);
                new_row[write_pos] = value;
                ++write_pos;
                if (write_pos >= topk) {
                    break;
                }
            }
        }
        last_req_ids[row] = req_id;
        return;
    }

    for (int slot = 0; slot < topk; ++slot) {
        const int64_t value = old_row[slot];
        if (LIKELY(value >= 0 && value < max_token)) {
            set_old(mark, static_cast<int32_t>(value), base);
        }
    }

    int miss_count = 0;
    for (int slot = 0; slot < topk; ++slot) {
        const int32_t value = new_row[slot];
        if (LIKELY(value >= 0 && value < max_token)) {
            set_new(mark, value, base);
            if (!has_old(mark, value, base)) {
                miss[miss_count] = value;
                ++miss_count;
                set_old(mark, value, base);
            }
        }
    }

    int miss_pos = 0;
    for (int slot = 0; slot < topk; ++slot) {
        new_row[slot] = -1;

        const int64_t old_value = old_row[slot];
        const int64_t old_raw = old_value;
        if (old_raw >= 0 && old_raw < max_token &&
            !has_new(mark, static_cast<int32_t>(old_raw), base)) {
            if (miss_pos < miss_count) {
                const int32_t replacement = miss[miss_pos];
                ++miss_pos;
                old_row[slot] = static_cast<int64_t>(replacement);
                new_row[slot] = replacement;
            } else {
                old_row[slot] = -1;
            }
        }
    }

    if (miss_pos < miss_count) {
        for (int slot = 0; slot < topk; ++slot) {
            if (old_row[slot] == -1) {
                const int32_t replacement = miss[miss_pos];
                ++miss_pos;
                old_row[slot] = static_cast<int64_t>(replacement);
                new_row[slot] = replacement;
                if (miss_pos >= miss_count) {
                    break;
                }
            }
        }
    }
    last_req_ids[row] = req_id;
}

FORCE_INLINE bool is_current_hash_marker(int32_t marker, int32_t base) {
    return marker >= base + OLD_FLAG && marker <= base + BOTH_FLAG;
}

FORCE_INLINE uint32_t hash_topk_token(int32_t token) {
    uint32_t value = static_cast<uint32_t>(token);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

FORCE_INLINE int find_hash_slot(
    int32_t* RESTRICT keys,
    int32_t* RESTRICT marks,
    int capacity,
    int32_t token,
    int32_t base,
    bool* found
) {
    const int mask = capacity - 1;
    int slot = static_cast<int>(hash_topk_token(token) & static_cast<uint32_t>(mask));
    for (int probe = 0; probe < capacity; ++probe) {
        const int32_t marker = marks[slot];
        if (!is_current_hash_marker(marker, base)) {
            *found = false;
            return slot;
        }
        if (keys[slot] == token) {
            *found = true;
            return slot;
        }
        slot = (slot + 1) & mask;
    }
    *found = false;
    return -1;
}

FORCE_INLINE bool has_old_hash(
    int32_t* RESTRICT keys,
    int32_t* RESTRICT marks,
    int capacity,
    int32_t token,
    int32_t base
) {
    bool found = false;
    const int slot = find_hash_slot(keys, marks, capacity, token, base, &found);
    if (UNLIKELY(slot < 0 || !found)) {
        return false;
    }
    const int32_t marker = marks[slot];
    return marker == base + OLD_FLAG || marker == base + BOTH_FLAG;
}

FORCE_INLINE bool has_new_hash(
    int32_t* RESTRICT keys,
    int32_t* RESTRICT marks,
    int capacity,
    int32_t token,
    int32_t base
) {
    bool found = false;
    const int slot = find_hash_slot(keys, marks, capacity, token, base, &found);
    if (UNLIKELY(slot < 0 || !found)) {
        return false;
    }
    const int32_t marker = marks[slot];
    return marker == base + NEW_FLAG || marker == base + BOTH_FLAG;
}

FORCE_INLINE void set_old_hash(
    int32_t* RESTRICT keys,
    int32_t* RESTRICT marks,
    int capacity,
    int32_t token,
    int32_t base
) {
    bool found = false;
    const int slot = find_hash_slot(keys, marks, capacity, token, base, &found);
    if (UNLIKELY(slot < 0)) {
        return;
    }
    if (!found) {
        keys[slot] = token;
        marks[slot] = base + OLD_FLAG;
        return;
    }

    const int32_t marker = marks[slot];
    if (UNLIKELY(marker == base + NEW_FLAG)) {
        marks[slot] = base + BOTH_FLAG;
    } else if (LIKELY(marker != base + OLD_FLAG && marker != base + BOTH_FLAG)) {
        marks[slot] = base + OLD_FLAG;
    }
}

FORCE_INLINE void set_new_hash(
    int32_t* RESTRICT keys,
    int32_t* RESTRICT marks,
    int capacity,
    int32_t token,
    int32_t base
) {
    bool found = false;
    const int slot = find_hash_slot(keys, marks, capacity, token, base, &found);
    if (UNLIKELY(slot < 0)) {
        return;
    }
    if (!found) {
        keys[slot] = token;
        marks[slot] = base + NEW_FLAG;
        return;
    }

    const int32_t marker = marks[slot];
    if (UNLIKELY(marker == base + OLD_FLAG)) {
        marks[slot] = base + BOTH_FLAG;
    } else if (LIKELY(marker != base + NEW_FLAG && marker != base + BOTH_FLAG)) {
        marks[slot] = base + NEW_FLAG;
    }
}

FORCE_INLINE void process_one_cache_miss_topk_hash_row(
    int row,
    const int topk,
    const int64_t max_token,
    const int hash_capacity,
    const int64_t* RESTRICT req_ids,
    int64_t* RESTRICT last_req_ids,
    int64_t* RESTRICT topk_indices_old,
    int32_t* RESTRICT topk_indices_new,
    int32_t* RESTRICT hash_keys,
    int32_t* RESTRICT hash_marks,
    int32_t* RESTRICT miss,
    int32_t* RESTRICT epoch
) {
    int32_t base = epoch[0] + 4;
    if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
        std::fill(hash_marks, hash_marks + hash_capacity, 0);
        base = 4;
    }
    epoch[0] = base;

    int64_t* RESTRICT old_row = topk_indices_old + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT new_row = topk_indices_new + static_cast<int64_t>(row) * topk;
    const int64_t req_id = req_ids[row];

    if (UNLIKELY(last_req_ids[row] != req_id)) {
        std::fill(old_row, old_row + topk, -1);
        int write_pos = 0;
        for (int slot = 0; slot < topk; ++slot) {
            const int32_t value = new_row[slot];
            new_row[slot] = -1;
            if (LIKELY(value >= 0 && value < max_token) &&
                !has_old_hash(hash_keys, hash_marks, hash_capacity, value, base)) {
                set_old_hash(hash_keys, hash_marks, hash_capacity, value, base);
                old_row[write_pos] = static_cast<int64_t>(value);
                new_row[write_pos] = value;
                ++write_pos;
                if (write_pos >= topk) {
                    break;
                }
            }
        }
        last_req_ids[row] = req_id;
        return;
    }

    for (int slot = 0; slot < topk; ++slot) {
        const int64_t value = old_row[slot];
        if (LIKELY(value >= 0 && value < max_token)) {
            set_old_hash(hash_keys, hash_marks, hash_capacity, static_cast<int32_t>(value), base);
        }
    }

    int miss_count = 0;
    for (int slot = 0; slot < topk; ++slot) {
        const int32_t value = new_row[slot];
        if (LIKELY(value >= 0 && value < max_token)) {
            set_new_hash(hash_keys, hash_marks, hash_capacity, value, base);
            if (!has_old_hash(hash_keys, hash_marks, hash_capacity, value, base)) {
                miss[miss_count] = value;
                ++miss_count;
                set_old_hash(hash_keys, hash_marks, hash_capacity, value, base);
            }
        }
    }

    int miss_pos = 0;
    for (int slot = 0; slot < topk; ++slot) {
        new_row[slot] = -1;

        const int64_t old_value = old_row[slot];
        const int64_t old_raw = old_value;
        if (old_raw >= 0 && old_raw < max_token &&
            !has_new_hash(hash_keys, hash_marks, hash_capacity, static_cast<int32_t>(old_raw), base)) {
            if (miss_pos < miss_count) {
                const int32_t replacement = miss[miss_pos];
                ++miss_pos;
                old_row[slot] = static_cast<int64_t>(replacement);
                new_row[slot] = replacement;
            } else {
                old_row[slot] = -1;
            }
        }
    }

    if (miss_pos < miss_count) {
        for (int slot = 0; slot < topk; ++slot) {
            if (old_row[slot] == -1) {
                const int32_t replacement = miss[miss_pos];
                ++miss_pos;
                old_row[slot] = static_cast<int64_t>(replacement);
                new_row[slot] = replacement;
                if (miss_pos >= miss_count) {
                    break;
                }
            }
        }
    }
    last_req_ids[row] = req_id;
}

HOT_FUNCTION void cache_miss_topk(
    uintptr_t req_ids_ptr,
    uintptr_t last_req_ids_ptr,
    uintptr_t topk_indices_old_ptr,
    uintptr_t topk_indices_new_ptr,
    uintptr_t mark_workspace_ptr,
    uintptr_t miss_workspace_ptr,
    uintptr_t epochs_ptr,
    int64_t num_reqs,
    int64_t topk,
    int64_t max_token,
    int64_t workspace_threads,
    int64_t requested_threads
) {
    auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
    auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
    auto* RESTRICT topk_indices_old = reinterpret_cast<int64_t*>(topk_indices_old_ptr);
    auto* RESTRICT topk_indices_new = reinterpret_cast<int32_t*>(topk_indices_new_ptr);
    auto* RESTRICT mark_workspace = reinterpret_cast<int32_t*>(mark_workspace_ptr);
    auto* RESTRICT miss_workspace = reinterpret_cast<int32_t*>(miss_workspace_ptr);
    auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);

    const int num_reqs_int = static_cast<int>(num_reqs);
    const int topk_int = static_cast<int>(topk);
    const int workspace_threads_int = static_cast<int>(workspace_threads);
    const int requested_threads_int = static_cast<int>(requested_threads);
    const int active_threads = choose_cache_miss_topk_threads(
        num_reqs_int,
        workspace_threads_int,
        requested_threads_int
    );

    if (active_threads == 1) {
        for (int row = 0; row < num_reqs_int; ++row) {
            process_one_cache_miss_topk_row(
                row,
                topk_int,
                max_token,
                req_ids,
                last_req_ids,
                topk_indices_old,
                topk_indices_new,
                mark_workspace,
                miss_workspace,
                epochs
            );
        }
        return;
    }

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id = omp_get_thread_num();
        int32_t* RESTRICT mark = mark_workspace + static_cast<int64_t>(thread_id) * max_token;
        int32_t* RESTRICT miss = miss_workspace + static_cast<int64_t>(thread_id) * topk_int;
        int32_t* RESTRICT epoch = epochs + thread_id;
        for (int row = thread_id; row < num_reqs_int; row += active_threads) {
            process_one_cache_miss_topk_row(
                row,
                topk_int,
                max_token,
                req_ids,
                last_req_ids,
                topk_indices_old,
                topk_indices_new,
                mark,
                miss,
                epoch
            );
        }
    }
}

HOT_FUNCTION void cache_miss_topk_hash(
    uintptr_t req_ids_ptr,
    uintptr_t last_req_ids_ptr,
    uintptr_t topk_indices_old_ptr,
    uintptr_t topk_indices_new_ptr,
    uintptr_t hash_keys_workspace_ptr,
    uintptr_t hash_marks_workspace_ptr,
    uintptr_t miss_workspace_ptr,
    uintptr_t epochs_ptr,
    int64_t num_reqs,
    int64_t topk,
    int64_t max_token,
    int64_t hash_capacity,
    int64_t workspace_threads,
    int64_t requested_threads
) {
    auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
    auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
    auto* RESTRICT topk_indices_old = reinterpret_cast<int64_t*>(topk_indices_old_ptr);
    auto* RESTRICT topk_indices_new = reinterpret_cast<int32_t*>(topk_indices_new_ptr);
    auto* RESTRICT hash_keys_workspace = reinterpret_cast<int32_t*>(hash_keys_workspace_ptr);
    auto* RESTRICT hash_marks_workspace = reinterpret_cast<int32_t*>(hash_marks_workspace_ptr);
    auto* RESTRICT miss_workspace = reinterpret_cast<int32_t*>(miss_workspace_ptr);
    auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);

    const int num_reqs_int = static_cast<int>(num_reqs);
    const int topk_int = static_cast<int>(topk);
    const int hash_capacity_int = static_cast<int>(hash_capacity);
    const int workspace_threads_int = static_cast<int>(workspace_threads);
    const int requested_threads_int = static_cast<int>(requested_threads);
    if (hash_capacity_int <= 0 ||
        (hash_capacity_int & (hash_capacity_int - 1)) != 0 ||
        hash_capacity_int < topk_int * 4) {
        throw std::invalid_argument("hash_capacity must be a power of two and at least topk * 4");
    }
    const int active_threads = choose_cache_miss_topk_threads(
        num_reqs_int,
        workspace_threads_int,
        requested_threads_int
    );

    if (active_threads == 1) {
        for (int row = 0; row < num_reqs_int; ++row) {
            process_one_cache_miss_topk_hash_row(
                row,
                topk_int,
                max_token,
                hash_capacity_int,
                req_ids,
                last_req_ids,
                topk_indices_old,
                topk_indices_new,
                hash_keys_workspace,
                hash_marks_workspace,
                miss_workspace,
                epochs
            );
        }
        return;
    }

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id = omp_get_thread_num();
        int32_t* RESTRICT hash_keys = hash_keys_workspace + static_cast<int64_t>(thread_id) * hash_capacity_int;
        int32_t* RESTRICT hash_marks = hash_marks_workspace + static_cast<int64_t>(thread_id) * hash_capacity_int;
        int32_t* RESTRICT miss = miss_workspace + static_cast<int64_t>(thread_id) * topk_int;
        int32_t* RESTRICT epoch = epochs + thread_id;
        for (int row = thread_id; row < num_reqs_int; row += active_threads) {
            process_one_cache_miss_topk_hash_row(
                row,
                topk_int,
                max_token,
                hash_capacity_int,
                req_ids,
                last_req_ids,
                topk_indices_old,
                topk_indices_new,
                hash_keys,
                hash_marks,
                miss,
                epoch
            );
        }
    }
}

// std::tuple<at::Tensor, at::Tensor>
void
get_kv_topk_kernel(
    const at::Tensor& k_nope_h, // [block_num, block_size, head_kv, nope_dim]
    const at::Tensor& k_pe_h, // [block_num, block_size, head_kv, rope_dim]
    const at::Tensor& topk_indices, // [num_token, 1, topk]
    const at::Tensor& actual_seq_lengths_query, // [num_reqs], == num_token if without spec decode
    const at::Tensor& block_table, // [num_reqs, max_block_num]
    const int& thread_num,
    at::Tensor& k_nope_topk,
    at::Tensor& k_pe_topk
) {
    auto k_nope = k_nope_h.contiguous();
    auto k_pe = k_pe_h.contiguous();

    TORCH_CHECK(topk_indices.size(1) == 1, "topk_indices must have shape [N, 1, S]");
    auto topk_indices_squeezed = topk_indices.squeeze(1); // [N, S]

    int32_t token_num = topk_indices.size(0);  // Batch size
    int32_t pe_dim = k_pe.size(3);    // Embedding dimension of pe
    int32_t num_page = k_nope.size(0);    // Number of pages
    int32_t page_size = k_nope.size(1);   // Page size
    int32_t head_kv = k_nope.size(2);      // Number of heads of key/value
    int32_t value_dim = k_nope.size(3); // head dim of k_nope & value
    int32_t sqen_len = topk_indices_squeezed.size(1);  // Source sequence length (TopK)
    int32_t max_num_pages = block_table.size(1);  // Max number of pages per request
    int32_t num_requests = actual_seq_lengths_query.size(0);

    // auto cpu_pinned_opt = k_nope_h.options()
    //     .device(at::kCPU)         // 限定CPU设备
    //     .pinned_memory(true);     // 开启页锁定内存
    // at::Tensor k_nope_cpu_pinned = at::empty({token_num, sqen_len, head_kv, value_dim}, cpu_pinned_opt);
    // at::Tensor k_pe_cpu_pinned = at::empty({token_num, sqen_len, head_kv, pe_dim}, cpu_pinned_opt);

    bfloat16_t* k_nope_ptr = static_cast<bfloat16_t*>(k_nope.data_ptr());
    bfloat16_t* k_pe_ptr = static_cast<bfloat16_t*>(k_pe.data_ptr());
    int32_t* topk_indices_ptr = static_cast<int32_t*>(topk_indices_squeezed.data_ptr());
    int32_t* actual_seq_lengths_query_ptr = static_cast<int32_t*>(actual_seq_lengths_query.data_ptr());
    int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());

    // bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_cpu_pinned.data_ptr());
    // bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_cpu_pinned.data_ptr());
    bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_topk.data_ptr());
    bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_topk.data_ptr());

    auto req_ids = at::empty({token_num}, at::kInt);   // at::empty，分配一块指定形状和数据类型的内存空间，创建一维张量，int32类型
    auto page_indices = at::empty({token_num, sqen_len}, at::kInt);   //创建二维张量
    auto pos_in_page = at::empty({token_num, sqen_len}, at::kInt);

    int32_t* req_ids_ptr = req_ids.data_ptr<int32_t>();
    int32_t* page_indices_ptr = page_indices.data_ptr<int32_t>();
    int32_t* pos_in_page_ptr = pos_in_page.data_ptr<int32_t>();

    for (int32_t r = 0; r < num_requests; ++r) {
        int32_t start = (r == 0) ? 0 : actual_seq_lengths_query_ptr[r - 1];
        int32_t end = actual_seq_lengths_query_ptr[r];

        std::fill(req_ids_ptr + start, req_ids_ptr + end, r);
    }

    // multi thread related. Currently parallel in seq_len(topk) dim.
    // int thread_num = 16;
    int size_per_thread = sqen_len / thread_num;
    std::vector<int> thread_size(thread_num, size_per_thread);
    thread_size[thread_num - 1] += sqen_len - size_per_thread * thread_num;
    std::vector<int> thread_start(thread_num);
    int cumsum = 0;
    for(int i = 0; i < thread_num; ++i) {
        thread_start[i] = cumsum;
        cumsum += thread_size[i];
    }
    // std::cout << "thread start: "; for(auto i : thread_start) std::cout << i << " "; std::cout << std::endl;
    // std::cout << "thread size: "; for(auto i : thread_size) std::cout << i << " "; std::cout << std::endl;

    at::set_num_threads(thread_num);
    omp_set_num_threads(thread_num);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int t_start = thread_start[tid];
        int t_end = t_start + thread_size[tid];
        for (int32_t i = 0; i < token_num; ++i) {
            int32_t req_id = req_ids_ptr[i];
            // for (int32_t s = 0; s < sqen_len; ++s) {
            for (int32_t s = t_start; s < t_end; ++s) {
                int32_t global_token_pos = topk_indices_ptr[i * sqen_len + s];

                if (global_token_pos < 0) {
                    continue;
                    // page_indices_ptr[i * sqen_len + s] = 0;
                    // pos_in_page_ptr[i * sqen_len + s] = 0;
                    // for (int32_t h = 0; h < head_kv; ++h) {
                    //     for (int32_t d = 0; d < value_dim; ++d) {
                    //         k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                    //                          s * head_kv * value_dim +
                    //                          h * value_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    //     for (int32_t d = 0; d < pe_dim; ++d) {
                    //         k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                    //                        s * head_kv * pe_dim +
                    //                        h * pe_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    // }
                    // continue;
                }

                int32_t rel_page_idx = global_token_pos / page_size;
                int32_t offset_in_page = global_token_pos % page_size;

                int32_t physical_page_id = block_table_ptr[req_id * max_num_pages + rel_page_idx];

                page_indices_ptr[i * sqen_len + s] = physical_page_id;
                pos_in_page_ptr[i * sqen_len + s] = offset_in_page;

                for (int32_t h = 0; h < head_kv; ++h) {
                    for (int32_t d = 0; d < value_dim; ++d) {
                        k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                                    s * head_kv * value_dim +
                                    h * value_dim + d] =
                            k_nope_ptr[physical_page_id * page_size * head_kv * value_dim +
                                    offset_in_page * head_kv * value_dim +
                                    h * value_dim + d];
                    }
                    for (int32_t d = 0; d < pe_dim; ++d) {
                        k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                                    s * head_kv * pe_dim +
                                    h * pe_dim + d] =
                            k_pe_ptr[physical_page_id * page_size * head_kv * pe_dim +
                                    offset_in_page * head_kv * pe_dim +
                                    h * pe_dim + d];
                    }
                }
            }
        }
    }

    return;
    // return std::make_tuple(k_nope_cpu_pinned, k_pe_cpu_pinned);
}

void
get_kv_topk(
    uintptr_t k_nope_h_ptr,
    uintptr_t k_pe_h_ptr,
    uintptr_t topk_indices_ptr,
    uintptr_t actual_seq_lengths_query_ptr,
    uintptr_t block_table_ptr,
    uintptr_t k_nope_topk_ptr,
    uintptr_t k_pe_topk_ptr,
    std::vector<int64_t> k_nope_h_shape, // [block_num, block_size, head_kv, nope_dim]
    std::vector<int64_t> k_pe_h_shape, // [block_num, block_size, head_kv, rope_dim]
    std::vector<int64_t> topk_indices_shape, // [num_token, 1, topk]
    std::vector<int64_t> actual_seq_lengths_query_shape, // [num_reqs], == num_token if without spec decode
    std::vector<int64_t> block_table_shape, // [num_reqs, max_block_num]
    std::vector<int64_t> k_nope_topk_shape, // output, [num_reqs, topk, head_kv, nope_dim]
    std::vector<int64_t> k_pe_topk_shape, // output, [num_reqs, topk, head_kv, rope_dim]
    const int& thread_num
) {
    at::Tensor k_nope_h = restore_tensor(k_nope_h_ptr, k_nope_h_shape);
    at::Tensor k_pe_h = restore_tensor(k_pe_h_ptr, k_pe_h_shape);
    at::Tensor topk_indices = restore_tensor(topk_indices_ptr, topk_indices_shape, torch::kInt);
    at::Tensor actual_seq_lengths_query = restore_tensor(actual_seq_lengths_query_ptr, actual_seq_lengths_query_shape, torch::kInt);
    at::Tensor block_table = restore_tensor(block_table_ptr, block_table_shape, torch::kInt);
    at::Tensor k_nope_topk = restore_tensor(k_nope_topk_ptr, k_nope_topk_shape);
    at::Tensor k_pe_topk = restore_tensor(k_pe_topk_ptr, k_pe_topk_shape);
    get_kv_topk_kernel(
        k_nope_h,
        k_pe_h,
        topk_indices,
        actual_seq_lengths_query,
        block_table,
        thread_num,
        k_nope_topk,
        k_pe_topk
    );
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def("cache_miss_topk", &cache_miss_topk, "CPU cache-miss topk with OpenMP row-level parallelism");
    m.def("cache_miss_topk_hash", &cache_miss_topk_hash, "CPU cache-miss topk hash-table experiment");
    m.def("get_kv_topk", &get_kv_topk, "High performance topk combine");
}
