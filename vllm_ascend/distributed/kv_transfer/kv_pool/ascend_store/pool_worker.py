import importlib
import math
import os
import shutil
import threading
import time
from typing import Optional
import collections
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.cpp_extension import load
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.utils import CpuGpuBuffer

import vllm_ascend.envs as envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LasyerMultiBlockReqMeta,
    ReqMeta,
    PoolKey,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)
from vllm_ascend.attention.cpu_cache_miss_topk import (
    CPUCacheMissTopKWorkspace,
    CPULRUKVWorkspace,
    make_cpu_cache_miss_topk_workspace,
    make_cpu_lru_kv_workspace,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, get_subscribed_compute_streams

backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
}

CPU_CACHE_MISS_TOPK_MAX_TOKEN = 131072
CPU_CACHE_MISS_TOPK_WORKSPACE_THREADS = max(
    1, envs.VLLM_ASCEND_CPU_CACHE_MISS_TOPK_WORKSPACE_THREADS)
CPU_CACHE_MISS_TOPK_REQUESTED_THREADS = (
    envs.VLLM_ASCEND_CPU_CACHE_MISS_TOPK_REQUESTED_THREADS)

# cpu sparse attn kernel related
# TODO maybe implement this in vllm custom op framework
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
# cache_dir = "/root/.cache/torch_extensions/py311_cpu/cpu_sparse_attn"
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
#     print(f"已清理缓存目录: {cache_dir}")
ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
npu_include_path = os.path.join(ascend_home, "include")
npu_lib_path = os.path.join(ascend_home, "lib64")
if not os.path.exists(npu_lib_path):
    npu_lib_path = os.path.join(ascend_home, "lib")
torch_npu_path = os.path.dirname(torch_npu.__file__)
torch_npu_include = os.path.join(torch_npu_path, "include")
torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
os.environ['CXX'] = 'clang++'
os.environ['CC'] = 'clang'
abs_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(abs_path, "cpu_sparse_attn.cpp")
logger.info(f'>>>>> load cpu_sparse_attn from src: {src_path}')
cpu_sparse_attn = None
cpu_sparse_attn = load(
    name="cpu_sparse_attn",
    sources=[src_path],
    extra_cflags=[
        "-O3",
        "-std=c++20",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-fopenmp",
        "-march=armv8.2-a+sve+fp16+bf16",
        # "-march=native",
        "-fPIC",
        f"-I{npu_include_path}",
        f"-I{torch_npu_include}",
    ],
    extra_ldflags=[
        "-fopenmp",
        f"-L{npu_lib_path}",
        "-lascendcl",
        f"-L{torch_npu_lib_path}",
        "-ltorch_npu",
    ],
    verbose=True,  # 添加 verbose 查看编译过程
)
CPU_CACHE_MISS_TOPK_AVAILABLE = (
    cpu_sparse_attn is not None and hasattr(cpu_sparse_attn, "cache_miss_topk")
)
CPU_LRU_KV_AVAILABLE = (
    cpu_sparse_attn is not None and hasattr(cpu_sparse_attn, "lru_kv_topk")
)
CPU_LRU_KV_REQUESTED_THREADS = CPU_CACHE_MISS_TOPK_REQUESTED_THREADS
CPU_LRU_KV_WORKSPACE_THREADS = CPU_CACHE_MISS_TOPK_WORKSPACE_THREADS

# _TEST_STREAM = None
# def load_cpu(args):
#     # logger.info(f'>>>>> load_cpu, args = {len(args)}, get_kv_topk = {cpu_sparse_attn.get_kv_topk}')
#     # if 1:
#     #     token_indices_cpu = args[2]
#     #     num_tokens_to_load = (token_indices_cpu != -1).sum().item()
#     #     logger.info(f'>>>>> load_cpu, num_tokens_to_load={num_tokens_to_load}')
#     global _TEST_STREAM
#     if _TEST_STREAM is None:
#         _TEST_STREAM = torch_npu.npu.Stream()
#     with torch_npu.npu.stream(_TEST_STREAM):
#         # test(*args)
#         cpu_sparse_attn.get_kv_topk(*args)
#         _TEST_STREAM.synchronize()


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.original_block_size = vllm_config.cache_config.block_size
        self.block_size = vllm_config.cache_config.block_size

        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
        self.current_layer = 0
        self.current_layer_load = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        soc_version = get_ascend_device_type()
        # be removed later
        if self.backend == "mooncake" and soc_version in {AscendDeviceType.A3}:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        self.metadata = KeyMetadata(
            model_config.model.rstrip("/").split("/")[-1],
            self.head_or_tp_rank,
            self.pcp_rank,
            self.dcp_rank,
            self.pp_rank,
        )

        partitions = None
        if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "prefill_pp_layer_partition", None
            )
            prefill_pp_size = int(vllm_config.kv_transfer_config.kv_connector_extra_config.get("prefill_pp_size", 1))

            if partition_list_str is not None:
                try:
                    partitions = [int(layer) for layer in partition_list_str.split(",")]
                except ValueError as err:
                    raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
                if len(partitions) != prefill_pp_size:
                    raise ValueError(f"{len(partitions)=} does not match {prefill_pp_size=}.")
                if sum(partitions) != num_hidden_layers:
                    raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
            else:
                layers_per_partition = num_hidden_layers // prefill_pp_size
                partitions = [layers_per_partition for _ in range(prefill_pp_size)]

                if remaining_layers := num_hidden_layers % prefill_pp_size:
                    for i in range(2, remaining_layers + 2):
                        partitions[-i] += 1

        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size, partitions)

        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config
        )
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

        ascend_config = get_ascend_config()
        self.use_offload = ascend_config.use_offload
        # kv offload
        self.model_name = model_config.model.split('/')[-1]
        self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        # 添加缓存字典，用于缓存process_layer_data的计算结果
        # 缓存键格式：(req_id, tuple(block_hashes), token_len_chunk, can_save, load_spec_cache_key, is_last_chunk)
        # 缓存值：(keys, starts, ends)
        self.process_layer_cache = {}
        # 缓存清理阈值，当缓存大小超过此值时，清理最早的缓存项
        self.cache_max_size = 1000
        self.cache_lru = collections.OrderedDict()  # 用于实现LRU缓存，使用OrderedDict提高性能
        # 请求历史状态追踪
        # key: req_id, value: (prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split)
        self.request_history = {}

        self.request_history = collections.defaultdict(tuple)

        # Add cache index for efficient cleanup by req_id
        self.cache_by_req_id = collections.defaultdict(set)  # Maps req_id to set of cache keys

        # sparse onload related
        # TODO get from config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        hot_kv_config = ascend_config.hot_kv_cache_config
        self.topk = hot_kv_config.buffer_size if hot_kv_config.enabled else 2048
        head_num = 1
        head_dim_k = 512
        head_dim_v = 64
        dtype = torch.bfloat16
        self.token_size_bytes_k = head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = head_num * head_dim_v * dtype.itemsize
        max_model_len = vllm_config.model_config.max_model_len
        max_block_num = cdiv(max_model_len, self.block_size)
        self.cpu_block_table = CpuGpuBuffer(self.max_num_reqs, max_block_num, dtype=torch.int32, device='npu', pin_memory=True)
        self.cpu_block_table_host_buffer = torch.zeros([self.max_num_reqs, max_block_num], dtype=torch.int32, device='cpu', pin_memory=True)
        self.actual_seq_len_q = torch.arange(self.max_num_reqs, dtype=torch.int32, device='cpu', pin_memory=True) + 1
        self.req_ids = []
        # self.req_id_to_block_ids: dict[str, list[int]] = {} # reserve for load_kv_token_wise usage

        self.cpu_sparse_attn = cpu_sparse_attn

        # self.token_wise_load_event = torch.npu.Event()
        self.load_stream = None
        self.save_stream = None
        self.side_compute_stream = torch_npu.npu.Stream()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        block_rank = 3
        self.block_len = []
        if self.use_mla or self.use_sparse:
            for i in range(len(first_kv_cache_tuple)):
                if i == 2:
                    break # only save kv_cache[0 & 1] (k_cache and v_cache)
                block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(first_kv_cache[i].element_size() * math.prod(block_shape))
        else:
            # [num_block, block_size, num_head, hidden_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len = [first_kv_cache.element_size() * math.prod(block_shape)]

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches = kv_caches # dict[str, tuple[torch.Tensor]]
        self.kv_caches_base_addr = []
        ptrs = []
        lengths = []
        length = len(self.block_len)
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            for i, cache in enumerate(cache_or_caches, 0):
                if i == 2:
                    break # only save kv_cache[0 & 1] (k_cache and v_cache)
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len[i % length]
                self.kv_caches_base_addr.append(base_addr)
                ptrs.append(base_addr)
                lengths.append(region_len)

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        if self.use_layerwise:
            self.get_event = threading.Event()
            self.layer_save_finished_events = [threading.Event() for _ in range(self.num_layers)]
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    ready_event_sending,
                    self.num_layers,
                    self.enable_kv_events,
                    self.layer_save_finished_events,
                )
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self.get_event,
            )
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store, self.token_database, self.block_size, self.tp_rank, self.dcp_size, ready_event
                )
                self.kv_recv_thread.start()
                ready_event.wait()

        if self.use_sparse and self.use_offload:
            # register topk buffer, get base addr
            # this part is for mmc based offload, useless now
            # addr_k_list = []
            # addr_v_list = []
            # for cache_or_caches in kv_caches.values():
            #     assert len(cache_or_caches) == 5
            #     topk_buffer_k = cache_or_caches[3]
            #     topk_buffer_v = cache_or_caches[4]
            #     base_addr_k = topk_buffer_k.data_ptr()
            #     base_addr_v = topk_buffer_v.data_ptr()
            #     addr_k = torch.arange(base_addr_k, base_addr_k + topk_buffer_k.nelement() * topk_buffer_k.element_size(), self.token_size_bytes_k)
            #     addr_v = torch.arange(base_addr_v, base_addr_v + topk_buffer_v.nelement() * topk_buffer_v.element_size(), self.token_size_bytes_v)
            #     assert addr_k.size(0) == self.max_num_reqs * self.topk
            #     assert addr_v.size(0) == self.max_num_reqs * self.topk
            #     addr_k_list.append(addr_k.reshape([self.max_num_reqs, self.topk]))
            #     addr_v_list.append(addr_v.reshape([self.max_num_reqs, self.topk]))
            # assert len(addr_k_list) == self.num_layers
            # self.addr_k = torch.stack(addr_k_list, dim=0) # [num_layers, max_num_seqs, topk]
            # self.addr_v = torch.stack(addr_v_list, dim=0)

            self.k_caches_npu: list[torch.Tensor] = []
            self.v_caches_npu: list[torch.Tensor] = []
            self.topk_buffers_k: list[torch.Tensor] = []
            self.topk_buffers_v: list[torch.Tensor] = []
            for cache_or_caches in kv_caches.values():
                assert len(cache_or_caches) == 5
                self.k_caches_npu.append(cache_or_caches[0])
                self.v_caches_npu.append(cache_or_caches[1])
                self.topk_buffers_k.append(cache_or_caches[3])
                self.topk_buffers_v.append(cache_or_caches[4])
            self.topk_indices_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.topk_indices_new_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.topk_indices_old_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            )
            self.req_ids_tensor_buffer_cpu = torch.empty(
                [self.max_num_reqs],
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            )
            self.last_req_ids_tensor_buffer_cpu = torch.empty(
                [self.max_num_reqs],
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            )
            self.cpu_cache_miss_topk_workspace: CPUCacheMissTopKWorkspace | None = None
            if CPU_CACHE_MISS_TOPK_AVAILABLE:
                self.cpu_cache_miss_topk_workspace = make_cpu_cache_miss_topk_workspace(
                    topk=self.topk,
                    max_token=CPU_CACHE_MISS_TOPK_MAX_TOKEN,
                    workspace_threads=CPU_CACHE_MISS_TOPK_WORKSPACE_THREADS,
                )
            self.lru_slot_to_token_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slots_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_current_slots_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_load_token_indices_buffer_cpu = torch.empty(
                [self.max_num_reqs, self.topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.cpu_lru_kv_workspace: CPULRUKVWorkspace | None = None
            if CPU_LRU_KV_AVAILABLE:
                self.cpu_lru_kv_workspace = make_cpu_lru_kv_workspace(
                    topk=self.topk,
                    capacity=self.topk,
                    max_token=CPU_CACHE_MISS_TOPK_MAX_TOKEN,
                    workspace_threads=CPU_LRU_KV_WORKSPACE_THREADS,
                )
            self.onload_topk_buffer_k_npu = torch.empty([self.max_num_reqs, self.topk, 1, 512], dtype=torch.bfloat16, device='npu')
            self.onload_topk_buffer_v_npu = torch.empty([self.max_num_reqs, self.topk, 1, 64], dtype=torch.bfloat16, device='npu')
            self.onload_topk_buffer_k_cpu = torch.empty([self.max_num_reqs, self.topk, 1, 512], dtype=torch.bfloat16, device='cpu', pin_memory=True)
            self.onload_topk_buffer_v_cpu = torch.empty([self.max_num_reqs, self.topk, 1, 64], dtype=torch.bfloat16, device='cpu', pin_memory=True)

            npu_block_num = self.num_blocks
            cpu_block_num_multiple = 4 # 512 / 128
            cpu_block_num = npu_block_num * cpu_block_num_multiple
            cpu_cache_size_single_card = cpu_block_num * self.block_size * (512 + 64) * torch.bfloat16.itemsize * self.num_layers
            logger.info(f'KV offload allocate {cpu_block_num} cpu blocks, size = {cpu_cache_size_single_card / 1024 / 1024 / 1024} GB per rank')
            self.k_caches_cpu: list[torch.Tensor] = [torch.empty([cpu_block_num, self.block_size, 1, 512], dtype=torch.bfloat16, pin_memory=True) for _ in range(self.num_layers)]
            self.v_caches_cpu: list[torch.Tensor] = [torch.empty([cpu_block_num, self.block_size, 1, 64], dtype=torch.bfloat16, pin_memory=True) for _ in range(self.num_layers)]

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        # return
        self.current_layer = 0
        self.current_layer_load = 0
        req_id_to_block_ids: dict[str, list[int]] = {}
        for layer_save_task in self.layer_save_tasks:
            layer_save_task.clear()
        for request in metadata.requests:
            req_id_to_block_ids[request.req_id] = request.block_ids_cpu
            if not request.need_save:
                continue # no new blocks to save
            self.process_layer_data(request)
        self.num_save_tasks = len(self.layer_save_tasks[0])
        # if self.tp_rank == 0:
        #     logger.info(f'>>>>> start load kv, reqs num: {len(metadata.requests)}, save task num = {len(self.layer_save_tasks[0])}')

        # remove for async schedule
        # if len(self.layer_save_tasks[0]) == 0:
        #     # no save tasks
        #     for layer_id in range(self.num_layers):
        #         self.layer_save_tasks[layer_id].append(layer_id)

        # generate block_table for load
        # NOTE reqs in self.req_ids and metadata.requests may not be in same order,
        # use reqs from self.req_ids (order of actual batch) to compute block_table.
        num_reqs = len(self.req_ids)
        cpu_block_table_np = self.cpu_block_table.np[:num_reqs]
        cpu_block_table_np.fill(0)
        for i, req_id in enumerate(self.req_ids[:num_reqs]):
            cpu_block_ids = req_id_to_block_ids[req_id]
            cpu_block_table_np[i][:len(cpu_block_ids)] = np.array([cpu_block_ids], dtype=np.int32)
        self.cpu_block_table.copy_to_gpu(num_reqs)
        return
        self.current_layer = 0
        self.layerwise_retrievers = []
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load =0
                continue
            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = request.load_spec.kvpool_cached_tokens + 1
            else:
                token_len = request.load_spec.kvpool_cached_tokens
            request.load_spec.token_len = token_len
            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(request)
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if self.load_async:
                    self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                        request,
                    )
                else:
                    addr_list = []
                    size_list = []
                    key_list = []
                    mask_num = request.load_spec.vllm_cached_tokens // self.block_size * self.block_size
                    for start, end, key in self.token_database.process_tokens(
                        token_len, request.block_hashes, mask_num
                    ):
                        addr, size, _ = self.token_database.prepare_value(start, end, request.block_ids)
                        key_list.append(key.to_string())
                        addr_list.append(addr)
                        size_list.append(size)
                    key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                    addr_list_c = (
                        addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                    )
                    size_list_c = (
                        size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                    )
                    self.m_store.get(key_list_c, addr_list_c, size_list_c)

    def wait_for_layer_load(self) -> None:
        return
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug(f"Retrieved {num_retrieved_tokens} tokens")

    def save_cpu(self, args):
        if self.save_stream is None:
            self.save_stream = torch_npu.npu.Stream()
        with torch_npu.npu.stream(self.save_stream):
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            self.current_layer += 1
            if self.current_layer == self.num_layers:
                self.current_layer = 0

    def save_kv_offload(
        self,
        layer_name: str,
        capturing: bool = False,
    ):
        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)

        args = ()
        if capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.save_cpu,
                args,
            )
        else:
            self.save_cpu(args)

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # if len(self.layer_save_tasks[self.current_layer]) == 0:
        #     # Nothing to save in this step, set finish event here manually.
        #     self.layer_save_finished_events[self.current_layer].set()
        # else:
        #     # self.layer_save_finished_events[self.current_layer].set()
        #     self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        # self.current_layer += 1
        # if self.current_layer == self.num_layers:
        #     self.current_layer = 0
        # return
        if self.current_layer == 0:
            self.layerwise_storers = []
            current_event = None
            for request in connector_metadata.requests:
                can_save = request.can_save
                if can_save is None or not can_save:
                    continue
                current_event = torch.npu.Event()
                current_event.record()
                break
            for request in connector_metadata.requests:
                can_save = request.can_save
                if can_save is None or not can_save:
                    continue

                layerwise_storer = self.store_layer(request, current_event)
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception:
                raise
        self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        # return
        if self.use_sparse and self.use_offload:
            assert self.use_layerwise
            if self.num_save_tasks == 0:
                # no save tasks, no need to wait
                return
            for layer_id, event in enumerate(self.layer_save_finished_events):
                is_finish = event.wait(timeout=1)
                if not is_finish:
                    logger.info(f'>>>>> layer {layer_id} wait for save timeout')
                event.clear()
            return
        current_event = None
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            current_event.record()
            break

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )

    def retrieve_layer(
        self,
        request: ReqMeta,
    ) -> Generator[torch.Tensor | None, None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration.
        """
        token_len = request.token_len_chunk
        mask_num = (
            request.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        num_required_tokens = token_len - mask_num

        ret_mask = torch.zeros(token_len, dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        first_flag = True
        for start, end, key in self.token_database.process_tokens(token_len, request.block_hashes, mask_num):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys = [list(row) for row in zip(*keys)]  # [num_layer,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  # try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LasyerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends, request.block_ids, layer_id
                )
                self.kv_recv_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                first_flag = False
                yield None
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug(f"Retrieved {retrieved_tokens} out of {num_required_tokens} out of total {token_len} tokens")

        yield ret_mask

    def store_layer(
        self,
        request: ReqMeta,
        current_event: torch.npu.Event | None,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(request.token_len_chunk, request.block_hashes):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)  # [block_num,layer_num]

        if keys:
            keys = [list(row) for row in zip(*keys)]  # [layer_num,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta = LasyerMultiBlockReqMeta(
                    request.req_id,
                    keys_multi_chunk,
                    starts,
                    ends,
                    request.block_ids,
                    layer_id,
                    request.is_last_chunk,
                    current_event,
                )
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                yield
        else:
            for layer_id in range(self.num_layers):
                yield

    def generate_key(self, chunk_hash, layer_id):
        return (
            f"{self.model_name}"
            f"@pcp{self.pcp_rank}@dcp{self.dcp_rank}"
            f"@head_or_tp_rank:{self.head_or_tp_rank}"
            f"@{chunk_hash.hex()}@{layer_id}"
        )

    def set_req_ids(self, req_ids: list):
        self.req_ids = req_ids

    def cache_miss_topk_cpu(self, args):
        (
            req_ids_ptr,
            last_req_ids_ptr,
            topk_indices_old_ptr,
            topk_indices_new_ptr,
            mark_workspace_ptr,
            miss_workspace_ptr,
            epochs_ptr,
            num_reqs,
            topk,
            max_token,
            workspace_threads,
            requested_threads,
        ) = args

        if self.cpu_sparse_attn is None or not hasattr(
                self.cpu_sparse_attn, "cache_miss_topk"):
            print(
                "[SFA][cache_miss_prepare][worker_cpu] "
                "prepared=False reason=no_cpp_backend",
                flush=True,
            )
            return

        self.cpu_sparse_attn.cache_miss_topk(
            req_ids_ptr,
            last_req_ids_ptr,
            topk_indices_old_ptr,
            topk_indices_new_ptr,
            mark_workspace_ptr,
            miss_workspace_ptr,
            epochs_ptr,
            num_reqs,
            topk,
            max_token,
            workspace_threads,
            requested_threads,
        )

    def lru_kv_topk_cpu(self, args):
        (
            req_ids_ptr,
            last_req_ids_ptr,
            topk_indices_ptr,
            slot_to_token_ptr,
            lru_slots_ptr,
            current_slots_ptr,
            load_token_indices_ptr,
            token_mark_workspace_ptr,
            token_pos_workspace_ptr,
            hit_slot_workspace_ptr,
            evictable_slot_workspace_ptr,
            miss_token_workspace_ptr,
            miss_position_workspace_ptr,
            miss_slot_workspace_ptr,
            epochs_ptr,
            num_reqs,
            topk,
            capacity,
            max_token,
            workspace_threads,
            requested_threads,
        ) = args

        if self.cpu_sparse_attn is None or not hasattr(
                self.cpu_sparse_attn, "lru_kv_topk"):
            raise RuntimeError(
                "LRU KV CPU prepare requires cpu_sparse_attn.lru_kv_topk")

        self.cpu_sparse_attn.lru_kv_topk(
            req_ids_ptr,
            last_req_ids_ptr,
            topk_indices_ptr,
            slot_to_token_ptr,
            lru_slots_ptr,
            current_slots_ptr,
            load_token_indices_ptr,
            token_mark_workspace_ptr,
            token_pos_workspace_ptr,
            hit_slot_workspace_ptr,
            evictable_slot_workspace_ptr,
            miss_token_workspace_ptr,
            miss_position_workspace_ptr,
            miss_slot_workspace_ptr,
            epochs_ptr,
            num_reqs,
            topk,
            capacity,
            max_token,
            workspace_threads,
            requested_threads,
        )

    def prepare_lru_kv_topk(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices_npu: torch.Tensor,
        slot_to_token_npu: torch.Tensor,
        lru_slots_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        load_token_indices_npu: torch.Tensor,
        req_ids_tensor_npu: torch.Tensor,
        last_req_ids_tensor_npu: torch.Tensor,
        max_token: int,
        capturing: bool = False,
    ) -> bool:
        if self.cpu_lru_kv_workspace is None:
            raise RuntimeError(
                "LRU KV CPU prepare requires cpu_lru_kv_workspace")

        topk = int(topk_indices_npu.shape[1])
        capacity = int(slot_to_token_npu.shape[1])
        if capacity != int(load_token_indices_npu.shape[1]):
            raise RuntimeError("LRU KV load_token_indices capacity mismatch")
        if topk != int(current_slots_npu.shape[1]):
            raise RuntimeError("LRU KV current_slots topk mismatch")
        if capacity > self.topk or topk > self.topk:
            raise RuntimeError("LRU KV prepare shape exceeds worker buffers")

        topk_indices_cpu = self.topk_indices_buffer_cpu[:num_reqs, :topk]
        slot_to_token_cpu = self.lru_slot_to_token_buffer_cpu[:num_reqs, :capacity]
        lru_slots_cpu = self.lru_slots_buffer_cpu[:num_reqs, :capacity]
        current_slots_cpu = self.lru_current_slots_buffer_cpu[:num_reqs, :topk]
        load_token_indices_cpu = self.lru_load_token_indices_buffer_cpu[
            :num_reqs, :capacity]
        req_ids_tensor_cpu = self.req_ids_tensor_buffer_cpu[:num_reqs]
        last_req_ids_tensor_cpu = self.last_req_ids_tensor_buffer_cpu[:num_reqs]

        topk_indices_cpu.copy_(topk_indices_npu.to(torch.int32),
                               non_blocking=capturing)
        slot_to_token_cpu.copy_(slot_to_token_npu.to(torch.int32),
                                non_blocking=capturing)
        lru_slots_cpu.copy_(lru_slots_npu.to(torch.int32),
                            non_blocking=capturing)
        req_ids_tensor_cpu.copy_(req_ids_tensor_npu, non_blocking=capturing)
        last_req_ids_tensor_cpu.copy_(last_req_ids_tensor_npu,
                                      non_blocking=capturing)

        args = (
            req_ids_tensor_cpu.data_ptr(),
            last_req_ids_tensor_cpu.data_ptr(),
            topk_indices_cpu.data_ptr(),
            slot_to_token_cpu.data_ptr(),
            lru_slots_cpu.data_ptr(),
            current_slots_cpu.data_ptr(),
            load_token_indices_cpu.data_ptr(),
            self.cpu_lru_kv_workspace.token_mark_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.token_pos_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.hit_slot_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.evictable_slot_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.miss_token_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.miss_position_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.miss_slot_workspace.data_ptr(),
            self.cpu_lru_kv_workspace.epochs.data_ptr(),
            num_reqs,
            topk,
            capacity,
            max_token,
            self.cpu_lru_kv_workspace.workspace_threads,
            CPU_LRU_KV_REQUESTED_THREADS,
        )

        if capturing:
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.lru_kv_topk_cpu,
                args,
            )
        else:
            self.lru_kv_topk_cpu(args)

        slot_to_token_npu.copy_(slot_to_token_cpu, non_blocking=capturing)
        lru_slots_npu.copy_(lru_slots_cpu, non_blocking=capturing)
        current_slots_npu.copy_(current_slots_cpu, non_blocking=capturing)
        load_token_indices_npu.copy_(load_token_indices_cpu,
                                     non_blocking=capturing)
        last_req_ids_tensor_npu.copy_(last_req_ids_tensor_cpu,
                                      non_blocking=capturing)
        return True

    def prepare_cache_miss_topk(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices_new_npu: torch.Tensor,
        topk_indices_old_npu: torch.Tensor,
        req_ids_tensor_npu: torch.Tensor,
        last_req_ids_tensor_npu: torch.Tensor,
        capturing: bool = False,
    ) -> bool:
        if self.cpu_cache_miss_topk_workspace is None:
            print(
                "[SFA][cache_miss_prepare][worker] "
                f"layer={layer_name} capturing={capturing} "
                "prepared=False reason=no_workspace",
                flush=True,
            )
            return False

        topk_indices_new_cpu = self.topk_indices_new_buffer_cpu[:num_reqs]
        topk_indices_old_cpu = self.topk_indices_old_buffer_cpu[:num_reqs]
        req_ids_tensor_cpu = self.req_ids_tensor_buffer_cpu[:num_reqs]
        last_req_ids_tensor_cpu = self.last_req_ids_tensor_buffer_cpu[:num_reqs]

        topk_indices_new_cpu.copy_(topk_indices_new_npu, non_blocking=capturing)
        topk_indices_old_cpu.copy_(topk_indices_old_npu, non_blocking=capturing)
        req_ids_tensor_cpu.copy_(req_ids_tensor_npu, non_blocking=capturing)
        last_req_ids_tensor_cpu.copy_(last_req_ids_tensor_npu, non_blocking=capturing)

        args = (
            req_ids_tensor_cpu.data_ptr(),
            last_req_ids_tensor_cpu.data_ptr(),
            topk_indices_old_cpu.data_ptr(),
            topk_indices_new_cpu.data_ptr(),
            self.cpu_cache_miss_topk_workspace.mark_workspace.data_ptr(),
            self.cpu_cache_miss_topk_workspace.miss_workspace.data_ptr(),
            self.cpu_cache_miss_topk_workspace.epochs.data_ptr(),
            num_reqs,
            self.cpu_cache_miss_topk_workspace.topk,
            self.cpu_cache_miss_topk_workspace.max_token,
            self.cpu_cache_miss_topk_workspace.workspace_threads,
            CPU_CACHE_MISS_TOPK_REQUESTED_THREADS,
        )

        if capturing:
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            print(
                "[SFA][cache_miss_prepare][worker] "
                f"layer={layer_name} capturing=True action=launch_host_func",
                flush=True,
            )
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.cache_miss_topk_cpu,
                args,
            )
        else:
            self.cache_miss_topk_cpu(args)

        topk_indices_new_npu.copy_(topk_indices_new_cpu, non_blocking=capturing)
        topk_indices_old_npu.copy_(topk_indices_old_cpu, non_blocking=capturing)
        last_req_ids_tensor_npu.copy_(last_req_ids_tensor_cpu, non_blocking=capturing)
        print(
            "[SFA][cache_miss_prepare][worker] "
            f"layer={layer_name} capturing={capturing} prepared=True "
            "action=copy_h2d",
            flush=True,
        )
        return True

    def load_cpu(self, args):
        if self.load_stream is None:
            self.load_stream = torch_npu.npu.Stream()
        with torch_npu.npu.stream(self.load_stream):
            self.cpu_sparse_attn.get_kv_topk(*args)
            self.load_stream.synchronize()

    def load_kv_token_wise(
        self,
        layer_name: str,
        num_reqs: int,
        token_indices_npu: torch.tensor,
        cpu_mask: torch.tensor,
        capturing: bool = False,
    ):
        thread_num = 16
        token_indices_cpu = self.topk_indices_buffer_cpu[:num_reqs]
        cpu_mask = cpu_mask.unsqueeze(-1).unsqueeze(-1)
        token_indices_cpu.copy_(token_indices_npu, non_blocking=capturing)
        if not capturing and self.current_layer_load == 0 and self.tp_rank == 0:
            num_tokens_to_load = (token_indices_cpu != -1).sum().item()
            logger.info(f'>>>>> load kv tokenwise, num_tokens_to_load = {num_tokens_to_load}')

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)

        k_cache_cpu = self.k_caches_cpu[self.current_layer_load]
        v_cache_cpu = self.v_caches_cpu[self.current_layer_load]
        token_indices_cpu = token_indices_cpu.unsqueeze(1)
        actual_seq_len_q = self.actual_seq_len_q[:num_reqs]
        cpu_block_table = self.cpu_block_table_host_buffer[:num_reqs]
        cpu_block_table.copy_(self.cpu_block_table.gpu[:num_reqs], non_blocking=capturing)
        onload_topk_buffer_k_cpu = self.onload_topk_buffer_k_cpu[:num_reqs]
        onload_topk_buffer_v_cpu = self.onload_topk_buffer_v_cpu[:num_reqs]
        args = (
            k_cache_cpu.data_ptr(),
            v_cache_cpu.data_ptr(),
            token_indices_cpu.data_ptr(),
            actual_seq_len_q.data_ptr(),
            cpu_block_table.data_ptr(),
            onload_topk_buffer_k_cpu.data_ptr(),
            onload_topk_buffer_v_cpu.data_ptr(),
            k_cache_cpu.shape,
            v_cache_cpu.shape,
            token_indices_cpu.shape,
            actual_seq_len_q.shape,
            cpu_block_table.shape,
            onload_topk_buffer_k_cpu.shape,
            onload_topk_buffer_v_cpu.shape,
            thread_num,
        )
        if capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.load_cpu,
                args,
            )
        else:
            self.load_cpu(args)

        # load gpu & cpu kv and compute one sfa, need to combine cpu/npu kv here
        onload_topk_buffer_k_npu = self.onload_topk_buffer_k_npu[:num_reqs]
        onload_topk_buffer_v_npu = self.onload_topk_buffer_v_npu[:num_reqs]
        onload_topk_buffer_k_npu.copy_(onload_topk_buffer_k_cpu, non_blocking=capturing)
        onload_topk_buffer_v_npu.copy_(onload_topk_buffer_v_cpu, non_blocking=capturing)
        topk_buffer_k = self.topk_buffers_k[self.current_layer_load][:num_reqs]
        topk_buffer_v = self.topk_buffers_v[self.current_layer_load][:num_reqs]
        topk_buffer_k[...] = torch.where(cpu_mask, onload_topk_buffer_k_npu, topk_buffer_k)
        topk_buffer_v[...] = torch.where(cpu_mask, onload_topk_buffer_v_npu, topk_buffer_v)

        # compute gpu attn direcly, no need to combine cpu/gpu kv, directly copy to topk_buffer
        # topk_buffer_k = self.topk_buffers_k[self.current_layer_load][:num_reqs]
        # topk_buffer_v = self.topk_buffers_v[self.current_layer_load][:num_reqs]
        # topk_buffer_k.copy_(onload_topk_buffer_k_cpu, non_blocking=capturing)
        # topk_buffer_v.copy_(onload_topk_buffer_v_cpu, non_blocking=capturing)

        self.current_layer_load += 1
        if self.current_layer_load == self.num_layers:
            self.current_layer_load = 0

    def load_kv_token_wise_mmc(
        self,
        layer_name: str,
        token_indices: torch.Tensor, # [num_reqs, topk]
        req_ids,
    ):
        """
        mmc based onload, not used now.
        npu kv storage: (
            k_cache[
                block0[
                    k0[1, 512], k1[1, 512], ..., k127
                ],
                block1[],
                ...
            ],
            v_cache[
                block0[
                    v0[1, 64], v1[1, 64], ..., v127
                ],
                block1[],
                ...
            ]
        )

        cpu kv storage: {
            gva_block0[
                keys[
                    k0[1, 512], k1[1, 512], ..., k127
                ],
                values[
                    v0[1, 64], v1[1, 64], ..., v127
                ]
            ],
            gva_block1[
                keys[],
                values[]
            ],
            ...
        }
        """
        time_0 = time.time()
        PAD_ID = -1
        num_reqs = len(req_ids)
        token_indices = token_indices.to('cpu')
        token_mask = token_indices != PAD_ID
        if not token_mask.any():
            if self.tp_rank == 0 and self.current_layer == 0:
                logger.info(f'>>>>> load_kv_token_wise, load token number: 0')
            return
        assert token_indices.size(0) == num_reqs, "token_indices size mismatch" # TODO consider spec decode case
        assert len(self.req_id_to_meta) >= num_reqs, "req_id_to_meta size mismatch"
        time_1 = time.time()

        # addr_list (npu token addr)
        addr_list_k = self.addr_k[self.current_layer][:num_reqs][token_mask].numpy().tolist()
        addr_list_v = self.addr_v[self.current_layer][:num_reqs][token_mask].numpy().tolist()
        num_tokens_to_load = len(addr_list_k)
        assert len(addr_list_v) == num_tokens_to_load

        time_2 = time.time()

        # gvas_list (cpu token addr)
        gvas_list_k = []
        gvas_list_v = []
        gva_v_offset = self.block_len[0]

        req_tokens_to_load = token_mask.sum(dim=1).numpy().tolist()
        valid_token_indices = token_indices[token_mask]
        block_indices = (valid_token_indices // self.block_size).numpy()
        offset_in_block = (valid_token_indices % self.block_size).numpy().astype(np.uint64)

        slice_start = 0
        gvas_base = np.zeros([num_tokens_to_load], dtype=np.uint64)
        for i, req_id in enumerate(req_ids):
            slice_end = slice_start + req_tokens_to_load[i]
            req_block_indices = block_indices[slice_start : slice_end]
            req_gvas_all = self.req_id_to_meta[req_id].block_gvas[self.current_layer][self.pcp_rank][self.dcp_rank][self.head_or_tp_rank]
            req_gvas = req_gvas_all[req_block_indices]
            gvas_base[slice_start : slice_end] = req_gvas
            slice_start = slice_end
        gvas_base_k = gvas_base
        gvas_base_v = gvas_base + gva_v_offset
        gvas_k = gvas_base_k + offset_in_block * self.token_size_bytes_k
        gvas_v = gvas_base_v + offset_in_block * self.token_size_bytes_v
        gvas_list_k = gvas_k.tolist()
        gvas_list_v = gvas_v.tolist()
        assert len(gvas_list_k) == num_tokens_to_load
        assert len(gvas_list_v) == num_tokens_to_load

        time_3 = time.time()

        # size_list
        size_list_k = np.full([num_tokens_to_load], self.token_size_bytes_k).tolist()
        size_list_v = np.full([num_tokens_to_load], self.token_size_bytes_v).tolist()

        if self.tp_rank == 0 and self.current_layer == 0:
            logger.info(f'>>>>> load_kv_token_wise, load token number: {num_tokens_to_load}')
        # return
        time_4 = time.time()
        self.m_store.store.batch_copy(
            gvas_list_k + gvas_list_v,
            addr_list_k + addr_list_v,
            size_list_k + size_list_v,
            direct=1,
        )
        time_5 = time.time()
        time_2cpu = time_1 - time_0
        time_addr = time_2 - time_1
        time_gvas = time_3 - time_2
        time_size = time_4 - time_3
        # logger.info(f'>>>>> time, load_kv_token_wise layer {self.current_layer}, 2cpu = {time_2cpu}, time_addrs = {time_addr}, time_gvas = {time_gvas}, time_sizes = {time_size}, time_meta_all = {time_2cpu + time_addr + time_gvas + time_size}, time_batch_copy = {time_5 - time_4}')

        return

    def _prepare_req_meta_data(self, req_meta, starts, ends, request, layer_id):
        """
        Prepare common data for both save and load request metadata.

        :param req_meta: The request metadata object to prepare.
        :param starts: List of start positions.
        :param ends: List of end positions.
        :param request: Original request containing block_ids.
        :param layer_id: Current layer ID.
        """
        # Preallocate lists with known size
        keys_count = len(req_meta.keys)
        keys_str = []
        gvas = []  # This list is currently empty but maintained for compatibility
        addr_list = []
        size_list = []

        # Generate addr_list and size_list
        # TODO 缓存起来
        for i in range(keys_count):
            keys_str.append(req_meta.keys[i].to_string())
            addr, size = self.token_database.prepare_value_layer(
                starts[i], ends[i], request.block_ids, layer_id)
            addr_list.extend(addr)
            size_list.extend(size)
            gvas.append(request.key_gva_mapping[keys_str[i]])
            gvas.append(request.key_gva_mapping[keys_str[i]] + self.block_len[0])
            # src/npu_block_ids request.block_ids
            # dst/cpu_block_ids request.key_offload_block_ids[keys]

        # Avoid deepcopy when possible - use direct assignment since we're creating new lists
        req_meta.keys_str = keys_str
        req_meta.gvas = gvas
        req_meta.addr_list = addr_list
        req_meta.size_list = size_list
        return
        # req_meta.host_base_addr = request.host_base_addr
        all_blocks = 256
        # with num_transfer_tasks_lock:
        #     global NUM_TRANSFER_TASKS
        num_transfer_tasks = int(os.environ.get('NUM_TRANSFER_TASKS'))
        # logger.info(f"num_transfer_tasks: {num_transfer_tasks}")
        # logger.info(f"req_id : {request.req_id}")
        # num_transfer_tasks = os.getenv('NUM_TRANSFER_TASKS')
        # assert num_transfer_tasks is None, "num_transfer_tasks cannot be None"
        #
        # # num_transfer_tasks = 512
        blocks_per_task = all_blocks // num_transfer_tasks
        k_gvas = [request.host_base_addr + i * blocks_per_task * self.token_database.block_len[0]
                  for i in range(num_transfer_tasks)]
        v_gvas = [k_gvas[-1] + self.token_database.block_len[0] + i * blocks_per_task * self.token_database.block_len[1]
                  for i in range(num_transfer_tasks)]
        gvas = k_gvas + v_gvas

        k_addrs = [self.token_database.kv_caches_base_addr[0] + i * blocks_per_task * self.token_database.block_len[0]
                   for i in range(num_transfer_tasks)]
        v_addrs = [self.token_database.kv_caches_base_addr[1] + i * blocks_per_task * self.token_database.block_len[1]
                   for i in range(num_transfer_tasks)]
        addrs = k_addrs + v_addrs

        k_sizes = [blocks_per_task * self.token_database.block_len[0] - 1 for _ in range(num_transfer_tasks)]
        v_sizes = [blocks_per_task * self.token_database.block_len[1] - 1 for _ in range(num_transfer_tasks)]
        sizes = k_sizes + v_sizes

        req_meta.gvas = gvas
        req_meta.addr_list = addrs
        req_meta.size_list = sizes

    def _get_load_spec_cache_key(self, load_spec):
        """
        Generate a cache key for LoadSpec object.
        """
        if load_spec is None:
            return None
        return (
            load_spec.vllm_cached_tokens,
            load_spec.kvpool_cached_tokens,
            load_spec.can_load
        )

    def _get_cache_key(self, request: ReqMeta):
        """
        Generate a cache key for the process_layer_data method.
        """
        # 将block_hashes转换为元组以便作为字典键
        block_hashes_tuple = tuple(h.hex() if hasattr(h, 'hex') else h for h in request.block_hashes)
        load_spec_key = self._get_load_spec_cache_key(request.load_spec)
        return (
            request.req_id,
            block_hashes_tuple,
            request.token_len_chunk,
            request.can_save,
            load_spec_key,
            request.is_last_chunk
        )

    def _cleanup_cache(self):
        """
        Cleanup the cache when it exceeds the maximum size.
        Implements LRU (Least Recently Used) eviction policy.
        """
        if len(self.process_layer_cache) > self.cache_max_size:
            # 移除最旧的缓存项
            oldest_key, _ = self.cache_lru.popitem(last=False)
            # 从缓存中移除
            del self.process_layer_cache[oldest_key]
            # 从缓存索引中移除
            req_id = oldest_key[0]
            if req_id in self.cache_by_req_id:
                self.cache_by_req_id[req_id].discard(oldest_key)
                # 如果该req_id没有缓存项了，移除该req_id的索引
                if not self.cache_by_req_id[req_id]:
                    del self.cache_by_req_id[req_id]

    def process_layer_data(self, request: ReqMeta) -> Generator[
        Optional[torch.Tensor], None, None]:
        """
        A more efficient version of the layer-wise KV cache retrieval or storage.
        Implements incremental computation: only processes new blocks when they appear.

        :param request: The request containing meta information about the tokens and blocks.

        :return: A generator that yields either None (for store) or a tensor (for retrieve).
        """
        req_id = request.req_id
        current_block_hashes = request.block_hashes
        current_token_len = request.token_len_chunk
        num_new_offload_blocks = request.num_new_offload_blocks
        block_ids_npu = request.block_ids
        block_ids_cpu = request.block_ids_cpu
        if len(block_ids_npu) > len(block_ids_cpu):
            # in most cases block_ids_npu has one more unfull block, remove it
            block_ids_npu = block_ids_npu[:-1]
        assert len(block_ids_npu) == len(block_ids_cpu)
        block_ids_npu = block_ids_npu[-num_new_offload_blocks:]
        block_ids_cpu = block_ids_cpu[-num_new_offload_blocks:]

        for layer_id in range(self.num_layers):
            req_meta_save = LasyerMultiBlockReqMeta(
                request.req_id, [], [], [],
                request.block_ids, layer_id, request.is_last_chunk,
                block_ids_npu=block_ids_npu,
                block_ids_cpu=block_ids_cpu,
                cache_npu=(self.k_caches_npu[layer_id], self.v_caches_npu[layer_id]),
                cache_cpu=(self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]),
            )
            self.layer_save_tasks[layer_id].append(req_meta_save)
        return

        # mmc based offload meta
        # no cache
        starts, ends, keys = [], [], []
        for start, end, key in self.token_database.process_tokens(
            current_token_len,
            # current_block_hashes,
            current_block_hashes[-num_new_offload_blocks:],
            num_computed_blocks=len(current_block_hashes) - num_new_offload_blocks,
        ):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)  # [block_num, layer_num]
        self._build_layer_tasks(request, keys, starts, ends, unfull_key_split=None)

        return

        # 生成缓存键
        cache_key = self._get_cache_key(request)

        # 检查缓存中是否存在结果
        if cache_key in self.process_layer_cache:
            # 更新缓存项的使用时间（移到LRU列表末尾）
            self.cache_lru.move_to_end(cache_key)
            # 从缓存中获取结果
            keys, starts, ends, unfull_key_split = self.process_layer_cache[cache_key]
        else:
            # 检查请求历史记录，判断是否有新增的block
            has_new_block = False
            new_block_hashes = None
            new_token_start = None

            if req_id in self.request_history:
                prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split = \
                self.request_history[req_id]

                # 判断是否有新的block：block数量增加或token长度超过之前的block容量
                if len(current_block_hashes) > len(prev_block_hashes):
                    has_new_block = True
                    # 新block的信息
                    new_block_hashes = current_block_hashes[-1:]
                    new_token_start = prev_token_len // self.block_size * self.block_size

            if has_new_block and new_block_hashes and new_token_start is not None:
                # 只处理新增的block
                # 获取之前的计算结果
                prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split = \
                self.request_history[req_id]

                new_start = prev_token_len // self.block_size * self.block_size
                new_end = current_token_len // self.block_size * self.block_size

                new_hash = new_block_hashes[0]

                if not isinstance(new_hash, str):
                    new_hash = new_hash.hex()

                # 只取新增的block（最后一个）
                new_key = self.token_database._make_key_by_hash(new_hash)
                keys_multi_layer = new_key.split_layers(self.num_layers)
                prev_starts.append(new_start)
                prev_ends.append(new_end)
                prev_keys.append(keys_multi_layer)  # [block_num, layer_num]
            else:
                # 没有新的block或第一次处理该请求，完整计算
                starts, ends, keys = [], [], []
                for start, end, key in self.token_database.process_tokens(
                        current_token_len, current_block_hashes):
                    keys_multi_layer = key.split_layers(self.num_layers)
                    starts.append(start)
                    ends.append(end)
                    keys.append(keys_multi_layer)  # [block_num, layer_num]

            # 计算unfull_key_split
            unfull_key_split = None
            need_unfull_key = (request.load_spec is not None and request.load_spec.can_load and
                               (current_token_len - 1) % self.block_size != 0)

            if need_unfull_key:
                unfull_key = PoolKey(self.metadata,
                                     f"{req_id}"
                                     f"_lastblock"
                                     )
                unfull_key_split = unfull_key.split_layers(self.num_layers)

            # 更新请求历史记录
            self.request_history[req_id] = (current_block_hashes, current_token_len, keys, starts, ends,
                                            unfull_key_split)
            # 更新缓存
            self.process_layer_cache[cache_key] = (keys, starts, ends, unfull_key_split)
            self.cache_lru[cache_key] = None  # 添加到LRU末尾
            # 更新缓存索引
            self.cache_by_req_id[req_id].add(cache_key)
            # 检查并清理缓存
            self._cleanup_cache()
        self._build_layer_tasks(request, keys, starts, ends, unfull_key_split)

    def _build_layer_tasks(self, request: ReqMeta, keys: list, starts: list, ends: list, unfull_key_split):
        if not keys:
            return
        # Only process further if keys are present
        # TODO 要缓存这里计算之后的key，性能会更好
        keys = [list(row) for row in zip(*keys)]

        for layer_id, keys_multi_chunk in enumerate(keys):
            # save
            can_save = request.can_save
            req_meta_save = None
            req_meta_load = None

            if can_save is not None and can_save:
                req_meta_save = LasyerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends,
                    request.block_ids, layer_id, request.is_last_chunk
                )

            # load
            # load_spec = request.load_spec
            # if load_spec is not None and load_spec.can_load:  # load =0
            #     # 计算token_len
            #     token_len = request.token_len_chunk
            #     if (token_len - 1) % self.block_size == 0:
            #         req_meta_load = LasyerMultiBlockReqMeta(
            #             request.req_id, keys_multi_chunk[:-1], starts[:-1], ends[:-1],
            #             request.block_ids, layer_id
            #         )
            #     else:
            #         # Use cached unfull_key_split to avoid repeated split_layers calls
            #         req_meta_load = LasyerMultiBlockReqMeta(
            #             request.req_id, keys_multi_chunk[:-1] +
            #                             [unfull_key_split[layer_id]], starts, ends,
            #             request.block_ids, layer_id
            #         )

            if req_meta_save is not None:
                self._prepare_req_meta_data(req_meta_save, starts, ends, request, layer_id)
                self.layer_save_tasks[layer_id].append(req_meta_save)

            # if req_meta_load is not None:
            #     # For load, we might need to adjust starts and ends if we're using a subset
            #     token_len = request.token_len_chunk
            #     load_starts = starts[:-1] if (token_len - 1) % self.block_size == 0 else starts
            #     load_ends = ends[:-1] if (token_len - 1) % self.block_size == 0 else ends
            #     self._prepare_req_meta_data(req_meta_load, load_starts, load_ends, request, layer_id)
            #     self.layer_load_tasks[layer_id].append(req_meta_load)

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests(
            )
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.load_async
            else set()
        )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        for req_id in finished_req_ids:
            # 清理请求历史记录
            self.request_history.pop(req_id, None)
            # 清理相关缓存 - 使用缓存索引避免遍历所有缓存
            if req_id in self.cache_by_req_id:
                for cache_key in self.cache_by_req_id[req_id]:
                    self.process_layer_cache.pop(cache_key, None)
                    self.cache_lru.pop(cache_key, None)  # O(1)操作
                # 移除req_id的缓存索引
                del self.cache_by_req_id[req_id]
        return done_sending, done_recving

    def get_and_clear_finished_requests(self, finished_req_ids, meta: AscendConnectorMetadata) -> set[str]:
        finished_sending = set()
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                req_id
            )
        for req_id in self.kv_send_thread.stored_requests.copy(  # type: ignore[union-attr]
        ):
            if (
                self.kv_send_thread.stored_requests[  # type: ignore[union-attr]
                    req_id
                ]
                == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(  # type: ignore[union-attr]
                req_id
            )
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            res = self.m_store.exists(keys)  # type: ignore[assignment]

            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return starts[index]
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return end

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
                for item in keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1
                    )
                    multi_tp_keys.append(new_str)

            for i in range(1, self.pp_size):
                for item in keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@pp_rank:0", f"@pp_rank:{i}", 1
                    )
                    multi_tp_keys.append(new_str)

            res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
            num_block = len(keys)
            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
                num_block = len(keys) // self.num_layers
            multi_tp_values = [
                res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                for i in range(min(self.tp_size, self.num_kv_head) * self.pp_size)
            ]
            index = self.find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return end

    def check_all_layers_exists(self, res: list[int], num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    def find_min_first_non_one_index(self, arr):
        try:
            return min(idx for row in arr for idx, val in enumerate(row) if val != 1)
        except ValueError:
            return -1

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []
