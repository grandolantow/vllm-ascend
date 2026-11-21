"""Commit one MLA and Indexer row per TP8 attention layer."""

import logging

import torch
import torch_npu
import triton
import triton.language as tl


_LOG = logging.getLogger("vllm.ascend.mha_row_writeback")


@triton.jit
def _commit_rows(addresses, pool_boundary: tl.constexpr):
    layer = tl.program_id(0)
    src_mla = tl.load(addresses + layer * 5).to(tl.pointer_type(tl.uint16))
    dst_mla = tl.load(addresses + layer * 5 + 1).to(tl.pointer_type(tl.uint16))
    row = tl.arange(0, 512)
    tl.store(dst_mla + row, tl.load(src_mla + row))

    row = tl.arange(0, 128)
    dst_indexer = tl.load(addresses + layer * 5 + 3)
    if pool_boundary:
        src_pool = tl.load(addresses + layer * 5 + 2).to(tl.pointer_type(tl.uint16))
        tl.store(dst_indexer.to(tl.pointer_type(tl.uint16)) + row, tl.load(src_pool + row))
    else:
        key = tl.load(addresses + layer * 5 + 2).to(tl.pointer_type(tl.bfloat16))
        score = tl.load(addresses + layer * 5 + 4).to(tl.pointer_type(tl.uint32))
        dst_tail = dst_indexer.to(tl.pointer_type(tl.uint32))
        tl.store(dst_tail + row, tl.load(key + row).to(tl.float32).to(tl.uint32, bitcast=True))
        tl.store(dst_tail + 128 + row, tl.load(score + row))


def _row_binding(tensor, width, dtype, device):
    if (tensor.device != device or tensor.dtype != dtype
            or not (tensor.ndim == 3 or tensor.ndim == 4 and tensor.shape[2] == 1)
            or tensor.shape[-1] != width or tensor.shape[0] < 1 or tensor.shape[1] < 1
            or tensor.stride(-1) != 1 or tensor.stride(1) != width
            or tensor.stride(0) < tensor.shape[1] * width):
        return None
    if int(torch_npu.get_npu_format(tensor)) not in (0, 2):
        return None
    elem = tensor.element_size()
    start = tensor.data_ptr()
    page_bytes = tensor.stride(0) * elem
    row_bytes = width * elem
    if not start or start % 32 or page_bytes % 32:
        return None
    end = start + (tensor.shape[0] - 1) * page_bytes + tensor.shape[1] * row_bytes
    return (start, end), (start, page_bytes, row_bytes, tensor.shape[1])


def _overlap(regions):
    ordered = sorted(regions)
    return any(end > start for (_, end), (start, _) in zip(ordered, ordered[1:]))


class AttentionWriteback:
    @classmethod
    def create(cls, source_mla, source_pool, keys, scores, mla, indexer):
        values = tuple((source_mla[layer], mla[layer].latent, source_pool[layer],
                        indexer[layer].pool, indexer[layer].tail, keys[layer], scores[layer])
                       for layer in mla)
        if not values:
            return None
        device = values[0][0].device
        if device.type != "npu":
            return None
        bf16, fp32 = torch.bfloat16, torch.float32
        shapes = ((512, bf16), (512, bf16), (128, bf16), (128, bf16),
                  (256, fp32), (128, bf16), (128, fp32))
        kinds = ("MK MLA", "VA MLA", "MK KPool", "VA KPool", "VA tail", "MK tail key", "MK tail score")
        layouts = []
        regions = []
        for layer, tensors in zip(mla, values):
            rows = []
            for kind, tensor, (width, dtype) in zip(kinds, tensors, shapes):
                entry = _row_binding(tensor, width, dtype, device)
                if entry is None:
                    layout_format = torch_npu.get_npu_format(tensor) if tensor.device.type == "npu" else None
                    _LOG.warning("MK attention writeback unsupported layer=%s kind=%s shape=%s stride=%s dtype=%s "
                                 "device=%s format=%s", layer, kind, tuple(tensor.shape), tensor.stride(),
                                 tensor.dtype, tensor.device, layout_format)
                    return None
                region, row = entry
                regions.append(region)
                rows.append(row)
            layouts.append(tuple(rows))
        pinned = torch.empty((len(values), 5), dtype=torch.int64, device="cpu", pin_memory=True)
        addresses = torch.empty_like(pinned, device=device, pin_memory=False)
        return cls(tuple(mla), tuple(layouts), values, pinned, addresses, _overlap(regions))

    def __init__(self, layers, layouts, tensors, pinned, addresses, shared_backing):
        self._layers = layers
        self._layouts = layouts
        self._tensors = tensors  # Device address table does not own the cache storage.
        self._pinned = pinned
        self._host = pinned.numpy()
        self._addresses = addresses
        self._shared_backing = shared_backing
        self._reported_overlap = False
        self._reported_submit = False

    def run(self, position, page_table, mla_pages, indexer_addresses):
        # EP synchronizes cache writeback before reusing the pinned table.
        mk_page, mk_row = divmod(position, 64)
        tail_row = position % 4
        pool_boundary = tail_row == 3
        regions = [] if self._shared_backing else None
        if pool_boundary:
            pool_page, pool_row = divmod(position // 4, 64)
        for i, layer in enumerate(self._layers):
            src, dst, src_pool, dst_pool, tail, key, score = self._layouts[i]
            page, row = divmod(mla_pages[layer].slot(position), dst[3])
            src_ptr = src[0] + page_table[mk_page] * src[1] + mk_row * src[2]
            dst_ptr = dst[0] + page * dst[1] + row * dst[2]
            address = indexer_addresses[layer]
            if pool_boundary:
                a = src_pool[0] + page_table[4 * pool_page] * src_pool[1] + pool_row * src_pool[2]
                b = dst_pool[0] + address.pool_page * dst_pool[1] + address.pool_row * dst_pool[2]
                c = 0
            else:
                a = key[0] + tail_row * key[2]
                b = tail[0] + address.tail_page * tail[1] + address.tail_row * tail[2]
                c = score[0] + tail_row * score[2]
            self._host[i] = (src_ptr, dst_ptr, a, b, c)
            if regions is not None:
                regions.extend(((src_ptr, src_ptr + src[2]), (dst_ptr, dst_ptr + dst[2])))
                if pool_boundary:
                    regions.extend(((a, a + src_pool[2]), (b, b + dst_pool[2])))
                else:
                    regions.extend(((a, a + key[2]), (b, b + tail[2]), (c, c + score[2])))
        # VA can expose overlapping storage views for independent scheduler
        # groups. Only launch the parallel copy when this step's rows are disjoint.
        if regions is not None and _overlap(regions):
            if not self._reported_overlap:
                _LOG.warning("MK attention writeback rejected overlapping active rows at position=%d", position)
                self._reported_overlap = True
            return False
        self._addresses.copy_(self._pinned, non_blocking=True)
        _commit_rows[(len(self._layers),)](self._addresses, pool_boundary)
        if not self._reported_submit:
            _LOG.info("MK attention writeback submitted shared_backing=%s", self._shared_backing)
            self._reported_submit = True
        return True
