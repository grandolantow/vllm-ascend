"""Batch the TP8 KDA state writes to native VA storage on the current stream."""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _copy_convs(rows, row_count: tl.constexpr, transposed: tl.constexpr, block: tl.constexpr):
    layer = tl.program_id(0)
    if layer < row_count:
        src = tl.load(rows + layer * 4).to(tl.pointer_type(tl.uint16))
        dst = tl.load(rows + layer * 4 + 1).to(tl.pointer_type(tl.uint16))
        for start in range(0, 9216, block):
            idx = start + tl.arange(0, block)
            value = tl.load(src + idx, idx < 9216, other=0)
            if transposed:
                dst_index = (idx % 3072) * 3 + idx // 3072
            else:
                dst_index = idx
            tl.store(dst + dst_index, value, idx < 9216)


@triton.jit
def _transpose_recurrents(rows, row_count: tl.constexpr, block: tl.constexpr):
    tiles_per_head: tl.constexpr = (128 // block) ** 2
    tiles_per_layer: tl.constexpr = 8 * tiles_per_head
    for tile in range(tl.program_id(0), row_count * tiles_per_layer, tl.num_programs(0)):
        layer = tile // tiles_per_layer
        head = tile // tiles_per_head % 8
        row = tile // (128 // block) % (128 // block) * block + tl.arange(0, block)
        col = tile % (128 // block) * block + tl.arange(0, block)
        src = tl.load(rows + layer * 4 + 2).to(tl.pointer_type(tl.uint32))
        dst = tl.load(rows + layer * 4 + 3).to(tl.pointer_type(tl.uint32))
        value = tl.load(src + head * 16384 + row[:, None] * 128 + col[None, :])
        tl.store(dst + head * 16384 + col[:, None] * 128 + row[None, :], tl.trans(value))


class KDAWriteback:
    @classmethod
    def create(cls, sources, targets):
        if not sources or sources.keys() != targets.keys():
            return None
        pairs = tuple((*sources[layer], *targets[layer]) for layer in sources)
        device = pairs[0][0].device
        transposed = pairs[0][2].stride() == (1, 3)
        conv_stride = (1, 3) if transposed else (3072, 1)
        addresses = []
        memory = []
        for src_conv, src_rec, dst_conv, dst_rec in pairs:
            convs = (src_conv, dst_conv)
            recs = (src_rec, dst_rec)
            if (any(t.device != device for t in (*convs, *recs)) or device.type != "npu"
                    or any(t.shape != (3, 3072) or t.dtype != torch.bfloat16 for t in convs)
                    or any(t.shape != (8, 128, 128) or t.dtype != torch.float32 for t in recs)
                    or src_conv.stride() != (3072, 1) or dst_conv.stride() != conv_stride
                    or src_rec.stride() != (16384, 128, 1) or dst_rec.stride() != (16384, 1, 128)):
                return None
            values = (*convs, *recs)
            if any(int(torch_npu.get_npu_format(tensor)) not in (0, 2) for tensor in values):
                return None
            row = [tensor.data_ptr() for tensor in values]
            if any(ptr == 0 or ptr % 32 for ptr in row):
                return None
            addresses.append(row)
            memory.extend((ptr, ptr + tensor.numel() * tensor.element_size()) for ptr, tensor in zip(row, values))
        memory.sort()
        if any(end > start for (_, end), (start, _) in zip(memory, memory[1:])):
            return None
        return cls(torch.tensor(addresses, dtype=torch.int64, device=device), pairs, transposed)

    def __init__(self, rows, pairs, transposed):
        self._rows = rows
        # Storage is addressed by the device table. Hold these views until stream completion.
        self._pairs = pairs
        self._transposed = transposed

    def run(self):
        count = len(self._pairs)
        _copy_convs[(count,)](self._rows, count, self._transposed, 512)
        _transpose_recurrents[(min(64, 32 * count),)](self._rows, count, 64)
