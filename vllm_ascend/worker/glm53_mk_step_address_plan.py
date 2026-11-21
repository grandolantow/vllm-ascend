"""Rank-local CPU cache addresses for one already-established TP8 MK replay.

Call after VA has updated this schedule's batch and block tables and before
forward_round consumes its previous completion. None selects this rank's normal
metadata builder. A plan does not authorize a launch or a native fallback.
The caller owns eager/connector/offload/capture eligibility and the bridge's
common tensor layout, device and lifetime checks.
"""

from dataclasses import dataclass
from operator import index


class UnsupportedAddressPlan(ValueError):
    """The available CPU data cannot prove an address for this step."""


def _require(condition, reason):
    if not condition:
        raise UnsupportedAddressPlan(reason)


@dataclass(frozen=True)
class StepAddressPlan:
    request_id: str
    position: int
    kda_slots: dict
    mla_pages: dict
    idx_addresses: dict


@dataclass(frozen=True)
class _Binding:
    gid: int
    group: object
    builder: object
    names: tuple
    table: object
    cpu: object
    gpu: object
    slots: object


def _owners(runner):
    groups = runner.attn_groups
    tables = runner.input_batch.block_table
    for gid, members in enumerate(groups):
        table = tables[gid]
        for group in members:
            yield _Binding(gid, group, group.get_metadata_builder(0), tuple(group.layer_names),
                           table, table.get_numpy_array(), table.block_table.gpu, table.slot_mapping.gpu)


def _production_types():
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadataBuilder,
        AscendIndexerKPoolTailMetadataBuilder,
    )
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNHostMetadataBuilder,
    )
    from vllm_ascend.worker.va_mk_bridge import _IndexerAddress, _NativePages

    # The Host builder only changes chunk metadata for prefill. Steady
    # non-spec decode inherits the same column-zero state index.
    gdn_builders = (AscendGDNHostMetadataBuilder,)
    return ((gdn_builders, AscendSFAMetadataBuilder,
             AscendIndexerKPoolMetadataBuilder, AscendIndexerKPoolTailMetadataBuilder),
            _NativePages, _IndexerAddress)


class _Tables:
    """Own fresh row tuples for this call, shared only by identical sources."""

    def __init__(self, page_type):
        self.page_type = page_type
        self.rows = {}
        self.page_sets = {}

    def row(self, binding, columns):
        key = (id(binding.table), columns)
        if key not in self.rows:
            cpu = binding.cpu
            active = index(binding.table.num_blocks_per_row[0])
            _require(cpu.ndim == 2 and cpu.shape[0] >= 1 and cpu.dtype.kind in ('i', 'u')
                     and 0 <= columns <= active <= cpu.shape[1], "invalid active CPU block-table row")
            # get_numpy_array() is the post-expansion H2D source. Applying
            # blocks_per_phys_block again would double-expand physical IDs.
            self.rows[key] = tuple(cpu[0, :columns].tolist())
        return self.rows[key]

    def pages(self, binding, count, width, capacity, prev):
        _require(width > 0 and capacity > 0 and prev.rows_per_id == width
                 and prev.capacity == capacity, "page geometry changed since the committed step")
        columns = (count + width - 1) // width
        key = (id(binding.table), columns, width, capacity, id(prev))
        if key not in self.page_sets:
            ids = self.row(binding, columns)
            old = prev.ids
            if ids == old:
                self.page_sets[key] = prev
            else:
                _require(len(ids) >= len(old) and ids[:len(old)] == old, "committed page prefix changed")
                _require(all(0 <= p < capacity for p in ids) and len(set(ids)) == len(ids),
                         "out-of-range or duplicate active page")
                self.page_sets[key] = self.page_type(ids, width, capacity)
        return self.page_sets[key]


def early_indexer_cache(layer):
    """Resolve an unambiguous engine-zero binding before VA enters ForwardContext."""
    from torch import Tensor

    cache = layer.kv_cache
    if isinstance(cache, list):
        _require(len(cache) == 1, "early KPool lookup requires one virtual engine")
        cache = cache[0]
    if isinstance(cache, (list, tuple)):
        _require(len(cache) == 1, "early KPool lookup requires one key tensor")
        cache = cache[0]
    _require(isinstance(cache, Tensor), "early KPool tensor is not bound")
    return cache


class CpuStepAddressBuilder:
    """Bind reviewed builder types to the actual worker's cache groups.

    Inject builder types for each cache kind, native pages type, and native
    indexer address type for CPU-only tests. Production resolves its own VA types.
    """

    def __init__(self, runner, *, types=None):
        self.bindings = ()
        self.by_name = {}
        self.unavailable_reason = None
        try:
            kinds, self.page_type, self.index_type = types or _production_types()
            self.kinds = tuple(kind if isinstance(kind, tuple) else (kind,) for kind in kinds)
            reviewed_builders = set().union(*self.kinds)
            self.groups = runner.attn_groups
            self.tables = runner.input_batch.block_table
            bindings = tuple(_owners(runner))
            for binding in bindings:
                _require(type(binding.builder) in reviewed_builders,
                         f"unreviewed attention builder {type(binding.builder).__qualname__}")
                for name in binding.names:
                    _require(name not in self.by_name, "duplicate attention layer registration")
                    self.by_name[name] = binding
            self.bindings = bindings
        except (AttributeError, IndexError, KeyError, TypeError, ImportError, UnsupportedAddressPlan) as exc:
            self.by_name.clear()
            self.unavailable_reason = str(exc)

    def _check_bindings(self, runner):
        _require(self.bindings, self.unavailable_reason or "no attention groups")
        _require(runner.attn_groups is self.groups
                 and runner.input_batch.block_table is self.tables, "unbound or replaced attention groups")
        i = 0
        for gid, members in enumerate(self.groups):
            table = self.tables[gid]
            for group in members:
                b = self.bindings[i]
                _require(b.gid == gid and group is b.group and group.get_metadata_builder(0) is b.builder
                         and tuple(group.layer_names) == b.names and table is b.table
                         and table.get_numpy_array() is b.cpu and table.block_table.gpu is b.gpu
                         and table.slot_mapping.gpu is b.slots, "changed attention group backing")
                i += 1
        _require(i == len(self.bindings), "removed attention groups")

    def _binding(self, name, kind):
        b = self.by_name[name]
        _require(type(b.builder) in self.kinds[kind], "unexpected metadata builder for cache")
        return b

    def read(self, runner, position, kda, mla, indexer, *, scheduler_output=None):
        """Return current addresses or None, without device reads or state writes.

        Pass scheduler_output when available. Without it the caller must already
        have run update_requests and checked a one-token text decode schedule.
        """
        try:
            return self.read_checked(runner, position, kda, mla, indexer, scheduler_output=scheduler_output)
        except UnsupportedAddressPlan:
            return None

    def read_checked(self, runner, position, kda, mla, indexer, *, scheduler_output=None):
        """Like read, but expose the precise rank-local miss for opt-in diagnostics."""
        try:
            return self._read(runner, position, kda, mla, indexer, scheduler_output)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise UnsupportedAddressPlan(str(exc)) from exc

    def _read(self, runner, p, kda, mla, indexer, out):
        hook, batch = runner._glm53_mk_ep, runner.input_batch
        bridge = hook.bridge
        rid = batch.req_ids[0]
        _require(type(p) is int and 0 < p < bridge.max_context_tokens and batch.num_reqs == 1
                 and not runner.use_dcp and not runner.use_async_scheduling and not runner.use_async_spec_decode
                 and runner.speculative_config is None and not runner.cache_config.enable_prefix_caching
                 and runner.cache_config.mamba_cache_mode in ("all", "none")
                 and hook.tp_size == 8 and hook.plan.size == 8 and hook.plan.depth == 45
                 and not hook.invalid and not bridge._poisoned and hook.active is None
                 and hook.verified_graph is not None and bridge._graph is hook.verified_graph
                 and hook._last_completed_round == (rid, p - 1)
                 and (hook.request_id, hook.next_position) == (rid, p)
                 and (bridge._request, bridge._position) == (rid, p - 1)
                 and int(batch.num_computed_tokens_cpu[0]) == p and p >= int(batch.num_prompt_tokens[0])
                 and int(runner.optimistic_seq_lens_cpu[0]) == p + 1, "not a committed single-token continuation")
        if out is not None:
            _require(out.total_num_scheduled_tokens == 1 and out.num_scheduled_tokens == {rid: 1}
                     and not out.scheduled_spec_decode_tokens and not out.scheduled_new_reqs
                     and rid not in out.finished_req_ids and rid not in out.scheduled_cached_reqs.resumed_req_ids
                     and rid not in (getattr(out, "preempted_req_ids", None) or ())
                     and not getattr(out, "has_sync_kv_loads", False), "new or interrupted schedule")
        q = runner.query_start_loc.np
        _require((q[0], q[1]) == (0, 1), "unexpected query boundaries")
        if runner._has_gdn:
            q = runner.gdn_query_start_loc.np
            _require((q[0], q[1]) == (0, 1), "unexpected GDN query boundaries")
        self._check_bindings(runner)
        old_slots, _, old_indices = bridge._locations
        _require(kda.keys() == old_slots.keys() and mla.keys() == indexer.keys() == bridge._native_pages.keys()
                 and mla.keys() == old_indices.keys(), "missing cache descriptors")

        data = _Tables(self.page_type)
        slots, mla_pages, indices = {}, {}, {}
        for layer, cache in kda.items():
            b = self._binding(cache.prefix, 0)
            slot, = data.row(b, 1)
            _require(0 <= slot < min(cache.conv.shape[0], cache.recurrent.shape[0])
                     and slot == old_slots[layer][0], "invalid or changed KDA slot")
            slots[layer] = slot
        for layer, cache in mla.items():
            b = self._binding(cache.prefix, 1)
            sfa = b.builder
            width = sfa.kernel_block_size
            state = sfa.nope_states[None]
            rows = cache.latent.shape[1]
            factor = b.table.blocks_per_phys_block
            _require(sfa.nope and not sfa.use_pcp and sfa.speculative_config is None
                     and state.table_stride == 1 and state.block_size == width and width > 0
                     and not b.table.is_circular and b.table.block_size == width
                     and factor > 0 and b.table.physical_block_size == width * factor
                     and state.split > 0 and b.cpu.shape[1] % state.split == 0
                     and b.cpu.shape[1] <= state.block_table_buffer.shape[1]
                     and rows >= width and rows % width == 0, "unverified sparse MLA table view")
            prev_mla, prev_pool = bridge._native_pages[layer]
            mla_pages[layer] = data.pages(b, p + 1, width, cache.latent.shape[0] * (rows // width), prev_mla)
            idx = indexer[layer]
            pb = self._binding(idx.pool_prefix, 2)
            tb = self._binding(idx.tail_prefix, 3)
            indices[layer] = self._indexer(data, p, idx, pb, tb, prev_pool, old_indices[layer][0])
        return StepAddressPlan(rid, p, slots, mla_pages, indices)

    def _indexer(self, data, p, cache, pb, tb, previous, old_tail_page):
        pool, tail = pb.builder, tb.builder
        rows = cache.pool.shape[1]
        logical = pool.logical_block_size
        _require(pool.pcp_size == 1 and pool.kernel_blocks_per_logical_block == logical // 128 and pool.storage_block_size > 0
                 and pool.compress_ratio == 4 and logical > 0 and logical % 128 == 0
                 and pool.kernel_row_block_size == 32 and rows > 0
                 and not pb.table.is_circular, "unsupported KPool geometry")
        pools, open_rows = divmod(p + 1, 4)
        pages = data.pages(pb, pools, rows, cache.pool.shape[0], previous)
        pool_page = pool_row = None
        if not open_rows:
            raw_width = pb.table.block_size
            _require(raw_width > 0, "invalid KPool token page width")
            column = p // raw_width
            raw_id = data.row(pb, column + 1)[column]
            raw = raw_id * raw_width + p % raw_width
            slot = raw // logical * (logical // 4) + raw % logical // 4
            _require(raw_id >= 0 and slot == pages.slot(pools - 1), "inconsistent KPool write slot")
            pool_page, pool_row = divmod(slot, rows)
        tail_page, = data.row(tb, 1)
        _require(0 <= tail_page < cache.tail.shape[0] and tail_page == old_tail_page
                 and cache.tail.shape[-1] == 256 and cache.tail.shape[1] == tail.block_size == tail._pool_size == 4
                 and not tail._lookahead and tail._page_rows >= 4
                 and cache.tail.stride(0) == 256 * tail._page_rows, "invalid or changed KPool tail binding")
        # VA history and writes use page * physical stride plus compact ring
        # row. The bridge uses the page and relative rows separately.
        history = tuple(range(p % 4))
        last = open_rows - 1 if open_rows else None
        return self.index_type(pool_page, pool_row, tail_page, history, last, pages)
