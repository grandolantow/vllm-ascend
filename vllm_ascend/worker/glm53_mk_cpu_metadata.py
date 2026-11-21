"""Read steady TP8 MK cache addresses from VA's CPU block tables."""


_RATIO = 4


class UnsupportedMetadata(Exception):
    pass


def _pages(table, count):
    if not 0 <= count <= int(table.num_blocks_per_row[0]):
        raise UnsupportedMetadata("insufficient active block table columns")
    return tuple(table.get_numpy_array()[0, :count].tolist())


def _slot(pages, position, width):
    column, row = divmod(position, width)
    return pages[column] * width + row


class CpuDecodeMetadata:
    def __init__(self, runner):
        self.groups = runner.attn_groups
        self.block_tables = runner.input_batch.block_table
        self.by_layer = {}
        for gid, groups in enumerate(self.groups):
            table = self.block_tables[gid]
            for group in groups:
                entry = (table, group.get_metadata_builder(0))
                for name in group.layer_names:
                    if name in self.by_layer:
                        raise UnsupportedMetadata(f"duplicate attention group for {name}")
                    self.by_layer[name] = entry

    def read(self, runner, metadata, position, kda, mla, indexer):
        mode = runner.cache_config.mamba_cache_mode
        if (runner.attn_groups is not self.groups or runner.input_batch.block_table is not self.block_tables
                or mode not in {"none", "all", "align"} or runner.input_batch.num_reqs != 1
                or int(runner.input_batch.num_computed_tokens_cpu[0]) != position
                or runner.use_dcp or runner.use_async_scheduling or runner.use_async_spec_decode
                or runner.cache_config.enable_prefix_caching):
            return None
        if not isinstance(metadata, dict):
            return None
        store = _Prefetch(metadata, self.by_layer, position)
        try:
            for cache in kda.values():
                store.kda(cache.prefix, mode)
            for cache in mla.values():
                store.mla(cache.prefix)
            for cache in indexer.values():
                store.pool(cache.pool_prefix, cache.pool.shape[1])
                store.tail(cache.tail_prefix, cache.tail.stride(0) // cache.tail.shape[-1])
        except (AttributeError, IndexError, KeyError, UnsupportedMetadata):
            return None
        return store.reads


class _Prefetch:
    def __init__(self, metadata, by_layer, position):
        self.metadata = metadata
        self.by_layer = by_layer
        self.position = position
        self.reads = {}
        self.seen = set()

    def source(self, name, kind, shape=None):
        meta = self.metadata[name]
        table, builder = self.by_layer[name]
        key = (id(meta), id(table), id(builder), kind, shape)
        if key in self.seen:
            return None
        self.seen.add(key)
        return meta, table, builder

    def put(self, tensor, values, *, row=False):
        key = (id(tensor), len(values) if row else None)
        previous = self.reads.setdefault(key, (tensor, values))
        if previous[1] != values:
            raise UnsupportedMetadata("one device tensor maps to different CPU metadata")

    def kda(self, prefix, mode):
        source = self.source(prefix, "kda")
        if source is None:
            return
        meta, table, builder = source
        column = self.position // builder.kv_cache_spec.block_size if mode == "align" else 0
        slot = _pages(table, column + 1)[column]
        self.put(meta.non_spec_query_start_loc, (0, 1))
        self.put(meta.non_spec_state_indices_tensor, (slot,))

    def mla(self, prefix):
        source = self.source(prefix, "mla")
        if source is None:
            return
        meta, table, builder = source
        width = builder.kernel_block_size
        if width != meta.block_size:
            raise UnsupportedMetadata("inconsistent MLA block widths")
        pages = _pages(table, self.position // width + 1)
        self.put(meta.block_table, pages, row=True)
        self.put(meta.query_start_loc, (0, 1))
        self.put(meta.seq_lens, (self.position + 1,))
        self.put(meta.positions, (self.position,))
        self.put(meta.slot_mapping, (_slot(pages, self.position, width),))

    def pool(self, prefix, rows):
        source = self.source(prefix, "pool", rows)
        if source is None:
            return
        meta, table, builder = source
        pools, tail = divmod(self.position + 1, _RATIO)
        if builder.compress_ratio != _RATIO or rows <= 0:
            raise UnsupportedMetadata("unsupported Indexer pool geometry")
        pages = _pages(table, (pools + rows - 1) // rows)
        if tail:
            slot = -1
        else:
            raw_width = table.block_size
            column = 0 if table.is_circular else self.position // raw_width
            raw_slot = _pages(table, column + 1)[column] * raw_width + self.position % raw_width
            width = builder.logical_block_size
            slot = raw_slot // width * (width // _RATIO) + raw_slot % width // _RATIO
            if slot != _slot(pages, pools - 1, rows):
                raise UnsupportedMetadata("Indexer compact and cache row addresses disagree")
        self.put(meta.block_table, pages, row=True)
        self.put(meta.query_start_loc, (0, 1))
        self.put(meta.positions, (self.position,))
        self.put(meta.start_pos, (self.position,))
        self.put(meta.raw_seq_lens, (self.position + 1,))
        self.put(meta.seq_lens, (pools,))
        self.put(meta.pool_tail, (tail,))
        self.put(meta.slot_mapping, (slot,))
        self.put(meta.pooled_slot_mapping, (slot,))

    def tail(self, prefix, stride):
        source = self.source(prefix, "tail", stride)
        if source is None:
            return
        meta, table, builder = source
        if stride != builder._page_rows or stride < _RATIO:
            raise UnsupportedMetadata("inconsistent Indexer tail page stride")
        pages = _pages(table, 1)
        base = pages[0] * stride
        open_rows = (self.position + 1) % _RATIO
        self.put(meta.block_table, pages, row=True)
        self.put(meta.cann_inputs.history_indices, tuple(base + row for row in range(_RATIO)), row=True)
        self.put(meta.cann_inputs.write_indices,
                 tuple(base + row if row < open_rows else -1 for row in range(_RATIO)), row=True)
