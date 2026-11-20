"""Single-request GLM-5.3-Flash VA prefill to MK decode cache handoff."""

import logging
from dataclasses import dataclass, field
from threading import Lock

import torch


_LOG = logging.getLogger(__name__)
_PAGE = 64
_SCRATCH_PAGE = 1
_RATIO = 4
_CONV = (3, 24576)
_RECURRENT = (64, 128, 128)
_LATENT = 512
_INDEX = 128


@dataclass(frozen=True)
class KDACache:
    prefix: str
    conv: torch.Tensor
    recurrent: torch.Tensor
    conv_dim_first: bool


@dataclass(frozen=True)
class MLACache:
    prefix: str
    latent: torch.Tensor


@dataclass(frozen=True)
class IndexerCache:
    pool_prefix: str
    tail_prefix: str
    pool: torch.Tensor
    tail: torch.Tensor


@dataclass(frozen=True)
class _NativePages:
    ids: tuple[int, ...]
    rows_per_id: int

    def slot(self, row):
        page, offset = divmod(row, self.rows_per_id)
        return self.ids[page] * self.rows_per_id + offset

    def extends(self, previous):
        return self.rows_per_id == previous.rows_per_id and self.ids[:len(previous.ids)] == previous.ids


@dataclass(frozen=True)
class _IndexerAddress:
    pool_page: int | None
    pool_row: int | None
    tail_page: int
    history_rows: tuple[int, ...]
    tail_row: int | None
    pages: _NativePages = field(repr=False)


@dataclass(frozen=True)
class DecodeResult:
    """The caller must consume these MK-owned tensors before the next step."""

    logits: torch.Tensor | None
    hidden_states: torch.Tensor
    imported: bool
    graph_reused: bool


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class DecodePageLayout:
    """Disjoint latent and KPool table rows indexed by persistent request slot."""

    def __init__(self, max_context_tokens, request_slots=1):
        _require(type(max_context_tokens) is int and max_context_tokens > 0
                 and type(request_slots) is int and 1 <= request_slots <= 128, "invalid decode page layout")
        width = (max_context_tokens + _PAGE - 1) // _PAGE
        self.num_pages = _SCRATCH_PAGE + request_slots * width
        _require(self.num_pages <= 4096, "MK bridge exceeds the 4096 physical page capacity including padding")
        self.tables = tuple(tuple(range(start, start + width))
                            for start in range(_SCRATCH_PAGE, self.num_pages, width))


def _metadata_values(tensor, length, field):
    _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.shape[0] == length
             and tensor.dtype in (torch.int32, torch.int64), f"invalid {field}")
    return tuple(tensor.tolist())


def _value(tensor, field):
    return _metadata_values(tensor, 1, field)[0]


def _active_pages(table, count, rows_per_id, total_rows):
    _require(type(rows_per_id) is int and rows_per_id > 0 and total_rows % rows_per_id == 0,
             "invalid VA block granularity")
    columns = (count + rows_per_id - 1) // rows_per_id
    _require(table.ndim == 2 and table.shape[0] == 1 and table.shape[1] >= columns
             and table.dtype in (torch.int32, torch.int64), "invalid VA block table")
    ids = tuple(table[0, :columns].tolist())
    _require(all(0 <= page < total_rows // rows_per_id for page in ids), "VA page is out of range")
    _require(len(set(ids)) == len(ids), "duplicate active VA page in one request")
    return _NativePages(ids, rows_per_id)


def _copy_prefix(source, native, target, count, mk_pages, mk_page_stride):
    row = 0
    while row < count:
        src_page, src_offset = divmod(native.slot(row), source.shape[1])
        mk_page, mk_offset = divmod(row, _PAGE)
        n = min(count - row, native.rows_per_id - row % native.rows_per_id,
                source.shape[1] - src_offset, _PAGE - mk_offset)
        target[mk_pages[mk_page * mk_page_stride], mk_offset:mk_offset + n].copy_(
            source[src_page, src_offset:src_offset + n])
        row += n


def _row_cache(tensor, inner_shape, dtype, device):
    _require(tensor.ndim == len(inner_shape) + 2 and tensor.shape[2:] == inner_shape
             and tensor.dtype == dtype and tensor.device == device, "invalid VA row cache layout")
    width = inner_shape[-1]
    _require(tensor.stride(-1) == 1 and tensor.stride(1) == width
             and tensor.stride(0) >= tensor.shape[1] * width, "invalid VA cache strides")


def _layout(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride())


class VAMKBridge:
    """Decode a single non-speculative request on the VA model stream.

    Pass the real per-layer tensors from VA's cache owners and the active
    ForwardContext before the native model has executed the decode token.
    The worker must serialize prefill, this call and sampling on the same
    stream, and select this path only after native prefill for the request.
    Device-produced metadata must be ready on that stream before the call.
    """

    def __init__(self, model, *, ep2=False, ep_size=None, max_context_tokens=_PAGE):
        args = model.model_args
        if ep_size is None:
            ep_size = 2 if ep2 else 1
        depths = {1: 4, 2: 8, 6: 45, 8: 45}
        _require(ep_size in depths and (not ep2 or ep_size == 2), "unsupported VA/MK bridge profile")
        tp_size, tp_rank = model.tp_size, model.tp_rank
        depth = args.num_hidden_layers
        shared_experts = getattr(model, "native_routed_layers", ())
        valid_profile = ((tp_size, tp_rank) == (1, 0) and depth == depths[ep_size]
                         or tp_size == ep_size and tp_size in (2, 8) and tp_rank == args.ep_rank
                         and (depth in (4, 8) or depth == 45 and shared_experts == tuple(range(3, 45))))
        _require(valid_profile and args.ep_size == ep_size
                 and 0 <= args.ep_rank < ep_size and args.attention_impl == "mk_decode"
                 and args.num_hidden_layers == depth
                 and tuple(args.linear_layer_ids) == tuple(i for i in range(depth) if i % 4 != 3)
                 and tuple(args.full_attention_layer_ids) == tuple(range(3, depth, 4))
                 and args.indexer_skip_rope and args.index_kpool == _RATIO,
                 "VA bridge requires a supported TP/EP profile with NoPE and KPool=4")
        self._conv_shape = (_CONV[0], _CONV[1] // tp_size)
        self._recurrent_shape = (_RECURRENT[0] // tp_size, *_RECURRENT[1:])
        self._model = model
        self._state = model._mk_decode_state
        _require(self._state is not None and self._state._shared_pools is None
                 and self._state.batch.index_cos_sin_cache.shape[0] == _PAGE,
                 "VA bridge requires unbound MK state with a 64-row NoPE table")
        self._device = self._state.positions.device
        layout = DecodePageLayout(max_context_tokens)
        if max_context_tokens > _PAGE:
            _require((tp_size, ep_size) in ((1, 6), (2, 2), (8, 8)),
                 "paged VA bridge requires TP1/EP6, TP2/EP2 or TP8/EP8")
        self.max_context_tokens = max_context_tokens
        num_pages = layout.num_pages
        self._page_table = layout.tables[0]
        with torch.inference_mode(False), torch.no_grad():
            self._token = torch.empty(1, dtype=torch.int64, device=self._device)
            self._mla = {layer: torch.zeros((num_pages, _PAGE, 1, _LATENT), dtype=torch.bfloat16, device=self._device)
                         for layer in args.full_attention_layer_ids}
            self._pool = {layer: torch.zeros((num_pages, _PAGE, 1, _INDEX), dtype=torch.bfloat16, device=self._device)
                          for layer in args.full_attention_layer_ids}
            self._keys = {layer: torch.empty((_RATIO, _INDEX), dtype=torch.bfloat16, device=self._device)
                          for layer in args.full_attention_layer_ids}
            self._gates = {layer: torch.empty((_RATIO, _INDEX), dtype=torch.float32, device=self._device)
                           for layer in args.full_attention_layer_ids}
        self._request = None
        self._position = None
        self._locations = None
        self._native_pages = None
        self._graph = None
        self._poisoned = False
        self._lock = Lock()

    def step(self, forward_context, *, request_id, position, input_ids, kda, mla, indexer):
        """Consume one scheduled token and commit it to the VA-owned cache.

        ``kda``, ``mla`` and ``indexer`` are mappings keyed by MK layer id of
        the public cache descriptors. Their prefixes resolve in
        ``forward_context.attn_metadata``. Position is zero-based and equals
        the number of already committed tokens. A new request imports VA's
        native cache, even when it reuses the previous request's pages.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("VA bridge only supports serial decode")
        try:
            with torch.inference_mode(False), torch.no_grad():
                return self._step(forward_context, request_id, position, input_ids, kda, mla, indexer)
        finally:
            self._lock.release()

    def release(self, request_id):
        """Forget a finished or interrupted request while retaining graph storage."""
        with self._lock:
            if self._request == request_id:
                self._request = self._position = self._locations = self._native_pages = None

    def _step(self, context, request_id, position, input_ids, kda, mla, indexer):
        if self._poisoned:
            raise RuntimeError("VA bridge stopped after an MK execution or writeback failure")
        runtime_mode = getattr(context, "cudagraph_runtime_mode", None)
        _require(not getattr(context, "capturing", False)
                 and (runtime_mode is None or getattr(runtime_mode, "name", None) == "NONE"),
                 "VA bridge requires an eager VA ForwardContext")
        _require(isinstance(request_id, str) and request_id and type(position) is int
                 and 0 < position < self.max_context_tokens,
                 "VA bridge requires a request id and a nonzero position within its context capacity")
        _require(input_ids.shape == (1,) and input_ids.dtype in (torch.int32, torch.int64)
                 and input_ids.device == self._device, "VA bridge requires one device token")
        _require(kda.keys() == self._state.kda.conv.keys() and mla.keys() == indexer.keys() == self._mla.keys(),
                 "VA caches must cover the MK layers")
        metadata = context.attn_metadata
        _require(isinstance(metadata, dict), "VA bridge requires a per-layer ForwardContext metadata dict")
        conv_states = {layer: self._kda_target(cache, metadata[cache.prefix]) for layer, cache in kda.items()}
        mla_pages = {layer: self._mla_target(cache, metadata[cache.prefix], position) for layer, cache in mla.items()}
        idx_addresses = {
            layer: self._indexer_target(cache, metadata[cache.pool_prefix], metadata[cache.tail_prefix], position)
            for layer, cache in indexer.items()
        }
        locations = (
            {layer: (slot, _layout(kda[layer].conv), _layout(kda[layer].recurrent), kda[layer].conv_dim_first)
             for layer, (_, _, slot) in conv_states.items()},
            {layer: _layout(mla[layer].latent) for layer in mla_pages},
            {layer: (address.tail_page, _layout(indexer[layer].pool), _layout(indexer[layer].tail))
             for layer, address in idx_addresses.items()},
        )
        native_pages = {layer: (pages, idx_addresses[layer].pages) for layer, pages in mla_pages.items()}
        imported = request_id != self._request
        if not imported:
            pages_extend = all(new.extends(old) for layer, pair in native_pages.items()
                               for new, old in zip(pair, self._native_pages[layer]))
            _require(position == self._position + 1 and locations == self._locations and pages_extend,
                     "VA request must advance by one token with unchanged cache locations")

        # Validate every destination before touching any data. MK execution
        # uses bridge-owned buffers, then writeback commits to VA in stream order.
        try:
            initial = None
            if imported:
                initial = self._import_prefix(position, conv_states, mla, mla_pages, indexer, idx_addresses)
            tails = self._import_tail(position, indexer, idx_addresses)
            if self._graph is None:
                self._model.clear_graph_cache()
            paging = ({"page": _SCRATCH_PAGE} if self.max_context_tokens <= _PAGE
                      else {"page_table": self._page_table})
            batch, positions = self._state.prepare_shared(position, kda=initial, mla=self._mla, indexer=tails, **paging)
            self._token.copy_(input_ids)
            logits = self._model.forward(batch, self._token, positions, None)
            hidden = self._model.last_hidden_states
            graph = self._model._last_graph
            reused = self._graph is not None
            capture_head = getattr(self._model, "capture_lm_head", True)
            if (graph is None or (reused and graph is not self._graph)
                    or (logits is not None) != capture_head or hidden is None):
                raise RuntimeError("VA bridge requires usable outputs from one stable MK graph")
            # VA may allocate its cache tensors as inference tensors. MK's
            # tracked capture is complete by the time VA storage is updated.
            with torch.inference_mode():
                self._commit(position, conv_states, mla, mla_pages, indexer, idx_addresses)
        except Exception:
            self._poisoned = True
            raise

        self._request, self._position, self._locations, self._graph = request_id, position, locations, graph
        self._native_pages = native_pages
        _LOG.info("VA/MK decode request=%s position=%d imported=%s graph=%#x reused=%s pool_boundary=%s "
                  "kda_slots=%s mla_pages=%s indexer=%s", request_id, position, imported, id(graph), reused,
                  (position + 1) % _RATIO == 0,
                  {layer: slot for layer, (_, _, slot) in conv_states.items()},
                  {layer: pages.ids[-1] for layer, pages in mla_pages.items()}, idx_addresses)
        return DecodeResult(logits, hidden, imported, reused)

    def _kda_target(self, cache, metadata):
        _require((metadata.num_actual_tokens, metadata.num_decodes, metadata.num_decode_tokens,
                  metadata.num_prefills, metadata.num_spec_decodes) == (1, 1, 1, 0, 0)
                 and metadata.spec_sequence_masks is None
                 and _metadata_values(metadata.non_spec_query_start_loc, 2, "KDA query boundaries") == (0, 1),
                 "VA KDA metadata must represent one non-speculative decode token")
        slot = _value(metadata.non_spec_state_indices_tensor, "KDA state index")
        conv_shape = self._conv_shape[::-1] if cache.conv_dim_first else self._conv_shape
        _require(cache.conv.shape[1:] == conv_shape and cache.recurrent.shape[1:] == self._recurrent_shape
                 and cache.conv.dtype == torch.bfloat16 and cache.recurrent.dtype == torch.float32
                 and cache.conv.device == cache.recurrent.device == self._device
                 and 0 <= slot < min(cache.conv.shape[0], cache.recurrent.shape[0]), "invalid VA KDA cache or slot")
        conv = cache.conv[slot]
        if cache.conv_dim_first:
            conv = conv.transpose(-1, -2)
        return conv, cache.recurrent[slot].transpose(-1, -2), slot

    def _mla_target(self, cache, metadata, position):
        _row_cache(cache.latent, (1, _LATENT), torch.bfloat16, self._device)
        rows = cache.latent.shape[1]
        # The NoPE sparse MLA layer is registered with AscendSFABackend.
        _require((metadata.num_actual_tokens, metadata.num_input_tokens, metadata.num_decodes,
                  metadata.num_decode_tokens, metadata.num_prefills) == (1, 1, 1, 1, 0)
                 and isinstance(metadata.block_size, int) and 0 < metadata.block_size <= rows
                 and rows % metadata.block_size == 0 and metadata.sin is None and metadata.cos is None
                 and _metadata_values(metadata.query_start_loc, 2, "MLA query boundaries") == (0, 1)
                 and _value(metadata.seq_lens, "MLA length") == position + 1
                 and _value(metadata.positions, "MLA decode position") == position,
                 "VA sparse NoPE MLA metadata must represent the scheduled decode token")
        pages = _active_pages(metadata.block_table, position + 1, metadata.block_size, cache.latent.shape[0] * rows)
        slot = _value(metadata.slot_mapping, "MLA slot")
        _require(slot == pages.slot(position), "VA MLA slot disagrees with the active prefix page")
        return pages

    def _indexer_target(self, cache, pool, tail, position):
        _row_cache(cache.pool, (1, _INDEX), torch.bfloat16, self._device)
        _row_cache(cache.tail, (2 * _INDEX,), torch.float32, self._device)
        # The native KeyPool writer splits pooled_slot_mapping using cache.shape[1].
        # Its metadata.block_size describes the 128-token SFA kernel divided by 4.
        rows = cache.pool.shape[1]
        prior_pools, prior_tail = divmod(position, _RATIO)
        pools, open_rows = divmod(position + 1, _RATIO)
        _require(pool.num_actual_tokens == 1 and pool.query_metadata is None
                 and pool.compress_ratio == _RATIO
                 and _metadata_values(pool.query_start_loc, 2, "KPool query boundaries") == (0, 1)
                 and _value(pool.positions, "KPool position") == position
                 and _value(pool.start_pos, "KPool start") == position
                 and _value(pool.raw_seq_lens, "KPool token length") == position + 1
                 and _value(pool.seq_lens, "KPool compressed length") == pools
                 and _value(pool.pool_tail, "KPool tail length") == open_rows, "invalid VA KPool decode metadata")
        pages = _active_pages(pool.block_table, pools, rows, cache.pool.shape[0] * rows)
        slot = pages.slot(prior_pools) if open_rows == 0 else -1
        pool_page, pool_row = divmod(slot, rows) if slot >= 0 else (None, None)
        _require(_value(pool.slot_mapping, "KPool token slot") == slot
                 and _value(pool.pooled_slot_mapping, "KPool compact slot") == slot, "invalid VA KPool slot")

        tail_page = _active_pages(tail.block_table, 1, 1, cache.tail.shape[0]).ids[0]
        cann = tail.cann_inputs
        capacity = cache.tail.shape[1]
        stride, remainder = divmod(cache.tail.stride(0), 2 * _INDEX)
        _require(cann is not None and cann.retained_indices is None and remainder == 0
                 and cann.page_rows == stride and capacity == tail.block_size == _RATIO
                 and cann.history_indices.shape == cann.write_indices.shape == (1, _RATIO),
                 "invalid VA non-speculative CANN tail metadata or page stride")
        history = _metadata_values(cann.history_indices[0], _RATIO, "CANN tail history")
        writes = _metadata_values(cann.write_indices[0], _RATIO, "CANN tail writes")
        base = tail_page * stride
        history_rows = tuple((position - prior_tail + row) % capacity for row in range(prior_tail))
        expected_writes = tuple(base + (position + 1 - open_rows + row) % capacity if row < open_rows else -1
                                for row in range(_RATIO))
        _require(history[:prior_tail] == tuple(base + row for row in history_rows) and writes == expected_writes,
                 "VA CANN tail indices do not address the declared tail page and ring rows")
        tail_row = writes[open_rows - 1] - base if open_rows else None
        return _IndexerAddress(pool_page, pool_row, tail_page, history_rows, tail_row, pages)

    def _import_prefix(self, position, kda, mla, mla_pages, indexer, idx_addresses):
        for layer, pages in mla_pages.items():
            if self.max_context_tokens <= _PAGE:
                self._mla[layer][_SCRATCH_PAGE].zero_()
                self._pool[layer][_SCRATCH_PAGE].zero_()
            _copy_prefix(mla[layer].latent, pages, self._mla[layer], position, self._page_table, 1)
            _copy_prefix(indexer[layer].pool, idx_addresses[layer].pages, self._pool[layer],
                         position // _RATIO, self._page_table, _RATIO)
        return {layer: (conv.contiguous(), recurrent.contiguous()) for layer, (conv, recurrent, _) in kda.items()}

    def _import_tail(self, position, indexer, idx_addresses):
        count = position % _RATIO
        result = {}
        for layer, address in idx_addresses.items():
            keys, gates = self._keys[layer][:count], self._gates[layer][:count]
            for i, row in enumerate(address.history_rows):
                source = indexer[layer].tail[address.tail_page, row]
                keys[i].copy_(source[:_INDEX])  # FP32 CANN key -> BF16 MK key.
                gates[i].copy_(source[_INDEX:])  # Unmodified FP32 raw gate, before APE.
            result[layer] = (self._pool[layer], keys, gates)
        return result

    def _commit(self, position, kda, mla, mla_pages, indexer, idx_addresses):
        for layer, (conv, recurrent, _) in kda.items():
            conv.copy_(self._state.kda.conv[layer][0])
            recurrent.copy_(self._state.kda.temporal[layer][0])
        for layer, pages in mla_pages.items():
            page, row = divmod(pages.slot(position), mla[layer].latent.shape[1])
            mk_page, mk_row = divmod(position, _PAGE)
            mla[layer].latent[page, row].copy_(self._mla[layer][self._page_table[mk_page], mk_row])
            cache, address = indexer[layer], idx_addresses[layer]
            if address.pool_row is not None:
                mk_page, mk_row = divmod(position // _RATIO, _PAGE)
                cache.pool[address.pool_page, address.pool_row].copy_(
                    self._pool[layer][self._page_table[_RATIO * mk_page], mk_row])
            else:
                tail = cache.tail[address.tail_page, address.tail_row]
                mk_slot = position % _RATIO
                tail[:_INDEX].copy_(self._state.indexer.tail_k_buffer[layer][0, mk_slot])
                tail[_INDEX:].copy_(self._state.indexer.tail_score_buffer[layer][0, mk_slot])
