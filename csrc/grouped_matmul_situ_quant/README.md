# grouped_matmul_situ_quant (GroupedMatmulSituQuant A5 fused op)

`GroupedMatmul(MXFP8 x MXFP4) + SiTU + dynamic MX quant` single-launch fused
custom op for **Ascend950PR (arch35) only**. Fuses the production split chain
`npu_grouped_matmul + situ_mx_quant` with bit-exact outputs; contract geomean
2.650x (8/8 >= 1.0), graph-mode ratio 1.06-1.96, delivered 2026-08-25.

## Layout

- `op_kernel/gmm_situ_vcv_dev.cpp` — device kernel (device group_list,
  graph-capturable static grid, in-kernel pruning)
- `op_kernel/gmsq_vcv_controller.h`, `op_kernel/situ_epilogue.h` — controller + SiTU/MXQuant epilogue
- `op_kernel/vendor/{wqbmm,gmsq2}` — vendored official arch35 weight-quant VCV
  data path (self-contained, no external deps)
- `op_host/gmm_situ_quant_entries.cpp` — host tiling/launch; the four
  V2-aligned entries (aclnnGroupedMatmulSwigluQuantV2 API habits; our op
  itself carries no version suffix)
- `register.cpp` / `ops.h` — torch registration, delivery surface =
  `torch.ops.npu.gmm_situ_quant{,_list}` + `gmm_situ_quant_weight_nz{,.list}`
- Only the MX A8W4 combo is implemented (Kimi w4a8); `bias`/`smoothScale`
  unsupported by design

## Build / use

```bash
bash csrc/grouped_matmul_situ_quant/build.sh   # -> libgmm_situ_quant.so
```

Python surface: `vllm_ascend.ops.grouped_matmul_situ_quant` (lazy load, SOC
gate, ND/NZ dispatch, `to_weight_nz` helpers). Tests:
`tests/ut/ops/test_grouped_matmul_situ_quant.py`.

Wiring into the pip/cpack package build (like `csrc/gmm/grouped_matmul_swiglu_quant_v2`)
is a follow-up; today the .so is built by `build.sh` and discovered by the
wrapper (env `GMM_SITU_QUANT_LIB` overrides).

## Source of truth & sync

Kernel evolution happens in the evolution delivery tree
(`output/GroupedMatmulSituQuantA5-evo/p0_entries/kernel`, default; override
with `GMSQ_P0_KERNEL`). After an accepted kernel change:

```bash
bash csrc/grouped_matmul_situ_quant/sync_from_p0.sh
bash csrc/grouped_matmul_situ_quant/build.sh
pytest tests/ut/ops/test_grouped_matmul_situ_quant.py
```

`ops.h`/`register.cpp`/`CMakeLists.txt`/`build.sh` are integration-owned here
and are NOT synced.
