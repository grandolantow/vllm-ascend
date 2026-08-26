#!/usr/bin/env bash
# One-way sync of the V2-entry source subset from the evolution delivery tree
# (source of truth) into this vllm-ascend subtree. Kernel evolution continues
# in the evolution tree; re-run this after any accepted kernel change, then
# rebuild (build.sh) and re-run tests/ut/ops/test_grouped_matmul_situ_quant.py.
#
# NOTE: op_host/gmm_situ_quant_v2.cpp is renamed to gmm_situ_quant_entries.cpp
# on copy ("V2" belongs to the official aclnn family lineage, not to our op —
# registered torch.ops names carry no version suffix).
set -euo pipefail

SRC="${GMSQ_P0_KERNEL:-/home/s00886374/agent_generate/output/GroupedMatmulSituQuantA5-evo/p0_entries/kernel}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "${SRC}/op_kernel/gmm_situ_vcv_dev.cpp" ] || { echo "source of truth not found: ${SRC}"; exit 1; }

cp -f "${SRC}/op_kernel/gmm_situ_vcv_dev.cpp"        "${HERE}/op_kernel/"
cp -f "${SRC}/op_kernel/gmsq_vcv_controller.h"       "${HERE}/op_kernel/"
cp -f "${SRC}/op_kernel/situ_epilogue.h"             "${HERE}/op_kernel/"
rm -rf "${HERE}/op_kernel/vendor/wqbmm" "${HERE}/op_kernel/vendor/gmsq2"
cp -r  "${SRC}/op_kernel/vendor/wqbmm"               "${HERE}/op_kernel/vendor/"
cp -r  "${SRC}/op_kernel/vendor/gmsq2"               "${HERE}/op_kernel/vendor/"
cp -f "${SRC}/op_host/gmm_situ_quant_v2.cpp"         "${HERE}/op_host/gmm_situ_quant_entries.cpp"
cp -f "${SRC}/utils/torch_kernel_helper.h"           "${HERE}/utils/"

# ops.h / register.cpp / CMakeLists.txt / build.sh are integration-owned here
# (trimmed to the 4 V2 entries); they are NOT synced.
echo "[sync] V2 subset refreshed from ${SRC}"
echo "[sync] reminder: bash ${HERE}/build.sh && pytest tests/ut/ops/test_grouped_matmul_situ_quant.py"
