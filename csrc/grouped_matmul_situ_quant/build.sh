#!/usr/bin/env bash
# Build libgmm_situ_quant.so (GroupedMatmulSituQuant A5 fused custom op,
# Ascend950PR/arch35 only) and stage it for the python wrapper.
#
# Usage:  bash csrc/grouped_matmul_situ_quant/build.sh
# Env:    ASCEND_HOME_PATH (default /usr/local/Ascend/cann-9.1.0)
#         SOC_VERSION      (default Ascend950PR_9579)
# Output: csrc/grouped_matmul_situ_quant/build/libgmm_situ_quant.so
#         (+ copy to vllm_ascend/ops/_gmm_situ_quant_libs/ for lazy loading)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
export SOC_VERSION="${SOC_VERSION:-Ascend950PR_9579}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
[ -f "${ASCEND_HOME_PATH}/set_env.sh" ] && source "${ASCEND_HOME_PATH}/set_env.sh"

echo "[gmm_situ_quant] SOC_VERSION=${SOC_VERSION} CANN=${ASCEND_HOME_PATH}"
cmake -S "${HERE}" -B "${HERE}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${HERE}/build" -j"$(nproc)"

SO="${HERE}/build/libgmm_situ_quant.so"
[ -f "${SO}" ] || { echo "[gmm_situ_quant] ERROR: ${SO} not found"; exit 1; }

STAGE="${REPO_ROOT}/vllm_ascend/ops/_gmm_situ_quant_libs"
mkdir -p "${STAGE}"
cp -f "${SO}" "${STAGE}/"
echo "[gmm_situ_quant] built $(du -h "${SO}" | cut -f1) -> ${STAGE}/libgmm_situ_quant.so"
