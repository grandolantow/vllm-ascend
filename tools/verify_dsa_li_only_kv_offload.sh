#!/usr/bin/env bash
# Verify DSA fused-overlap LI-only KV offload changes on an NPU host.
#
# Typical usage:
#   tools/verify_dsa_li_only_kv_offload.sh
#
# If this checkout does not contain the fused-overlap smoke test source, pass
# the target explicitly:
#   NPU_SMOKE_TESTS='tests/ut/test_fused_sparse_attention_overlap_gqa.py' \
#     tools/verify_dsa_li_only_kv_offload.sh
#
# Useful switches:
#   PYTHON_BIN=python3
#   DEVICE_ID=0
#   RUN_PYTHON_UT=0
#   RUN_NPU_SMOKE=0
#   RUN_DEBUG_SMOKE=0
#   CHECK_DEBUG_MARKERS=0
#   LOG_DIR=/tmp/dsa_li_only_kv_verify
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/dsa_li_only_kv_verify}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_ID="${DEVICE_ID:-0}"

run_python_ut="${RUN_PYTHON_UT:-1}"
run_npu_smoke="${RUN_NPU_SMOKE:-1}"
run_debug_smoke="${RUN_DEBUG_SMOKE:-1}"
check_debug_markers="${CHECK_DEBUG_MARKERS:-1}"
debug_pytest_args="${DEBUG_PYTEST_ARGS:--s --log-cli-level=INFO}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

log() {
  printf '[verify_dsa_li_only] %s\n' "$*" >&2
}

run_and_log() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  log "running ${name}; log=${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

resolve_npu_smoke_tests() {
  if [[ -n "${NPU_SMOKE_TESTS:-}" ]]; then
    printf '%s\n' "${NPU_SMOKE_TESTS}"
    return 0
  fi

  local candidates=(
    "tests/ut/test_fused_sparse_attention_overlap_gqa.py"
    "tests/ut/attention/a2/test_fused_sparse_attention_overlap_gqa_precision.py"
    "tests/ut/test_fused_sparse_attention_overlap_sfa_precision.py"
  )
  local existing=()
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      existing+=("${candidate}")
    fi
  done
  if (( ${#existing[@]} == 0 )); then
    log "no default NPU smoke test exists in this checkout; set NPU_SMOKE_TESTS explicitly."
    log "example: NPU_SMOKE_TESTS='tests/ut/test_fused_sparse_attention_overlap_gqa.py'"
    return 1
  fi
  printf '%s\n' "${existing[*]}"
}

require_python_module() {
  local module="$1"
  "${PYTHON_BIN}" - <<PY
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${module}") is not None else 1)
PY
}

if [[ "${run_python_ut}" == "1" || "${run_npu_smoke}" == "1" || "${run_debug_smoke}" == "1" ]]; then
  if ! require_python_module pytest; then
    log "pytest is not available in ${PYTHON_BIN}; install test dependencies or set PYTHON_BIN."
    exit 1
  fi
fi

log "repo=${ROOT_DIR}"
log "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
log "device_id=${DEVICE_ID}"
log "log_dir=${LOG_DIR}"

npu_smoke_tests=""
if [[ "${run_npu_smoke}" == "1" || "${run_debug_smoke}" == "1" ]]; then
  npu_smoke_tests="$(resolve_npu_smoke_tests)"
  log "npu_smoke_tests=${npu_smoke_tests}"
fi

if [[ "${run_python_ut}" == "1" ]]; then
  run_and_log "python_ut" \
    "${PYTHON_BIN}" -m pytest -q \
      tests/ut/kv_offload/test_dsa_li_only_kv_cache.py \
      tests/ut/worker/a2/test_model_runner_v1.py::TestNPUModelRunnerKVCache \
      tests/ut/attention/a2/test_sfa_v1.py::test_prepare_dsa_full_kv_inputs_binds_kv_cache_as_fused_full_kv_source \
      tests/ut/attention/a2/test_sfa_v1.py::test_prepare_dsa_full_kv_inputs_skips_mooncake_gate_when_debug_disabled \
      tests/ut/attention/a2/test_sfa_v1.py::test_selection_cache_static_sizing_uses_configured_capacity \
      tests/ut/attention/a2/test_sfa_v1.py::test_selection_cache_static_sizing_rejects_too_many_runtime_tokens \
      tests/ut/attention/a2/test_sfa_v1.py::test_selection_cache_views_cover_capacity_stride_block_table
fi

if [[ "${run_npu_smoke}" == "1" ]]; then
  read -r -a npu_smoke_test_args <<< "${npu_smoke_tests}"
  run_and_log "npu_smoke" env \
    VLLM_ASCEND_DSA_PD_MOONCAKE_CPU_KV=1 \
    DEVICE_ID="${DEVICE_ID}" \
    "${PYTHON_BIN}" -m pytest -q "${npu_smoke_test_args[@]}"
fi

if [[ "${run_debug_smoke}" == "1" ]]; then
  read -r -a npu_smoke_test_args <<< "${npu_smoke_tests}"
  read -r -a debug_pytest_extra_args <<< "${debug_pytest_args}"
  run_and_log "npu_debug_smoke" env \
    VLLM_ASCEND_DSA_LI_ONLY_DEBUG=1 \
    VLLM_ASCEND_DSA_PD_MOONCAKE_CPU_KV=1 \
    DEVICE_ID="${DEVICE_ID}" \
    "${PYTHON_BIN}" -m pytest -q "${debug_pytest_extra_args[@]}" "${npu_smoke_test_args[@]}"

  debug_log="${LOG_DIR}/npu_debug_smoke.log"
  if [[ "${check_debug_markers}" == "1" ]]; then
    log "checking debug markers in ${debug_log}"
    required_markers=(
      "[DSA_LI_ONLY_DEBUG] spec"
      "[DSA_LI_ONLY_DEBUG] allocation_plan"
      "[DSA_LI_ONLY_DEBUG] allocation_result"
      "[DSA_LI_ONLY_DEBUG] reshape_result"
      "[DSA_LI_ONLY_DEBUG] fused_full_kv_inputs"
    )
    for marker in "${required_markers[@]}"; do
      if ! grep -Fq "${marker}" "${debug_log}"; then
        log "missing debug marker: ${marker}"
        exit 1
      fi
    done
  fi
fi

log "verification finished"
