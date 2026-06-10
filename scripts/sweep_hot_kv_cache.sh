#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'EOF'
Usage:
  scripts/sweep_hot_kv_cache.sh <dump_path> [out_dir]

Environment overrides:
  PYTHON_BIN        Python executable. Default: python3
  BUFFER_SIZES     Space-separated values. Default: "2048 4096 8192"
  RECENT_WINDOWS   Space-separated values. Default: "16 32 64"
  EMA_BETAS        Space-separated values. Default: "0.5 0.9"
  RECENT_WEIGHTS   Space-separated values. Default: "0.5 1.0 2.0"
  EMA_WEIGHTS      Space-separated values. Default: "0.0 0.5"
  AGE_WEIGHTS      Space-separated values. Default: "0.001 0.01"
  CANDIDATE_SIZES  Space-separated values. Default: "2048 4096"

Notes:
  - analyze_hot_kv_cache.py simulates the online effective candidate window:
    min(buffer_size, max(candidate_size, topk_width * 2)). CANDIDATE_SIZES are
    raw HotKVCacheConfig candidate_size values, not the final scan width.
  - Online HotKVCacheConfig requires buffer_size >= 2048. The offline simulator
    can run smaller values, but this sweep defaults to online-deployable sizes.
EOF
  exit 2
fi

cd "${REPO_ROOT}"

DUMP_PATH="$1"
OUT_DIR="${2:-/tmp/hot_kv_sweep_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BUFFER_SIZES="${BUFFER_SIZES:-2048 4096 8192}"
RECENT_WINDOWS="${RECENT_WINDOWS:-16 32 64}"
EMA_BETAS="${EMA_BETAS:-0.5 0.9}"
RECENT_WEIGHTS="${RECENT_WEIGHTS:-0.5 1.0 2.0}"
EMA_WEIGHTS="${EMA_WEIGHTS:-0.0 0.5}"
AGE_WEIGHTS="${AGE_WEIGHTS:-0.001 0.01}"
CANDIDATE_SIZES="${CANDIDATE_SIZES:-2048 4096}"

if [[ ! -f "${DUMP_PATH}" ]]; then
  echo "dump_path does not exist: ${DUMP_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
echo "buffer_size,recent_window,ema_beta,recent_weight,ema_weight,age_weight,candidate_size,requests,needed,hits,misses,hit_rate,out_dir" > "${SUMMARY_CSV}"

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "${value}"
}

extract_metric() {
  local line="$1"
  local key="$2"
  echo "${line}" | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p"
}

run_count=0
for buffer_size in ${BUFFER_SIZES}; do
  for recent_window in ${RECENT_WINDOWS}; do
    for ema_beta in ${EMA_BETAS}; do
      for recent_weight in ${RECENT_WEIGHTS}; do
        for ema_weight in ${EMA_WEIGHTS}; do
          for age_weight in ${AGE_WEIGHTS}; do
            for candidate_size in ${CANDIDATE_SIZES}; do
              run_name="bs$(sanitize_value "${buffer_size}")"
              run_name+="_rw$(sanitize_value "${recent_window}")"
              run_name+="_eb$(sanitize_value "${ema_beta}")"
              run_name+="_rwgt$(sanitize_value "${recent_weight}")"
              run_name+="_ewgt$(sanitize_value "${ema_weight}")"
              run_name+="_awgt$(sanitize_value "${age_weight}")"
              run_name+="_cs$(sanitize_value "${candidate_size}")"
              run_dir="${OUT_DIR}/${run_name}"
              mkdir -p "${run_dir}"

              echo "[HOT-KV-SWEEP] ${run_name}"
              "${PYTHON_BIN}" scripts/analyze_hot_kv_cache.py "${DUMP_PATH}" \
                --buffer-size "${buffer_size}" \
                --recent-window "${recent_window}" \
                --ema-beta "${ema_beta}" \
                --recent-weight "${recent_weight}" \
                --ema-weight "${ema_weight}" \
                --age-weight "${age_weight}" \
                --candidate-size "${candidate_size}" \
                --out-dir "${run_dir}" \
                | tee "${run_dir}/stdout.txt"

              global_line="$(grep '^ALL:' "${run_dir}/stdout.txt" | head -n 1)"
              requests="$(extract_metric "${global_line}" "requests")"
              needed="$(extract_metric "${global_line}" "needed")"
              hits="$(extract_metric "${global_line}" "hits")"
              misses="$(extract_metric "${global_line}" "misses")"
              hit_rate="$(extract_metric "${global_line}" "hit_rate")"

              echo "${buffer_size},${recent_window},${ema_beta},${recent_weight},${ema_weight},${age_weight},${candidate_size},${requests},${needed},${hits},${misses},${hit_rate},${run_dir}" >> "${SUMMARY_CSV}"
              run_count=$((run_count + 1))
            done
          done
        done
      done
    done
  done
done

echo "[HOT-KV-SWEEP] completed ${run_count} runs"
echo "[HOT-KV-SWEEP] summary: ${SUMMARY_CSV}"
