#!/usr/bin/env bash
# Canonical student-solver runner with progress bar.
#   ./run.sh <name>          # 20-ep public benchmark → runs/<name>
#   ./run.sh <name> 2        # 2-ep smoke test
#   ./run.sh <name> 20 16    # 20-ep, max_tool_rounds=16
set -euo pipefail

NAME="${1:?usage: ./run.sh <name> [n_episodes=20] [max_tool_rounds=14]}"
N_EP="${2:-20}"
ROUNDS="${3:-14}"
OUT="runs/${NAME}"

PY="${PY:-/opt/anaconda3/envs/mldl_mac/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${ROOT}/${OUT}"

echo "→ ${NAME}: ${N_EP} ep, max_tool_rounds=${ROUNDS}, max_output_tokens=900"
echo "→ output: ${OUT}/"

"${PY}" "${ROOT}/run_llm_baselines.py" \
  --config "${ROOT}/llm_eval_config.json" --systems student_solver \
  --skip-hidden --skip-ablations \
  --set student_solver.max_tool_rounds="${ROUNDS}" \
  --set student_solver.max_output_tokens=900 \
  --limit-public "${N_EP}" --output-dir "${ROOT}/${OUT}" \
  > "${ROOT}/${OUT}/run.log" 2>&1 &
RUN_PID=$!

"${PY}" "${ROOT}/watch_progress.py" "${ROOT}/${OUT}" --total "${N_EP}" || true

wait "${RUN_PID}"
EXIT=$?

echo
echo "─── summary (${OUT}/llm_eval_summary_v2.md) ───"
awk '/^## Main Results/{flag=1} flag && /^## /{if(seen) exit; seen=1} flag' \
  "${ROOT}/${OUT}/llm_eval_summary_v2.md"

exit "${EXIT}"
