#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

LABELS="outputs/e6_d0_labels/e55_window_labels.csv"
FULLTRAJ="outputs/e6_d0_fulltraj/main8"
OUT="outputs/e6_d0_xwp_diagnostic/main8"
CKPT="ckpts/PixelGen_XL_80ep.ckpt"

"${PYTHON_CMD[@]}" scripts/09_e6d0_build_labels.py \
  --e55-dir outputs/e5_5_multi_skip_pis/e5_5_main8_windows70_fp32 \
  --out "${LABELS}"

if [[ -f "${CKPT}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_CMD[@]}" scripts/10_e6d0_dump_fulltraj.py \
    --device cuda:0 \
    --num-samples 8 \
    --shard-id 0 \
    --num-shards 2 \
    --out "${FULLTRAJ}" \
    --no-autocast \
    --resume \
    --check-replay \
    --preview-count 4 &
  PID0=$!

  CUDA_VISIBLE_DEVICES=1 "${PYTHON_CMD[@]}" scripts/10_e6d0_dump_fulltraj.py \
    --device cuda:0 \
    --num-samples 8 \
    --shard-id 1 \
    --num-shards 2 \
    --out "${FULLTRAJ}" \
    --no-autocast \
    --resume \
    --check-replay \
    --preview-count 4 &
  PID1=$!

  wait "${PID0}"
  wait "${PID1}"
else
  echo "[E6-D0 main8] checkpoint missing: ${CKPT}; skip GPU trajectory dump."
  echo "[E6-D0 main8] after adding the checkpoint, rerun: bash scripts/run_e6d0_main8_2gpu.sh"
  exit 0
fi

SAMPLE_COUNT=0
if compgen -G "${FULLTRAJ}/sample_*.pt" > /dev/null; then
  SAMPLE_COUNT=$(find "${FULLTRAJ}" -maxdepth 1 -name 'sample_*.pt' | wc -l)
fi
if [[ "${SAMPLE_COUNT}" -lt 8 ]]; then
  echo "[E6-D0 main8] found ${SAMPLE_COUNT}/8 trajectory files in ${FULLTRAJ}; skip risk compute."
  echo "[E6-D0 main8] check summary_shard*.json, then rerun after GPU dump succeeds."
  exit 0
fi

"${PYTHON_CMD[@]}" scripts/11_e6d0_compute_window_risks.py \
  --labels "${LABELS}" \
  --fulltraj "${FULLTRAJ}" \
  --out "${OUT}" \
  --device cuda:0 \
  --proxy-size 64

"${PYTHON_CMD[@]}" scripts/12_e6d0_evaluate_scores.py \
  --risk-table "${OUT}/window_risk_table.csv" \
  --out "${OUT}/eval" \
  --danger-top-frac 0.2 \
  --budgets 0.2,0.3,0.4,0.5

echo "[E6-D0 main8] labels: ${LABELS}"
echo "[E6-D0 main8] trajectories: ${FULLTRAJ}"
echo "[E6-D0 main8] risks: ${OUT}/window_risk_table.csv"
echo "[E6-D0 main8] report: ${OUT}/eval/report.md"

# To experiment with 3 GPUs later, change --num-shards to 3 and launch a
# third process with CUDA_VISIBLE_DEVICES=2. The default intentionally uses
# only 2 GPUs.
