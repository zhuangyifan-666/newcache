#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

LABELS="outputs/e6_d0_labels/e55_window_labels.csv"
FULLTRAJ="outputs/e6_d0_fulltraj/smoke"
OUT="outputs/e6_d0_xwp_diagnostic/smoke"
CKPT="ckpts/PixelGen_XL_80ep.ckpt"

"${PYTHON_CMD[@]}" scripts/09_e6d0_build_labels.py \
  --e55-dir outputs/e5_5_multi_skip_pis/e5_5_main8_windows70_fp32 \
  --out "${LABELS}"

if [[ -f "${CKPT}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_CMD[@]}" scripts/10_e6d0_dump_fulltraj.py \
    --device cuda:0 \
    --num-samples 2 \
    --shard-id 0 \
    --num-shards 1 \
    --out "${FULLTRAJ}" \
    --no-autocast \
    --resume \
    --check-replay \
    --preview-count 2
else
  echo "[E6-D0 smoke] checkpoint missing: ${CKPT}; skip GPU trajectory dump."
fi

if compgen -G "${FULLTRAJ}/sample_*.pt" > /dev/null; then
  "${PYTHON_CMD[@]}" scripts/11_e6d0_compute_window_risks.py \
    --labels "${LABELS}" \
    --fulltraj "${FULLTRAJ}" \
    --out "${OUT}" \
    --max-rows 20 \
    --device cuda:0 \
    --proxy-size 64

  "${PYTHON_CMD[@]}" scripts/12_e6d0_evaluate_scores.py \
    --risk-table "${OUT}/window_risk_table.csv" \
    --out "${OUT}/eval" \
    --danger-top-frac 0.2 \
    --budgets 0.2,0.3,0.4,0.5

  echo "[E6-D0 smoke] report: ${OUT}/eval/report.md"
else
  echo "[E6-D0 smoke] no trajectory files found; run dump after placing checkpoint, then rerun smoke."
fi
