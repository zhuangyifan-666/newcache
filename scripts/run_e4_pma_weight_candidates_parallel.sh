#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REF_TENSOR="outputs/e4_oracle_schedule_cache/e4_reference_test_fp32/reference_images.pt"
if [[ ! -f "${REF_TENSOR}" ]]; then
  echo "[E4-candidate] Missing ${REF_TENSOR}"
  echo "[E4-candidate] Generate it first with:"
  echo "CUDA_VISIBLE_DEVICES=3 conda run -n pixelgen python scripts/04_e4_oracle_schedule_cache_rerun.py --device cuda:0 --split test --target-rrs 0.30 --schedule-methods sea_oracle --no-autocast --reference-only --save-reference-tensor ${REF_TENSOR} --run-id e4_reference_test_fp32"
  exit 1
fi

COMMON_ARGS=(
  --split test
  --target-rrs 0.30,0.40,0.50
  --schedule-methods pma_stageaware_oracle
  --no-autocast
  --reference-tensor "${REF_TENSOR}"
  --save-image-count 16
)

run_candidate() {
  local gpu="$1"
  local candidate="$2"
  local run_id="$3"
  local schedule_dir="outputs/e4_pma_weight_candidates/e4_pma_weight_candidates_from_e2_fp32_calib64/${candidate}/matched_schedules"
  local log_dir="outputs/e4_oracle_schedule_cache/${run_id}"
  mkdir -p "${log_dir}"
  echo "[E4-candidate] GPU ${gpu}: ${candidate} -> ${run_id}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n pixelgen python scripts/04_e4_oracle_schedule_cache_rerun.py \
    --device cuda:0 \
    --schedule-dir "${schedule_dir}" \
    --run-id "${run_id}" \
    "${COMMON_ARGS[@]}" \
    > "${log_dir}/run.log" 2>&1
}

run_candidate 0 candidate_a e4_candidate_a_rr030_rr040_rr050_fp32 &
pid_a=$!
run_candidate 1 candidate_b e4_candidate_b_rr030_rr040_rr050_fp32 &
pid_b=$!
run_candidate 2 candidate_c e4_candidate_c_rr030_rr040_rr050_fp32 &
pid_c=$!

echo "[E4-candidate] Started PIDs: A=${pid_a}, B=${pid_b}, C=${pid_c}"
status=0
wait "${pid_a}" || status=1
wait "${pid_b}" || status=1
wait "${pid_c}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "[E4-candidate] At least one candidate rerun failed. Check run.log files under outputs/e4_oracle_schedule_cache/e4_candidate_*"
  exit "${status}"
fi
echo "[E4-candidate] All candidate reruns finished."

conda run -n pixelgen python scripts/06_e4_compare_pma_weight_candidates.py
echo "[E4-candidate] Comparison written under outputs/e4_pma_weight_candidates/comparison/compare_candidates_vs_main"
