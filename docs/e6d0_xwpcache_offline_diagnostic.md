# E6-D0 xWPCache-v2 Offline Diagnostic

E6-D0 checks whether xWPCache-v2 risk scores predict the causal E5.5 continuous-skip PIS labels. It is only an offline diagnostic. It does not implement online cache, does not choose a refresh/skip sampler, and does not report final acceleration.

## Inputs And Outputs

Inputs:

- E5.5 labels: `outputs/e5_5_multi_skip_pis/e5_5_main8_windows70_fp32/`
- PixelGen config: `configs_c2i/PixelGen_XL_without_CFG.yaml`
- PixelGen checkpoint: `ckpts/PixelGen_XL_80ep.ckpt`

Main outputs:

- Labels: `outputs/e6_d0_labels/e55_window_labels.csv`
- Full trajectories: `outputs/e6_d0_fulltraj/main8/sample_*.pt`
- Risk table: `outputs/e6_d0_xwp_diagnostic/main8/window_risk_table.csv`
- Evaluation report: `outputs/e6_d0_xwp_diagnostic/main8/eval/report.md`

Large trajectory `.pt` files are ignored by git.

## How To Run Smoke

```bash
bash scripts/run_e6d0_smoke.sh
```

The smoke run builds labels, dumps two trajectories if the checkpoint exists, computes at most 20 risk rows, and evaluates them. If the checkpoint is missing, the script prints a clear message and skips GPU trajectory work.

If the shell's default `python` is not the PixelGen environment, run:

```bash
PYTHON_BIN="conda run -n pixelgen python" bash scripts/run_e6d0_smoke.sh
```

## How To Run Main8

```bash
bash scripts/run_e6d0_main8_2gpu.sh
```

Or with an explicit conda environment:

```bash
PYTHON_BIN="conda run -n pixelgen python" bash scripts/run_e6d0_main8_2gpu.sh
```

The default uses only two GPUs:

- process 0: `CUDA_VISIBLE_DEVICES=0`, internal `--device cuda:0`
- process 1: `CUDA_VISIBLE_DEVICES=1`, internal `--device cuda:0`

Do not use GPU 2 or GPU 3 unless explicitly requested.

## Script Roles

- `scripts/09_e6d0_build_labels.py`: reads E5.5 `pis_window_summary.csv`, keeps valid interventions, and adds `pis_total_z` plus `pis_total_rank`.
- `scripts/10_e6d0_dump_fulltraj.py`: runs full no-cache sampling and saves each denoiser call input `x_t`, effective guided clean prediction `xhat`, timestep, step index, and call kind.
- `scripts/11_e6d0_compute_window_risks.py`: computes time-only, raw, SEA-like, Wiener, xWP, ODE, vector, uncertainty, and full-xhat oracle risks for each label window.
- `scripts/12_e6d0_evaluate_scores.py`: evaluates Spearman, ROC-AUC, PR-AUC, CapturedPIS@budget, FNR@budget, controlled correlation, and false negatives.

## Core Score Meanings

- `R_time_len`, `R_time_sum_ode`: time/window baselines.
- `R_raw_anchor`: relative L1 between current sampler states and the anchor state.
- `R_sea_anchor`: SEA-like frequency-normalized filtered state distance.
- `R_xw_anchor`: Wiener clean proxy compared to cached anchor xhat.
- `R_xwp_anchor`: Wiener proxy plus perceptual frequency weighting.
- `R_xwp_ode_scalar`: perceptual anchor distance weighted by the x-pred ODE factor.
- `R_xwp_ode_vector`: vector accumulated solver residual.
- `R_xwp_ode_vector_u_eta0p1`: vector risk plus proxy uncertainty penalty.
- `R_oracle_xhat_vector`: upper bound using the true full-trajectory current xhat.

## Reading `report.md`

The report answers:

- which score has the best Spearman correlation,
- which score has the best PR-AUC,
- which score captures the most PIS at 30% refresh budget,
- whether xWPCache beats time-only/raw/SEA baselines,
- whether full-xhat oracle is higher than Wiener proxy,
- whether vector beats scalar,
- whether uncertainty reduces false negatives.

If xWPCache does not beat time-only/raw/SEA, treat that as a real diagnostic result. Do not convert E5.5 windows into hand-written online refresh ranges.
