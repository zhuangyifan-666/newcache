# E6-D0 xWPCache Offline Diagnostic Report

Valid evaluated windows: 560

## Best Scores

- Best Spearman vs pis_total_rank: `R_xw_adj` (0.8096).
- Best PR-AUC for dangerous_total: `R_oracle_xhat_vector_u_eta0p1` (0.7355).
- Best CapturedPIS@30: `R_oracle_xhat_vector_u_eta0p1` (0.4648).

## Required Questions

- xWPCache vs time/raw/SEA baseline: Spearman does not win (R_xwp_ode_vector_u_eta0p1 `0.3987` vs baseline best `0.7460`); PR-AUC does not win (`0.5125` vs `0.5492`); CapturedPIS@30 does not win (`0.3995` vs `0.4440`).
- Controlled correlation after time features: R_xwp_ode_vector_u_eta0p1 LOSO residual Spearman `-0.0791`.
- Full-xhat oracle vs Wiener proxy: oracle `0.7458`, Wiener vector `0.2808`.
- Vector vs scalar: vector `0.2808`, scalar `0.4095`.
- Uncertainty effect: vector+eta0.1 Spearman `0.3987`, base vector `0.2808`; FNR@30 base `0.4554`, eta0.1 `0.3750`.

If xWPCache does not beat time-only/raw/SEA here, treat that as a diagnostic failure or a proxy-quality problem before attempting online cache.

## Files

- Metrics: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/main8/eval/score_metrics.csv`
- Captured PIS: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/main8/eval/captured_pis_at_budget.csv`
- False negatives: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/main8/eval/failure_false_negative.csv`
- Plots: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/main8/eval/plots`
