# E6-D0 xWPCache Offline Diagnostic Report

Valid evaluated windows: 20

## Best Scores

- Best Spearman vs pis_total_rank: `R_uncertainty_mean` (0.9170).
- Best PR-AUC for dangerous_total: `R_time_len` (0.8875).
- Best CapturedPIS@30: `R_xw_adj` (0.4045).

## Required Questions

- xWPCache vs time/raw/SEA baseline: Spearman wins (R_xwp_ode_vector_u_eta0p1 `0.5549` vs baseline best `0.5371`); PR-AUC does not win (`0.7214` vs `0.8875`); CapturedPIS@30 does not win (`0.3938` vs `0.3974`).
- Controlled correlation after time features: R_xwp_ode_vector_u_eta0p1 LOSO residual Spearman `n/a`.
- Full-xhat oracle vs Wiener proxy: oracle `-0.6872`, Wiener vector `0.1008`.
- Vector vs scalar: vector `0.1008`, scalar `0.1353`.
- Uncertainty effect: vector+eta0.1 Spearman `0.5549`, base vector `0.1008`; FNR@30 base `1.0000`, eta0.1 `0.2500`.

If xWPCache does not beat time-only/raw/SEA here, treat that as a diagnostic failure or a proxy-quality problem before attempting online cache.

## Files

- Metrics: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/synthetic/eval/score_metrics.csv`
- Captured PIS: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/synthetic/eval/captured_pis_at_budget.csv`
- False negatives: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/synthetic/eval/failure_false_negative.csv`
- Plots: `/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/outputs/e6_d0_xwp_diagnostic/synthetic/eval/plots`
