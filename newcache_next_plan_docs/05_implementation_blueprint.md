# 05. 代码实现蓝图

## 0. 总原则

你的仓库已经有一个可用的 cache controller 结构，不建议推倒重写。下一步应该做“增量扩展”：

- 保留现有 `OnlineInputCacheController`、`UniformCacheController`、`AlwaysRefreshController`；
- 新增 PMA 相关 controller 和 proxy fitting 脚本；
- 保留 E0–E4 的日志格式，新增字段而不是替换字段；
- 所有实验都要能保存 `per_call_decisions.npz`，方便后续画图和复现实验。

---

## 1. 推荐新增文件结构

建议新增：

```text
src/diffusion/flow_matching/
  pma_cache.py
  pma_features.py
  pma_proxy.py
  pma_schedule.py
  cache_logging.py

scripts/
  07_e5_build_proxy_dataset.py
  08_e5_fit_pma_proxy.py
  09_e6_online_pma_cache.py
  10_e6_compare_online_methods.py
  11_e7_solver_forecast_ablation.py
  12_e8_generate_npz_for_fid.py
  13_e8_make_figures.py

evaluations/
  paired_metrics.py
  fid_eval.py
  bootstrap_ci.py
```

如果你不想拆太多文件，最小版本可以只新增：

```text
src/diffusion/flow_matching/pma_cache.py
scripts/07_e5_build_proxy_dataset.py
scripts/08_e5_fit_pma_proxy.py
scripts/09_e6_online_pma_cache.py
```

---

## 2. `pma_features.py`

职责：从相邻 call 的 online proxy 中构造 cheap features。

### 2.1 Feature dataclass

```python
from dataclasses import dataclass

@dataclass
class PMATransitionMeta:
    sample_idx: int
    call_idx: int
    prev_call_idx: int
    t_prev: float
    t_cur: float
    dt: float
    call_frac: float
    is_predictor: bool
    is_corrector: bool
    transition_kind: str  # 'pc', 'cp', 'other'
    stage_bin: str        # 'early', 'middle', 'late'
    cache_age: int
```

### 2.2 Feature builder

```python
def build_pma_features(prev_proxy, cur_proxy, meta, sea_beta=2.0):
    raw = relative_l1_distance(cur_proxy, prev_proxy)
    sea = relative_l1_distance(
        apply_sea_filter(cur_proxy, meta.t_cur, beta=sea_beta),
        apply_sea_filter(prev_proxy, meta.t_prev, beta=sea_beta),
    )

    delta = cur_proxy - prev_proxy
    feat = {
        'raw': raw.item(),
        'sea': sea.item(),
        'log1p_sea': np.log1p(sea.item()),
        't_prev': meta.t_prev,
        't_cur': meta.t_cur,
        'dt': meta.dt,
        'call_frac': meta.call_frac,
        'is_predictor': float(meta.is_predictor),
        'is_corrector': float(meta.is_corrector),
        'cache_age': float(meta.cache_age),
        'proxy_norm': cur_proxy.abs().mean().item(),
        'delta_mean': delta.abs().mean().item(),
        'delta_p95': torch.quantile(delta.abs().flatten(), 0.95).item(),
    }
    return feat
```

注意：为了速度，在线推理时不要频繁做很重的统计。`delta_p95` 如果开销太大，可以在 ablation 里关掉。

---

## 3. `pma_proxy.py`

职责：保存 / 加载 / 运行 proxy model。

### 3.1 最小 Ridge proxy

```python
class RidgePMAProxy:
    def __init__(self, weights, bias, feature_names):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)
        self.feature_names = list(feature_names)

    def predict(self, feat_dict):
        x = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32)
        return float(np.dot(self.weights, x) + self.bias)
```

可以用 sklearn 拟合，然后导出 JSON：

```json
{
  "type": "ridge",
  "feature_names": ["log1p_sea", "t_cur", "dt", "is_predictor", "cache_age"],
  "weights": [...],
  "bias": 0.123,
  "normalizer": {...}
}
```

### 3.2 Bootstrap uncertainty proxy

```python
class BootstrapRidgePMAProxy:
    def __init__(self, models):
        self.models = models

    def predict_mu_sigma(self, feat_dict):
        vals = np.array([m.predict(feat_dict) for m in self.models])
        return float(vals.mean()), float(vals.std())
```

---

## 4. `pma_schedule.py`

职责：score normalization、delta calibration、schedule simulation。

### 4.1 Stage-kind normalizer

```python
class StageKindNormalizer:
    def __init__(self, table, eps=1e-6):
        self.table = table
        self.eps = eps

    def __call__(self, score, stage_bin, transition_kind):
        key = f"{stage_bin}_{transition_kind}"
        stat = self.table.get(key, self.table['global'])
        med = stat['median']
        mad = stat['mad']
        z = (score - med) / (mad + self.eps)
        return float(np.log1p(np.exp(z)))
```

### 4.2 Delta calibration

```python
def simulate_rr(scores, delta, forced_mask=None, max_skip=4):
    refresh = np.zeros_like(scores, dtype=bool)
    for sample in range(scores.shape[0]):
        acc, age = 0.0, 0
        for c in range(scores.shape[1]):
            acc += scores[sample, c]
            force = forced_mask[sample, c] if forced_mask is not None else False
            if force or acc >= delta or age >= max_skip:
                refresh[sample, c] = True
                acc, age = 0.0, 0
            else:
                age += 1
    return refresh.mean(), refresh


def calibrate_delta(scores, target_rr, lo=1e-6, hi=100.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        rr, _ = simulate_rr(scores, mid)
        if rr > target_rr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
```

---

## 5. `pma_cache.py`

职责：真实在线 controller。

### 5.1 Config

```python
@dataclass
class PMAOnlineConfig:
    target_rr: float
    delta: float
    warmup_calls: int = 5
    max_skip_calls: int = 4
    force_final: bool = True
    sea_beta: float = 2.0
    use_uncertainty: bool = False
    uncertainty_k: float = 0.5
    uncertainty_threshold: float | None = None
    reuse_mode: str = "velocity"  # velocity | xhat_convert | velocity_forecast | xhat_forecast
```

### 5.2 Controller skeleton

```python
class PMAOnlineCacheController:
    def __init__(self, cfg, proxy, normalizer):
        self.cfg = cfg
        self.proxy = proxy
        self.normalizer = normalizer
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.cache_age = 0
        self.prev_proxy = None
        self.last_refresh_output = None
        self.prev_refresh_output = None
        self.logs = []

    def should_refresh(self, cur_proxy, meta):
        if self.prev_proxy is None:
            self.prev_proxy = cur_proxy.detach()
            return True

        feat = build_pma_features(self.prev_proxy, cur_proxy, meta, sea_beta=self.cfg.sea_beta)

        if self.cfg.use_uncertainty:
            mu, sigma = self.proxy.predict_mu_sigma(feat)
            raw_score = mu + self.cfg.uncertainty_k * sigma
        else:
            raw_score = self.proxy.predict(feat)
            sigma = None

        score = self.normalizer(raw_score, meta.stage_bin, meta.transition_kind)
        self.acc += score

        refresh = False
        reason = "skip"
        if meta.call_idx < self.cfg.warmup_calls:
            refresh, reason = True, "warmup"
        elif self.cfg.force_final and meta.is_final_call:
            refresh, reason = True, "final"
        elif self.cache_age >= self.cfg.max_skip_calls:
            refresh, reason = True, "max_skip"
        elif self.acc >= self.cfg.delta:
            refresh, reason = True, "acc_delta"
        elif sigma is not None and self.cfg.uncertainty_threshold is not None and sigma >= self.cfg.uncertainty_threshold:
            refresh, reason = True, "uncertainty"

        self.logs.append({
            "call_idx": meta.call_idx,
            "score": score,
            "raw_score": raw_score,
            "sigma": sigma,
            "acc": self.acc,
            "refresh": refresh,
            "reason": reason,
            **feat,
        })

        self.prev_proxy = cur_proxy.detach()
        if refresh:
            self.acc = 0.0
            self.cache_age = 0
        else:
            self.cache_age += 1
        return refresh
```

---

## 6. 修改 sampler 的最小方式

你现有 HeunSamplerJiT 已经能在每个 denoiser call 调 controller。建议保持接口类似：

```python
out, did_refresh = maybe_run_or_reuse(model, x, t, y, controller, call_meta)
```

### 6.1 helper 函数

```python
def denoise_with_cache(model, x, t, y, controller, meta):
    proxy = extract_jit_modulated_proxy(model, x, t, y)
    refresh = controller.should_refresh(proxy, meta)

    if refresh:
        xhat = model(x, t, y)
        v = (xhat - x) / (1.0 - t).clamp_min(controller.t_eps)
        controller.update_cache(xhat=xhat, v=v, meta=meta)
        return v, xhat, True
    else:
        v, xhat = controller.get_cached_output(x=x, t=t, meta=meta)
        return v, xhat, False
```

这样 sampler 主体只需要把原来的 model forward 替换为 `denoise_with_cache`。

---

## 7. E5 脚本设计

### 7.1 `07_e5_build_proxy_dataset.py`

输入：full trajectory run 或 E2 distance bank。

输出：

```text
outputs/e5_proxy_dataset/
  features_train.csv
  features_val.csv
  features_test.csv
  labels_train.csv
  labels_val.csv
  labels_test.csv
  meta.json
```

字段：

```text
sample_idx, call_idx, transition_kind, stage_bin,
raw, sea, log1p_sea, t_prev, t_cur, dt, call_frac,
proxy_norm, delta_mean, delta_p95,
y_dino, y_lpips, y_pma
```

命令示例：

```bash
python scripts/07_e5_build_proxy_dataset.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --ckpt ckpts/PixelGen_XL_80ep.ckpt \
  --num-samples 256 \
  --split 64,64,128 \
  --out outputs/e5_proxy_dataset
```

### 7.2 `08_e5_fit_pma_proxy.py`

功能：拟合 heuristic / ridge / bootstrap ridge，导出 JSON。

命令：

```bash
python scripts/08_e5_fit_pma_proxy.py \
  --dataset outputs/e5_proxy_dataset \
  --label y_pma \
  --model ridge \
  --target-rr 0.3 0.4 0.5 \
  --out outputs/e5_proxy_models/ridge_y_pma
```

输出：

```text
model.json
normalizer.json
deltas.json
proxy_eval.csv
proxy_scatter.png
topk_recall.png
```

---

## 8. E6 脚本设计

### 8.1 `09_e6_online_pma_cache.py`

功能：真实在线采样，记录 paired metrics。

命令：

```bash
python scripts/09_e6_online_pma_cache.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --ckpt ckpts/PixelGen_XL_80ep.ckpt \
  --proxy outputs/e5_proxy_models/ridge_y_pma/model.json \
  --normalizer outputs/e5_proxy_models/ridge_y_pma/normalizer.json \
  --delta outputs/e5_proxy_models/ridge_y_pma/deltas_rr030.json \
  --target-rr 0.30 \
  --num-samples 192 \
  --out outputs/e6_online_pma/rr030
```

输出：

```text
summary.json
per_sample_metrics.csv
per_call_logs.csv
per_call_decisions.npz
samples_ref/
samples_cache/
worst_cases.png
```

### 8.2 `10_e6_compare_online_methods.py`

功能：统一跑所有 online methods。

```bash
python scripts/10_e6_compare_online_methods.py \
  --methods uniform raw sea pma_proxy \
  --rr 0.3 0.4 0.5 \
  --num-samples 192 \
  --out outputs/e6_compare
```

---

## 9. E7 脚本设计

### 9.1 Solver-aware ablation

```bash
python scripts/11_e7_solver_forecast_ablation.py \
  --gate pma_proxy \
  --schedule call_acc step_pair two_threshold paired_refresh \
  --reuse-mode velocity xhat_convert velocity_forecast xhat_forecast \
  --rr 0.3 0.5 \
  --num-samples 192 \
  --out outputs/e7_solver_forecast
```

---

## 10. E8 脚本设计

### 10.1 FID 生成

```bash
python scripts/12_e8_generate_npz_for_fid.py \
  --method pma_proxy \
  --rr 0.3 \
  --num-samples 10000 \
  --batch-size 16 \
  --out outputs/e8_fid/pma_rr030_10k
```

### 10.2 画图

```bash
python scripts/13_e8_make_figures.py \
  --runs outputs/e6_compare outputs/e7_solver_forecast outputs/e8_fid \
  --out outputs/paper_figures
```

---

## 11. 日志字段必须统一

每次实验都保存：

```json
{
  "method": "pma_proxy",
  "target_rr": 0.3,
  "actual_rr": 0.297,
  "num_samples": 192,
  "num_calls": 99,
  "sampler": "HeunSamplerJiT",
  "guidance": 1.0,
  "timeshift": 2.0,
  "fp32": true,
  "ema": true,
  "warmup_calls": 5,
  "max_skip_calls": 4,
  "sea_beta": 2.0,
  "proxy_model": "ridge_y_pma",
  "seed": 0
}
```

这样以后写论文、补实验、做 rebuttal 都不会乱。

---

## 12. 最小可运行版本

如果你今天就要开工，先只做四件事：

1. 在 E2 bank 上生成 `features.csv`。
2. 用 sklearn Ridge 拟合 `y_pma`。
3. 把 Ridge JSON 接到 `OnlineInputCacheController` 的 score 位置。
4. 跑 192 samples，比较 `SEAInput-online` vs `PMA-Ridge-online`。

只要这一步有结果，你就知道整条路是否值得继续。

