# E6：第一次真正在线 PMA-Cache 实验

> 一句话目标：把 E5 学到的小预测器放进真实 PixelGen cache loop，看看它能不能在真实推理时超过 SEAInput-online。

---

## 0. E6 是什么

E5 只是做表格预测，还没有真正生成图片。

E6 要开始真正跑 PixelGen 推理：

```text
每到一个 denoiser call：
  先算便宜特征
  用 E5 预测风险 risk
  把 risk 加到累计器 accumulator
  如果 accumulator 超过阈值 delta：刷新 denoiser
  否则：复用上一次 denoiser 输出
```

这一步非常重要，因为它是你的方法第一次从“分析”变成“真实加速方法”。

---

## 1. 用生活例子理解 E6

想象你在画一张图。每次 denoiser refresh 就像“认真重画一次”，skip 就像“先沿用上次的画法”。

如果每一步都认真重画，图像最接近原始 full model，但很慢。

如果太多步偷懒复用，图像可能崩。

E6 的小预测器就像一个提醒器：

```text
这一步变化可能很大，别偷懒，刷新！
这一步变化可能很小，可以复用。
```

---

## 2. E6 要回答的科学问题

E6 只回答一个问题：

> E5 的便宜预测器，在真实 online cache 中，是否能比 SEAInput-online 更好？

重点比较：

- 同样 RR≈0.30 / 0.40 / 0.50。
- 同样 sample split。
- 同样 full reference。
- 看 PSNR、SSIM、LPIPS、speedup。

---

## 3. E6 为什么不能再用 DINO / LPIPS

E5 可以用 DINO / LPIPS 当训练答案。

但是 E6 真正在线推理时，不能每一步都算 DINO / LPIPS。原因很简单：

1. 它们需要 `xhat`，而 `xhat` 要跑 denoiser 才有。
2. 如果为了判断是否刷新而先跑 denoiser，就已经没有 cache 加速意义。
3. DINO / LPIPS 本身也很贵。

所以 E6 在线时只能用：

```text
SEAInput
RawInput
step位置
call_kind
cache_age 等便宜信息
```

绝对不要在 E6 的 online loop 中调用 DINO / LPIPS。

---

## 4. E6 的方法名字建议

可以暂时叫：

```text
ProxyPMA-online
```

或者更论文一点：

```text
PMA-Cache
Perceptual-Manifold-Aware Cache
```

中文可以叫：

```text
感知流形感知 Cache
```

但实验代码里建议先用清晰名字：

```text
proxy_pma_online
```

---

## 5. E6 的 cache 规则

继续使用你 E1/E4 已经用过的 accumulated distance rule。

### 5.1 普通 SEAInput-online 规则

```text
score = SEAInput distance
accumulator += score
if accumulator > delta:
    refresh
    accumulator = 0
else:
    skip
```

### 5.2 E6 的 ProxyPMA-online 规则

```text
features = [SEA, Raw, step_fraction, call_kind, stage]
risk = proxy_model(features)
risk = max(risk, 0)   # 防止预测出负数
accumulator += risk
if accumulator > delta:
    refresh
    accumulator = 0
else:
    skip
```

这就是 E6 的核心。

---

## 6. E6 需要实现什么代码

### 6.1 新增一个小模型读取模块

建议新建：

```text
src/diffusion/flow_matching/e5_proxy_model.py
```

功能：

```text
读取 proxy_model_weights.json
输入一行 online feature
输出 risk 分数
```

如果你用的是线性模型，`proxy_model_weights.json` 可以长这样：

```json
{
  "feature_names": ["log1p_sea", "raw_norm", "call_fraction", "is_pred_to_corr", "is_corr_to_pred"],
  "weights": [0.8, 0.1, 0.2, 0.5, -0.1],
  "bias": 0.03,
  "normalization": {
    "log1p_sea": {"mean": 0.1, "std": 0.2}
  }
}
```

不要太纠结具体数值，这只是格式例子。

### 6.2 在 cache 代码里加一个模式

你现在的核心 cache 模块是：

```text
src/diffusion/flow_matching/e1_cache.py
```

可以加一个 mode：

```text
cache_metric = "proxy_pma"
```

里面做：

```text
if cache_metric == "sea":
    score = sea_distance
elif cache_metric == "proxy_pma":
    score = proxy_model.predict(features)
```

### 6.3 新增一个 E6 脚本

建议新建：

```text
scripts/09_e6_online_proxy_pma_cache.py
```

它可以参考你的 E1 脚本：

```text
scripts/01_e1_online_cache.py
```

区别只是多了一个 `proxy_pma_online` 方法。

---

## 7. E6 的实验设置

### 7.1 先跑 pilot，不要一上来跑大实验

先跑 64 samples：

```text
Samples: 64
RR targets: 0.30 / 0.40 / 0.50
Precision: fp32, no autocast
Sampler: Heun exact
Calls/sample: 99
```

这个阶段的目标不是论文最终结果，而是确认代码没问题。

### 7.2 pilot 通过后再跑主实验

主实验建议沿用 E4 test split：

```text
Samples: 192
Sample indices: 64-255
RR targets: 0.30 / 0.40 / 0.50
```

这样可以和 E4 oracle rerun 对齐。

---

## 8. E6 要比较哪些方法

最少比较这些：

```text
Full reference
Uniform RR0.30 / RR0.40 / RR0.50
RawInput-online
SEAInput-online
ProxyPMA-online
```

如果时间够，再加：

```text
ProxyPMA-no-stage
ProxyPMA-no-callkind
ProxyPMA-candidateA-label
```

但这些也可以留到 E7。

---

## 9. E6 的阈值 delta 怎么定

你要让不同方法的 actual RR 尽量接近目标 RR。

建议还是用 calibration samples。

例如：

```text
Calibration samples: 8 或 16
Target RR: 0.30
搜索 delta，让 calibration 上 actual RR 接近 0.30
然后固定 delta，跑 test samples
```

不要在 test set 上调 delta，否则 reviewer 会说你偷看测试集。

简单理解：

- calibration 是用来调开关灵敏度的小集合。
- test 是真正考试。

---

## 10. E6 输出文件

建议输出到：

```text
outputs/e6_online_proxy_pma/e6_pilot_64_fp32/
outputs/e6_online_proxy_pma/e6_test_192_fp32/
```

每个目录里放：

```text
summary.json
method_summary.csv
per_sample_metrics.csv
refresh_trace.csv
stage_refresh_density.csv
stage_kind_refresh_density.csv
```

### 10.1 method_summary.csv 应该包含

```text
method
target_rr
actual_rr
refresh_per_sample
speedup_vs_full
psnr
ssim
lpips
lpips_worst10_mean
```

`lpips_worst10_mean` 很有用，它表示最差 10% 样本的平均 LPIPS。

cache 方法很容易平均值不错，但少数图崩得很厉害。这个指标能帮你发现 tail failure。

---

## 11. E6 推荐命令模板

### 11.1 pilot

```bash
python scripts/09_e6_online_proxy_pma_cache.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --checkpoint ckpts/PixelGen_XL_80ep.ckpt \
  --proxy-model outputs/e5_proxy_fitting/e5_main_from_e2_fp32/no_gate_sea_time_kind/proxy_model_weights.json \
  --num-samples 64 \
  --target-rr 0.30 0.40 0.50 \
  --calib-samples 8 \
  --precision fp32 \
  --no-autocast \
  --outdir outputs/e6_online_proxy_pma/e6_pilot_64_fp32
```

### 11.2 主实验

```bash
python scripts/09_e6_online_proxy_pma_cache.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --checkpoint ckpts/PixelGen_XL_80ep.ckpt \
  --proxy-model outputs/e5_proxy_fitting/e5_main_from_e2_fp32/no_gate_sea_time_kind/proxy_model_weights.json \
  --sample-start 64 \
  --num-samples 192 \
  --target-rr 0.30 0.40 0.50 \
  --calib-samples 16 \
  --precision fp32 \
  --no-autocast \
  --outdir outputs/e6_online_proxy_pma/e6_test_192_fp32
```

按你的脚本参数实际情况改。

---

## 12. E6 怎么判断成功

### 12.1 最理想结果

```text
RR0.30：ProxyPMA 接近 SEAInput-online，不明显差
RR0.40：ProxyPMA 超过 SEAInput-online
RR0.50：ProxyPMA 明显超过 SEAInput-online
```

这和 E4 的 oracle 发现一致：感知 PMA 在高 refresh budget 更有价值。

### 12.2 可以接受的结果

```text
RR0.30：ProxyPMA 略差于 SEA
RR0.40：ProxyPMA 持平 SEA
RR0.50：ProxyPMA 优于 SEA
```

这也很有论文价值，因为它证明：

> online proxy 虽然不是 oracle，但能在某些预算下把 perceptual branch 的价值带到真实推理中。

### 12.3 危险结果

```text
ProxyPMA 三个 RR 全部弱于 SEAInput-online
```

这说明 E5 预测器还不够好。不要放弃，回到 E5 或做 E7 诊断。

---

## 13. 如果 E6 不好，优先检查什么

### 检查 1：有没有泄漏或偷看

确认 E6 在线 loop 没有调用 DINO / LPIPS。

### 检查 2：actual RR 是否匹配

如果 ProxyPMA 的 RR 比 SEA 低很多，那不公平。

例如：

```text
SEA actual RR = 0.40
Proxy actual RR = 0.34
```

这时 Proxy 差不一定是方法差，而是刷新次数少。

### 检查 3：预测分数是不是有负数

如果线性模型输出负数，accumulator 会乱。建议：

```text
risk = max(pred, 0)
```

或者：

```text
risk = softplus(pred)
```

简单起见先用 `max(pred, 0)`。

### 检查 4：normalization 是否一致

E5 训练时怎么归一化，E6 推理时必须完全一样。

例如 E5 用了：

```text
x_norm = (x - mean) / std
```

E6 也必须用同样的 mean / std，不能重新在 test 上算。

---

## 14. E6 最小可运行版本

如果你想最快跑起来，先只做：

1. 使用 E5 的 `label_pma_no_gate` 线性模型。
2. 特征只用 `log1p_sea + call_fraction + call_kind`。
3. 只跑 64 samples。
4. 只比较 RR0.40 和 RR0.50。
5. 先不做所有 ablation。

如果这个版本 RR0.50 能超过 SEAInput-online，就说明方向有戏。

---

## 15. E6 结束后你要写下什么结论

E6 结束后，请用下面模板总结：

```text
在真实 online cache setting 下，ProxyPMA-online 在 RR=__ 时相对 SEAInput-online：
PSNR 提升/下降 __ dB，LPIPS 降低/升高 __，speedup 为 __。

这说明 E5 学到的便宜 proxy 能/不能把 E4 oracle PMA 的上限价值转移到真实推理。
```

这个总结会直接变成论文实验部分的核心段落。
