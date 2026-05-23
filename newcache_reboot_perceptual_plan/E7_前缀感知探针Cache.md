# E7｜前缀感知探针 Cache：用模型前几层预测真正的感知风险

## 1. 为什么需要 E7？

E6 直接用 DINO / LPIPS 看 `x_t`，它更感知，但有两个问题：

```text
1. DINO / LPIPS online 可能太慢。
2. x_t 不是 xhat，不能完全代表模型这一步真正会输出什么。
```

E7 的目标是做一个更强的 online 方法：

```text
只跑模型前几层，预测当前 call 的感知风险。
```

这个方法叫：

```text
Prefix Perceptual Probe Cache
中文：前缀感知探针 Cache
```

---

## 2. 这个方法和旧 proxy 方案有什么区别？

旧方案是：

```text
SEAInput / timestep / age 这些便宜信号
↓
训练一个小模型预测 PMA
```

它的问题是：SEAInput 本身不一定代表感知变化。

E7 新方案是：

```text
当前 x_t, t, condition 进入 PixelGen
↓
只跑前 K 个 transformer block
↓
用一个很小的 probe head 预测：
    1. full xhat 的 DINO embedding
    或
    2. E5 里得到的 PIS 感知风险
```

也就是说，E7 不是随便用 SEAInput 猜感知变化。

它是让模型自己的前几层先看一眼当前输入，然后预测真正的感知目标。

这个目标可以是：

```text
DINO(xhat_full)
LPIPS/VGG(xhat_full)
PIS：skip 这一步造成的最终感知损伤
```

这就靠谱很多。

---

## 3. 模型结构怎么拆？

PixelGen / JiT 是 transformer。

你可以把它拆成两段：

```text
prefix：embedding + 前 K 个 block
suffix：剩下的 block + final layer
```

例如 PixelGen-XL 有 28 个 block。

可以试：

```text
K = 2
K = 4
K = 6
K = 8
```

K 越小，越快，但预测可能不准。

K 越大，越准，但 skip 的收益变小。

---

## 4. 很重要：refresh 时不要重复算 prefix

E7 的正确流程是：

```text
1. 当前 call 先跑 prefix。
2. probe head 给出 risk。
3. 如果 risk 小：skip，复用 cached output，不跑 suffix。
4. 如果 risk 大：refresh，直接接着刚才的 prefix hidden 继续跑 suffix。
```

这点很重要。

因为如果 refresh 时你重新完整跑一遍模型，就浪费了 prefix 计算。

正确实现应该是：

```text
prefix hidden 不丢掉，refresh 时继续用。
```

---

## 5. probe head 预测什么？

推荐做两个版本。

---

### 版本 A：预测 DINO(xhat_full)

目标：

```text
target = DINO(xhat_full)
```

probe 输入：

```text
前 K 层 hidden tokens
```

输出：

```text
一个 DINO embedding，例如 768 维
```

训练 loss：

```text
cosine loss
```

online 判断：

```text
pred_z_current = probe(prefix_hidden_current)

risk = 1 - cosine(pred_z_current, pred_z_at_last_refresh)
```

如果 risk 超过阈值，就 refresh。

这个版本语义清楚：

```text
它直接预测模型当前 xhat 在 DINO 感知空间里的位置。
```

---

### 版本 B：直接预测 PIS risk

目标来自 E5：

```text
target = PIS_LPIPS 或 PIS_combined
```

probe 输出一个标量：

```text
risk_score
```

训练 loss 可以用：

```text
MSE：预测具体 PIS 数值
或者 BCE：预测 high-risk / low-risk
或者 ranking loss：让危险 call 排在前面
```

最推荐：

```text
ranking loss + BCE
```

因为 cache 更关心“哪些 call 最该 refresh”，不一定需要精确预测数值。

---

## 6. 训练数据从哪里来？

来自 E2 和 E5。

你需要保存：

```text
sample_id
call_index
t
call_kind
prefix_hidden_after_K_blocks
xhat_full
DINO(xhat_full)
LPIPS/VGG features of xhat_full
PIS_LPIPS / PIS_DINO / PIS_PSNR
```

训练集不要太大。

先用：

```text
64 samples * 99 calls
```

就有 6336 条训练样本。

对于一个小 probe head 来说够用了。

---

## 7. 怎么判断 probe 好不好？

不要只看训练 loss。

至少看四个指标。

### 指标 1：DINO embedding cosine similarity

预测的 DINO embedding 和真实 DINO(xhat_full) 越接近越好。

---

### 指标 2：PIS Spearman rank correlation

Spearman 相关看的是排序。

它回答：

```text
probe 认为危险的 call，真的在 PIS 里也危险吗？
```

这比普通 Pearson 更适合 cache。

---

### 指标 3：Top-k recall

例如目标 RR=0.30。

看：

```text
PIS 最危险的 30% call
有多少被 probe 选进 refresh 的 30% call？
```

这个指标很关键。

---

### 指标 4：真实 cache rerun final LPIPS

最终还是要跑真实 cache。

同样 RR 下比较：

```text
PrefixProbe-cache vs SEAInput-online vs DINO-State-cache
```

---

## 8. Online cache 流程

伪代码：

```text
for each denoiser call c:
    if c in forced calls:
        run full model
        cache output
        store probe feature
        continue

    prefix_hidden = run_prefix(x, t, condition, K)
    pred_feature_or_risk = probe(prefix_hidden)

    risk = compare_to_last_refresh(pred_feature_or_risk)

    if risk > threshold or max_skip reached:
        output = run_suffix(prefix_hidden)
        cache output
        update last_refresh_probe_feature
    else:
        output = cached_output
```

---

## 9. E7 的重点实验表

你需要做一个表：

```text
K | target | actual RR | speedup | PSNR | SSIM | LPIPS | overhead
```

例如：

```text
K=2, target=DINO-xhat
K=4, target=DINO-xhat
K=6, target=DINO-xhat
K=4, target=PIS
K=6, target=PIS
```

比较对象：

```text
SEAInput-online
DINO-State-online
LPIPS-State-online
```

---

## 10. E7 的最小成功标准

E7 成功不要求一开始就非常快。

最小成功标准：

```text
PrefixProbe 在相同 RR 下 final LPIPS 明显低于 SEAInput-online。
```

如果它质量好但速度一般，也可以继续优化 K。

如果 K=2 不准，K=4/6 准，说明前几层确实包含感知风险信息。

这就是论文里的重要发现。

---

## 11. 这个方法的论文说法

不要说：

```text
我们训练了一个小模型预测 cache。
```

要说：

```text
我们发现 x-prediction 模型中的 early prefix 已经包含关于 clean-image perceptual trajectory 的信息。
因此，我们用一个轻量 perceptual probe 在完整 denoising 前预测 skip 的感知风险。
```

中文就是：

```text
模型前几层已经能看出这一步对最终感知质量是否重要，
我们用这个信息决定要不要继续跑完整模型。
```

