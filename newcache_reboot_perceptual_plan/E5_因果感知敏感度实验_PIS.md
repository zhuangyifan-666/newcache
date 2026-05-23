# E5｜因果感知敏感度实验：先找出“哪些 call 真的不能 skip”

## 1. 这个实验要解决什么问题？

以前我们看的是：

```text
相邻两个 xhat 之间的 DINO / LPIPS 距离
```

但现在我们换一个更靠谱的问题：

```text
如果只跳过某一个 denoiser call，最终图片会不会变差？
```

这就是 E5 的目标。

E5 的输出不是一个 online cache 方法，而是一个“老师”。

这个老师会告诉我们：

```text
哪些 call 是真正危险的？
哪些 call 其实可以 skip？
```

---

## 2. 核心定义：PIS

PIS 全称：

```text
Perceptual Intervention Score
中文：感知干预分数 / 感知损伤分数
```

定义：

```text
PIS(sample, call_c)
= 只在 call_c 上 skip 一次以后，最终图和 full reference 的感知距离
```

可以用三个指标表示：

```text
PIS_LPIPS：最终图 LPIPS，越大越危险
PIS_DINO ：最终图 DINO 距离，越大越危险
PIS_PSNR ：最终图 PSNR，越低越危险
```

最后可以合成一个分数：

```text
PIS = normalize(LPIPS) + normalize(DINO) - normalize(PSNR)
```

实际做实验时，先不要急着复杂融合。

最小版本只用：

```text
PIS_LPIPS
```

因为 LPIPS 很直观：越大表示人眼感知差异越大。

---

## 3. 为什么这个实验更靠谱？

因为它不是在猜。

它直接做了一个“干预”：

```text
原来这一步正常算。
现在我故意只跳过这一步。
然后看最终结果坏没坏。
```

这类似医学里的“做实验看因果关系”。

旧方案更像是：

```text
SEAInput 和 LPIPS 好像有相关性，所以猜 SEAInput 可以代表感知变化。
```

新方案是：

```text
我真的把这一步 skip 掉，然后看它对最终图片造成了什么后果。
```

这比相关性分析强很多。

---

## 4. 最小可运行版本

你先不要一上来做 192 或 256 张图。

先做：

```text
samples = 8
calls per sample = 99
```

也就是最多需要测试：

```text
8 * 99 = 792 个单点 skip 实验
```

听起来很多，但可以分批跑。

如果觉得还是太多，可以先做：

```text
samples = 4
只测每隔 2 个 call：0, 2, 4, ..., 98
```

先观察趋势。

---

## 5. 实验步骤

### Step 1：跑 full reference，并保存每个 call 前的状态

你需要保存：

```text
sample_id
call_index
current x_t 或 x_call_input
current t
call_kind：predictor / corrector
full output：xhat 或 v
最终 full image
```

最重要的是保存每个 call 开始前的 `x`。

因为后面做单点 skip 时，可以从这个 `x` 开始重放 suffix，不必从第 0 步重新跑。

---

### Step 2：对每个 call 做一次 single-skip intervention

对某个 sample 的某个 call c：

```text
1. 从 full trajectory 里读取 call c 开始前的 x。
2. 在 call c 这里不要跑 denoiser。
3. 复用上一次可用的 cached output。
4. 从 call c+1 开始，后面所有 call 都正常 full compute。
5. 得到一个 final image。
6. 和 full final image 比较 LPIPS / DINO / PSNR。
```

注意：

```text
只 skip 一个 call。
不要连续 skip 多个。
```

这样才能知道“这个 call 自己”的影响。

---

### Step 3：得到 PIS bank

输出一个文件：

```text
outputs/e5_pis_bank/pis_bank.npz
```

里面至少保存：

```text
pis_lpips: [num_samples, 99]
pis_dino:  [num_samples, 99]
pis_psnr:  [num_samples, 99]
call_kind: [99]
t_values:  [99]
```

再输出一个 CSV：

```text
outputs/e5_pis_bank/pis_summary.csv
```

里面每行是：

```text
sample_id, call_index, t, call_kind, pis_lpips, pis_dino, pis_psnr
```

---

## 6. 你要画哪些图？

### 图 1：平均 PIS 曲线

横轴：call index，0 到 98。

纵轴：平均 PIS_LPIPS。

这张图回答：

```text
哪些 timestep / call 区域最危险？
```

---

### 图 2：PIS 热力图

横轴：call index。

纵轴：sample。

颜色：PIS_LPIPS。

这张图回答：

```text
危险 call 是所有样本都一样，还是每张图都不一样？
```

如果每张图差异很大，说明 fixed schedule 不够，dynamic cache 有必要。

---

### 图 3：predictor / corrector 分开统计

你的 Heun exact 有：

```text
predictor calls
corrector calls
```

分别统计它们的 PIS。

这张图回答：

```text
到底 predictor 更危险，还是 corrector 更危险？
```

这是你论文里很有价值的分析点。

---

### 图 4：PIS 和旧指标的 top-k overlap

比较：

```text
PIS 最危险的前 30% call
和 SEAInput / DINO-xhat / LPIPS-xhat 认为最危险的前 30% call
重合多少？
```

不要只看 Pearson 相关系数。

因为 cache 更关心：

```text
最应该 refresh 的那些 call 有没有选对？
```

所以 top-k overlap 更重要。

---

## 7. 判断结果好坏的标准

### 情况 A：PIS 有明显结构

例如：

```text
某些 early/middle call 特别危险
corrector 比 predictor 更危险
或者每个 sample 的危险位置不一样
```

这说明你的研究有很好的切入点。

下一步就做 E6/E7，设计 online 方法去预测这个 PIS。

---

### 情况 B：PIS 和旧 PMA 很接近

这说明你之前的 PMA oracle 其实还不错。

但现在你有了更强的证据：

```text
PMA 不是只看起来合理，它确实接近 single-skip final perceptual damage。
```

这会让论文更稳。

---

### 情况 C：PIS 和旧 PMA 差很多

这也不是坏事。

这说明之前的 PMA 方向确实有问题，而你现在找到了更真实的目标。

然后 E7 就应该直接学习 PIS，而不是学习 PMA。

---

## 8. 建议的命名

你可以把这个实验叫做：

```text
Causal Perceptual Sensitivity Analysis
```

中文：

```text
因果感知敏感度分析
```

这个名字比 “distance bank” 更像论文。

---

## 9. 最小成功标准

E5 不要求你马上超过 SeaCache 或 TeaCache。

E5 成功的标准是：

```text
你能清楚画出：哪些 call skip 后真的伤害最终图。
```

只要这件事做出来，你后面的 online cache 就有了可靠目标。

