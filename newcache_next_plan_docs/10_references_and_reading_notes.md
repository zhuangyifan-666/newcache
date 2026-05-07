# 10. 参考论文与阅读笔记

这份笔记不是完整 survey，而是围绕你当前项目最相关的论文，说明每篇对你的方法有什么启发、应该借什么、不应该被什么带偏。

---

## 1. PixelGen

### 核心内容

PixelGen 是 pixel diffusion + x-prediction + perceptual losses 的代表。它强调：pixel space 里完整图像流形太复杂，模型应该更关注 perceptual manifold。它用 LPIPS 捕捉局部纹理，用 DINO-based loss 捕捉全局语义。

### 对你的启发

PixelGen 的训练阶段发现可以转化成推理阶段问题：

> 如果训练时 perceptual manifold 更重要，那么推理 cache 时 refresh 判据是否也应该定义在 perceptual manifold 上？

这就是你项目的核心动机。

### 需要注意

PixelGen 的 DINO / LPIPS 是训练 loss，不是推理时每步都跑的 cache gate。你的方法必须强调：DINO / LPIPS 只用于 oracle analysis / calibration label / evaluation，在线不跑。

---

## 2. SeaCache

### 核心内容

SeaCache 指出 diffusion trajectory 有 spectral evolution：早期形成低频结构，后期补高频细节。它用 timestep-dependent SEA filter 过滤 input features，在频谱对齐空间里计算距离，再做 cache refresh。

### 对你的启发

SEAInput 是你当前最强的 online baseline，也是连接 noisy input 和 perceptual drift 的桥。你的 E2 已经显示 SEA 与 LPIPS/DINO 的相关性显著强于 Raw。

### 需要注意

SeaCache 的主目标是“频谱演化感知”的 cache，而你的目标是“x-pred clean-image perceptual trajectory”。不要把自己写成 SeaCache 的小改。

---

## 3. TeaCache

### 核心内容

TeaCache 使用 timestep embedding modulated input 的变化来估计 denoiser output 变化，并通过累积距离决定何时 cache。它的卖点是训练自由、适配 DiT/视频扩散。

### 对你的启发

你可以把 RawInput-online 看作 TeaCache 思路在 PixelGen 上的简化版本。为了更公平，可以加 polynomial fitted input distance，作为 `TeaCache-style` baseline。

### 可借鉴点

- accumulated-distance rule；
- threshold 控制 cache ratio；
- 用 input-side proxy 避免完整 denoiser forward。

---

## 4. MagCache

### 核心内容

MagCache 利用 successive residual outputs 的 magnitude ratio 规律来做 cache，强调 magnitude-aware 和单样本校准。

### 对你的启发

x-pred PixelGen 可以定义：

```text
residual = xhat_t - x_t
velocity = residual / (1-t)
```

然后分析 residual norm / velocity norm 是否有稳定演化规律。如果有，就加入 PMA proxy；如果没有，也可以作为负结果说明 x-pred pixel diffusion 与 latent video diffusion 的差异。

---

## 5. TaylorSeer

### 核心内容

TaylorSeer 不只是 reuse，而是用 Taylor 展开预测未来特征，从“复用”走向“预测”。

### 对你的启发

在低 RR 下，单纯复用可能误差太大。你可以在 E7 测试：

```text
xhat_forecast = xhat_last + alpha * (xhat_last - xhat_prev)
```

如果 forecast 有效，你的方法会更强：既知道何时 refresh，也知道 skip 时如何更好地近似。

---

## 6. DiCache

### 核心内容

DiCache 强调让 diffusion model 自己决定 cache，通过 shallow / intermediate block probe 预测最终 output 变化，并做动态轨迹对齐。

### 对你的启发

你现在用 first-block AdaLN-modulated proxy。可以做一个 ablation：first block vs middle block vs first+middle。如果 first block 足够强，就说明低开销 proxy 是合理选择；如果 middle block 明显更准但更慢，就可以写成 speed-quality trade-off。

---

## 7. ToCa

### 核心内容

ToCa 是 token-wise feature caching，发现不同 token 对质量的敏感度不同，有些 token 错误复用破坏性更大。

### 对你的启发

不建议你主线做 token-wise cache，因为工程复杂。但可以做 spatial risk map：把 SEA delta reshape 成空间图，看哪些区域容易造成 final error。

---

## 8. FreqCa

### 核心内容

FreqCa 更细地分析频率 band：不同频段的 similarity 和 continuity 不同，低频、高频适合不同复用策略。

### 对你的启发

你的 SEAInput 已经是频谱方向。FreqCa 可以启发两个 ablation：

- static low-pass filter vs SEA filter；
- low/mid/high band 的 distance 与 LPIPS/DINO 的相关性。

这能证明 SEA 的 timestep-dependent 设计比简单低通更合理。

---

## 9. SpeCa / speculative caching

### 核心内容

Speculative caching 通常是先预测或复用后续特征，再用某种验证机制判断是否可靠。

### 对你的启发

你可以把 uncertainty gate 解释成 lightweight verification：当 proxy 对风险不确定时，就 refresh，相当于保守验证。

---

## 10. 你的相关工作章节建议分类

建议分四类写：

### 10.1 Diffusion inference acceleration

包括减少 step、蒸馏、量化、稀疏注意力、token pruning。简单带过，因为你不是做这些。

### 10.2 Caching-based acceleration

重点写：DeepCache、TeaCache、SeaCache、MagCache、TaylorSeer、DiCache、ToCa。

### 10.3 Pixel diffusion and x-prediction

重点写：JiT / PixelGen。说明 x-pred 模型直接输出 clean image prediction，是你的方法基础。

### 10.4 Perceptual supervision / perceptual metrics

写 LPIPS、DINO / DINOv2、PixelGen 的 perceptual losses。说明你把训练时的 perceptual manifold 观点转到推理时 cache decision。

---

## 11. 你论文里最关键的对比句

可以这样写：

```text
Unlike prior cache methods that measure redundancy in noisy input or hidden feature spaces, we exploit the clean-image prediction trajectory exposed by x-prediction models and approximate its perceptual drift with a lightweight online proxy.
```

中文：

> 与以往在带噪输入或隐藏特征空间中度量冗余的 cache 方法不同，我们利用 x-prediction 模型暴露出的 clean-image prediction 轨迹，并用轻量在线代理近似其感知变化。

---

## 12. 阅读顺序建议

如果你基础较弱，建议按这个顺序读：

1. PixelGen：理解 x-pred 和 perceptual manifold。
2. SeaCache：理解 spectral evolution 和 cache distance。
3. TeaCache：理解 accumulated-distance dynamic cache。
4. MagCache：理解 residual magnitude 作为风险信号。
5. TaylorSeer：理解 forecast 替代 reuse。
6. DiCache / ToCa / FreqCa：作为补充，不要一开始深陷。

---

## 13. 论文写作时的引用策略

在 Introduction 里：

- 引 PixelGen 说明 x-pred pixel diffusion 和 perceptual losses；
- 引 SeaCache 说明 spectral evolution-aware cache；
- 引 TeaCache / MagCache / TaylorSeer / DiCache 说明现有 cache 方法主要在 raw/noisy/intermediate spaces 里做刷新或预测。

在 Method 里：

- 引 LPIPS / DINO 说明感知空间来源；
- 引 SeaCache 说明 SEA filter 作为 proxy component；
- 引 TeaCache 说明 accumulated-distance rule。

在 Experiments 里：

- 诚实说明哪些是 official baseline，哪些是 style implementation。

