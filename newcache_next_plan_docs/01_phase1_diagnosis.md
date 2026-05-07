# 01. 你已经做出的结果意味着什么

## 1. 当前项目的核心发现

你的仓库已经完成了一个很完整的第一阶段。用论文语言概括，它已经证明了下面几件事。

第一，PixelGen / JiT 这类 x-pred pixel diffusion 的采样轨迹有一个特殊性质：denoiser 每次不是直接预测噪声，也不是只输出 velocity，而是先预测 clean image estimate，即 `xhat_t = f_theta(x_t, t, c)`，再通过 `(xhat_t - x_t)/(1-t)` 转成速度。因此 refresh 判据不一定只能建立在 noisy input 或 hidden feature 上，也可以建立在 `xhat_t` 的感知变化上。

第二，你的 E1 已经说明 SEAInput-online 明显优于 RawInput-online 和 Uniform，尤其在 RR≈0.30 的低刷新预算下，SEAInput-online 比 RawInput-online 的 PSNR 高约 5.82 dB，LPIPS 从 0.1102 降到 0.0461。这说明：单纯看未滤波的 early token proxy 很容易被无意义 drift 干扰，而 spectral filtering 确实是一个更稳的 cache signal。

第三，E2 的 distance bank 给出了非常关键的诊断：Raw 与 DINO / LPIPS 的相关性弱，而 SEA 与 DINO / LPIPS 的相关性明显更强。尤其在 predictor→corrector 且去掉第一个巨大 spike 后，SEA-LPIPS correlation 达到 0.813；Raw-LPIPS 和 Raw-DINO 甚至接近无关或轻微负相关。这一点是你论文里非常有价值的“机制证据”。

第四，E2 还暴露了 Heun exact 的特殊结构：99 个 call opportunity 中，distance mass 几乎集中在 predictor→corrector transition。这个发现非常重要，因为它说明你的 cache 策略不应该只按 50 个 solver step 设计，而应该 solver-aware，也就是意识到 predictor call 和 corrector call 的角色不同。

第五，E4 的真实 cache rerun 已经证明 refresh 的位置比数量重要。Uniform 在相同 RR 下明显弱于 oracle schedules；RawInput-oracle 也明显弱于 SEAInput-oracle，特别是 RR≈0.30 / 0.40。

第六，PMA clean-image perceptual branch 有上限价值，但强烈依赖 refresh budget。RR≈0.50 时 PMA-no-gate 明显超过 SEAInput-oracle；RR≈0.30 时 SEAInput-oracle 仍然最强。这个现象说明：DINO / LPIPS early spike 不是纯噪声，也不是永远可信；它需要更细的 stage / call-kind / budget 适配。

## 2. 目前离一篇论文还缺什么

目前结果是非常好的探索性结果，但还没有形成一个真正可部署的 cache 方法。关键原因是：E2–E4 中的 DINO / LPIPS / PMA schedule 是 oracle，即它们提前知道 full uncached trajectory 上的 `xhat_t`。真实推理时，你如果为了决定“当前 call 是否 refresh”而先跑完整 denoiser 得到 `xhat_t`，那就已经失去 cache 的意义了。

所以，下一阶段的核心目标应该是：

> 用 cheap online proxy 预测 clean-image perceptual drift，然后用这个预测值做 refresh gate。

这里的 cheap online proxy 可以来自：

- SEA-filtered first-block proxy；
- call kind：predictor / corrector；
- timestep / timeshift 后的 t；
- cache age：连续复用次数；
- residual magnitude：最近一次 denoiser output 的范数或变化率；
- 最近两次 refresh 的 xhat / velocity 变化；
- 小型 calibration regressor 的输出。

注意，这里的“小型 regressor”不一定是重训练模型。它可以是 64 个 calibration samples 上拟合的 ridge regression、isotonic regression、quantile regression 或很浅的 MLP。论文里可以叫 lightweight calibration 或 calibration-only predictor。和训练扩散模型不是一个量级。

## 3. 你现在最有潜力的论文卖点

不要把论文包装成“我把 SeaCache 和 PixelGen 拼起来”。这样创新会显得不够。更好的卖点是：

> x-prediction pixel diffusion exposes a clean-image prediction trajectory, and this trajectory admits a perceptual-manifold-aware cache criterion. We show that spectral input drift is a bridge to this perceptual manifold, and we design an online proxy to approximate perceptual drift without running the full denoiser.

中文解释：

- x-pred 模型天然给出 `xhat`，它比 noisy input 更接近人眼看到的图像；
- 感知流形不是抽象口号，而是由 DINO / LPIPS / spectral signal 共同定义的“对人眼和语义重要的变化空间”；
- SEA 是从 noisy feature 到 perceptual drift 的桥；
- online PMA 是真正可部署的刷新策略。

## 4. 当前代码结构中的优势和风险

### 优势

你的 `src/diffusion/flow_matching/e1_cache.py` 已经实现了比较干净的 cache controller 框架，包括 AlwaysRefresh、Uniform、OnlineInputCacheController。它已经能做：

- 统一统计 refresh ratio / hit rate / denoiser time；
- first-block AdaLN-modulated proxy 抽取；
- SEA filter；
- accumulated-distance refresh rule；
- warmup calls / final call forced refresh / max skip。

这意味着你后续最好不要推倒重来，而是在这个 controller 框架上扩展 `PMAOnlineCacheController`。

### 风险

第一，当前主结果大多是 paired full-reference fidelity，即“cache 后的结果是否接近 uncached full trajectory”。这对 cache paper 很重要，但还不够。你还需要补生成质量指标，例如 ImageNet FID / IS / Precision / Recall，至少 5k quick FID，最好 10k 或 50k。

第二，目前样本数偏小。E4 test 是 192 samples，适合快速验证，但顶会主表最好至少 512 / 1k paired samples；FID 则需要更大规模。

第三，现有 PMA-stage-aware 是手工权重。手工权重可以作为 heuristic，但论文主方法最好有更系统的来源，例如：

- calibration set 自动拟合权重；
- per-stage / per-call-kind 的 rank normalization；
- uncertainty-aware gate；
- threshold controller 保证目标 RR。

第四，真实 wall-clock speedup 需要非常谨慎。refresh ratio 不等于真实加速比。DINO / LPIPS 如果在线运行，会显著增加开销；所以主方法必须明确“不在在线推理时运行 DINO / LPIPS”。

## 5. 下一阶段的总目标

建议把下一阶段命名为：

- E5: Perceptual proxy fitting
- E6: Online PMA cache
- E7: Solver-aware and forecast-aware cache
- E8: Paper-scale evaluation

一句话目标：

> 从 oracle perceptual upper bound 走向 deployable online perceptual-manifold-aware cache，并在 matched latency / matched RR 下超过 SEAInput-online、TeaCache-style、MagCache-style 和 Uniform。

