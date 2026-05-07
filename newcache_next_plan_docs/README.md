# newcache 下一阶段实验方案包

这份方案包面向“专为 x-prediction / PixelGen 类模型设计的感知流形 cache 策略”。它不是泛泛的读论文建议，而是按你当前仓库已经完成的 E0–E4 结果继续往下推进。

建议阅读顺序：

1. `01_phase1_diagnosis.md`：先把你已经做出的结果翻译成“论文证据链”和“下一步缺口”。
2. `02_e5_to_e8_experiment_plan.md`：给出 E5–E8 的主实验路线。
3. `03_online_pma_method_design.md`：把 oracle PMA 变成真正可部署 online cache 的方法设计。
4. `04_baselines_and_ablations.md`：应该补哪些 baseline、ablation 和表格。
5. `05_implementation_blueprint.md`：代码层面怎么改、增加哪些脚本、关键伪代码。
6. `06_metrics_and_evaluation_protocol.md`：怎么评估才像一篇顶会论文。
7. `07_paper_story_aaai_checklist.md`：论文包装、标题、贡献点、风险与 rebuttal 准备。
8. `08_concepts_glossary.md`：一些关键概念的中文解释。
9. `09_minimal_publishable_plan.md`：算力有限时的最小可发表路径。

核心判断：你现在最重要的下一步，不是继续堆更多 oracle 权重，而是把“clean-image perceptual drift 的 oracle 上限”转化为“无需完整 denoiser / 无需运行 DINO-LPIPS 的在线刷新判据”。只有这一步做出来，才会从“有趣现象”变成“可部署方法”。

