# AxCore 阅读笔记与思考

更新时间：2026-06-27 12:06 +08:00

## 为什么先从 Profile/Figure 2 开始

Profile/Figure 2 虽然不是完整 AxCore 实验，但它很适合作为第一个切入口。它回答了一个基本系统问题：

在大规模自回归 LLM decode 中，随着序列长度增加，计算量究竟主要还在 Linear/GEMM 层，还是已经明显转向 Attention？

这点很重要，因为 AxCore 的近似计算故事主要围绕 GEMM/Linear 层展开。在考虑替换乘法器之前，先要理解为什么 GEMM 是值得优化的目标。

## 当前概念性结论

对于 OPT-175B 和 LLaMA-3.1-405B，脚本结果显示：

- 较短和中等序列长度下，Linear OPs 占绝对主导。
- 随着上下文长度增长，Attention OPs 占比上升。
- 即使到较长上下文，Linear OPs 仍然可能占据很大比例。

这支持 AxCore 论文优化低比特 GEMM 的动机，但也引出一个值得继续思考的问题：

如果目标是长上下文 LLM 推理，那么“优化 GEMM/乘法器”和“优化 Attention/KV cache/访存移动”的边界在哪里？

## 后续可能的替换/插入层次

后续需要继续读代码，不能过早下结论。不过从当前仓库结构看，可能的插入层次包括：

- 软件数值模拟：`Software\AxCore\approximation_computation`
- 性能/能耗仿真：`Software\axcore_simulator\AxCore`
- 硬件实现：`Hardware\AxCore\hw\spinal\AxCore`

## 需要持续追踪的问题

1. AxCore 的 software evaluation 是在张量层面模拟近似 INT8 计算，还是依赖自定义 kernel？
2. simulator 是否显式建模 multiplier 类型，还是只通过吞吐、面积、能耗参数间接体现？
3. LUT-based approximate multiplier 是否可以先用 Python lookup table 表示，然后再迁移到 SpinalHDL？
4. 最小 LLM 量化实验应该用什么 accuracy 指标：perplexity、小规模 zero-shot subset，还是 layerwise error？
5. 如何设计一个最小实验，把 Adana 的量化思路和 AxCore 的 approximate GEMM 思路连接起来？

## 第一次 simulator 尝试带来的启发

第一次 simulator 失败不是单纯的环境问题，它暴露了 AxCore simulator 实际耦合了两个层次：

- 面向 LLM layer 的架构级 cycle/memory traffic simulation；
- 通过 Cacti-derived table 做 SRAM energy modeling。

对学习和项目推进来说，这两个层次可以暂时拆开：

- 如果目标是理解 dataflow、cycles 和 benchmark 结构，可以先使用近似或缓存的 SRAM cost。
- 如果目标是复现论文中的 energy/TOPS/W 结果，就必须正确构建并使用 Cacti。

后续写任何报告时都应该保留这个区分：

bring-up experiment 不等于 paper-faithful reproduction。
