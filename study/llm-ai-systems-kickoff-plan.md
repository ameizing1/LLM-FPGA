# LLM AI Systems 14 天启动计划

更新时间：2026-06-20

## 目标

用 14 天完成从“知道大模型推理很重要”到“能跑通、测量、解释一个 LLM 推理瓶颈”的启动闭环。

这 14 天只围绕一个能力增量：

**理解并测量 LLM 推理中的上下文长度、batch size、KV Cache 与显存/延迟/吞吐之间的关系。**

暂时不追求：

- 系统学习全部机器学习
- 训练大模型
- 完整学习 CUDA
- 设计芯片架构
- 写论文

## 为什么先学这个

你未来想走 AI Systems / 软硬件协同 / AI 芯片方向，第一步不是先啃所有 AI 算法，而是先建立一个系统直觉：

> 大模型推理为什么慢、为什么贵、为什么吃显存，硬件瓶颈在哪里？

这个问题会连接你后面所有方向：

- KV Cache 压缩
- Attention 优化
- 量化
- GPU/NPU 架构
- 存储层次
- HBM 带宽
- 编译器和 kernel 优化
- 推理服务调度

## 当前起点判断

你的优势：

- 微电子和集成电路背景强。
- 学过 C++、数据结构、操作系统、组成原理、网络等 CS 课程。
- 毕设已经涉及长文本大模型、压缩注意力和软硬件协同。

当前最需要补的不是“会不会写代码”，而是：

1. Transformer 推理阶段的关键结构。
2. KV Cache 为什么成为长上下文瓶颈。
3. 如何用真实实验测量 latency、throughput、memory。
4. 如何把测量结果解释成软硬件协同问题。

## 第一天先学什么

今天只学四个概念：

1. **Transformer Decoder**
2. **Attention**
3. **Prefill / Decode**
4. **KV Cache**

### 要达到的标准

今天结束时，你应该能用自己的话讲清楚：

- 为什么 LLM 生成不是一次性输出全文，而是 token by token。
- prefill 阶段和 decode 阶段分别在做什么。
- KV Cache 缓存的 K 和 V 是什么。
- 为什么上下文越长，KV Cache 显存越大。
- 为什么 decode 阶段经常是 memory-bound，而不只是 compute-bound。

### 今天不用学

- 反向传播
- 训练流程
- RLHF
- MoE 细节
- CUDA kernel 优化
- 芯片 RTL

这些都以后再说。

## 14 天安排

### Day 1：理解 LLM 推理的最小框架

学习：

- Transformer decoder-only 模型结构
- Attention 中 Q、K、V 的作用
- prefill 和 decode 的区别
- KV Cache 的目的

产出：

- 写一页笔记：《LLM 推理为什么需要 KV Cache》

检查：

- 能画出 input tokens -> prefill -> KV Cache -> decode -> next token 的流程。

### Day 2：补 KV Cache 的显存估算

学习：

- KV Cache size 的粗略公式
- batch size、sequence length、hidden size、num layers、precision 如何影响显存

产出：

- 写一个小表格：不同 sequence length 下 KV Cache 大小如何增长。

检查：

- 能解释为什么 128K context 会给推理系统带来巨大压力。

### Day 3：跑通一个最小 LLM 推理 demo

学习：

- Python 环境
- Hugging Face transformers 或 vLLM quickstart
- 模型加载和生成

产出：

- 跑通一个小模型生成。
- 记录模型名称、显存占用、输入长度、输出长度、生成耗时。

检查：

- 能区分模型权重显存和 KV Cache 显存。

### Day 4：测量 latency 和 throughput

学习：

- latency
- throughput
- tokens/s
- TTFT：time to first token
- TPOT：time per output token

产出：

- 对同一个 prompt 跑 3 次，记录平均生成速度。

检查：

- 能说明为什么只看总耗时不够。

### Day 5：改变 prompt 长度

学习：

- context length 对 prefill 的影响
- long context 对 attention 和 KV Cache 的影响

产出：

- 构造短、中、长三个 prompt，记录 TTFT、TPOT、显存。

检查：

- 能解释为什么长 prompt 通常会明显增加 TTFT。

### Day 6：改变 batch size

学习：

- batching
- batch size 对吞吐和显存的影响
- latency-throughput trade-off

产出：

- batch size = 1、2、4 的实验表格。

检查：

- 能解释为什么 batch 变大可能提高吞吐，但也可能增加单请求延迟和显存压力。

### Day 7：第一周复盘

产出：

- 写一份 2-3 页短报告：《LLM 推理基础实验复盘》

报告结构：

- 实验环境
- 模型和框架
- 测量指标
- 三个变量：context length、batch size、输出长度
- 你观察到的瓶颈
- 你还不确定的问题

### Day 8：学习 PagedAttention / continuous batching

学习：

- 为什么传统 KV Cache 管理会浪费显存
- vLLM 的 PagedAttention 解决什么问题
- continuous batching 解决什么问题

产出：

- 写一页笔记：PagedAttention 和操作系统虚拟内存分页的类比。

检查：

- 能解释为什么这个问题和操作系统知识有关。

### Day 9：学习量化对推理的影响

学习：

- FP16、BF16、INT8、INT4 的基本区别
- weight quantization 和 KV Cache quantization 的区别

产出：

- 对比至少两种精度或查阅一个量化实验结果，记录准确率、显存、速度的 trade-off。

检查：

- 能说明“量化不等于一定更快”，因为可能受 kernel 和硬件支持影响。

### Day 10：学习 profiling 思维

学习：

- GPU utilization
- memory bandwidth
- compute-bound vs memory-bound
- roofline model 的直觉

产出：

- 用 nvidia-smi 或 profiler 记录一次推理过程中的显存和 GPU 利用率。

检查：

- 能初步判断某个实验更像算力瓶颈还是访存/显存瓶颈。

### Day 11：读一篇系统论文或技术文档

推荐：

- vLLM / PagedAttention 相关论文或文档
- FlashAttention 相关论文或技术解读
- TensorRT-LLM 文档中的 performance 章节

产出：

- 写半页论文笔记：它解决了什么瓶颈，怎么解决，有什么代价。

### Day 12：把问题转成软硬件协同语言

学习：

- 算法优化、系统优化、硬件优化之间的区别
- 为什么同一个模型优化不一定带来真实硬件收益

产出：

- 选一个观察到的瓶颈，写出三种可能优化：
  - 算法层
  - 系统/运行时层
  - 硬件层

### Day 13：形成课题候选清单

产出：

写 3 个候选课题，每个课题包含：

- 研究问题
- 为什么重要
- baseline
- 可能方法
- 实验指标
- 需要的资源

建议候选：

1. 长上下文 LLM 的 KV Cache 压缩与显存带宽分析
2. 面向 decode 阶段的 attention/KV 访存优化
3. 面向 LLM 推理的 batch 调度和显存管理实验

### Day 14：整理给导师看的材料

产出：

- 一份 5-8 页技术报告
- 一个实验表格
- 一个课题候选清单
- 一个后续 1 个月计划

标题建议：

《长上下文 LLM 推理瓶颈分析与软硬件协同优化机会》

## 第一周最小阅读清单

优先级从高到低：

1. Transformer / Attention 的最小教程
2. Hugging Face text generation 或 vLLM quickstart
3. vLLM PagedAttention 介绍
4. FlashAttention 的直观解释
5. KV Cache memory formula 相关文章或文档

不建议第一周读太多论文。先跑实验，再读论文，会更容易看懂。

## 第一个练习题

不用查资料，先尝试回答：

> 为什么 LLM 推理时，prefill 阶段更像“大矩阵并行计算”，而 decode 阶段更容易被 KV Cache 访存拖慢？

如果能回答清楚，说明你已经进入这个方向的正确门口。

## 验收标准

14 天后，最低验收标准：

- 跑通一个 LLM 推理 demo。
- 记录至少 3 组上下文长度实验。
- 记录至少 3 组 batch size 实验。
- 能解释 TTFT、TPOT、tokens/s、显存占用。
- 能写出 KV Cache 显存估算公式。
- 能提出 2-3 个软硬件协同优化课题。

优秀验收标准：

- 有一份 5-8 页技术报告。
- 有一张清晰实验表格。
- 有一个可复现脚本。
- 能和导师讨论具体课题，而不是泛泛说“我想做 AI 芯片”。
