# 入学前学习计划：大模型推理优化、AI Systems 与软硬件协同

更新时间：2026-06-20

## 目标

入学前的目标不是提前把整个研究生课题确定下来，而是完成一个小闭环：

**理解大模型推理的基本过程，能用系统和体系结构视角解释其主要性能瓶颈，并做出一个可复现的小实验报告。**

更具体地说，入学前最好做到：

1. 能讲清 Transformer 推理、Attention、prefill/decode、KV Cache 的基本机制。
2. 能理解 latency、throughput、TTFT、TPOT、显存占用、带宽瓶颈等推理系统指标。
3. 能跑通一个开源推理框架或简单推理 demo。
4. 能做一组小实验，观察上下文长度、batch size、量化方式对延迟、吞吐、显存的影响。
5. 能把实验结果整理成 5-8 页技术报告，用来和导师或师兄师姐讨论后续方向。

## 当前定位

入学前建议先围绕这个方向准备：

**大模型推理优化、AI Systems、体系结构与软硬件协同**

这个方向暂时不等于最终研究课题。它只是一个适合你当前背景的探索入口：

- 它和微电子、集成电路、计算机系统基础有交集。
- 它不要求你一开始就转成纯 AI 算法研究。
- 它能帮助你理解 AI 产业和科研中真实的系统瓶颈：推理成本、显存、带宽、延迟、吞吐和能耗。
- 它后续可以自然延伸到 AI 加速器、体系结构、AI 编译器、算子优化、模型压缩硬件友好性等方向。

## 入学前不急着做什么

这些内容重要，但不建议作为入学前第一优先级：

- 系统学习全部机器学习。
- 从零训练大模型。
- 完整系统学习 CUDA。
- 直接设计一个芯片架构。
- 过早确定硕士论文题目。

原因不是它们不重要，而是它们范围太大。入学前更适合先抓住一条主线：**大模型推理到底慢在哪里、贵在哪里、硬件和系统瓶颈在哪里。**

## 能力拆解

### 能力 1：理解大模型推理流程

需要掌握：

- Transformer decoder-only 结构
- Attention 中 Q/K/V 的含义
- FFN、LayerNorm、RoPE 的基本作用
- prefill 和 decode 的区别
- token-by-token generation
- KV Cache 的作用和显存开销

掌握标准：

- 能画出 input prompt -> prefill -> KV Cache -> decode -> next token 的流程。
- 能解释为什么 prefill 更像矩阵并行计算，而 decode 更容易受访存和 KV Cache 影响。

### 能力 2：理解推理系统性能指标

需要掌握：

- latency
- throughput
- TTFT: time to first token
- TPOT: time per output token
- tokens/s
- GPU memory usage
- GPU utilization
- compute-bound vs memory-bound

掌握标准：

- 能说明为什么只看总耗时不够。
- 能区分用户体验相关指标和系统吞吐相关指标。
- 能初步判断瓶颈更可能来自计算、显存、带宽还是调度。

### 能力 3：跑通和测量一个小实验

需要掌握：

- Python/PyTorch 基础调用
- Hugging Face transformers 或 vLLM 的基本使用
- 简单 benchmark 脚本
- 显存和耗时记录
- 实验表格整理

掌握标准：

- 能跑通一个开源模型的推理。
- 能记录上下文长度、batch size、输出长度、耗时、显存等信息。
- 能把结果整理成表格，而不是只停留在“跑起来了”。

### 能力 4：建立软硬件协同视角

需要掌握：

- 为什么模型优化不一定带来真实硬件收益。
- 为什么量化、稀疏、KV Cache 压缩要看硬件和系统支持。
- 为什么 data movement、memory bandwidth、cache/on-chip memory 很关键。
- 算法层、系统层、硬件层分别能做什么。

掌握标准：

- 能对一个瓶颈分别提出算法、系统、硬件三个层面的可能优化。
- 能说明某个优化可能带来的收益和代价。

## 6 周学习安排

如果入学前时间充裕，建议按 6 周执行；如果时间不足，可以压缩成 4 周，只保留每周的核心产出。

### 第 1 周：Transformer 与推理流程

学习重点：

- Transformer decoder
- Attention、Q/K/V
- FFN、LayerNorm、RoPE
- prefill / decode
- KV Cache

推荐资料：

- The Illustrated Transformer
- Hugging Face LLM Course
- Hugging Face KV Cache 文档

产出：

- 一页笔记：《LLM 推理为什么需要 KV Cache》
- 一张流程图：prompt -> prefill -> KV Cache -> decode

验收标准：

- 能用自己的话解释 prefill 和 decode 的区别。
- 能解释 KV Cache 为什么会随着上下文长度增长。

### 第 2 周：推理框架与最小 demo

学习重点：

- Hugging Face transformers 或 vLLM quickstart
- 模型加载和生成
- 输入长度、输出长度、生成耗时
- 显存占用

推荐资料：

- Hugging Face text generation 文档
- vLLM 官方文档

产出：

- 跑通一个小模型推理 demo。
- 记录至少 3 次生成实验。

验收标准：

- 能区分模型权重显存和 KV Cache 显存。
- 能说明模型大小、上下文长度、输出长度分别影响什么。

### 第 3 周：性能指标与 benchmark

学习重点：

- latency、throughput
- TTFT、TPOT
- tokens/s
- batch size
- GPU memory / utilization

推荐资料：

- vLLM benchmark 相关文档
- TensorRT-LLM performance 相关文档
- Nsight Systems 入门材料

产出：

- 一个简单 benchmark 表格。
- 至少比较 3 组不同上下文长度。
- 至少比较 3 组不同 batch size。

验收标准：

- 能解释为什么长 prompt 通常增加 TTFT。
- 能解释为什么 batch size 变大可能提高吞吐，但也增加显存压力。

### 第 4 周：KV Cache、PagedAttention 与显存管理

学习重点：

- KV Cache size 粗略公式
- PagedAttention 的基本思想
- continuous batching
- 显存碎片和显存管理

推荐资料：

- Hugging Face KV Cache 文档
- vLLM / PagedAttention 论文或解读

产出：

- 一张 KV Cache 显存估算表。
- 一页笔记：《PagedAttention 解决了什么问题》

验收标准：

- 能解释为什么 KV Cache 管理像操作系统的内存管理问题。
- 能说明 PagedAttention 主要想减少什么浪费。

### 第 5 周：模型压缩与硬件友好性

学习重点：

- quantization
- sparsity
- low-rank 的直觉
- KV Cache compression
- 为什么压缩不一定等于真实加速

推荐资料：

- MIT 6.5940 中 efficient inference / quantization 相关内容
- SmoothQuant、GPTQ 或相关技术解读
- Efficient Processing of Deep Neural Networks 中 data movement / precision 相关材料

产出：

- 一页对比笔记：量化、稀疏、KV Cache 压缩分别优化什么。
- 选择一个压缩方向，写出它的硬件收益和潜在代价。

验收标准：

- 能解释为什么 INT4/INT8 不一定在所有硬件上都更快。
- 能说明模型压缩需要同时看准确率、延迟、显存、带宽和硬件支持。

### 第 6 周：整理技术报告与方向问题

学习重点：

- 汇总前 5 周实验和笔记。
- 从系统和硬件角度解释瓶颈。
- 形成后续可讨论的方向问题。

产出：

- 一份 5-8 页技术报告。
- 一个实验表格。
- 2-3 个后续可能探索的问题。

报告建议题目：

《大模型推理瓶颈分析与软硬件协同优化机会》

报告建议结构：

1. 背景：为什么关注 LLM inference。
2. 基础：prefill/decode、KV Cache、性能指标。
3. 实验设置：模型、框架、硬件环境、变量。
4. 实验结果：上下文长度、batch size、量化方式对性能的影响。
5. 分析：瓶颈可能来自计算、显存、带宽还是调度。
6. 后续问题：可进一步探索的 2-3 个研究切入点。

验收标准：

- 报告能让导师或师兄师姐看出你已经完成入门调研和基本实验。
- 能根据反馈决定后续是偏推理系统、算子优化、体系结构、AI 加速器，还是模型压缩硬件友好性。

## 推荐资料顺序

### 第一层：马上开始

1. The Illustrated Transformer
2. Hugging Face LLM Course
3. Hugging Face KV Cache 文档
4. vLLM 官方文档

### 第二层：实验时查

1. TensorRT-LLM 官方文档
2. Nsight Systems / Nsight Compute 入门
3. vLLM benchmark 文档

### 第三层：方向深化

1. Stanford CS336: Language Modeling from Scratch
2. CMU 10-414/714 Deep Learning Systems
3. MIT 6.5940 TinyML and Efficient Deep Learning Computing
4. Efficient Processing of Deep Neural Networks
5. Computer Architecture: A Quantitative Approach

## 每周时间分配

如果每周有 10 小时：

- 4 小时看课程/文档。
- 3 小时跑实验或写脚本。
- 2 小时整理笔记和表格。
- 1 小时复盘，记录问题。

如果每周有 20 小时：

- 7 小时看课程/文档。
- 7 小时实验。
- 4 小时写报告/笔记。
- 2 小时读论文或和同学讨论。

## 和导师沟通前应准备的内容

不需要准备得很完美，但最好带着这些内容去问：

- 你想准备的方向：大模型推理优化、AI Systems、体系结构与软硬件协同。
- 你已经学到的概念：Transformer、Attention、prefill/decode、KV Cache、推理性能指标。
- 你计划做的最小实验：不同上下文长度、batch size、量化方式对延迟、吞吐、显存的影响。
- 你希望导师帮忙判断的问题：这个方向是否适合作为入学前准备，是否有更推荐的资料或优先级。

## 给导师的简短表述

可以这样说：

> 老师您好，打扰您一下。我最近在规划入学前的学习安排，初步想围绕“大模型推理优化、AI Systems、体系结构与软硬件协同”做一些准备。我的想法是先补 Transformer 和大模型推理基础，比如 Attention、prefill/decode、KV Cache、模型压缩等；同时了解推理系统中的延迟、吞吐、显存、带宽等性能瓶颈，并尝试做一个小实验报告。想请您帮我简单判断一下，这样准备是否合适？如果方便的话，也想请您推荐 1-2 个更值得优先看的课程、论文或资料。您简单给我关键词也可以，我再自己细化学习。谢谢老师！

## 不要踩的坑

- 不要把入学前计划做成完整硕士课题。
- 不要一开始就追求读很多论文。
- 不要只看概念不跑实验。
- 不要只看模型准确率，忽略延迟、显存、带宽和吞吐。
- 不要过早纠结必须走哪一个具体方向。

## 最小可执行版本

如果你只想先执行 14 天：

1. 第 1-2 天：学 Transformer、Attention、prefill/decode、KV Cache。
2. 第 3-5 天：跑通 Hugging Face 或 vLLM 推理 demo。
3. 第 6-9 天：测上下文长度、batch size 对性能的影响。
4. 第 10-12 天：学 PagedAttention、KV Cache 显存管理和量化基础。
5. 第 13-14 天：整理 2-3 页短报告，列出后续问题。
