# 大模型推理系统相关课程视频映射

更新时间：2026-07-02

## 关注问题

本文件整理三门课中和以下主题密切相关的视频：

- 大模型推理的计算行为
- 大模型推理的访存行为
- 大模型推理的通信行为
- 分布式推理原理
- 推理调度原理

涉及课程：

- CS224N
- CS336
- MIT 6.5940

## 总体结论

如果目标是理解 LLM 推理系统，三门课的重要性排序是：

```text
CS336 > MIT 6.5940 > CS224N
```

CS224N 主要补模型结构和 Transformer 基础，不系统讲推理系统。

CS336 是主线，覆盖 architecture、GPU、kernel、parallelism、inference、evaluation。

MIT 6.5940 是补充，重点在量化、压缩、部署、长上下文和高效推理。

## 主题一：推理计算行为

核心问题：

- LLM 推理中哪些算子最重？
- attention、MLP、GEMM 分别占什么角色？
- prefill 和 decode 的计算模式有什么不同？
- FLOPs 如何估算？

### CS224N

相关视频：

- P8：Lecture 8 - Self-Attention and Transformers

对应内容：

- self-attention 计算流程。
- `QK^T`、softmax、weighted sum。
- MLP / feed-forward network。
- residual、LayerNorm。

作用：

- 建立 Transformer block 的基本计算结构。
- 但不深入讲 FLOPs、GPU、kernel。

### CS336

相关视频：

- P3：Lecture 3: Architectures 重制版
- P5：Lecture 5: GPUs, TPUs 重制版
- P10：Lecture 10: Inference 重制版

对应内容：

- P3：Transformer 架构、attention、MLP、参数规模、主要计算模块。
- P5：GPU/TPU 计算模型、FLOPs、硬件执行效率。
- P10：prefill/decode 阶段的推理计算行为。

重点：

```text
prefill:
一次处理整段 prompt，并行度高，矩阵乘更大。

decode:
一次生成一个 token，batch 和 KV cache 变得关键，容易受访存限制。
```

### MIT 6.5940

相关视频：

- P12：Lecture 12 - Transformer and LLM
- P13：Lecture 13 - LLM Deployment Techniques

对应内容：

- Transformer / LLM 的计算瓶颈。
- 部署时如何理解算力、显存和带宽成本。
- 量化、压缩对计算量和硬件执行的影响。

## 主题二：推理访存行为

核心问题：

- 为什么 LLM 推理不只是算力问题？
- 显存占用来自哪里？
- 权重、activation、KV cache 各自如何占显存？
- memory bandwidth 为什么会成为瓶颈？

### CS224N

相关视频：

- P8：Lecture 8 - Self-Attention and Transformers

对应内容：

- 能看到 attention 里需要保存和使用 Q/K/V。
- 能理解 attention matrix 和 token 序列长度相关。

局限：

- 不系统讲显存、带宽、KV cache。

### CS336

相关视频：

- P5：Lecture 5: GPUs, TPUs 重制版
- P6：Lecture 6: Kernels, Triton, XLA 重制版
- P10：Lecture 10: Inference 重制版

对应内容：

- P5：memory hierarchy、HBM、SRAM/cache、bandwidth、arithmetic intensity。
- P6：kernel、tiling、memory access、fusion。
- P10：KV cache、batching、prefill/decode、memory footprint。

重点：

```text
权重访存:
每层 GEMM 需要读取模型权重。

KV cache:
decode 阶段每生成一个 token 都要访问历史 K/V。

activation:
推理时比训练轻，但中间结果仍影响峰值显存和带宽。

memory bandwidth:
如果算子 arithmetic intensity 不够高，就会受访存限制。
```

### MIT 6.5940

相关视频：

- P5：Lecture 5 - Quantization Part I
- P6：Lecture 6 - Quantization Part II
- P13：Lecture 13 - LLM Deployment Techniques
- P15：Lecture 15 - Long-Context LLM

对应内容：

- P5/P6：量化如何减少权重和 activation 的存储/访存。
- P13：部署中的显存、带宽和延迟问题。
- P15：长上下文中 KV cache 和 attention 的显存压力。

重点：

```text
低比特量化:
减少权重大小，降低显存占用和 bandwidth pressure。

activation quantization:
进一步降低中间激活和算子访存成本，但更容易影响精度。

long context:
KV cache 随 batch size、层数、head 数、head_dim、context length 增长。
```

## 主题三：推理通信行为

核心问题：

- 多卡推理为什么需要通信？
- tensor parallel、pipeline parallel 会通信什么？
- 通信和计算如何重叠？
- 通信瓶颈如何影响 latency / throughput？

### CS224N

相关视频：

- 基本没有密切相关内容。

说明：

- CS224N 主要是 NLP / Transformer 模型基础，不讲分布式推理通信。

### CS336

相关视频：

- P7：Lecture 7: Parallelism 重制版
- P8：Lecture 8: Parallelism 重制版
- P10：Lecture 10: Inference 重制版

对应内容：

- P7/P8：data parallel、tensor parallel、pipeline parallel、可能涉及 sequence/context parallel。
- P10：推理场景下并行和 serving 相关问题。

重点：

```text
tensor parallel:
把矩阵乘切到多张卡上，常见通信是 all-reduce / all-gather / reduce-scatter。

pipeline parallel:
把不同层放到不同设备上，通信主要是层间 activation 传递。

decode 阶段:
每 token 一步，通信延迟更敏感。

prefill 阶段:
矩阵较大，计算密集，更容易摊薄通信成本。
```

### MIT 6.5940

相关视频：

- P19：Lecture 19 - Distributed Training Part 1
- P20：Lecture 20 - Distributed Training Part 2
- P13：Lecture 13 - LLM Deployment Techniques

对应内容：

- P19/P20：分布式训练通信原理，虽然不是纯推理，但有助于理解 parallelism 通信模式。
- P13：部署侧可能涉及推理系统和服务成本。

说明：

- 如果你主要关心分布式推理，CS336 P7/P8/P10 优先级高于 MIT P19/P20。

## 主题四：分布式推理原理

核心问题：

- 为什么一个模型要切到多卡？
- 权重如何切？
- batch / token / layer 如何切？
- 不同并行方式适合什么场景？

### CS224N

相关视频：

- 没有系统相关内容。

### CS336

相关视频：

- P7：Lecture 7: Parallelism 重制版
- P8：Lecture 8: Parallelism 重制版
- P10：Lecture 10: Inference 重制版

对应内容：

- 分布式并行基本分类。
- 大模型为什么需要多设备。
- 推理和训练的并行差异。
- serving 中 batch、latency、throughput 的权衡。

学习目标：

```text
能解释：
1. data parallel 为什么对单请求推理帮助有限。
2. tensor parallel 为什么适合切大矩阵。
3. pipeline parallel 为什么会引入 bubble。
4. decode 阶段为什么更怕同步和通信延迟。
```

### MIT 6.5940

相关视频：

- P13：Lecture 13 - LLM Deployment Techniques
- P19：Lecture 19 - Distributed Training Part 1
- P20：Lecture 20 - Distributed Training Part 2

对应内容：

- LLM 部署技巧。
- 分布式训练的基础通信模式。
- 有助于建立 distributed systems 的直觉。

## 主题五：推理调度原理

核心问题：

- 多个请求如何 batching？
- 为什么 decode 阶段调度很重要？
- continuous batching / dynamic batching 是什么？
- KV cache 如何影响调度？
- latency 和 throughput 如何权衡？

### CS224N

相关视频：

- 基本没有密切相关内容。

说明：

- CS224N 最多补 generation / decoding 概念，不讲 serving scheduler。

### CS336

相关视频：

- P10：Lecture 10: Inference 重制版
- P12：Lecture 12: Evaluation 重制版

对应内容：

- P10：推理 serving、batching、KV cache、prefill/decode、latency/throughput。
- P12：评价指标，有助于理解调度优化不能只看 tokens/s。

重点：

```text
static batching:
简单，但请求长度不一时效率低。

continuous batching:
decode 过程中动态加入/移除请求，提高 GPU 利用率。

prefill/decode scheduling:
prefill 更计算密集，decode 更访存/延迟敏感，混合调度会影响 TTFT 和 TPOT。

KV cache management:
请求越多、上下文越长，KV cache 越占显存，调度必须考虑显存容量。
```

### MIT 6.5940

相关视频：

- P13：Lecture 13 - LLM Deployment Techniques
- P15：Lecture 15 - Long-Context LLM

对应内容：

- LLM 部署技巧。
- 长上下文下 KV cache 和 attention 的系统压力。

说明：

- 如果课程没有深入 scheduler，可以后续单独补 vLLM / PagedAttention / continuous batching 资料。

## 最推荐观看组合

如果你只围绕“LLM 推理系统”看，建议：

```text
CS224N:
P8 Self-Attention and Transformers

CS336:
P3 Architectures
P5 GPUs, TPUs
P6 Kernels, Triton, XLA
P7 Parallelism
P8 Parallelism
P10 Inference
P12 Evaluation

MIT 6.5940:
P5 Quantization Part I
P6 Quantization Part II
P12 Transformer and LLM
P13 LLM Deployment Techniques
P15 Long-Context LLM
P19/P20 Distributed Training 选看
```

如果进一步压缩：

```text
CS224N P8
CS336 P3, P5, P6, P10
MIT P5, P6, P13
```

如果专门补分布式和调度：

```text
CS336 P7, P8, P10
MIT P13, P19, P20
```

## 课程之外需要单独补的内容

这三门课能建立主干，但下面内容可能需要额外读文档/论文：

- vLLM / PagedAttention
- continuous batching
- TensorRT-LLM serving
- speculative decoding
- FlashAttention
- FlashInfer
- KV cache compression
- multi-GPU inference serving
- NCCL collective communication
- disaggregated prefill/decode

这些更接近当前工业界 LLM inference system 的细节。

