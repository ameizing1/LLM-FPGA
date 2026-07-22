# CS336 B站课程观看清单

更新时间：2026-07-01

课程链接：

```text
https://www.bilibili.com/video/BV11LEA6eEuj/
```

## 使用原则

这个合集里前 18 个分 P 是重制版或新版内容，P19-P35 基本是旧版或重复上传内容。建议优先看 P1-P18，不要新旧版本重复刷。

当前学习目标不是完整掌握大模型训练全链路，而是服务于：

- LLM inference optimization
- AI Systems / AI Infra
- quantization
- GEMM / kernel / accelerator-aware optimization
- 后续 Adana / AxCore / AM-LUT 项目

因此优先级排序是：

```text
模型结构与推理机制
→ GPU / kernel / systems
→ inference / evaluation
→ parallelism / scaling
→ data / post-training / alignment 选择性了解
```

## 必看

### P1：Lecture 1: Overview, Tokenization

作用：

- 建立 CS336 的整体地图。
- 理解 tokenizer、token、vocabulary、next-token prediction。
- 连接 CS224N 里的 language model 和真正 LLM 工程实现。

需要掌握：

- 文本如何变成 token。
- token 如何进入 embedding。
- LM 训练目标为什么是 next-token prediction。

### P2：Lecture 2: PyTorch(einops) 重制版

作用：

- 补张量操作、shape、einops、PyTorch 实现习惯。
- 对后面看 Transformer from scratch、attention、kernel 很重要。

需要掌握：

- tensor shape 如何变化。
- batch、sequence length、hidden dimension 的含义。
- einsum / einops 这类表达方式在 attention 中怎么用。

说明：

- PyTorch 细节可以放到“学习Pytorch”对话里消化，但这节仍建议看。

### P3：Lecture 3: Architectures 重制版

作用：

- 这是最核心的模型结构课。
- 理解 decoder-only Transformer、attention、MLP、residual、LayerNorm、参数规模。

需要掌握：

- 一个 LLM block 的组成。
- attention 和 MLP 的计算角色。
- 为什么主要计算量来自矩阵乘。
- 从 token 到 logits 的完整路径。

### P4：Lecture 4: Attention Alternatives 重制版

作用：

- 理解标准 attention 的局限。
- 对长上下文、KV cache、attention 优化、稀疏/线性 attention 很有帮助。

需要掌握：

- 标准 attention 的复杂度。
- 为什么长上下文会带来计算和显存压力。
- 各类 attention alternative 大致想解决什么问题。

### P5：Lecture 5: GPUs, TPUs 重制版

作用：

- 这是 AI Systems 主线的关键课。
- 帮你从硬件角度理解 LLM 为什么贵、慢、耗显存。

需要掌握：

- GPU/TPU 的基本执行模型。
- memory hierarchy。
- bandwidth、FLOPs、arithmetic intensity。
- 为什么 GEMM 适合 GPU，为什么访存会成为瓶颈。

### P6：Lecture 6: Kernels, Triton, XLA 重制版

作用：

- 直接连接 kernel optimization、Triton、AI compiler。
- 和你后续做低比特 GEMM、近似计算、硬件友好优化高度相关。

需要掌握：

- kernel 是什么。
- operator fusion、tiling、memory access 的基本思想。
- Triton / XLA 试图解决什么问题。

### P10：Lecture 10: Inference 重制版

作用：

- 这是和你方向最相关的一节。
- 直接对应 LLM 推理优化、serving、KV cache、latency、throughput。

需要掌握：

- prefill / decode 的区别。
- KV cache 的作用和代价。
- batching / continuous batching。
- latency、throughput、TTFT、TPOT。
- 量化为什么能影响显存、带宽、推理速度。

### P12：Lecture 12: Evaluation 重制版

作用：

- 帮你建立模型质量指标。
- 对读量化论文和做实验评价很重要。

需要掌握：

- perplexity。
- benchmark。
- zero-shot / few-shot。
- accuracy 和 loss 的区别。
- 为什么系统优化不能只看速度，也要看模型质量。

## 建议快看

### P7：Lecture 7: Parallelism 重制版

作用：

- 理解训练和推理中的并行基础。
- 对 AI Infra 岗位有帮助。

观看要求：

- 重点理解 data parallel、tensor parallel、pipeline parallel 分别解决什么问题。
- 不需要一开始就深挖分布式训练实现细节。

### P8：Lecture 8: Parallelism 重制版

作用：

- 是 P7 的延续。
- 帮你理解大模型为什么需要切分到多卡/多机。

观看要求：

- 快看主线概念。
- 如果后续做 serving / distributed inference，再回头精读。

### P9：Lecture 9: Scaling Laws 重制版

作用：

- 理解模型规模、数据量、计算量之间的经验规律。

观看要求：

- 了解即可。
- 对你当前低层推理/量化项目不是第一优先级。

### P11：Lecture 11: Scaling Laws 重制版

作用：

- P9 的延续。

观看要求：

- 快看结论，不必花太多时间推细节。

### P18：Lecture 18: Guest Lecture Dan Fu 重制版

作用：

- 作为 systems/efficient ML 方向的拓展讲座看。
- 如果内容涉及高效序列建模、长上下文、kernel 或 systems，可以重点听。

观看要求：

- 可在 P1-P12 后看。
- 根据是否贴近推理优化决定是否做详细笔记。

## 可选了解

### P13：Lecture 13: Data (Sources, Datasets) 重制版

作用：

- 了解预训练数据来源、数据集构成。

为什么不是优先：

- 你的当前项目主要在 inference / quantization / hardware-aware optimization，不是数据工程或预训练数据研究。

### P14：Lecture 14: Data 重制版

作用：

- P13 的延续。

观看方式：

- 有时间快看，不作为当前主线。

### P15：Lecture 15: Mid/Post-Training 重制版

作用：

- 了解 SFT、instruction tuning、post-training 等。

为什么不是优先：

- 对大模型全貌有用，但和你当前做推理低层优化不是同一个切口。

### P16：Lecture 16: Post-Training - RLVR 重制版

作用：

- 了解 RLVR、reasoning model 后训练。

观看方式：

- 如果只是补 AI Systems 主线，可以暂时跳过。

### P17：Lecture 17: Alignment - Multimodality 重制版

作用：

- 了解 alignment 和 multimodality。

观看方式：

- 暂时不是主线。后续如果方向转向多模态模型，再补。

## 可以跳过

P19-P35 基本是旧版或重复内容：

- P19：Lecture 2: PyTorch (einops)
- P20：Lecture 3: Architectures
- P21：Lecture 4: Attention Alternatives
- P22：Lecture 5: GPUs, TPUs
- P23：Lecture 6: Kernels, Triton, XLA
- P24：Lecture 7: Parallelism
- P25：Lecture 8: Parallelism
- P26：Lecture 9: Scaling Laws
- P27：Lecture 10: Inference
- P28：Lecture 11: Scaling Laws
- P29：Lecture 12: Evaluation
- P30：Lecture 13: Data (Sources, Datasets)
- P31：Lecture 14: Data
- P32：Lecture 15: Mid/Post-Training
- P33：Lecture 16: Post-Training - RLVR
- P34：Lecture 17: Alignment - Multimodality
- P35：Guest Lecture: Dan Fu

除非重制版音画或字幕有问题，否则不需要重复看。

## 推荐观看顺序

### 标准版

```text
P1 → P2 → P3 → P4 → P5 → P6 → P10 → P12
```

然后快看：

```text
P7 → P8 → P9 → P11 → P18
```

最后按需看：

```text
P13 → P14 → P15 → P16 → P17
```

### 如果时间紧

```text
P1 → P3 → P5 → P6 → P10 → P12
```

其中 P10 不建议跳过。

### 如果想更贴近导师项目

```text
P1 → P3 → P5 → P6 → P10 → P12 → P4
```

学习时重点把这些内容映射到：

```text
LLM workload
→ Transformer block
→ GEMM / attention / MLP
→ GPU memory hierarchy / bandwidth
→ kernel optimization
→ inference bottleneck
→ quantization / low-bit GEMM / hardware-aware evaluation
```

## 每节课的最低产出

P1：

- 画出文本到 token，再到 next-token prediction 的流程。

P2：

- 整理常见 shape：batch、seq_len、d_model、num_heads、head_dim。

P3：

- 画出 decoder-only Transformer block。

P4：

- 写出标准 attention 为什么是 `O(n^2)`。

P5：

- 写出 FLOPs、memory bandwidth、arithmetic intensity 的区别。

P6：

- 用自己的话解释 kernel、tiling、fusion。

P10：

- 写出 prefill / decode / KV cache / batching 的关系图。

P12：

- 做一张模型质量指标和系统指标对照表。

