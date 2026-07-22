# MIT 6.5940 量化与高效深度学习观看清单

更新时间：2026-07-02

课程链接：

```text
https://www.bilibili.com/video/BV1c8wNe1ErX/
```

## 当前定位

这门课适合补：

- 模型压缩
- 量化
- 稀疏化
- LLM 部署
- 长上下文 LLM
- 高效推理和硬件友好优化

但它不是一门纯 LLM 量化课，也不是纯 AI Infra 课。对你当前路线，应该选择性看。

## 必看

### P5：Lecture 5 - Quantization Part I

核心作用：

- 量化入门主课。
- 建立 scale、zero-point、bit-width、quantization error 等基础概念。

重点关注：

- 为什么量化能降低模型大小和访存成本。
- FP32/FP16 到 INT8/INT4 的基本思想。
- 量化误差来自哪里。
- weight quantization 和 activation quantization 的区别。

### P6：Lecture 6 - Quantization Part II

核心作用：

- 量化进阶主课。
- 衔接 QAT/PTQ、硬件执行、低比特推理。

重点关注：

- post-training quantization, PTQ。
- quantization-aware training, QAT。
- integer-only inference。
- per-tensor / per-channel / group-wise quantization。
- 量化对 accuracy、latency、memory、energy 的影响。

### P12：Lecture 12 - Transformer and LLM

核心作用：

- 把高效深度学习方法放到 Transformer / LLM 场景里。

重点关注：

- Transformer 主要计算瓶颈。
- attention 和 MLP 的计算量。
- LLM 为什么对显存、带宽和算力敏感。
- 和 CS336 的 architecture / inference 内容对齐。

### P13：Lecture 13 - LLM Deployment Techniques

核心作用：

- 和你的 AI Systems / 推理优化路线强相关。

重点关注：

- LLM 部署瓶颈。
- KV cache。
- batching。
- quantization / compression 在部署中的角色。
- latency、throughput、memory footprint。

### P15：Lecture 15 - Long-Context LLM

核心作用：

- 和长上下文、attention / KV cache 优化相关。
- 对你早期长文本注意力压缩方向也有参考价值，但不需要强绑定。

重点关注：

- 长上下文为什么困难。
- attention `O(n^2)` 问题。
- KV cache 显存压力。
- 长上下文推理中的系统瓶颈。

## 建议快看

### P1：Lecture 1 - Introduction

作用：

- 看课程地图，了解这门课覆盖哪些 efficient ML 技术。

观看方式：

- 快看即可，不需要详细做笔记。

### P3：Lecture 3 - Pruning and Sparsity Part I

作用：

- 稀疏化和量化同属模型压缩主线。
- 对理解 sparse attention、稀疏 GEMM、硬件不规则性有帮助。

观看方式：

- 重点看 pruning / sparsity 为什么不一定直接带来硬件加速。

### P4：Lecture 4 - Pruning and Sparsity Part II

作用：

- P3 延续。

观看方式：

- 快看结构化稀疏、非结构化稀疏、硬件收益之间的区别。

### P9：Lecture 9 - Knowledge Distillation

作用：

- 了解另一类模型压缩方法。

观看方式：

- 了解 teacher-student、soft label、distillation loss 即可。
- 当前不作为主线。

### P19：Lecture 19 - Distributed Training Part 1

作用：

- 对 AI Infra 全貌有帮助。

观看方式：

- 如果 CS336 parallelism 已看，可以跳过或快看。

### P20：Lecture 20 - Distributed Training Part 2

作用：

- P19 延续。

观看方式：

- 非当前量化主线，按需看。

## 可选了解

### P10：Lecture 10 - MCUNet and TinyML

适合了解端侧极限资源部署，但和 LLM 推理不是同一规模。

### P11：Lecture 11 - TinyEngine

适合了解 tiny runtime / edge deployment。

### P14：Lecture 14 - LLM Post-Training

了解 post-training 全貌即可，不是量化核心。

### P16：Lecture 16 - Vision Transformer

如果你后续看 FPGA ViT accelerator 或视觉 Transformer，可以补。

### P21：Lecture 21 - On-device Training

端侧训练方向，暂时不是主线。

### P24：Introduction to Deep Compression Autoencoder

短视频，可作为模型压缩拓展了解。

### P25：Introduction to SVDQuant for 4-bit Diffusion Models

短视频，可了解低比特扩散模型量化。和 LLM 不完全同线，但 SVDQuant 思路可作为拓展。

## 可以跳过

这些和当前 LLM 量化 / 推理系统 / 硬件友好优化主线距离较远：

- P2：Basics of Neural Networks
- P7：Neural Architecture Search Part I
- P8：Neural Architecture Search Part II
- P17：GAN, Video, Point Cloud
- P18：Diffusion Models
- P22：Quantum Machine Learning Part 1
- P23：Quantum Machine Learning Part 2

如果基础神经网络已经通过 CS224N 补过，P2 不需要重复看。

## 推荐观看顺序

### 标准版

```text
P1 快看
→ P5
→ P6
→ P12
→ P13
→ P15
→ P3/P4 快看
→ P9 快看
```

### 时间紧版本

```text
P5
→ P6
→ P13
→ P12
```

### 和导师项目最贴合版本

```text
P5 Quantization Part I
→ P6 Quantization Part II
→ P12 Transformer and LLM
→ P13 LLM Deployment Techniques
→ P15 Long-Context LLM
```

## 学完后的最低产出

看完 P5/P6 后，应该能写清楚：

- uniform quantization 是什么。
- scale / zero-point 是什么。
- weight quantization 和 activation quantization 的区别。
- PTQ 和 QAT 的区别。
- per-tensor、per-channel、group-wise quantization 的区别。
- 为什么低比特量化能降低访存和硬件成本。

看完 P12/P13/P15 后，应该能写清楚：

- LLM 推理瓶颈来自哪些部分。
- quantization 如何影响 memory footprint、bandwidth、latency、energy。
- KV cache 为什么会成为长上下文推理瓶颈。
- 量化实验为什么必须同时报告模型质量和系统指标。

