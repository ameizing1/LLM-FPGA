# LLM AI Systems 学习资源地图

更新时间：2026-06-20

## 选资源原则

这条路线的核心不是“把所有 AI 课程刷完”，而是围绕一个主线组织资源：

**大模型推理瓶颈 -> 测量与解释 -> 系统优化 -> 软硬件协同 -> 架构/编译器/芯片方向深化**

因此资源优先级是：

1. 先理解 Transformer、Attention、KV Cache、LLM inference。
2. 再学 vLLM / TensorRT-LLM / profiling，能跑实验。
3. 再补 ML Systems、CUDA/Triton、体系结构、DNN accelerator。
4. 最后根据课题选择深入训练系统、AI 编译器或芯片架构。

## 第一优先级：立刻开始用

### Hugging Face LLM Course

链接：https://huggingface.co/learn/llm-course/en/chapter1/1

用途：

- 快速补 LLM、Transformer、Hugging Face 工具链。
- 适合第一周建立最小背景。

建议看：

- Chapter 1：LLM 和 Transformer 基础
- Transformer architecture 相关章节
- Optimized Inference Deployment

不要一开始全刷。先服务于“跑通推理 demo”。

### Hugging Face Transformers KV Cache 文档

链接：https://huggingface.co/docs/transformers/en/kv_cache

用途：

- 理解 generation 里 cache 是怎么被管理的。
- 对你的长上下文、KV Cache、显存瓶颈主线很直接。

建议看：

- Cache strategies
- Cache explanation
- LLM inference optimization

### The Illustrated Transformer

链接：https://jalammar.github.io/illustrated-transformer/

用途：

- 图解 Transformer 和 self-attention。
- 适合第一天快速建立直觉。

注意：

- 它适合作为直观入门，不适合作为最终理论深度来源。

### vLLM 文档与 PagedAttention 论文

vLLM 文档：https://docs.vllm.ai/

PagedAttention 论文：https://arxiv.org/abs/2309.06180

用途：

- 理解 LLM serving 中 KV Cache 管理为什么重要。
- PagedAttention 是你连接 OS、显存管理和 LLM 推理的关键材料。

建议顺序：

1. 先跑 vLLM quickstart。
2. 再看 PagedAttention 论文的 introduction 和 motivation。
3. 最后看实验部分，关注 throughput、latency、sequence length。

## 第二优先级：2-8 周内系统学

### Stanford CS336: Language Modeling from Scratch

链接：https://cs336.stanford.edu/spring2025/

YouTube：https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

用途：

- 认可度很高的语言模型课程。
- 从 tokenization、Transformer、training、scaling、systems 等角度建立完整语言模型观。

建议学法：

- 不要一上来全部做完。
- 先看和 inference、GPU/TPU、Transformer、attention 相关内容。
- 等你第一轮实验跑完，再回来看 training 和 scaling。

### CMU 10-414/714 Deep Learning Systems

链接：https://dlsyscourse.org/

课程讲义：https://dlsyscourse.org/lectures/

用途：

- ML Systems 方向非常经典的公开课程。
- 从自动微分、深度学习框架、硬件加速到部署，能帮你理解 PyTorch/vLLM/Triton 背后的系统层。

建议学法：

- 如果时间有限，先看硬件加速、算子、自动微分、框架执行相关讲义。
- 作业 Needle 很有价值，但不必在 14 天启动期做。

### MIT 6.5940 TinyML and Efficient Deep Learning Computing

链接：https://hanlab.mit.edu/courses/2024-fall-65940

用途：

- 高效深度学习、模型压缩、量化、部署、LLM efficiency 的强相关课程。
- 适合你从“模型压缩”走向“硬件友好优化”。

建议学法：

- 优先看 pruning、quantization、efficient inference、LLM efficiency 相关课。
- 课程作业可以作为后续项目来源。

### MLSys Book

链接：https://mlsysbook.ai/

用途：

- 建立 ML Systems 的整体工程框架。
- 适合查漏补缺：数据、模型、部署、优化、硬件约束。

建议学法：

- 当工具书看。
- 不要从第一页线性读完；根据当前实验问题查相关章节。

## 第三优先级：体系结构、GPU、硬件深化

### NVIDIA CUDA C++ Programming Guide

链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

用途：

- CUDA 官方权威文档。
- 后续做 kernel / 算子优化必须查。

建议学法：

- 启动期先不全读。
- 先理解 thread/block/grid、memory hierarchy、shared memory、global memory、occupancy 的概念。

### Triton Tutorials

链接：https://triton-lang.org/main/getting-started/tutorials/

用途：

- 比 CUDA 更适合较快写出高性能 GPU kernel 原型。
- 对 LLM attention、softmax、matmul、kernel fusion 很有用。

建议顺序：

1. Vector Addition
2. Fused Softmax
3. Matrix Multiplication
4. FlashAttention 类材料

### NVIDIA Nsight Systems

链接：https://docs.nvidia.com/nsight-systems/UserGuide/index.html

用途：

- 学会 profiling，而不是凭感觉说瓶颈。
- 用于观察 CPU/GPU timeline、kernel、显存与调度问题。

建议学法：

- 先会跑最小 profile。
- 先回答：时间花在哪里？GPU 有没有忙？是否有明显数据搬运或等待？

### Computer Architecture: A Quantitative Approach

官方页：https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-443-15406-5

用途：

- 体系结构经典教材。
- 适合你补“定量分析”和 memory hierarchy、parallelism、performance trade-off。

建议学法：

- 重点看 performance、memory hierarchy、multiprocessors、warehouse-scale / parallelism 相关章节。
- 不必第一阶段整本读。

### MIT 6.5900 / 6.823 Computer System Architecture

链接：https://csg.csail.mit.edu/6.5900/

讲义：https://csg.csail.mit.edu/6.5900/lecnotes.html

用途：

- 高质量体系结构课程。
- 帮你补没选过的体系结构。

### ETH / Onur Mutlu Computer Architecture Lectures

链接：https://people.inf.ethz.ch/omutlu/lecture-videos.html

用途：

- 体系结构、memory systems、硬件/软件协同的高质量视频材料。
- 特别适合后面深入 memory wall、近存计算、可靠性和系统瓶颈。

## 第四优先级：AI 芯片 / DNN Accelerator

### Efficient Processing of Deep Neural Networks

教程页：https://eyeriss.mit.edu/tutorial.html

Survey：https://arxiv.org/abs/1703.09039

用途：

- DNN accelerator 和软硬件协同的经典材料。
- 讲清楚 data movement、energy efficiency、throughput、latency、precision、sparsity、硬件/算法协同。

建议学法：

- 先看 tutorial slides 或 survey。
- 重点看 metrics、data movement、DNN accelerator、precision、sparsity。
- 后续再读书。

## 深度学习基础备用资源

### Dive into Deep Learning

链接：https://d2l.ai/

用途：

- 代码 + 数学 + 图示结合，适合补 PyTorch/深度学习基础。

建议学法：

- 查缺补漏，不要全刷。
- 重点看 attention、transformer、optimization、GPU 相关内容。

### MIT 6.S191 Introduction to Deep Learning

链接：https://introtodeeplearning.com/

用途：

- 快速补深度学习全景。
- 适合你如果觉得神经网络基础不稳，用 1-2 周快速过一遍。

## 不建议现在优先投入的资源

### Goodfellow / Bengio / Courville《Deep Learning》

链接：https://www.deeplearningbook.org/

原因：

- 经典且权威，但太重。
- 更适合作为理论参考书，不适合作为你现在启动 AI Systems 路线的第一本主教材。

### 从零训练大模型教程

原因：

- 训练链路重、成本高。
- 你现在的主线是 inference bottleneck 和软硬件协同。
- 可以通过 CS336 理解训练，但不用第一阶段动手训练大模型。

### 完整 CUDA 大课

原因：

- CUDA 很重要，但启动期目标是理解 bottleneck 和做最小实验。
- 等你确定要做 kernel / attention 优化后，再系统投入。

## 推荐学习顺序

### 第 1 周

1. The Illustrated Transformer
2. Hugging Face LLM Course 的 Transformer 基础
3. Hugging Face KV Cache 文档
4. vLLM quickstart

目标：

- 能解释 Attention、prefill/decode、KV Cache。
- 能跑通一个推理 demo。

### 第 2 周

1. vLLM / PagedAttention
2. Hugging Face LLM inference optimization
3. Nsight Systems 入门
4. Triton vector add / fused softmax tutorial

目标：

- 能测 TTFT、TPOT、tokens/s、显存。
- 能解释 KV Cache 管理为什么像 OS memory management。

### 第 3-8 周

1. Stanford CS336 选看
2. CMU Deep Learning Systems 选看
3. MIT 6.5940 选看
4. Triton matmul / softmax
5. Computer Architecture 选看 memory hierarchy

目标：

- 形成一个小型研究问题。
- 写出 5-8 页技术报告。

## 最推荐的组合

如果只选 5 个资源：

1. Hugging Face LLM Course
2. Hugging Face KV Cache 文档
3. vLLM + PagedAttention 论文
4. Stanford CS336
5. MIT 6.5940 或 CMU 10-414/714 二选一

如果偏 AI 芯片：

- 加 Efficient Processing of Deep Neural Networks
- 加 Computer Architecture: A Quantitative Approach

如果偏系统/算子：

- 加 Triton Tutorials
- 加 NVIDIA Nsight Systems
- 加 CUDA Programming Guide
