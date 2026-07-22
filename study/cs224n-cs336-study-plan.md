# CS224N 与 CS336 学习计划

更新时间：2026-06-30

## 定位

当前目标不是系统学完整 NLP，而是为后续大模型推理优化、量化、AI Systems、软硬件协同方向补齐必要基础。

建议策略：

1. CS224N 作为桥接课程，选择性学习。
2. CS336 作为主线课程，重点学习语言模型从 tokenizer、Transformer、训练、系统优化、推理到评测的完整流程。
3. 学习产出要尽量和导师给出的 LLM 量化、GEMM、近似计算、硬件友好优化方向连接。

官方依据：

- CS224N 官方页面说明该课程覆盖深度学习 NLP 基础、LLM 前沿、PyTorch 实现、word vectors、Transformer、LLM benchmarking/evaluation 等内容。
- CS336 官方页面说明该课程以从零构建语言模型为主线，覆盖 pre-training 数据、Transformer 架构、训练、评测、部署前流程，并强调 PyTorch、系统优化、GPU、多机效率。

## 总体顺序

推荐顺序：

```text
李沐 attention / Transformer 已完成
        ↓
CS224N 精选桥接 2-3 周
        ↓
CS336 主线 6-8 周
        ↓
回到导师项目：LLM 量化 / GEMM / AxCore / AM-LUT
```

不建议先完整刷完 CS224N 再开始 CS336。CS224N 内容很大，很多传统 NLP 任务对你当前项目不是最高优先级。

## CS224N 精选计划

### 目标

用 CS224N 补齐进入 CS336 前最小必要基础：

- word vectors / word2vec / negative sampling
- 神经网络、反向传播、loss、梯度下降
- language modeling、perplexity
- self-attention、Transformer
- pretraining、LLM 基本范式
- evaluation / benchmarking

### B 站分 P 观看清单

当前课程链接：

```text
https://www.bilibili.com/video/BV1vQMBz6EvP
```

这个合集一共有 23 个分 P。按你当前目标，不需要全看。建议按下面优先级执行。

#### 必看

这些直接服务于进入 CS336 和后续 LLM / AI Systems 项目：

1. P1：Lecture 1 - Intro and Word Vectors
2. P2：Lecture 2 - Word Vectors and Language Models
3. P3：Lecture 3 - Backpropagation, Neural Network
4. P7：Lecture 7 - Attention, Final Projects and LLM Intro
5. P8：Lecture 8 - Self-Attention and Transformers
6. P9：Lecture 9 - Pretraining
7. P12：Lecture 11 - Benchmarking by Yann Dubois

对应学习目标：

- P1-P2：补 Word2Vec、softmax、negative sampling、language model、perplexity。
- P3：补神经网络、反向传播、loss、梯度下降。
- P7-P8：补 attention、self-attention、Transformer、LLM 基本结构。
- P9：补 pretraining、BERT/GPT、迁移学习范式。
- P12：补 benchmark、evaluation、模型质量指标。

#### 建议快看

这些不需要深挖，但有助于理解 LLM 发展脉络：

1. P5：Lecture 5 - Recurrent Neural Networks
2. P6：Lecture 6 - Sequence to Sequence Models
3. P10：Lecture 11 - Natural Language Generation
4. P11：Lecture 10 - Post-training by Archit Sharma
5. P23：Hugging Face Tutorial, Eric Frankel

观看方式：

- P5-P6：重点看 RNN / Seq2Seq 为什么曾经重要，以及 attention 为什么替代它们。不建议花大量时间推 RNN/LSTM 细节。
- P10：了解 decoding、generation、sampling 等概念即可。
- P11：了解 SFT、RLHF、DPO 等 post-training 名词即可，不作为当前主线。
- P23：如果后面要跑小模型、加载 tokenizer、做 benchmark，可以快看。

#### 按需补

这些只在你卡住对应基础时再看：

1. P13：Lecture 12 - Neural Network by Shikhar Murty
2. P21：Python Tutorial, Manasi Sharma
3. P22：PyTorch Tutorial, Drew Kaul

观看方式：

- 如果 P3 的神经网络、反向传播、矩阵维度还不清楚，再看 P13。
- Python 基础不够再看 P21。
- PyTorch 基础薄弱可以看 P22；涉及 PyTorch 代码细节时，优先放到“学习Pytorch”对话里消化。

#### 可以跳过

这些和你当前的 LLM 推理优化、量化、AI Systems 主线关系较远：

1. P4：Lecture 4 - Dependency Parsing
2. P14：Lecture 13 - Interfaces by Chaofei Fan
3. P15：Lecture 14 - Reasoning and Agents by Shikhar Murty
4. P16：Lecture 15 - After DPO by Nathan Lambert
5. P17：Lecture 16 - ConvNets
6. P18：Lecture 18 - NLP, Linguistics, Philosophy
7. P19：Lecture 16 - Multimodal Deep Learning, Douwe Kiela
8. P20：Model Interpretability & Editing, Been Kim

如果后续方向转向 agent、multimodal、interpretability，再回头补。

#### 推荐观看顺序

```text
P1 → P2 → P3 → P7 → P8 → P9 → P12
```

如果你想更稳一点：

```text
P1 → P2 → P3 → P5 快看 → P6 快看 → P7 → P8 → P9 → P12
```

如果你想尽快转 CS336：

```text
P1 → P2 → P3 → P8 → P9 → P12
```

最低限度不要跳过 P2、P3、P8、P12。

### 必看内容

1. Word Vectors

重点理解：

- one-hot 到 word embedding 的动机
- distributional semantics
- Word2Vec skip-gram
- softmax objective
- negative sampling
- GloVe 可以了解，不必深挖

产出：

- 能解释 `P(o|c)`、softmax、negative sampling、两个词向量矩阵 `U/V`。

2. Neural Network Basics / Backpropagation

重点理解：

- logistic regression
- sigmoid / softmax
- cross entropy / negative log likelihood
- hidden layer
- backpropagation 的链式法则直觉
- 参数、梯度、优化器

产出：

- 能看懂 `h = f(Wx + b)`、`s = u^T h`、`loss.backward()` 在数学上对应什么。

3. Language Models

重点理解：

- 语言模型是在建模 `P(w_t | w_<t)`
- next-token prediction
- perplexity
- RNN/LSTM 只需理解历史脉络，不需要投入太多

产出：

- 能解释为什么 GPT 本质上是 next-token language model。

4. Self-Attention and Transformers

重点理解：

- Q/K/V
- scaled dot-product attention
- multi-head attention
- positional encoding / RoPE 可之后补
- Transformer block
- causal mask
- decoder-only LM

产出：

- 能画出 decoder-only Transformer 的数据流。

5. Pretraining / LLMs

重点理解：

- pretraining
- fine-tuning / SFT
- instruction tuning / RLHF / DPO 了解即可
- BERT 和 GPT 的区别

产出：

- 能解释 encoder-only、encoder-decoder、decoder-only 的区别。

6. Evaluation / Benchmarking

重点理解：

- perplexity
- downstream benchmark
- zero-shot / few-shot
- accuracy 和 loss 的区别
- 为什么量化论文常同时看 perplexity、zero-shot accuracy、latency、memory、energy

产出：

- 能把模型质量指标和系统指标区分开。

### 可以跳过或快速浏览

- dependency parsing 的细节
- 传统序列标注任务细节
- 机器翻译 encoder-decoder 细节
- speech / multimodal / human-centered NLP
- 过多 NLP 应用型论文

这些不是没价值，而是和你当前的 AI Systems / LLM 推理优化主线距离较远。

### 作业取舍

优先级：

1. Assignment 1：word vectors，建议做或至少读懂。
2. Assignment 2：神经网络基础和 tensor derivatives，建议做核心部分。
3. Assignment 3：self-attention and Transformers，强烈建议做。
4. Assignment 4：LLM benchmarking and evaluation，强烈建议做或迁移到自己的小实验。

不建议为了刷课完整度，把大量时间投入传统 NLP 任务的高分实现。

### 进入 CS336 的切换标准

当你能独立回答下面问题，就可以进入 CS336：

1. Word2Vec 为什么要用 negative sampling？
2. softmax 和 sigmoid 分别在什么场景下使用？
3. cross entropy / NLL 为什么可以作为训练目标？
4. language model 在建模什么概率？
5. perplexity 越低意味着什么？
6. self-attention 里 Q/K/V 分别起什么作用？
7. Transformer block 里主要计算量来自哪些部分？
8. 为什么 LLM 推理中 GEMM、attention、KV cache 会成为系统瓶颈？

## CS336 主线计划

### 目标

CS336 是后续研究和项目的主线课程。你需要从中获得三类能力：

1. 能从零理解一个语言模型的组成。
2. 能看懂训练、推理、评测的系统瓶颈。
3. 能把量化、kernel、GEMM、硬件效率问题放回 LLM 真实工作负载里解释。

### 第一阶段：语言模型基础与实现框架

建议内容：

- Overview, tokenization
- PyTorch / einops
- FLOPs、memory、arithmetic intensity
- Transformer architecture
- hyperparameters

重点产出：

- 写一页笔记：一个 decoder-only LM 从文本到 logits 的完整流程。
- 能解释 tokenizer、embedding、Transformer block、LM head、loss 的关系。
- 能初步估算一个矩阵乘法的 FLOPs 和 memory traffic。

### 第二阶段：Attention、MoE 与系统瓶颈

建议内容：

- attention alternatives
- mixture of experts
- GPUs / TPUs
- kernels / Triton
- resource accounting

重点产出：

- 写一页笔记：prefill 和 decode 阶段的计算与访存差异。
- 能解释为什么 attention、GEMM、KV cache、memory bandwidth 会影响推理效率。
- 能看懂 profiling 指标的基本含义。

### 第三阶段：并行、推理与 scaling

建议内容：

- data / tensor / pipeline parallelism
- inference
- scaling laws
- evaluation

重点产出：

- 写一页笔记：LLM inference 的 latency、throughput、TTFT、TPOT、memory footprint。
- 能解释模型规模、batch size、sequence length、量化 bit-width 如何影响推理成本。

### 第四阶段：数据与后训练，选择性学习

建议内容：

- data sources
- filtering / deduplication
- SFT / RLHF / RLVR
- alignment / multimodality

优先级低于前面三阶段。你需要理解基本概念，但不需要现在深入做 RLHF 或数据清洗系统。

## 8 周执行版本

### 第 1 周：CS224N Word Vectors

任务：

- 看 word vectors / word2vec / negative sampling。
- 整理 softmax、sigmoid、NLL、negative sampling 的关系。

产出：

- 一份 1-2 页笔记：Word2Vec objective 与 negative sampling。

### 第 2 周：CS224N 神经网络基础

任务：

- 看 neural network basics / backpropagation。
- 补 cross entropy、gradient descent、matrix calculus 最小必要部分。

产出：

- 能手写一个二分类网络的数据流和 loss。

### 第 3 周：CS224N Transformer 与 LLM

任务：

- 看 self-attention / Transformer。
- 看 pretraining / LLM introduction。
- 快速看 evaluation / benchmarking。

产出：

- 一份 decoder-only Transformer 结构图。
- 一份 perplexity / benchmark / latency 指标对照表。

### 第 4 周：CS336 基础与 Transformer from Scratch

任务：

- 学 CS336 overview、tokenization、architecture。
- 开始看 Assignment 1 的结构。

产出：

- 能描述 tokenizer 到 loss 的完整训练路径。

### 第 5 周：CS336 Resource Accounting

任务：

- 学 FLOPs、memory、arithmetic intensity。
- 学 PyTorch / einops 中和张量形状相关的内容。

产出：

- 对一个 Transformer layer 粗略估算 GEMM FLOPs 和激活显存。

### 第 6 周：CS336 GPU / Kernel / Triton

任务：

- 学 GPUs / TPUs、kernels、Triton。
- 重点理解 matrix multiplication、memory hierarchy、tiling。

产出：

- 写一份 GEMM 为什么是 LLM 推理核心算子的笔记。

### 第 7 周：CS336 Inference

任务：

- 学 inference、batching、KV cache、prefill/decode。
- 把它和量化、低比特 GEMM、近似乘法联系起来。

产出：

- 一份 LLM 推理瓶颈图：compute、memory、bandwidth、latency。

### 第 8 周：CS336 Evaluation + 项目对齐

任务：

- 学 evaluation。
- 回看 Adana / AxCore / AM-LUT 方向。
- 整理你的项目评价指标。

产出：

- 一份项目评价表：
  - perplexity
  - zero-shot accuracy
  - latency / throughput
  - memory footprint
  - compute density
  - LUT / DSP / delay / power / energy

## 和导师项目的连接方式

学习 CS224N / CS336 时，不要只停留在课程理解，而要不断映射到项目：

```text
Word2Vec / LM objective
        → 理解模型质量指标和 loss

Transformer
        → 找到 GEMM、attention、KV cache 等核心算子

Resource accounting
        → 判断瓶颈来自算力、显存、带宽还是调度

Inference
        → 理解量化和近似计算为什么能降低成本

Evaluation
        → 建立 accuracy / perplexity / latency / energy 的联合评价

AxCore / Adana / AM-LUT
        → 把低比特量化、近似 GEMM、FPGA LUT multiplier 放回 LLM 推理场景
```

## 使用 AI 的规则

可以用 AI：

- 解释概念
- 检查公式推导
- 帮你读课程讲义
- 帮你整理笔记
- 帮你 debug 报错
- 帮你设计实验表格

不建议用 AI：

- 直接替你完成作业核心实现
- 直接给你完整答案后照抄
- 在没理解的情况下让 AI 生成大段代码

推荐流程：

1. 先自己写出思路或代码骨架。
2. 让 AI 检查逻辑和补充解释。
3. 自己复述关键概念。
4. 隔一天不看答案重新实现或推导一次。

## 最小验收标准

完成这份计划后，你至少应该能做到：

1. 解释 LLM 从文本输入到 next-token logits 的完整流程。
2. 解释 Transformer 推理中 GEMM、attention、KV cache 的角色。
3. 解释量化为什么会影响模型质量、显存、带宽和硬件实现。
4. 看懂一篇 LLM quantization / inference acceleration 论文的主要实验指标。
5. 能把导师项目表述为：

```text
面向 LLM 推理的硬件友好量化与近似 GEMM 加速
```

而不是孤立的：

```text
做一个 FPGA 乘法器优化
```
