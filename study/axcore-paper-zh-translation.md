# AxCore 论文中文翻译稿

论文原题：AxCore: A Quantization-Aware Approximate GEMM Unit for LLM Inference

中文译名：AxCore：一种面向大语言模型推理的量化感知近似 GEMM 单元

更新时间：2026-07-04

译稿说明：

- 本文档是面向学习和实验推进的中文翻译稿。
- 公式编号、图表编号、关键英文术语和缩写尽量保留，便于和原论文对照。
- PDF 文本抽取中部分图内文字存在乱码，图中具体数值请以原 PDF 图像和已复现的 `axcore_res.csv` 为准。
- 参考文献列表未逐条翻译。

## 摘要

大语言模型已经成为现代自然语言处理的基础，但其巨大的计算需求和存储需求，给高效推理带来了主要障碍。基于 Transformer 的 LLM 大量依赖浮点通用矩阵乘法（FP-GEMM），而 FP-GEMM 同时主导计算量和带宽需求。

本文提出 AxCore，这是一种量化感知的近似 GEMM 单元。它结合 weight-only quantization 和 floating-point multiplication approximation（FPMA），以实现高效且精确的 LLM 推理。不同于传统 GEMM 单元，AxCore 完全消除了乘法器，并在一种新的 systolic array 中用低比特整数加法替代乘法。

AxCore 具有几个关键创新：

1. 一种基于 FPMA 的 mixed-precision processing element，可以直接在压缩权重和高精度激活上计算。
2. 一套轻量级精度保持策略，包括 subnormal number handling、error compensation 和 format-aware quantization。
3. 一系列 systolic array 优化，包括共享 correction logic 和 normalization logic。

在开源 LLM 上的评估表明，AxCore 相比传统 FP GEMM 单元可以实现最高 6.3 倍到 12.5 倍的 compute density 提升。与最先进的 INT4-based 加速器 FIGLUT 和 FIGNA 相比，AxCore 分别提升 53% 和 70% 的 compute density，同时获得更低的 perplexity。

AxCore 已开源：https://github.com/CLab-HKUST-GZ/micro58-axcore

关键词：Large Language Model，Approximate Computing，Weight-only Quantization，Hardware Accelerator

## 1 Introduction

大语言模型已经显著推动了自然语言理解、翻译和生成等任务。这些模型通常由多层堆叠的 Transformer 组成，参数规模从数十亿到数千亿不等，因此带来了巨大的存储和计算需求。例如，GPT-3 有 1750 亿参数，如果使用 FP16 表示，需要约 350GB 内存，远超许多标准硬件加速器的容量。

LLM 的核心计算瓶颈来自 Transformer 架构，其中 general matrix-matrix multiplication（GEMM）操作主导了算术吞吐和内存带宽需求。这些 GEMM kernel 通常使用浮点算术（如 FP16 或 BF16）实现，硬件代价较高，从而限制了推理效率。

Quantization 是缓解这些问题的关键技术。它用低精度数据类型表示高精度浮点值。特别是 weight-only quantization，即将模型权重量化为低比特格式（如 INT4 或 FP4），同时保留较高精度的激活（如 FP16），已经被广泛用于 LLM 推理。这样做有效，是因为模型权重通常比激活消耗更多内存，而激活是动态且输入相关的，激进量化通常会损害模型精度。

然而，weight-only quantization 并不能自动解决计算单元的硬件代价。常见方式是先将低比特权重 dequantize 回浮点值，再用传统 FP GEMM 计算。这会引入额外的 dequantization 开销，并且仍然依赖昂贵的浮点乘法器。另一类方式是 direct mixed-precision GEMM（mpGEMM），即直接用低比特权重和高精度激活进行计算。这种方式硬件上更有吸引力，但需要专门的计算单元支持不同精度操作数之间的高效乘加。

另一方面，floating-point multiplication approximation（FPMA）通过整数加法近似浮点乘法，已经被关注为一种高效推理方案。Mitchell logarithm approximation 表明，浮点数可以解释到对数数系中；相关工作进一步说明浮点乘法可以用整数加法近似。这意味着 GEMM 单元中代价高昂的浮点乘法器可以被更简单的整数加法器替代，从而节省大量硬件资源。

但是，现有 FPMA 方法通常针对 uniform-precision 设置，例如 FP16 x FP16。当它们被直接应用到低比特量化 LLM 中时，会出现严重问题。LLM weight-only quantization 需要处理 FP16 activation x FP4 weight 这类 mixed-precision 计算；此外，低比特浮点格式中 subnormal values 更常见，FPMA 的系统误差也会在深层模型中累积，导致精度下降。

本文提出 AxCore，一种面向 LLM 推理的量化感知近似 mpGEMM 单元。它将低比特量化和 FPMA 融合起来，提供高效、无乘法器、mixed-precision matrix multiplication，同时保持端到端模型精度。

本文主要贡献如下：

1. 提出一种基于 mpFPMA 的 systolic array GEMM 单元，支持直接在低比特压缩权重和高精度激活之间计算。
2. 提出轻量级精度保持机制，包括 subnormal number conversion、mean-based error compensation 和 format-aware quantization。
3. 提出 AxCore 架构优化，包括 PreAdd correction advancing、shared normalization 和 FPMA-based scaling，从而降低 PE 复杂度和硬件开销。
4. 在 LLM accuracy 和 hardware efficiency 上进行评估，证明 AxCore 相比 FP GEMM 单元和现有 INT4-based 加速器具有更高 compute density、更低能耗和更好或相近的 perplexity。

## 2 Background

### 2.1 GEMM in LLM Inference

LLM 通常由多层 Transformer decoder block 组成，每个 block 包含 masked self-attention 和 linear transformation layers。尽管 attention mechanism 提供了 Transformer 的推理能力，但 linear layers，包括 feed-forward networks 和 attention projections，主导了 LLM 推理中的计算负载，并贡献了大部分模型参数。

这些层高度依赖 GEMM，因为它们本质上是 dense linear transformations，将高维输入激活映射到输出激活。

论文 Figure 2 展示了在 OPT-175B 和 LLaMA-3.1-405B 中，attention mechanism 和 linear layers 在不同 sequence length 下的相对操作量。随着 sequence length 增长，attention 的计算占比会增加，但在实际 sequence length（10k 到 20k tokens）下，linear layers 中的 GEMM 仍然主导计算，占比约 69% 到 99%。此外，在 prefill 阶段，attention 中也主要使用 GEMM，因此真实 GEMM 占比甚至更大。这说明优化 linear-layer GEMM 仍然是提升大规模 LLM 推理效率的关键。

### 2.2 Weight-only Quantization

为了减少存储和计算开销，quantization 技术被广泛用于 LLM 推理。Quantization 将高精度权重映射为紧凑的低比特格式，例如 INT4 或 FP4，从而降低模型大小和算术位宽。

Weight-only quantization 特别适合 LLM 推理。模型权重通常消耗大量内存，因此量化权重可以显著减少内存占用和带宽需求。相比之下，激活是动态且输入相关的，强行低比特量化通常会导致明显精度损失。因此，低比特权重加高精度激活（如 FP16 或 BF16）已经成为学术界和工业界常用实践。

典型量化会将权重 `w` 通过 scale `s` 映射为低比特表示 `w_q`。为了在激进低比特量化下保持精度，通常采用 group quantization，即将权重 tensor 划分为较小 group，每个 group 使用自己的 scale factor。这样能更好地捕获局部分布，降低量化误差。

### 2.3 Quantization-Aware GEMM

在 LLM 推理中，GEMM 涉及大规模权重矩阵和激活矩阵的乘法。没有量化时，GEMM 使用全精度操作数，例如 FP16 x FP16。

使用 weight-only quantization 后，有两种常见执行方式：

1. Indirect GEMM：先将量化权重 dequantize，也就是乘以 scaling factor，恢复为浮点值，再执行 GEMM。
2. Direct mpGEMM：直接在低比特权重和 FP activation 之间执行 GEMM，随后对累计输出做 dequantization。

Direct mpGEMM 更具硬件效率，因为它避免了逐权重 dequantization 的开销，并让 GEMM 单元可以使用更轻量的 datapath。

许多硬件加速器采用 uniform quantization formats，例如 INT4 或 INT8，因为它们简单。然而，uniform formats 均匀分布数值表示，和 LLM 权重常见的 Gaussian-like 分布并不完全匹配。相比之下，non-uniform formats，例如 FP4 和 FP8，在接近零的区域分配更多表示，具有更高精度潜力。本文证明，基于 FP 的 quantization 不仅在精度上有优势，也可以通过 AxCore 架构带来更高硬件效率。

### 2.4 FP Multiplication Approximation with Integer Addition

浮点乘法是许多应用中的基础操作，但由于涉及符号、指数、尾数、归一化和舍入，硬件面积代价较高。

根据 IEEE 754 标准，normalized floating-point number `x` 可表示为：

$$
x = (-1)^{S_x} \cdot 2^{E_x - B} \cdot (1 + M_x)
$$

其中 `Sx` 是符号位，`Ex` 是指数，`B` 是 exponent bias，`Mx` 是 mantissa。Bias 可写为：

$$
B = 2^{N_e - 1} - 1
$$

FPMA 的思想是用更简单的整数加法替代昂贵乘法。基于 Mitchell logarithm approximation，浮点数可近似映射到对数域：

$$
\log_2(|x|) = E_x - B + \log_2(1 + M_x) \approx E_x - B + M_x
$$

于是浮点乘法 `r = x * y` 可近似为：

$$
\log_2(|r|) \approx (E_x + M_x) + (E_y + M_y) - 2B
$$

如果结果 `r` 也表示为：

$$
\log_2(|r|) \approx E_r + M_r - B
$$

则可得到近似乘法：

$$
R = X + Y - B
\tag{5}
$$

其中 `X`、`Y`、`R` 分别定义为：

$$
X = E_x + M_x,\quad Y = E_y + M_y,\quad R = E_r + M_r
$$

所有运算都可以用整数加法完成，从而不再需要复杂乘法器。

FPMA 的代价是近似误差。误差来自下面这个线性化近似：

$$
\log_2(1 + M) \approx M
$$

此外，传统 FPMA 假设是 normalized floating-point numbers，不能直接处理 subnormal numbers，因为 subnormal 没有隐含 leading one。

## 3 Adopting FPMA for Quantized LLM Inference

### 3.1 Challenges

尽管 FPMA 可以通过用轻量整数加法器替代 FP multiplier 来提升硬件效率，但将它用于 quantized LLM inference 仍然面临挑战。

**挑战 1：支持 FPMA-based mpGEMM 的硬件。**

现代 LLM 通常使用 weight-only quantization，这会产生 mpGEMM 操作，例如 FP16 activation 乘 FP4 或 INT4 weight。FPMA 可以用于 indirect GEMM，即先 dequantize，再近似全精度浮点乘法；但这种方式会抵消量化带来的效率收益。

为了充分利用低比特表示，FPMA 必须支持 direct mpGEMM，即权重在计算过程中保持压缩格式。然而，传统 FPMA 方法只支持 uniform precision，例如 FP16 x FP16。将它扩展到 mpFPMA 需要重新设计 PE 和 datapath。难点包括：

- 对齐不同格式操作数。
- 处理 exponent bias 不匹配。
- 在整数近似计算中保持足够精度。
- 在 systolic array 中高效支持混合格式。

**挑战 2：以低成本保持精度。**

FPMA 依赖如下近似，因此会引入系统误差：

$$
\log_2(1 + M) \approx M
$$

这个误差会在深层模型中累积。论文 Figure 4 展示，在 OPT 不同规模模型上使用 FPMA 会导致 perplexity 上升。FP4 本身相比 FP16 已有一定精度损失，而加入 FPMA 后 perplexity 会进一步变差。通常 perplexity 差异超过 1% 就会被认为显著。

此外，低比特 FP 格式（如 FP4）会产生更多 subnormal numbers。由于 exponent bits 很少，低比特格式中的 subnormal 并不一定是可以忽略的极小值。相反，它们可能代表相对较大的数值区间。如果不处理 subnormal，FPMA 的数学假设会失效，导致显著精度损失。

### 3.2 Our Solution - AxCore

为解决这些问题，本文提出 AxCore，一种量化感知的近似 mpGEMM 单元，它将 FPMA 与低比特量化紧密结合，用于高效准确的 LLM 推理。

**特性 1：通过 mpFPMA-based systolic array 实现高效 mpGEMM。**

AxCore 引入一组优化的 mpFPMA PE，可以直接在压缩低比特权重和高精度激活之间计算。为了减少 datapath 宽度和 PE 复杂度，AxCore 使用 correction advancing，将 correction terms 预计算并在 PE 外共享。Normalization 被推迟到共享单元中，以减少每个 PE 的逻辑；FPMA-based dequantization 则消除了 post-GEMM multipliers。

**特性 2：轻量级精度保持机制。**

为处理 FPMA 误差，AxCore 采用专门面向 mixed-precision 场景的轻量 compensation mechanism。它包含预计算的 bias 和 correction terms，用于稳定不同 operand combinations 下的输出。

每个 PE 还集成 subnormal number conversion（SNC）逻辑，用于检测并转换 subnormal values，将其映射到最近的 normalized representations，防止 malformed mantissa 导致精度下降。

此外，AxCore 采用 adaptive format-aware quantization，在每个 weight group 上动态选择最合适的 FP4 encoding，例如 E1M2、E2M1 或 E3M0。这种细粒度适应性进一步提升了不同层、不同数值分布下的量化精度。

这些技术共同让 AxCore 在保持 LLM 推理精度的同时，获得高硬件效率。

## 4 Accuracy-Preserved mpFPMA for LLM

### 4.1 Extending FPMA to mpFPMA

为了让 quantized LLM inference 能高效执行 mixed-precision GEMM，本文将 FPMA 扩展到不同精度操作数之间。虽然近似公式和传统 FPMA 类似，但 bit width、fixed-point alignment 和 bias correction 都必须重新设计。

以 FP16 activation `A` 和 FP4 quantized weight `Wq` 的乘法为例。mpFPMA 首先将操作数对齐到共同的 fixed-point 表示。由于 FP4 的 mantissa bits 少于 FP16，需要将 FP4 operand 的 mantissa 左移，也就是补零，以匹配 FP16 的分辨率。论文将对齐后的表示写作：

$$
\mathrm{Align}(W_q) = W_q \ll (\mathrm{Mantissa}_{\mathrm{FP16}} - \mathrm{Mantissa}_{\mathrm{FP4}})
\tag{6}
$$

这保证两个操作数的 radix point 对齐。

但是，由于不同格式的 exponent bias 不同，例如 FP16 的 bias 是 15，而 FP4 E2M1 的 bias 是 1，因此需要 format-aware bias correction term `B1`：

$$
B_1 = B_a + B_{w_q} - B_r
\tag{7}
$$

其中 `Ba`、`Bwq` 和 `Br` 分别是 activation、quantized weight 和 result 的 exponent bias。对于 activation 和 result 都是 FP16 的常见情况，`B1` 可简化为 `Bwq`。

结合 alignment 和 bias correction，mixed-precision product 的近似结果为：

$$
R = A + \mathrm{Align}(W_q) - B_1
\tag{8}
$$

为了进一步改善数值精度，特别是在量化噪声和近似误差下，AxCore 引入 constant compensation term `C1`。最终 mpFPMA 表达式为：

$$
R = A + \mathrm{Align}(W_q) - B_1 + C_1
\tag{9}
$$

这个公式让 AxCore 能用整数加法高效近似 mixed-precision multiplication。

### 4.2 Handling Subnormal Numbers in mpFPMA

随着 LLM 推理中浮点格式位宽降低，特别是 FP4 这类格式，subnormal values 的处理变得非常关键。

#### 4.2.1 Problems with Subnormal Numbers

在浮点格式中，subnormal values 用于表示比最小 normalized exponent 更小的数值。它们有助于保留接近零附近的分辨率。Subnormal number 的指数为 0，并且没有隐含 leading one：

$$
x_{\mathrm{sub}} = (-1)^s \cdot 2^{1-B} \cdot M
\tag{10}
$$

和 normalized number 相比，subnormal 移除了 mantissa 中的 `1+M`。

在 FP16 或 FP32 中，subnormal 通常非常小。但在 FP4 这种低比特格式中，subnormal 可以代表相对较大的值。例如 E1M2 中一个 subnormal encoding 可能代表 0.75。由于低比特 FP 的 exponent bits 少，subnormal 在量化权重中会频繁出现。因此，subnormal 不再是边缘情况，必须被正确处理。

如果直接将 subnormal 输入 FPMA，由于缺少 implicit leading one，FPMA 的 approximation 会不成立，从而导致较大误差。

#### 4.2.2 Subnormal Number Conversion

AxCore 设计了 subnormal number conversion（SNC）机制。它根据 mantissa width（如 M1、M2、M3）将 subnormal encoding 转换为附近的 normalized encoding 或 0。论文 Table 1 给出转换表。

这个转换主要影响 mantissa。表中列出了 subnormal encoding、对应数值，以及转换后的 normal 或 zero value。

SNC 的目标不是做复杂精确恢复，而是在硬件成本较低的情况下避免 FPMA 对 subnormal 的错误解释。它将 subnormal 映射到更适合 FPMA datapath 的形式，从而显著缓解 perplexity 退化。

### 4.3 Error Compensation for mpFPMA

FPMA 的误差来自下面这个近似：

$$
\log_2(1 + M) \approx M
$$

为了理解误差分布，论文定义了 element-wise error，用来表示 exact floating-point product 和 approximate product 之间的差异。

#### 4.3.1 Analysis of Error Distribution

论文分析了不同 activation mantissa 和 weight mantissa pair 下的误差分布。Figure 6 展示了应用 compensation 前后的 square error distribution。

观察是：误差分布具有一定规律性，可以用一个较简单的平均补偿值来降低整体误差。

#### 4.3.2 Mean-Based Constant Compensation

AxCore 使用 mean-based constant compensation。它在所有有效 mantissa combinations 上计算 expected error 的平均值，得到 correction term。这个 correction term 可以在离线阶段预计算，并在硬件中作为常数使用。

这样做的优势是：

- 硬件开销很小。
- 不需要对每个乘法单独复杂校正。
- 能明显改善 mpFPMA 在 LLM 上的 perplexity。

### 4.4 Adaptive Format-Aware Quantization

#### 4.4.1 Block-wise Format Selection

不同层、不同 weight groups 的数值分布不同。固定使用一种 FP4 格式并不一定最优。

AxCore 在每个 weight group 上从候选 FP4 格式中选择最合适的一种，例如：

- E3M0
- E2M1
- E1M2

选择目标是最小化实际输入 activation distribution 下的 mean squared error。论文使用 calibration dataset 来避免过拟合。

Figure 7 展示了 LLaMA2-7B 不同层 attention output tensor 的 weight distribution。某些层的权重分布有尖峰，更适合 power-of-two-like encoding，因此会选择 E3M0；另一些层分布更宽、更均匀，则 E1M2 或 E2M1 更合适。

#### 4.4.2 Integration with FPMA

AxCore 还将 format-aware quantization 与 FPMA 结合。传统 floating-point quantization 和 reconstruction 通常依赖 division 和 multiplication，这会引入数值偏移和硬件开销。

AxCore 重新定义 quantization 和 dequantization，使其符合 FPMA-style approximation。通过在 quantization 和 dequantization 中使用补偿常数，误差可以相互抵消，从而更好保留原始数值并降低硬件成本。

## 5 AxCore Architecture

### 5.1 Overview

AxCore 使用 weight-stationary dataflow。低比特量化权重（如 FP4）被预加载并保持在每列 PE 中，而高精度激活（如 FP16）沿每行水平传播。

一个集中式 PreAdd unit 会预先计算 intermediate value `T`。它对 activation 应用 correction terms：

$$
T = A - B_1 + C_1
$$

其中 `A` 是高精度 activation，`B1` 是 exponent bias correction，`C1` 是 format-specific compensation constant。计算出的 `T` 沿行传播，以减少 PE 中重复逻辑。

在每个 PE 内部，低比特权重首先进入 SNC 单元。SNC 检测 subnormal values，并按格式将其映射到附近的 normalized representations。SNC 输出会被统一为共享内部格式，例如 S1E3M2，使后续逻辑与具体 FP 格式无关。这允许阵列同时支持多种 FP 格式，例如 FP4 的 E3M0、E2M1、E1M2。

对齐后的权重与预计算的 `T` 相加。这个操作只需要轻量整数加法器，从而替代传统乘法器。

后处理包括三阶段：

1. Normalization：将结果调整为标准浮点格式。
2. AxScale：用 FPMA-based addition logic 替代 dequantization multiplier，实现高效 scaling。
3. Accumulator：将 scaled partial sums 与先前存储值累加。

### 5.2 mpFPMA Processing Elements

#### 5.2.1 Overview

如 Figure 9 所示，每个 PE 逻辑上由两个连续模块组成：

1. Approximate Multiplication block。
2. Accumulation block。

PE 接收两个主要输入：

- low-bit quantized weight `Wq`
- 来自 PreAdd unit 的预计算 intermediate value `T`

`T` 在 PE 外部计算，并沿行传播到所有 PE。进入 PE 后，`Wq` 首先被 SNC 处理。SNC 后的 weight 进行 mantissa alignment。由于权重精度通常低于激活，weight mantissa 会被 zero-pad，以匹配 activation 的 fixed-point domain。

之后，aligned weight 和 `T` 相加，得到 approximate multiplication result。

#### 5.2.2 Subnormal Number Conversion Unit

Figure 10 展示了 SNC unit。以 FP4 为例，输入 `Wq` 可能属于 E1M2、E2M1 或 E3M0 等格式。`FormatSel` 信号选择对应格式的 decoder。Decoder 中的小逻辑表检查 subnormal encoding，并将其映射到 nearby normalized values。

SNC 的输出被统一为内部格式，例如 S1E3M2。这使后续计算逻辑可以复用，而不必为每种 FP4 格式分别设计完整 datapath。

论文还提到随机化舍入相关机制。对于某些 subnormal mapping，Zero Flag 会控制舍入方向。必要时可以利用 activation mantissa 的高位作为 stochastic bit，从而在不同输入之间交替舍入，降低系统性偏差。

### 5.3 Systolic Array Optimizations

AxCore 进一步采用多种资源共享优化，以降低 PE 面积和功耗。

#### Correction Advancing with PreAdd

在 baseline 设计中，每个 PE 都需要执行多个 correction 相关计算。例如，需要将 activation、aligned weight、bias correction 和 compensation 都在 PE 内完成。这会使 PE datapath 更宽、更复杂。

AxCore 将 correction advancing 到 PreAdd unit 中。PreAdd 在行级计算：

$$
T = A - B_1 + C_1
$$

然后将 `T` 广播给该行所有 PE。这样每个 PE 只需执行：

$$
R = T + \mathrm{Align}(W_q)
$$

这显著简化 PE datapath，减少加法器位宽和数量。

#### Postponed Normalization

Normalization 如果在每个 PE 内执行，会带来重复逻辑。AxCore 将 normalization 推迟到共享单元中处理，从而减少 PE 内面积。

#### AxScale

在 weight-only quantization 中，通常需要 scaling factor 进行 dequantization。传统方式会引入乘法器。AxScale 使用 FPMA-style addition logic 将 dequantization scaling 转换为低成本加法操作，从而避免 post-GEMM multiplier。

### 5.4 AxCore-Powered LLM Inference Accelerator

Figure 13 展示了基于 AxCore 的 LLM inference accelerator。它围绕 quantized model 的 GEMM pipeline 组织。

系统核心是 GEMM Unit（AxCore），由二维 processing tiles 组成，每个 tile 包含多个 mpFPMA PEs。Weight Buffer 存储量化模型权重，Unified Buffer 处理 activation 和 intermediate data。控制逻辑负责调度数据流和执行不同 layer 的 GEMM。

整体设计目标是将低比特量化、近似乘法和 systolic array 结合起来，实现高吞吐、低面积和低能耗的 LLM GEMM 加速。

## 6 Evaluation

### 6.1 Experimental Setup

#### 6.1.1 Accuracy Evaluation Setup

论文在两个常用 LLM families 上评估 AxCore 和 baseline designs：

- OPT
- LLaMA2

所有模型使用已有 weight-only quantization 方法量化到 4-bit。OPT 的 group size 设置为 128，LLaMA2 的 group size 设置为 64。对于 block-wise adaptive format quantization，使用来自 Pile dataset 的少量 calibration set，以避免过拟合。OPT 的 block size 为 128 x 64，LLaMA2 的 block size 为 64 x 64。

论文按照已有工作，在 WikiText-2 上评估 perplexity，sequence length 为 2048。Perplexity 越低表示精度越好。

此外，论文还使用 lm-eval-harness 在四个 zero-shot benchmark 上评估：

- ARC-e
- HellaSwag
- PiQA
- Winogrande

#### 6.1.2 Hardware Evaluation Setup

为了评估硬件效率，作者用 SpinalHDL 实现 AxCore，并用 Synopsys Design Compiler 在 TSMC 28nm 工艺节点下综合生成的 Verilog RTL。

所有设计使用相同 target frequency：1GHz，并归一化到相同 peak throughput（TOPS）。为公平比较，baseline 和 AxCore 共享 64 x 64 systolic array configuration 和 4 x 4 tilings。

评估覆盖不同 precision settings，包括 weight types（INT4、FP4、INT8、FP8）和 activation formats（FP16、BF16、FP32）。

作者基于开源 cycle-level simulator DNNWeaver 开发 simulator，用于性能评估。SRAM module 的 power 使用 CACTI 模拟。所有 accelerator designs 配置相同 SRAM sizes。

#### 6.1.3 Baselines

论文比较了 AxCore 和四种代表性 GEMM accelerator baselines：

- FPC：floating-point GEMM core。每个 PE 使用标准 floating-point fused-multiply-add（FMA），使用 FP32 accumulator。
- FPMA：用原始 FPMA logic 替代 FP multiplier。对于 FP16/BF16 activation 使用 FP16/BF16 adder；对于 FP32 activation 使用 FP32 adder。
- FIGNA：一种面向 weight-only quantized LLM 的 FP-INT mixed-precision GEMM unit。
- FIGLUT：一种面向 LLM 的 LUT-based FP-INT GEMM design。
- Tender：一种 INT-based non-mix-precision GEMM design。

### 6.2 Area Efficiency

#### 6.2.1 Area Efficiency of mpFPMA PEs

Figure 14 展示了不同数据类型配置下单个 PE 的 normalized area breakdown。Breakdown 包括：

- multiplication logic
- addition logic
- subnormal number conversion（SNC）
- other components

FPC 面积最高，因为它使用昂贵的 floating-point units。FPMA 通过近似减少 multiplier area。AxCore 在所有格式下 PE 面积最小，原因是它采用 mpFPMA design，消除了 multiplier。

与 FIGLUT 相比，AxCore 在 W4-FP32 下最多减少 34% PE area，在 W4-FP16 和 W4-BF16 下分别减少 31% 和 22%。与 FIGNA 相比，AxCore 在 4-bit formats 下减少 32% 到 39% PE area，在 8-bit formats 下减少 43% 到 56%。值得注意的是，SNC unit 引入的额外开销很小，平均只占 total PE area 的约 3.5%。

#### 6.2.2 Area Efficiency Across GEMM Designs

Figure 15 展示了不同 designs 和 data formats 下 GEMM unit 的 normalized area breakdown。面积被分为两类：

- PE array。
- Others：沿 activation data path 的 preprocessing 和 post-processing modules。

AxCore 在所有设置中都实现最低面积，并优于 FIGNA 和 FIGLUT。在 4-bit weight scenarios 下，与 FIGLUT 相比，AxCore 在 W4-FP16、W4-BF16、W4-FP32 下分别减少 31%、26%、34% total area；与 FIGNA 相比，分别减少 37%、36%、29%。在 8-bit settings 下，AxCore 相比 FIGLUT 平均减少 25% area，相比 FIGNA 减少超过 55% area。

### 6.3 Compute Density

Figure 16 展示了六种 input format configurations 下 GEMM array 的 normalized compute density（TOPS/mm2）。结果归一化到传统 FP32 design（FPC）。

AxCore 在所有格式下都达到最高 compute density，原因包括：

- compact mpFPMA datapath
- multiplier-free design
- centralized correction logic

在 W4-FP16 设置下，AxCore 相比 FPC 提升 6.7 倍 compute density，并显著优于 FIGNA（4.0 倍）和 FIGLUT（4.3 倍）。在 W4-FP32 设置下，AxCore 相比 FPC 提升 12.5 倍，并分别优于 FIGNA 和 FIGLUT 1.4 倍和 1.5 倍。其他格式也观察到类似趋势，例如 W4-BF16 下 AxCore 达到 5.3 倍，W8-FP16 下达到 6.2 倍。即使在 W8-FP32 等较高精度配置下，AxCore 仍然保持约 10 倍的 density gain。

### 6.4 Energy Efficiency

Figure 17 展示了 AxCore 和 baseline accelerators 在多种 input data types 以及两个 OPT models（13B 和 30B）上的 normalized energy breakdown 和 TOPS/W。

论文测量 decoding phase 的 energy，batch size 为 32，output sequence length 为 1，并和 baselines 对齐。所有设计都提供足够 bandwidth。

在所有评估配置中，AxCore 均表现出更优能效，具有最低 energy consumption 和最高 TOPS/W。

FIGNA 和 FIGLUT 在 8-bit scenarios 下 energy consumption 明显增加。原因是 FIGNA 的 multiplier overhead 随 computation bit-width 近似二次增长，而 FIGLUT 的 bit-serial architecture 需要更多 computation cycles，从而增加 energy expenditure。

平均而言，AxCore 相比 FPC、FPMA、FIGNA 和 FIGLUT 分别实现约 2.2 倍、1.5 倍、1.1 倍和 1.3 倍 total energy reduction，并实现约 6.4 倍、3.1 倍、1.4 倍和 2.0 倍 TOPS/W improvement。

这部分与我们已经复现的 Figure 17 直接对应。我们的 `axcore_res.csv` 中 `Static / Dram / Buffer / Core` 以及 `Time`、`TOPS/W` 即来自这类 simulator 和 synthesis/CACTI 参数链路。

### 6.5 Accuracy Evaluation

#### 6.5.1 End-to-end Model Accuracy

Table 2 比较了 AxCore 与 baseline accelerators 在 perplexity 上的表现，并展示了 AxCore 优化项的 ablation study：

- mpFPMA：基础 mixed-precision FPMA。
- mpFPMA+S：mpFPMA + SNC。
- mpFPMA+S+C：mpFPMA + SNC + compensation。
- AxCore：mpFPMA + SNC + compensation + format-aware quantization。
- AxCore-KV：AxCore + KV cache quantization。

FPMA 使用 FP4 round-to-nearest quantization；FIGNA 使用 GPTQ quantization；FIGLUT 结果来自其论文。所有方法使用 symmetric quantization，OPT group size 为 128，LLaMA2 group size 为 64。

由于 FIGNA 和 FIGLUT 不量化 attention layers，所以 accuracy 反映的是 linear layer quantization。

结果显示，AxCore 在不同模型规模上都达到竞争性甚至更好的 perplexity。对于 OPT 2.7B 到 30B，AxCore 匹配或优于现有 4-bit accelerator designs。在 OPT-6.7B 和 OPT-13B 上，AxCore 获得最低 perplexity。对于 LLaMA2 7B 和 70B，AxCore 精度接近 FP16，并优于 FIGNA 和 FPMA。

#### 6.5.2 KV Cache Quantization

除了 linear layers，attention mechanism 对端到端推理也很重要。为了支持 AxCore 上的端到端推理，论文将 KV cache 量化到 4-bit。OPT 模型使用 group size 64；K cache 使用 E1M2，V cache 使用 E3M0。LLaMA2 模型中 K cache 使用 E2M1，V cache 使用 E3M0。

结果显示，AxCore-KV 引入的精度损失很小。这说明数据格式选择会显著影响 KV cache quantization 的精度，因此针对 KV cache 的 data format calibration 是有价值的未来方向。

#### 6.5.3 Accuracy Improvement Breakdown

Table 2 还展示了 AxCore 各个设计特性如何提升 accuracy。

从 mpFPMA 开始，如果只使用 E2M1 format 而不使用 constant compensation 和 SNC，会观察到较高 perplexity。加入 SNC 后 perplexity 下降，说明 subnormal number conversion 有效果。进一步加入 constant compensation 后，accuracy 继续改善。最终 AxCore 将上述优化与 format-aware quantization 结合，在 4-bit designs 中获得最佳结果。

#### 6.5.4 Zero-shot Performance

论文还在四个 zero-shot benchmark 上评估 AxCore：

- ARC-e
- HellaSwag
- PiQA
- Winogrande

对于 LLaMA2 70B，AxCore 获得平均 accuracy 81.78%，接近 FP16 baseline 的 81.91%，并优于 INT4 和 FP4 quantization implementations。对于 OPT 30B，AxCore 平均 accuracy 为 70.76%，也接近 FP16 baseline 的 71.05%。

#### 6.5.5 Numerical Accuracy

论文还用 signal-to-noise ratio（SNR）评估 AxCore 的 numerical accuracy。SNR 定义为 exact matrix multiplication power 与 approximation noise power 的比值，单位为 decibels。SNR 越高，说明 approximate results 在 magnitude 和 direction 上保留得越好。

测试覆盖典型 fan-in values，从 128 到 32,768。结果显示，SNC、compensation 和 format-aware quantization 可以改善数值精度。

### 6.6 Comparison with Non-mpGEMM Designs

论文还将 AxCore 与非 mpGEMM designs 进行比较，尤其是 Tender 这类 INT-based 方案。Tender 采用 weight-activation quantization，并使用 chunking 和 reordering 处理 activation 和 KV cache 中的 outliers。

结果显示，在端到端 LLM inference accuracy 上，AxCore 相比 Tender 有更好表现。尤其是在低比特场景中，AxCore 的 format-aware FP quantization 和 mpFPMA 设计更适合在保持精度的同时降低硬件成本。

## 7 Conclusion

本文提出 AxCore，一种量化感知的近似 GEMM 单元，用于高效 mixed-precision matrix multiplication 和 LLM inference。通过结合 FPMA 与低比特 floating-point quantization，AxCore 消除了 multiplier，并显著简化了每个 PE 的逻辑。

据作者所知，AxCore 是第一个将 FPMA 潜力用于 LLM inference 的架构。AxCore 集成了一组轻量但有效的技术：

- subnormal number conversion
- mean-based error compensation
- adaptive format-aware quantization

评估表明，AxCore 相比 FP baselines 获得最高 12.5 倍 compute density 提升，并相比 INT4 accelerators 获得 50% 到 70% area savings，同时实现更低 perplexity。

AxCore 当前处理标准 low-bit FP formats。将其扩展到 custom data types 或 block-based formats 是有价值的未来方向。

## A Artifact Appendix

### A.1 Abstract

该 artifact 包含复现本文关键结果所需的组件，包括：

1. AxCore hardware design 的 SpinalHDL RTL code。
2. 用于 LLM accuracy 的 evaluation scripts。
3. 用于 end-to-end performance evaluation 的 cycle-accurate simulator。

这些组件可以用于复现 Table 2、Table 3、Figure 2 和 Figure 17 的数据。

### A.2 Artifact Check-list

主要信息包括：

- Compilation：NVCC 12.4，GCC 11.4.0。
- Models：OPT-2.7B、OPT-6.7B、OPT-13B、OPT-30B、LLaMA2-7B、LLaMA2-70B。
- Datasets：WikiText-2、ARC-e、HellaSwag、PiQA、Winogrande、Pile。
- Runtime environment：Ubuntu 22.04.5 LTS，CUDA 12.4，PyTorch 2.5.1。
- Hardware：一台带 x86 processor 和四张 NVIDIA RTX 6000 Ada GPU 的服务器。
- Output：model perplexity、accuracy、simulator energy 和 performance。
- Disk space：约 270GB。
- Environment preparation time：约 30 分钟。

### A.3 Description

Artifact 由几个部分组成：

- `Hardware/AxCore`：AxCore hardware design 和 functional verification。
- `Software/AxCore`：基于 PyTorch 的 LLM accuracy evaluation framework。
- `Software/axcore_simulator`：用于 performance 和 energy evaluation 的 cycle-accurate simulator。
- `Profile`：不同 sequence length 下 OPT 和 LLaMA models 的 GEMM operation percentage。

### A.4 Installation

作者在 GitHub 仓库中为各个实验提供 README 文件：

https://github.com/CLab-HKUST-GZ/micro58-axcore

### A.5 Experiment Workflow

Artifact evaluation 被分为三部分，每部分用于复现论文中的一类结果。

1. AxCore hardware functional verification：
   - 按 `Hardware/AxCore/README.md` 执行。

2. LLM accuracy evaluation：
   - 复现 Table 2 和 Table 3。
   - 按 `Software/AxCore` 的说明创建 Conda environment。
   - 执行对应 shell script。脚本会自动下载所需 Hugging Face models 和 datasets，或使用缓存，然后执行 AxCore evaluation。

3. Performance and energy evaluation：
   - 使用 `Software/axcore_simulator`。
   - 运行 simulator 并绘制结果，最终生成 `results/fig_17.pdf`。

4. GEMM operations percentage：
   - 复现 Figure 2。
   - 运行 profiling script 分析 workload distribution。
   - 生成 `figure2.pdf`。

### A.6 Evaluation and Expected Results

实验主要包括：

- AxCore hardware design evaluation。
- LLM accuracy evaluation。
- AxCore simulator performance evaluation。

仓库的 README 文件中提供了 Table 2、Table 3、Figure 2 和 Figure 17 的 expected results。

### A.7 Methodology

Artifact review 和 badging methodology 参考 ACM artifact review policy 和 cTuning artifact evaluation 相关方法。

## 术语表

| 英文术语 | 中文理解 |
|---|---|
| GEMM | 通用矩阵乘法，LLM 中 linear layer 和 projection 的核心计算 |
| FP-GEMM | 使用浮点操作数的 GEMM |
| mpGEMM | mixed-precision GEMM，例如 FP16 activation x FP4 weight |
| weight-only quantization | 只量化权重，激活保留较高精度 |
| indirect GEMM | 先 dequantize 权重，再做普通 GEMM |
| direct mpGEMM | 直接在低比特权重和高精度激活之间做 GEMM |
| FPMA | 用整数加法近似浮点乘法 |
| mpFPMA | mixed-precision 场景下的 FPMA |
| SNC | subnormal number conversion，用于处理低比特 FP 的 subnormal |
| format-aware quantization | 根据 weight group 分布选择不同低比特 FP 格式 |
| systolic array | 常见矩阵乘硬件阵列结构，由 PE 组成 |
| PE | processing element，阵列中的基本计算单元 |
| weight-stationary | 权重保持在 PE 中，激活流动的数据流 |
| CACTI | SRAM 能耗、面积估算工具 |
| TOPS/W | 每瓦特 tera operations，衡量能效 |
| TOPS/mm2 | 每平方毫米 tera operations，衡量计算密度 |

## 对当前实验推进的直接启发

1. AxCore 本身已经是 multiplier-free design，所以后续不能简单表述为“把 AxCore 的乘法器换成 AM-LUT”。
2. 第一阶段 AM-LUT 更适合定位为一个 alternative approximate GEMM / FIGLUT-like design，与 AxCore 做参数层比较。
3. 如果要进入 AxCore 内部改造，需要明确 AM-LUT 替代的是 SNC、Align、Add、Norm、AxScale 中的哪一段逻辑，而不是抽象地替代 multiplier。
4. Figure 17 的复现主要依赖 simulator、synthesis CSV 和 CACTI。短期最稳妥实验仍然是通过 `params/systolic_array_synth_*.csv` 做 sensitivity study。
5. 如果未来做 accuracy/perplexity，则必须回到 Section 4 的 SNC、compensation 和 format-aware quantization，而不仅是硬件面积/功耗。
