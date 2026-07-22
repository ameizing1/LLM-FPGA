# AxCore MICRO 2025 论文阅读与实验推进笔记

论文：AxCore: A Quantization-Aware Approximate GEMM Unit for LLM Inference

更新时间：2026-06-28

本笔记用于帮助阅读论文，并把论文内容和当前 AxCore simulator 复现实验、后续 AM-LUT/低比特近似计算实验连接起来。

## 1. 一句话理解

AxCore 的核心不是“把某个乘法器换成近似乘法器”，而是设计一种面向 weight-only quantized LLM 的 multiplier-free mixed-precision GEMM unit。

它把低比特浮点权重（如 FP4/FP8）和高精度激活（如 FP16/BF16/FP32）直接做 mixed-precision GEMM，并用 FPMA（Floating-Point Multiplication Approximation）把传统浮点乘法近似成整数加法。为了让这种近似在 LLM 上不崩精度，它又加入了 subnormal number conversion、constant compensation 和 format-aware quantization。

## 2. 论文要解决的问题

LLM 推理中的主要计算来自 Transformer 里的 GEMM，尤其是 linear projection 和 FFN。权重量很大，所以 weight-only quantization 很常见：权重低比特，激活保持较高精度。

传统做法有两类：

1. Indirect GEMM：先把低比特权重 dequantize 回浮点，再做普通 GEMM。
2. Direct mpGEMM：直接用低比特权重和高精度激活做 mixed-precision GEMM。

AxCore 选择第二类，因为它能避免 dequantization 之后再用完整浮点乘法器计算。

论文的基本判断是：

- 低比特权重可以显著降低存储和带宽。
- 但是如果计算单元仍然依赖完整 FP multiplier，硬件成本仍然高。
- FPMA 可以用整数加法近似浮点乘法，但直接用于 LLM 会有精度问题。
- AxCore 的贡献就是让 FPMA 能用于 low-bit weight + high-precision activation 的 LLM mpGEMM。

## 3. FPMA 基础：为什么可以不用乘法器

浮点数可写成：

```text
x = (-1)^S * 2^(E-B) * (1+M)
```

FPMA 的近似来自：

```text
log2(1 + M) ~= M
```

于是两个浮点数相乘可以在近似意义下变成 exponent/mantissa 表示上的加法：

```text
R = X + Y - B
```

其中 `X = E_x + M_x`，`Y = E_y + M_y`，`R = E_r + M_r`。

这就是 AxCore 能把乘法器替换成低成本整数加法器的基础。

但问题是：原始 FPMA 主要面向相同精度浮点数，例如 FP16 x FP16。LLM weight-only quantization 里常见的是 FP16 activation x FP4 weight，这就变成 mixed precision。AxCore 的 Section 4 就是在解决这个扩展问题。

## 4. AxCore 的三个关键技术点

### 4.1 mpFPMA：把 FPMA 扩展到 mixed precision

AxCore 要直接计算：

```text
FP16 activation x FP4 weight
```

低比特权重和高精度激活的 exponent bias、mantissa width 不同，所以不能直接照搬普通 FPMA。

论文做了两件事：

- mantissa alignment：把低比特 weight mantissa 左移/补零，对齐到 activation 的 fixed-point 表示。
- format-aware bias correction：根据 activation、weight、result 的 exponent bias 加修正项。

论文中的近似表达大致是：

```text
R = A + Align(Wq) - B1 + C1
```

其中：

- `A`：高精度 activation 的近似二进制表示。
- `Wq`：低比特量化权重。
- `B1`：format/bias correction。
- `C1`：constant compensation，用于补偿 FPMA 系统误差。

实验推进意义：如果后续要把 AM-LUT 接进 AxCore，不能只问“乘法器怎么换”，要先确认 AxCore 本来就没有传统 multiplier。更合理的问题是：AM-LUT 是要作为 AxCore 的替代方案，还是要改进某个 baseline（如 FIGLUT/FPC/FPMA）或扩展 AxCore 的 approximate multiplication block？

### 4.2 SNC：Subnormal Number Conversion

低比特 FP 格式里 subnormal 很重要。普通 FP16/FP32 里 subnormal 通常极小，但 FP4/FP8 的 exponent bits 少，subnormal 可能表示相对不小的值，而且在量化权重中很常见。

原始 FPMA 假设 normalized floating point 有隐含 leading one；subnormal 没有这个 leading one，所以直接套 FPMA 会造成明显数值错误。

AxCore 用 SNC 单元把 subnormal 映射到附近的 normalized representation 或 0。论文给了不同 mantissa width（M1/M2/M3）的转换表。

实验推进意义：如果我们后续只做参数层 energy sensitivity，可以暂时不碰 SNC。但如果进入 accuracy/perplexity，就必须理解 SNC，因为它是 AxCore 保精度的核心之一。

### 4.3 Constant Compensation

FPMA 的误差来自 `log2(1+M) ~= M` 的线性近似。这个误差会在 GEMM 和多层网络中累积。

AxCore 做 mean-based constant compensation：离线分析不同 mantissa pair 的误差分布，取一个平均补偿常数，加入 mpFPMA 表达式。

实验推进意义：这对应“近似计算误差模型”。如果我们后续做 AM-LUT，不仅要看 power/area，还要回答：AM-LUT 的误差是否有可补偿结构？能否像 AxCore 一样用常数、分组或格式感知策略补偿？

### 4.4 Format-aware Quantization

AxCore 不固定使用一种 FP4 格式，而是在候选 FP4 类型中选择更适合某个 weight group 的格式，例如：

- E3M0
- E2M1
- E1M2

选择目标是最小化实际 activation distribution 下的误差。论文给出的直觉是：不同层、不同 group 的 weight distribution 不同，有些更适合 power-of-two-like encoding，有些更适合更多 mantissa bits。

实验推进意义：这和 Adana 或其他 adaptive quantization 方向有连接。我们后续可以关注“量化格式选择”和“硬件近似单元”之间是否可以协同，而不是只做固定 W4/W8 的硬件参数替换。

## 5. AxCore 架构怎么工作

AxCore 使用 weight-stationary systolic array：

- 低比特量化权重预加载并保持在 PE 中。
- 高精度 activation 沿行传播。
- 每行有共享 PreAdd 单元，提前计算修正后的 activation-like term。
- 每个 PE 内部用 SNC 处理低比特权重，再用低比特整数加法完成 approximate multiplication。
- 后处理包括 Norm、AxScale、Accumulator。

重要模块：

- `PreAdd`：在行级共享计算 `A - B1 + C1`，减少每个 PE 内重复逻辑。
- `SNC`：处理低比特 FP 中的 subnormal。
- `Low-Bit Adder`：替代传统 multiplier。
- `Norm`：把近似结果规格化。
- `AxScale`：用 FPMA-style addition 替代 dequantization scaling 中的乘法。
- `Accumulator`：累加 partial sums。

实验推进意义：AxCore 的硬件节省不只是“乘法器变小”，还来自 correction advancing、shared normalization、dequantization multiplier removal 等系统性设计。因此在解释 Figure 17 时，不要把所有收益都归因于一个 multiplier。

## 6. 评估设置与我们复现的关系

论文硬件评估设置：

- RTL：SpinalHDL。
- 综合：Synopsys Design Compiler。
- 工艺：TSMC 28nm。
- 目标频率：1GHz。
- systolic array：64 x 64，4 x 4 tilings。
- simulator：基于 DNNWeaver 的 cycle-level simulator。
- SRAM power：CACTI。
- baselines：FPC、FPMA、FIGNA、FIGLUT、Tender。

这和我们当前已经跑通的 `Software/axcore_simulator` 对上：

- 我们复现的是 Figure 17。
- 输出是 `results/axcore_res.csv` 和 `results/fig_17.pdf`。
- simulator 中 `params/systolic_array_synth_*.csv` 对应综合得到的 area/power 参数。
- Cacti 影响 SRAM/Buffer energy。
- benchmark 当前主要是 OPT13B/OPT30B 的 linear/GEMM workload。

Figure 17 评估条件：

- OPT 13B 和 OPT 30B。
- decoding phase。
- batch size = 32。
- output sequence length = 1。
- bandwidth provisioned adequately。
- 展示 normalized energy breakdown 和 TOPS/W。

我们复现结果中的重要现象：

- `Time` 全部是 1.00。
- 差异主要来自 energy，尤其是 `Core`。
- 这说明第一阶段做 AM-LUT/近似计算扩展时，最合理入口仍然是 synthesis CSV 参数层 sensitivity study，而不是先改 cycle model 或硬件 RTL。

## 7. 你读论文时的推荐顺序

不要从头逐字读。建议按这个顺序：

1. Abstract + Introduction：弄清楚目标和贡献。
2. Section 2.3 + 2.4：理解 direct mpGEMM 和 FPMA。
3. Section 3：理解为什么 naive FPMA 用在 LLM 上会出问题。
4. Section 4.1：理解 mpFPMA 公式。
5. Section 4.2：理解 subnormal 为什么重要。
6. Section 4.3：理解 constant compensation。
7. Section 4.4：理解 format-aware quantization。
8. Section 5：看 Figure 8/9/10/11/12/13，理解架构。
9. Section 6.1/6.4：对照我们复现的 Figure 17。
10. Section 6.5：理解 accuracy/perplexity 为什么后续不能完全跳过。
11. Appendix A：理解 artifact 结构，对照 repo 目录。

## 8. 当前阶段最该回答的问题

为了推进接下来的实验，你读完后应能回答：

1. AxCore 为什么是 multiplier-free，而不是普通 approximate multiplier accelerator？
2. FPMA 近似乘法为什么可以用加法实现？
3. 为什么 low-bit FP 的 subnormal 会严重影响 LLM accuracy？
4. `SNC`、`constant compensation`、`format-aware quantization` 分别解决什么误差？
5. Figure 17 的 energy breakdown 来自哪些硬件/模拟参数？
6. 为什么我们现在不应该直接改 SpinalHDL？
7. 如果做 AM-LUT，第一版应该定位为 AxCore 的替代对比、baseline 改造，还是 AxCore 内部 approximate block 的扩展？

## 9. 和后续 AM-LUT 实验的关系

当前建议仍然不变：第一阶段先做参数层 sensitivity study。

但读完论文后，需要更精确地定义实验假设：

### 路线 A：AM-LUT 作为 AxCore baseline 对比

把 AM-LUT 看成另一种 approximate GEMM/FP-INT GEMM 单元，和 AxCore/FIGLUT/FIGNA/FPC/FPMA 比较 area/power/energy。

适合第一阶段，因为它可以从 `params/systolic_array_synth_*.csv` 入手。

### 路线 B：AM-LUT 改造 FIGLUT 类 LUT-based baseline

因为 FIGLUT 本身是 LUT-based FP-INT GEMM design，AM-LUT 和 FIGLUT 的概念更接近。可以先思考 AM-LUT 是否更适合作为 FIGLUT-like baseline 的改进，而不是直接插进 AxCore。

### 路线 C：AM-LUT 扩展 AxCore 内部 approximate multiplication block

这条路线更难，因为 AxCore 的 approximate multiplication 已经主要是 integer addition，不是传统 LUT multiplier。除非 AM-LUT 能替代 SNC/Align/Add/Norm 中某个具体逻辑，或者支持新的 low-bit/custom format，否则强行“替换乘法器”会概念不清。

当前最稳妥判断：

- 短期：参数层比较 AM-LUT-like 设计和 AxCore/FGLUT/FIGNA。
- 中期：补 accuracy/error model，证明 AM-LUT 的误差能被 LLM 接受或补偿。
- 后期：再考虑 RTL/FPGA/SpinalHDL。

## 10. 本地辅助材料

已复制论文到 ASCII 路径，方便脚本处理：

`work\papers\axcore-micro25.pdf`

已抽取文本：

`outputs\axcore-micro25-extracted.txt`

已渲染关键页：

`outputs\axcore-paper-pages\page-04.png`

`outputs\axcore-paper-pages\page-08.png`

`outputs\axcore-paper-pages\page-11.png`

`outputs\axcore-paper-pages\page-12.png`

其中：

- page-04：direct mpGEMM 和 FPMA 基础。
- page-08：format-aware quantization 和 AxCore array overview。
- page-11：area、compute density、energy evaluation setup。
- page-12：compute density、perplexity、zero-shot accuracy。

