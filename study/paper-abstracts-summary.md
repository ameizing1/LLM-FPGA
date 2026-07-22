# 参考论文摘要梳理

更新时间：2026-06-26

本文档覆盖以下 15 篇论文：

- AM-LUT 文件夹中的 10 篇
- LLM-FPGA 文件夹中的 4 篇
- 补充的 Adana 论文 1 篇

整理方式：主要依据论文标题、摘要、关键词、引言前几页和作者给出的贡献描述。每篇按“研究问题、核心方法、主要结果、与当前课题的关系”梳理。

## 总体脉络

这些论文可以分成三条线：

1. **FPGA 上的乘法器与近似乘法器设计**
   - 关注 LUT、carry chain、DSP、softcore multiplier、approximate multiplier、低位宽乘法器。
   - 代表论文：ARITH15、TCAD22、TCASI24、ARITH24、TC23、QUADOL。

2. **神经网络/Transformer 的量化与 LUT-friendly 计算**
   - 关注低比特量化、硬件友好数值表示、RCCM、LUT-based operations、FPGA 资源利用。
   - 代表论文：AddNet、More AddNet、ViT FPGA accelerator、Adana。

3. **LLM 推理的硬件/系统加速**
   - 关注 LLM inference、GEMM、KV cache、FPGA mapping、CXL、近似 GEMM、量化微架构。
   - 代表论文：AxCore、FlightLLM、CXL-SpecKV、Adana。

对你当前最相关的主线是：

**LLM 低比特量化 -> 近似/低成本 GEMM -> FPGA/LUT-based 乘法器实现 -> 软硬件协同评估。**

---

# AM-LUT 方向

## 1. AddNet: Deep Neural Networks Using FPGA-Optimized Multipliers

**来源**：IEEE TVLSI 2020  
**关键词**：FPGA, DNN, RCCM, quantization, multiplier

### 研究问题

低精度算术常用于在 FPGA 上加速深度学习，但低精度会带来精度损失。传统做法通常把乘法器统一降位宽，但这并不一定最适合 FPGA 的 LUT 结构。本文关注的问题是：是否可以用更适合 FPGA 的乘法器形式，在节省资源的同时尽量保持神经网络精度。

### 核心方法

论文提出使用 **reconfigurable constant coefficient multipliers, RCCMs** 替代普通低精度乘法器。RCCM 用加法、减法、移位和多路选择器实现乘常数操作，能够更好映射到 FPGA 逻辑资源。为了降低量化造成的信息损失，作者还提出训练方法，将 RCCM 可表示的系数集合映射到神经网络权重分布。

### 主要结果

在 AlexNet、ResNet-18 和 ResNet-50 上实验，RCCM 方案相对传统 8-bit 量化网络最高可节省约 50% 资源，并带来速度和功耗收益。最低资源版本的 RCCM 精度超过 6-bit fixed-point；其他 RCCM 实现至少能达到接近 8-bit 均匀量化的精度，同时显著节省硬件资源。

### 与当前课题的关系

这篇论文说明：**神经网络权重表示可以和 FPGA 乘法器结构联合设计**。它为你后续理解“不是简单 INT8/INT4，而是让数值表示适配硬件结构”提供了基础思想。

---

## 2. An Efficient Softcore Multiplier Architecture for Xilinx FPGAs

**来源**：ARITH 2015  
**关键词**：softcore multiplier, Xilinx FPGA, LUT, carry chain, Booth encoding

### 研究问题

FPGA 内置 DSP 数量有限，而且位宽固定。对于小位宽乘法或不匹配 DSP 位宽的乘法，直接使用 DSP 会浪费资源。因此本文研究如何用 FPGA slice/LUT/carry chain 高效实现 softcore multiplier。

### 核心方法

论文提出一种 array-like softcore multiplier。与传统“部分积生成 + compressor tree”不同，该结构让每一行生成一个部分积，并直接通过快速 carry chain 与前面行结果相加。论文使用 radix-4 Booth 编码/解码减少部分积行数，并使 Booth encoder 和 decoder 能映射到单个 6-input LUT。

### 主要结果

该结构完全避免 compressor tree，形成规则阵列结构，并且可以在不增加额外资源的情况下支持 MAC 操作。相较此前方法，最多可减少约 50% slice 资源。

### 与当前课题的关系

这是理解 **Xilinx FPGA 上 LUT/carry-chain 乘法器实现** 的基础论文。后续看 AM-LUT、TCASI24 或设计 INT8 近似乘法器时，这篇有助于理解 FPGA 结构约束。

---

## 3. Deep Learning with INT8 Optimization on Xilinx Devices

**来源**：Xilinx White Paper WP486, 2017  
**关键词**：INT8, DSP48E2, Xilinx FPGA, deep learning inference

### 研究问题

深度学习推理可以使用较低精度来降低计算和存储开销。本文关注 Xilinx UltraScale/UltraScale+ FPGA 中 DSP48E2 如何高效支持 INT8 深度学习计算。

### 核心方法

白皮书分析了 Xilinx DSP48E2 slice 如何执行 INT8 MACC 操作。其关键点是：DSP48E2 可以在共享 kernel weights 的情况下处理两个并发 INT8 MACC，并解释了相关位宽约束，例如为什么 24-bit 输入是利用该技术的最小尺寸。

### 主要结果

文中声称 Xilinx DSP 架构在 INT8 深度学习 OPS 上，相比同资源数量的其他 FPGA DSP 架构可达到约 1.75 倍峰值方案级性能。

### 与当前课题的关系

这是理解 **INT8 在 Xilinx FPGA DSP 上如何高效实现** 的工程资料。它可以作为你比较 LUT-based multiplier 和 DSP-based INT8 MAC 的背景材料。

---

## 4. Hardware-Efficient Multipliers With FPGA-Based Approximation for Error-Resilient Applications

**来源**：IEEE TCAS-I 2024  
**关键词**：approximate computing, FPGA, approximate multiplier, LUT sharing, carry switching

### 研究问题

近似乘法器能为误差容忍型应用节省硬件，但许多已有近似乘法器主要面向 ASIC，直接用于 FPGA 不一定有同样收益。本文研究如何面向 FPGA 结构设计准确和近似乘法器。

### 核心方法

论文提出 FPGA-based accurate and approximate 4x4 multipliers，核心方法包括 **LUT sharing** 和 **carry switching**。LUT sharing 通过共享输入相同的两个 LUT 来提高 LUT 利用率；carry switching 则部分保留被截断的进位，将其切换到相邻计算中以补偿精度损失。作者进一步提出三种近似加法器，用于构建更高阶乘法器。

### 主要结果

作者组合构造了 140 种 8x8 乘法器，为不同应用需求提供选择。与 Xilinx exact multiplier 相比，提出的近似 8x8 乘法器最高可获得约 38.75% 功耗、17.29% 延迟、28.17% 面积改进。论文还扩展到 16x16 乘法器，并在图像处理任务上验证实用性。作者提供了开源 multiplier library。

### 与当前课题的关系

这是你当前最关键的 AM-LUT 论文之一。它给出了 **FPGA LUT-based 近似乘法器的具体结构、误差/资源权衡和开源 Verilog**，可能成为替换 AxCore 中 INT8/低位宽乘法单元的候选。

---

## 5. High-Performance Accurate and Approximate Multipliers for FPGA-Based Hardware Accelerators

**来源**：IEEE TCAD 2022  
**关键词**：FPGA, softcore multiplier, approximate multiplier, LUT, carry chain, hardware accelerator

### 研究问题

乘法是图像/视频处理、机器学习等加速器中的核心运算。FPGA DSP 数量有限、位置固定，且对小位宽乘法可能低效。已有 FPGA soft multiplier IP 仍有优化空间。本文研究如何设计更高性能、更省资源的准确和近似 softcore multipliers。

### 核心方法

论文提出通用的、面积优化的、低延迟的准确和近似 softcore multiplier 架构，利用 FPGA 的 LUT 结构和 fast carry chains 来降低 critical path delay 和资源使用。

### 主要结果

相较 Xilinx multiplier LogiCORE IP，提出的 unsigned 和 signed 准确乘法器分别最高减少约 25% 和 53% LUT。对于 unsigned approximate multipliers，相比 LogiCORE IP 可最高降低约 51% critical path delay，且输出精度损失很小。论文还把这些乘法器用于图像和视频应用加速器，并开源了准确和近似乘法器库。

### 与当前课题的关系

这篇可作为 **FPGA soft multiplier 设计和近似乘法器对比基线**。如果你要选择候选近似乘法器，它能提供资源、延迟、精度权衡参考。

---

## 6. Low-power Approximate Multipliers Using Encoded Partial Products and Approximate Compressors

**来源**：IEEE JETCAS 2018  
**关键词**：approximate computing, approximate multiplier, compressor, low power

### 研究问题

近似计算适合图像、多媒体、通信等误差容忍应用，而乘法是这些应用的重要操作。本文研究如何通过近似 compressor 和部分积编码构造低功耗近似乘法器。

### 核心方法

作者首先提出一个近似 4:2 compressor。该 compressor 初始误差较大，但通过使用 generate/propagate 信号对输入进行编码，显著减少 truth table 中的错误行。基于改进 compressor，作者设计两种 4x4 乘法器，并进一步扩展到 16x16 和 32x32。论文还构建了基于该 compressor 的 signed Booth multiplier。

### 主要结果

在 MRED 指标下，最准确的 16x16 unsigned 设计相较同精度水平的其他设计可降低约 44% PDP。radix-4 signed Booth 乘法器相较其他近似 Booth 乘法器，在 PDP-MRED 产品上降低约 52%。论文还在图像锐化、JPEG 和 MIMO 通信系统中验证了应用效果。

### 与当前课题的关系

这篇是 **ASIC/通用近似乘法器思想** 的重要背景。它不一定直接适合 FPGA，但能帮助你理解误差指标、compressor-based approximate multiplier 和应用级误差评估。

---

## 7. More AddNet: A Deeper Insight into DNNs Using FPGA-Optimized Multipliers

**来源**：ISCAS 2023  
**关键词**：FPGA, DNN accelerator, RCCM, training flow, low precision

### 研究问题

AddNet 已证明 RCCM 能比普通低精度乘法更适合 FPGA，但仍需要进一步提高精度、扩大可选硬件表示空间，并改进训练流程。本文研究更完善的 DNN-to-FPGA RCCM 训练工具流。

### 核心方法

论文提出改进的训练工具流，包括 layer-wise weight quantization、更大的 RCCM coefficient set 搜索空间，以及优化的 retraining。RCCM 仍然用 shift-and-add 替代昂贵通用乘法器。

### 主要结果

相比原 AddNet，改进工具流提高了网络精度，同时每次乘法只需要 1 到 3 个加法器，降低整体复杂度和内存带宽需求。实验在多个 ResNet 和 ImageNet 数据集上进行。

### 与当前课题的关系

这篇展示了 **硬件友好的权重量化/训练流程如何和乘法器结构共同设计**。对于你理解“量化不是孤立算法，而是要服务硬件实现”很有帮助。

---

## 8. QUADOL: A Quality-Driven Approximate Logic Synthesis Method Leveraging Dual-Output LUTs for Modern FPGAs

**来源**：DATE 2026  
**关键词**：approximate logic synthesis, dual-output LUT, FPGA, LUT merging

### 研究问题

现代 FPGA 支持 dual-output LUT，可减少设计面积。已有工作尝试把 dual-output LUT 用于近似计算，但多局限在小规模算术电路。本文研究如何将 dual-output LUT 用于更通用、更大规模的 approximate logic synthesis。

### 核心方法

论文提出 QUADOL，一种 quality-driven approximate logic synthesis 方法，可以把两个 single-output LUT 近似合并为一个 dual-output LUT。LUT 选择问题被建模为 maximum matching，以最大化面积节省。作者还提出 QUADOL+，用于把 QUADOL 集成到已有 ALS 方法中。

### 主要结果

实验显示 QUADOL+ 相较先前方法能获得显著面积降低。其意义在于将 dual-output LUT 的近似合并能力从特定小算术电路扩展到更通用的近似逻辑综合流程。

### 与当前课题的关系

它不直接针对 LLM 或乘法器，但有助于理解 **FPGA LUT 级近似逻辑优化**。如果后续要自动生成或优化近似乘法器/PE，QUADOL 提供了逻辑综合层面的思路。

---

## 9. Small Logic-based Multipliers with Incomplete Sub-Multipliers for FPGAs

**来源**：ARITH 2024  
**关键词**：small multiplier, tiling, incomplete sub-multiplier, FPGA, ILP

### 研究问题

AI 推理正向 8-bit 甚至更低精度发展，小位宽乘法需求增加。FPGA 的大 DSP block 对许多小乘法并不高效。本文研究如何设计更适合 FPGA 的小型 logic-based multipliers。

### 核心方法

论文基于 multiplier tiling 方法。传统 tiling 通常使用完整矩形 N x K sub-multiplier。作者提出关键观察：sub-multiplier 不必一定是完整矩形，使用 incomplete/non-rectangular sub-multipliers 可能更高效。论文通过 ILP 寻找高效的不规则 incomplete sub-multipliers，并在 FPGA 综合实验中评估。

### 主要结果

论文表明，对于小位宽乘法器，可以通过不完整子乘法器获得优于已有 state-of-the-art 设计的资源效率。该方法尤其适合低精度 AI inference 中的大量小乘法。

### 与当前课题的关系

这是 **低位宽 FPGA 乘法器设计的前沿结果**。如果你的目标是 INT8 或更低位宽乘法器替换，这篇是重要对比对象。

---

## 10. Towards Globally Optimal Design of Multipliers for FPGAs

**来源**：IEEE TC 2023  
**关键词**：FPGA multiplier, ILP, multiplier tiling, compressor tree, final adder

### 研究问题

传统乘法器设计通常分三步：部分积生成、compressor tree 设计、final adder 选择。但在 FPGA 上，这三个步骤相互影响。逐步优化可能错过全局最优。本文研究如何联合优化这些设计步骤，得到更优 FPGA 乘法器。

### 核心方法

作者提出整体优化框架，把 multiplier tiling、compressor tree generation 和 final adder selection 合并到一个 ILP formulation 中。目标是在给定 sub-multipliers、compressors 和 final adders 的候选集合下，找到 LUT 数最少且 pipeline stage 最优的全局解。

### 主要结果

在 Xilinx FPGA 上，论文展示了实际位宽下可以求得全局最优解，并显著减少 LUT 使用。packing density 实验表明，在同一器件上可映射更多乘法器实例。

### 与当前课题的关系

这篇提供了 **乘法器自动设计和全局优化方法**。如果后续不只是手工选近似乘法器，而是想系统搜索适合 LLM GEMM 的 FPGA 乘法器结构，这篇很有价值。

---

# LLM-FPGA 方向

## 11. An FPGA Accelerator for Vision Transformer with Quantization and LUT-Based Operations

**来源**：IEICE Transactions 2026  
**关键词**：Vision Transformer, FPGA, quantization, LUT-based operations

### 研究问题

Vision Transformer 计算和存储需求高，在资源受限设备上部署困难。本文研究如何通过量化和 LUT-based operations 在 FPGA 上高效加速 ViT。

### 核心方法

论文提出两个改进量化方法，在较低 bit-width 下保持较好精度；同时设计线性和非线性单元以支持 ViT 中多种操作。整个 accelerator 采用 LUT-based design，在 FPGA 上实现和评估。

### 主要结果

在 ImageNet 上，提出的量化方法在 2-bit 宽度下达到 80.74% 精度，比若干 state-of-the-art ViT 量化方法高约 0.1% 到 0.5%。硬件方面，FPGA accelerator 达到 7.06 FPS/W 和 246 GOPS/W 的峰值能效。

### 与当前课题的关系

虽然目标模型是 ViT 而非 LLM，但它展示了 **Transformer-like 模型如何用量化和 LUT-based operations 在 FPGA 上部署**。可作为把 LUT 乘法/低比特量化迁移到 LLM 的参考。

---

## 12. AxCore: A Quantization-Aware Approximate GEMM Unit for LLM Inference

**来源**：MICRO 2025  
**关键词**：LLM inference, approximate GEMM, FPMA, quantization, systolic array

### 研究问题

LLM 推理计算和内存需求巨大，Transformer-based LLM 严重依赖 FP-GEMM，而 FP-GEMM 同时主导计算吞吐和内存带宽。本文研究如何在保持 LLM 精度的同时，降低 GEMM 单元的硬件成本。

### 核心方法

论文提出 AxCore，一种 quantization-aware approximate GEMM unit。它结合 weight-only quantization 和 floating-point multiplication approximation, FPMA。不同于传统 GEMM，AxCore 完全消除乘法器，用低位整数加法替代乘法，并构建新的 systolic array。关键技术包括：

- mixed-precision FPMA-based PE，支持压缩权重和高精度激活直接计算；
- subnormal number handling、error compensation、format-aware quantization；
- systolic array 级优化，包括共享 correction 和 normalization logic。

### 主要结果

在开源 LLM 上评估，AxCore 相比传统 FP GEMM units 获得约 6.3x 到 12.5x 更高 compute density。相比 state-of-the-art INT4 accelerators FIGLUT 和 FIGNA，compute density 分别提升约 53% 和 70%，同时保持更低或可比 perplexity。论文提供开源代码。

### 与当前课题的关系

这是当前最核心论文之一。它是你后续“以 AxCore 为平台，替换近似 INT8/LUT-based 乘法器或 GEMM 单元”的直接基础。

---

## 13. CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving

**来源**：FPGA 2026  
**关键词**：CXL, FPGA, LLM serving, KV cache, memory disaggregation, speculative execution

### 研究问题

LLM 在数据中心部署时，KV cache 占用大量 GPU 显存，限制 batch size 和系统吞吐。本文研究如何通过 CXL 和 FPGA 加速器缓解 LLM serving 中的 KV cache memory wall。

### 核心方法

论文提出 CXL-SpecKV，一种 disaggregated KV-cache architecture。核心包括：

- CXL-based memory disaggregation，将 KV cache offload 到远端 FPGA memory；
- speculative KV-cache prefetching，预测并预取未来 token 的 cache entries；
- FPGA-accelerated KV-cache compression/decompression engine，将内存带宽需求最高降低约 4x。

### 主要结果

在 LLM workload 上评估，CXL-SpecKV 相比 GPU-only baseline 最高达到约 3.2x 吞吐提升，内存成本降低约 2.8x，并保持准确率。论文表明 memory disaggregation 与 speculative execution 可以缓解大规模 LLM serving 的 memory wall。

### 与当前课题的关系

这是后续扩展方向，偏 **KV cache、CXL、数据中心 LLM serving、内存系统**。相比当前的 AxCore/Adana/AM-LUT 主线，它更偏系统和互连，短期可作为了解，不必优先深入。

---

## 14. FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs

**来源**：FPGA 2024  
**关键词**：LLM, FPGA, complete mapping flow, sparsity, quantization, DSP48, on-chip decode

### 研究问题

LLM 计算和存储开销巨大，压缩技术如稀疏化和量化能降低开销，但现有 GPU 和 Transformer accelerator 难以高效处理压缩后的 LLM，存在计算效率低、内存带宽利用不足、编译开销大等问题。本文研究如何在 FPGA 上完整映射并高效执行 LLM 推理。

### 核心方法

论文提出 FlightLLM，一个面向 FPGA 的 LLM inference complete mapping flow。核心设计包括：

- configurable sparse DSP chain，支持不同稀疏模式，提高计算效率；
- always-on-chip decode scheme，通过 mixed precision 支持提升内存带宽利用；
- length-adaptive compilation，降低动态输入长度带来的编译/指令存储开销。

### 主要结果

在 Xilinx Alveo U280 上，FlightLLM 相比商业 GPU 如 NVIDIA V100S，在 LLaMA2-7B、vLLM、SmoothQuant、batch size=1 设置下实现约 6.0x 更高能效和 1.8x 更好成本效率。在 Versal VHK158 上，FlightLLM 相比 NVIDIA A100 获得约 1.2x 更高吞吐。

### 与当前课题的关系

这是理解 **LLM 在 FPGA 上完整部署流程** 的核心参考。它使用 DSP-based 方案，而你当前可能更关注 LUT/近似乘法器/量化，但 FlightLLM 能帮你建立系统级视角。

---

# LLM 量化方向

## 15. Adana: Accelerating Large Language Models via Adaptive Nonuniform Asymmetric Quantization

**来源**：DAC 2026  
**关键词**：LLM, low-bit quantization, group-wise quantization, adaptive numeric type, accelerator

### 研究问题

LLM 参数规模大，推理需要大量存储和计算。低比特量化可以降低内存和计算开销，但 LLM 数据分布具有 long-tailed 特性，尤其在 small group-wise quantization 中，数据分布会出现非均匀和非对称特征。已有 adaptive numeric type 往往假设分布较对称，且对低比特 activation quantization 和高效在线量化支持不足。

### 核心方法

论文提出 Adana，一个 hardware-software co-design 框架。其核心包括：

- adaptive nonuniform asymmetric numeric type，刻画 small group 中数据的非均匀性和非对称性；
- approximate quantization error metric，用于高效实现 online adaptive activation quantization；
- 专用 LLM acceleration microarchitecture，包括 QUANT engine 和 MM engine，用于支持低比特 group-wise quantization。

### 主要结果

相比 state-of-the-art 设计，Adana 在多个 LLM 上实现约 1.42x 到 2.10x 加速，并节省约 18.9% 到 48.5% 功耗，同时保持更好的准确率。论文同时评估了 perplexity、量化单元面积/功耗、加速器性能和能耗。

### 与当前课题的关系

Adana 是你理解 LLM 量化思路的关键论文。它补上 AxCore 和 AM-LUT 之前的上游问题：**为什么 LLM 低比特量化难、数据分布如何影响数值类型、在线 activation quantization 如何影响硬件微架构**。

---

# 建议阅读顺序

## 第一阶段：理解 LLM 量化和计算核心

1. Adana
2. AxCore
3. FlightLLM

目标：理解 LLM inference、量化、GEMM、硬件加速、模型精度与硬件代价之间的关系。

## 第二阶段：理解 FPGA 乘法器和近似乘法器

1. TCASI24 Hardware-Efficient Multipliers
2. ARITH15 Efficient Softcore Multiplier
3. ARITH24 Small Logic-based Multipliers
4. TC23 Globally Optimal Multiplier Design
5. TCAD22 High-Performance Accurate and Approximate Multipliers

目标：理解 LUT、carry chain、DSP、softcore multiplier、approximate multiplier 的设计方法和对比指标。

## 第三阶段：理解神经网络到 FPGA 的硬件友好映射

1. AddNet
2. More AddNet
3. ViT FPGA Accelerator
4. Xilinx INT8 White Paper

目标：理解数值表示、训练/量化流程和 FPGA 硬件结构如何共同设计。

## 第四阶段：后续拓展

1. CXL-SpecKV
2. QUADOL
3. Low-power approximate multipliers

目标：分别补 KV cache 系统方向、approximate logic synthesis、通用近似乘法器背景。

# 和当前任务最相关的核心链条

如果只抓最核心的 5 篇，应优先看：

1. **Adana**：LLM 低比特 group-wise quantization 和量化微架构。
2. **AxCore**：LLM approximate GEMM 和开源平台。
3. **TCASI24 Hardware-Efficient Multipliers**：FPGA LUT-based approximate multiplier。
4. **FlightLLM**：LLM 在 FPGA 上的完整部署视角。
5. **ARITH24 Small Logic-based Multipliers**：低位宽 FPGA 乘法器前沿设计。

这五篇能串成一条完整路线：

**LLM 量化问题 -> LLM GEMM 近似计算 -> FPGA 近似乘法器实现 -> FPGA LLM 部署和评估。**
