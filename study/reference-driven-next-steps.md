# 基于导师参考文献的下一步计划

更新时间：2026-06-23

## 结论

导师认可原学习计划，并强调“需要对大模型很熟悉”。结合她给的参考文献和两个 readme，接下来方向应从泛泛的 “AI Systems / AI Infra” 收敛为：

**面向 LLM 推理的低比特量化、近似计算、GEMM/乘法器优化与 FPGA/软硬件协同实现。**

更具体地说，短期重点不是先做 KV Cache 或 vLLM serving，而是：

1. 熟悉 LLM 的基本结构和推理计算热点。
2. 精读 Adana，理解 LLM 量化的思路：group-wise quantization、非均匀/非对称数值类型、online activation quantization、微架构评估。
3. 理解 LLM 中 GEMM / matrix multiplication 的计算占比和硬件开销。
4. 精读 AxCore，跑通其开源平台。
5. 理解 AM-LUT 方向的 FPGA LUT-based / approximate multiplier。
6. 尝试将近似 INT8 乘法器替换进 AxCore 或其模拟/验证流程。

## 新增文献：Adana 的作用

老师补充的 **Adana: Accelerating Large Language Models via Adaptive Nonuniform Asymmetric Quantization** 是组内师兄 2026 DAC 的 LLM 量化论文，并提名最佳论文。它在整个文献链条中的位置很重要：

**Adana 负责告诉你：LLM 低比特量化为什么难、数据分布有什么特点、怎样设计硬件友好的量化格式和在线量化单元。**

它和 AxCore / AM-LUT 的关系可以理解为：

- **Adana：量化算法 + 数值类型 + 在线量化微架构**
- **AxCore：量化感知的近似 GEMM 单元，用于 LLM inference**
- **AM-LUT：FPGA LUT-based / approximate multiplier 的底层实现候选**

所以接下来的学习顺序应调整为：

1. 先读 Adana，理解 LLM 量化问题和软硬件协同思路。
2. 再读 AxCore，理解近似 GEMM 如何用于 LLM。
3. 再读 AM-LUT，寻找可替换进 GEMM/PE 的近似 INT8 乘法器实现。

### Adana 需要重点读什么

重点关注：

- group-wise quantization
- weight quantization vs activation quantization
- online activation quantization
- adaptive numeric type
- nonuniform / asymmetric data distribution
- small group size 下的数据分布变化
- Adana numeric type 的设计动机
- quantization error metric
- QUANT engine 和 MM engine
- perplexity、latency、energy、area/power 的评价方式

不要只看它的最终 speedup，而要学习它的研究思路：

1. 从 LLM 数据分布出发发现问题。
2. 设计新的数值类型。
3. 设计可高效实现的量化流程。
4. 给出硬件微架构。
5. 同时评估模型准确率和硬件性能。

## 文献给出的方向信号

### 1. AM-LUT 文件夹

readme 明确写到：

- TCASI24：基于 FPGA 内 LUT 的乘法器实现优化，包含基本原理、近似实现方法和开源 Verilog 代码。
- ARITH24、ARITH25：低位宽乘法器的 FPGA 实现优化，可作为前沿结果对比。
- 其他论文作为辅助参考。

这说明 AM-LUT 这一组材料的重点是：

- FPGA 上的 softcore multiplier
- LUT-based multiplier
- approximate multiplier
- low-bit multiplier
- resource / delay / power / accuracy trade-off
- Verilog 实现与综合结果

### 2. LLM-FPGA 文件夹

readme 明确写到：

- AxCore：近似乘法器在 LLM 中的应用，先研究开源平台，实现近似 INT8 乘法器替换。
- ViT FPGA accelerator：FPGA 部署 Transformer-like 模型的设计，可作为最终目标形式参考，但目标是 LLM。
- FlightLLM：LLM 在 FPGA 上部署，计算单元基于 FPGA DSP IP 核。
- CXL-SpeckKV：互连接口优化，后续研究方向。

这说明 LLM-FPGA 这一组材料的短期重点是：

- LLM inference
- quantization-aware approximate GEMM
- LLM 中 GEMM 的硬件优化
- FPGA 上 Transformer / LLM 部署
- LUT / DSP / systolic array / GEMM unit
- 后续可能扩展到 KV Cache、CXL、memory disaggregation

## 文献优先级

### 第一优先级：必须精读

1. **Adana: Accelerating Large Language Models via Adaptive Nonuniform Asymmetric Quantization**
   - 重点：LLM 低比特 group-wise quantization、非均匀/非对称数值类型、online activation quantization、QUANT engine、MM engine、accuracy/hardware co-evaluation。
   - 任务：理解师兄论文的完整思路，整理一页“问题-观察-方法-硬件-评估”笔记。

2. **AxCore: A Quantization-Aware Approximate GEMM Unit for LLM Inference**
   - 重点：LLM 中 GEMM bottleneck、weight-only quantization、FPMA、approximate GEMM、systolic array、perplexity 评估、开源平台。
   - 任务：跑通其软件评估、理解硬件结构、明确可替换乘法器的位置。

3. **Hardware-Efficient Multipliers With FPGA-Based Approximation for Error-Resilient Applications - TCASI24**
   - 重点：FPGA LUT-based approximate multiplier 的原理、Verilog 实现、误差和资源指标。
   - 任务：理解近似乘法器结构，查看开源 Verilog，判断如何改成 INT8 或适配 AxCore。

4. **FlightLLM**
   - 重点：LLM inference 在 FPGA 上的整体 mapping flow，prefill/decode、量化、稀疏、on-chip decode、DSP chain。
   - 任务：建立 LLM-FPGA 系统级视角，不必一开始复现完整设计。

### 第二优先级：实现和对比参考

5. **Small Logic-based Multipliers with Incomplete Sub-Multipliers for FPGAs - ARITH24**
6. **Towards Globally Optimal Design of Multipliers for FPGAs - TC23**
7. **High-Performance Accurate and Approximate Multipliers for FPGA-Based Hardware Accelerators - TCAD22**
8. **An Efficient Softcore Multiplier Architecture for Xilinx FPGAs - ARITH15**
9. **Deep Learning with INT8 Optimization on Xilinx Devices**

这些用于补：

- FPGA 低位宽乘法器设计
- LUT 和 carry chain 使用
- DSP48 / INT8 映射
- 资源、时延、功耗对比

### 第三优先级：后续扩展

10. **An FPGA Accelerator for Vision Transformer with Quantization and LUT-Based Operations**
   - 用于理解 LUT-based operations 如何从 ViT 迁移到 Transformer/LLM。

11. **CXL-SpecKV**
   - 后续方向：KV Cache、CXL、memory disaggregation、datacenter LLM serving。
   - 现阶段不作为主线。

## 接下来 4 周建议

### 第 1 周：补 LLM 推理和 Adana 量化背景

目标：

- 对 LLM 结构、推理流程和 GEMM 占比建立基本认识。
- 能讲清 Adana 想解决什么量化问题。

学习内容：

- Transformer decoder
- Attention、MLP/FFN
- prefill / decode
- weight-only quantization
- activation quantization
- group-wise quantization
- nonuniform / asymmetric quantization
- W4A16、W8A16 等表示
- W4A4、W3A3 等表示
- perplexity / zero-shot evaluation 的基本含义
- GEMM 在 LLM 中的位置和开销

任务：

- 精读 Adana abstract、introduction、motivation、numeric type 和 microarchitecture。
- 整理一页笔记：《Adana 的问题、观察、方法和评价指标》。
- 画出 Adana 的数据路径：FP16 activation -> online quantization -> Adana numeric type -> MM engine -> dequantization/accumulation。

验收：

- 能解释为什么 small-group quantization 会出现非对称分布问题。
- 能解释 Adana 为什么同时强调 nonuniform 和 asymmetric。
- 能解释 online activation quantization 为什么影响硬件设计。

### 第 2 周：精读 AxCore 并跑通开源平台

目标：

- 理解 AxCore 如何把量化和近似 GEMM 结合到 LLM inference。
- 搭建 AxCore 环境。

任务：

- 精读 AxCore abstract、introduction、background、method overview。
- 整理一页笔记：《AxCore 的问题、方法和评价指标》。
- 画出 AxCore 的计算路径：activation、quantized weight、approximate GEMM、output。
- 克隆 AxCore GitHub 仓库。
- 阅读 `Software/AxCore`、`Software/axcore_simulator`、`Hardware/AxCore` 的 README。
- 优先跑软件侧最小实验，确认依赖、模型、数据集和输出。
- 记录环境配置、报错、解决方法和结果。

验收：

- 能说明 AxCore 仓库分为哪些部分。
- 能跑出至少一个最小结果，或明确记录卡点。
- 能定位“乘法器/GEMM 单元/PE”相关代码或硬件模块。
- 能解释 AxCore 和 Adana 的区别：一个偏 approximate GEMM，一个偏 adaptive quantization numeric type + microarchitecture。

### 第 3 周：精读 AM-LUT 和近似 INT8 乘法器

目标：

- 理解 FPGA LUT-based approximate multiplier 的结构和指标。
- 找到可以作为替换候选的 INT8 近似乘法器设计。

学习内容：

- exact multiplier vs approximate multiplier
- LUT-based multiplier
- softcore multiplier
- error metrics：MED、MRED、NMED、error rate 等
- FPGA resource：LUT、FF、DSP、delay、power

任务：

- 精读 TCASI24 的结构和实验部分。
- 查看其开源 Verilog 代码。
- 选择 1-2 个低位宽乘法器设计作为候选。
- 整理表格：位宽、资源、延迟、误差、是否有 Verilog、是否适合 INT8 GEMM。

验收：

- 能解释近似乘法器如何用 FPGA LUT 实现。
- 能说明为什么近似乘法器适合 error-resilient workload。
- 能提出一个初步替换方案：替换位置、输入输出位宽、预期收益、可能误差。

### 第 4 周：形成替换方案和小报告

目标：

- 把 AxCore 和 AM-LUT 连接起来。
- 形成一个可以向老师汇报的初步方案。

任务：

- 明确 AxCore 中可替换的计算单元。
- 设计一个 approximate INT8 multiplier 替换路径。
- 如果时间允许，做一个最小 functional simulation。
- 如果暂时无法实现，至少完成接口分析和实验计划。

报告题目建议：

《面向 LLM 推理的近似 INT8 乘法器替换方案初步调研》

报告结构：

1. 背景：LLM inference 中 GEMM 和低比特量化的重要性。
2. Adana：非均匀/非对称 group-wise quantization 的思路。
3. AxCore：量化感知近似 GEMM 的核心思想。
4. AM-LUT：FPGA LUT-based approximate multiplier 的基本方法。
5. 连接点：如何把近似 INT8 乘法器替换进 AxCore，或如何先用 simulator 验证。
6. 实验计划：perplexity / zero-shot accuracy、resource、delay、compute density、energy。
7. 当前卡点和需要老师确认的问题。

## 你现在最该问老师的问题

下一次和老师同步时，可以问：

1. 短期是否以 AxCore 开源平台复现和乘法器替换为主要任务？
2. Adana 是否需要作为量化基础重点精读？是否需要复现其中某些量化实验？
3. 近似 INT8 乘法器是希望偏 Verilog/FPGA 实现，还是可以先用 CUDA/Python simulator 验证误差和 accuracy？
4. 评价指标优先看什么：perplexity、zero-shot accuracy、compute density、LUT/DSP、delay、power，还是综合指标？
5. 是否有指定的 FPGA 平台、综合工具或代码仓库？
6. 第一个月希望做到什么程度：读懂论文、跑通平台、完成替换接口分析，还是实现一个 baseline？

## 对原入学前学习计划的调整

原计划仍然有效，但优先级需要调整。

### 保留

- Transformer、Attention、prefill/decode
- LLM inference 基础
- quantization
- latency / throughput / memory / bandwidth
- 简单实验报告

### 降低优先级

- vLLM serving benchmark
- PagedAttention / continuous batching
- KV Cache 深入优化
- CXL / memory disaggregation

### 新增并置顶

- AxCore 论文和代码
- Adana 论文
- LLM 中 GEMM workload 分析
- weight-only quantization
- activation quantization / online quantization
- group-wise quantization / adaptive numeric type
- approximate GEMM / FPMA
- FPGA LUT-based multiplier
- INT8 approximate multiplier 替换

## 建议的近期定位

可以把自己当前任务表述为：

**先熟悉 LLM 推理与低比特量化基础，重点学习 Adana 的量化思路；然后以 AxCore 为平台，探索 FPGA LUT-based 近似 INT8 乘法器在 LLM GEMM 中的替换和评估。**

这个定位比泛泛说 “AI Infra” 更贴合老师给的文献，也更容易形成第一个可执行任务。
