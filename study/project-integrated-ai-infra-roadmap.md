# 将 AI Infra 能力自然整合进当前项目的路线

更新时间：2026-06-23

## 核心思路

不额外开一条 “AI Infra 学习线”，而是把 AI Infra 需要的能力嵌入当前项目：

**面向低比特 LLM 推理的硬件友好量化与近似 GEMM 加速。**

项目主线保持不变：

1. Adana：理解 LLM 低比特量化和硬件友好数值类型。
2. AxCore：理解并复现 approximate GEMM / LLM accuracy 评估平台。
3. AM-LUT：引入 FPGA LUT-based approximate INT8 multiplier。
4. 替换与评估：观察 accuracy、latency、throughput、resource、energy 的变化。

AI Infra 能力以项目需求的形式自然出现。

## 能力嵌入表

| AI Infra 能力 | 不单独学，而是在项目里这样学 | 项目产出 |
|---|---|---|
| LLM inference 基础 | 为解释 AxCore/Adana，梳理 Transformer 中 GEMM、attention、MLP 的计算占比 | LLM workload breakdown |
| vLLM / TensorRT-LLM | 用它们跑一个 baseline，理解真实推理指标，不深挖 serving 系统 | baseline benchmark 表格 |
| profiling | 对 baseline 推理或 GEMM kernel 记录 latency、memory、GPU utilization | profiling report |
| CUDA/Triton | 先用 Python/CUDA/Triton 写 approximate INT8 GEMM 的软件模拟或 kernel 原型 | approximate GEMM prototype |
| quantization | 复现或简化 Adana/AxCore 的量化流程 | quantization error + perplexity 对比 |
| serving 指标 | 把论文指标转成 AI Infra 常用表达：TTFT、TPOT、tokens/s、memory footprint、energy/token | 统一指标表 |
| hardware-software co-design | 连接量化格式、近似乘法器、GEMM 单元和 LLM accuracy | co-design analysis |
| 系统工程能力 | 把实验脚本、配置、环境、结果表格整理成可复现 repo | reproducible artifact |

## 项目版本设计

### V0：Adana/AxCore 对照基线

目标：

- 为后续 Adana/AxCore/近似乘法器实验建立对照组，而不是单独做一个推理 benchmark。
- 明确后续所有优化要对比的基准：原始模型、常规量化、AxCore/Adana 风格量化或近似计算。

任务：

- 选择与 AxCore/Adana 论文相近的模型或小规模替代模型，如 OPT、LLaMA/TinyLlama、Qwen 小模型。
- 选择与论文相近的数据集或评估方式，如 WikiText-2 perplexity，后续再扩展 zero-shot。
- 跑三类 baseline：
  - FP16 baseline
  - 常规 INT8/INT4 或 weight-only quantization baseline
  - 如果 AxCore/Adana 代码可跑，加入其原始实现结果
- 记录 perplexity、模型大小、显存占用、GEMM 相关耗时或推理耗时。

自然学到：

- Hugging Face generation
- perplexity evaluation
- 量化对模型效果和性能的影响
- AI Infra 常见指标

产出：

- `baseline_results.csv`
- 一页 baseline 分析笔记：《后续近似乘法器替换要和谁比较》

与主线的连接：

- 后续 approximate INT8 multiplier 替换后，必须回到这个 baseline 比较 perplexity / error / latency / resource。
- 如果没有这个 baseline，乘法器优化就只能说明“硬件变小了”，无法说明“对 LLM 推理是否仍然可用”。

### V1：复现 Adana/AxCore 的软件评估链路

目标：

- 理解低比特量化和 approximate GEMM 如何影响 LLM accuracy。

任务：

- 精读 Adana，整理它的数据分布观察和量化方法。
- 跑通 AxCore 的 `Software/AxCore` 或最小 accuracy evaluation。
- 记录模型、数据集、量化格式、perplexity/zero-shot 结果。

自然学到：

- group-wise quantization
- online activation quantization
- weight-only vs weight-activation quantization
- W4A16、W4A4、W3A3
- perplexity / zero-shot benchmark

产出：

- `quantization_eval.md`
- AxCore 环境配置记录
- 一张 Adana vs AxCore 对比表

### V2：approximate INT8 GEMM 软件原型

目标：

- 在不急着上 FPGA 的情况下，先验证近似乘法器对 GEMM 误差和模型评估的影响。

任务：

- 用 Python/NumPy/PyTorch 实现 exact INT8 multiply 和 approximate INT8 multiply。
- 替换一个 toy GEMM 或 AxCore simulator 中的乘法逻辑。
- 比较 output error、MSE、relative error。
- 如果可行，再写一个简单 CUDA/Triton kernel 或使用 PyTorch custom op 思路验证。

自然学到：

- GEMM 数据流
- 低比特整数计算
- CUDA/Triton 的最小使用场景
- error propagation

产出：

- `approx_int8_gemm.py`
- error analysis 表格
- 一个小型 GEMM benchmark

### V3：AM-LUT 近似乘法器接入方案

目标：

- 从软件原型走向硬件实现。

任务：

- 精读 TCASI24 和 AM-LUT 相关 Verilog。
- 选择 1-2 个候选近似乘法器。
- 整理接口：输入位宽、输出位宽、有符号/无符号、延迟、资源。
- 与 AxCore 的 GEMM/PE 结构对齐。

自然学到：

- LUT-based multiplier
- FPGA resource model
- Verilog functional simulation
- hardware/software interface

产出：

- `multiplier_candidates.md`
- Verilog simulation 结果
- 接入方案图

### V4：统一评估框架

目标：

- 让项目看起来像 AI Systems / AI Infra，而不是单点硬件优化。

任务：

- 汇总模型效果：perplexity / zero-shot accuracy。
- 汇总系统指标：latency、throughput、memory footprint。
- 汇总硬件指标：LUT、DSP、FF、delay、power、energy。
- 给出 trade-off：accuracy loss vs resource/power/speedup。

自然学到：

- AI Infra 指标体系
- hardware-aware model evaluation
- benchmark design
- 论文式实验组织

产出：

- `evaluation_dashboard.csv`
- `final_report.md`
- trade-off 图表

## 这样做后，简历如何表达

项目标题可以写：

**Hardware-aware Quantization and Approximate GEMM Acceleration for LLM Inference**

或中文：

**面向 LLM 推理的硬件友好量化与近似 GEMM 加速**

项目描述可以突出：

- 基于 Adana / AxCore 思路研究 LLM 低比特量化与 approximate GEMM。
- 构建 LLM 推理和量化评估 baseline，评估 perplexity、latency、memory footprint。
- 设计/接入 FPGA LUT-based approximate INT8 multiplier。
- 分析 accuracy、resource、delay、energy 之间的 trade-off。
- 使用 Python/PyTorch、simulator、Verilog/CUDA/Triton 进行软硬件协同验证。

对应 AI Infra 关键词：

- LLM inference
- quantization
- GEMM optimization
- approximate computing
- hardware-software co-design
- profiling
- latency / throughput / memory footprint
- accelerator-aware evaluation

## 每阶段该学什么

### 不要单独学 vLLM

只在 V0 用它或 Hugging Face 跑 baseline。

最低要求：

- 能启动模型推理。
- 能记录 tokens/s、显存、延迟。
- 能知道 serving 框架和硬件优化之间的关系。

### 不要单独系统学 CUDA/Triton

只在 V2 为 approximate GEMM 写一个最小原型。

最低要求：

- 会看 GEMM kernel 的输入输出。
- 会理解 memory access 和 compute 的基本瓶颈。
- 能解释为什么硬件乘法器优化可能影响 GEMM 性能。

### 不要单独学分布式 serving

项目早期不需要。

后续如果项目扩展到 CXL-SpecKV 或 KV Cache，再补：

- KV Cache memory footprint
- batching
- TTFT / TPOT
- memory offloading
- CXL / memory disaggregation

## 建议优先级

近期最优顺序：

1. Adana：建立 LLM 量化思路。
2. AxCore：跑通开源平台或最小软件评估。
3. V0 对照基线：围绕 AxCore/Adana 的模型、数据集和量化设置建立 baseline。
4. V2 approximate GEMM：用 Python 先模拟近似乘法误差。
5. AM-LUT：选择硬件乘法器候选。
6. V4：统一评估，形成 AI Systems 风格报告。

## 关键判断

只要项目最后能回答这几个问题，它就自然具备 AI Infra 含金量：

1. 这个 LLM workload 的计算瓶颈在哪里？
2. 量化和近似计算如何改变 GEMM 的计算和数据表示？
3. 近似乘法器带来了多少硬件收益？
4. 它对 perplexity / accuracy 造成多少影响？
5. 它对 latency / throughput / energy / memory footprint 有什么意义？
6. 这个优化能否被放到真实推理框架或加速器设计语境中解释？

如果能回答这些，你学到的就不是孤立的硬件模块，而是一套底层 AI Infra / AI Systems 能力。
