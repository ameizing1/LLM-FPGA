# AI Infra 实习要求与无法实习时的替代方案

更新时间：2026-06-24

## 结论

AI Infra / LLM inference 方向的正式校招确实很看：

- 顶会论文
- 大厂/创业公司实习
- 开源贡献
- 高质量系统项目

但实习生岗位本身不一定要求你已经有顶会论文。更常见的门槛是：

1. 能写代码，尤其是 Python / PyTorch，最好有 C++ / CUDA / Triton。
2. 理解 LLM inference 基础：prefill/decode、KV Cache、batching、attention、quantization。
3. 有 profiling 和 benchmark 意识：latency、throughput、memory、tokens/s。
4. 做过相关项目：量化、推理优化、kernel、serving、模型压缩、AI accelerator。
5. 能讲清实验设计和 trade-off。

## 实习岗位常见要求

### 底层 LLM inference / AI Systems 实习

常见关键词：

- LLM inference
- vLLM / SGLang / TensorRT-LLM
- KV Cache
- batching / scheduling
- quantization
- CUDA / Triton / C++
- profiling
- latency / throughput optimization
- memory optimization

项目证明：

- 跑过 vLLM / TensorRT-LLM benchmark。
- 做过量化或推理优化实验。
- 写过或改过 kernel / GEMM / attention / quantization 代码。
- 能解释优化前后性能变化。

### AI accelerator / hardware-aware ML systems 实习

常见关键词：

- AI accelerator
- GEMM / systolic array
- quantization
- approximate computing
- FPGA / ASIC / NPU
- hardware-software co-design
- Verilog / HLS / simulator
- resource / power / delay / energy

项目证明：

- 做过量化硬件友好性评估。
- 做过 GEMM / PE / multiplier / accelerator 模块。
- 能把模型精度和硬件指标放在一起评估。

### 平台型 AI Infra 实习

常见关键词：

- distributed inference
- resource scheduling
- Kubernetes
- request routing
- model serving
- observability / monitoring
- multi-tenant serving
- high-performance networking

项目证明：

- 搭过推理服务。
- 了解 OpenAI-compatible API、Triton Inference Server、vLLM serving。
- 做过调度、监控、队列、部署或服务稳定性工作。

这类和你当前课题关联较弱，需要额外补系统工程。

## 你的当前项目如何对齐实习要求

你的项目天然对齐第二类和部分第一类：

**面向 LLM 推理的硬件友好量化与近似 GEMM 加速**

能对齐的能力：

- LLM inference 基础
- quantization
- GEMM optimization
- approximate computing
- hardware-software co-design
- FPGA / accelerator
- accuracy vs hardware cost trade-off

需要补强的能力：

- 真实推理框架：vLLM / TensorRT-LLM
- profiling：Nsight / PyTorch profiler / nvidia-smi
- GPU kernel 或 Triton 最小经验
- 可复现代码仓库

## 申请实习前最低准备标准

如果目标是底层 AI Infra / LLM inference 实习，最低要有：

1. 一个完整项目 repo
   - README 清晰
   - 环境配置
   - 一键或少步骤复现实验
   - 结果表格

2. 一个 LLM inference baseline
   - FP16 / INT8 / INT4 对比
   - perplexity 或 accuracy
   - latency / throughput / memory footprint

3. 一个核心修改点
   - 量化方式修改
   - approximate GEMM
   - 近似乘法器替换
   - simulator 或 kernel 原型

4. 一份技术报告
   - 问题背景
   - 方法
   - 实验设计
   - 结果
   - trade-off

5. 面试可讲清楚
   - 为什么这么做
   - baseline 是否公平
   - 误差怎么传播
   - 性能瓶颈在哪里
   - 后续怎么优化

## 如果导师不放实习怎么办

### 先判断“不放”的原因

常见原因：

- 担心影响论文进度。
- 课题组项目紧。
- 实习方向和组内课题不匹配。
- 学生还没做出阶段成果。
- 老师不希望长期离组。

不同原因对应不同策略。

### 策略 1：先做出组内成果，再谈实习

最稳。

你可以先争取：

- 跑通 AxCore / Adana baseline。
- 完成近似乘法器替换方案。
- 写出阶段报告。
- 让老师看到你能推进课题。

然后再谈：

> 如果后续有和课题相关的短期实习，是否可以作为课题延伸？

### 策略 2：找“课题相关实习”

不要说：

> 我想去做 AI Infra。

更稳的说法：

> 如果有和 LLM 推理优化、量化、GEMM/算子优化、AI 加速器相关的实习，我希望它能反过来支持组内课题。

相关实习包括：

- LLM inference optimization
- model compression / quantization
- AI accelerator
- kernel / operator optimization
- ML systems performance

### 策略 3：短期远程/寒暑假实习

如果长期实习不现实，可以争取：

- 暑期 2-3 个月
- 寒假短期
- 远程开源/合作项目
- 每周固定时间同步

关键是让老师觉得可控。

### 策略 4：用开源贡献替代实习

如果完全不能实习，可以做：

- 给 AxCore 复现写 issue / PR
- 做 vLLM / SGLang / TensorRT-LLM 文档或 benchmark 贡献
- 开源自己的 approximate GEMM / quantization eval repo
- 复现论文 artifact 并整理结果

开源贡献虽然不等于实习，但对 AI Infra 岗位很有说服力。

### 策略 5：把课题做成“可面试项目”

即使没有实习，也要让项目像实习经历一样可讲：

- 有真实代码
- 有 benchmark
- 有 profiling
- 有 baseline
- 有 trade-off
- 有复现实验
- 有文档

面试官更关心你是否真的做过系统，而不是只看标题。

## 如何和导师谈实习

不要一开始就问：

> 老师能不能放我实习？

建议等有阶段成果后问：

> 老师，我后续想了解一下，如果课题进展比较顺利，是否可以考虑找和课题相关的实习，比如 LLM 推理优化、量化、算子/GEMM 优化或 AI 加速器方向。我的想法是实习内容最好能反过来支持组里的课题，而不是和课题脱节。想请教老师，您对这方面有什么建议或要求？

如果老师谨慎，可以继续问：

> 那我是否可以先以完成组内阶段目标为前提，比如先跑通平台、完成 baseline 和初步替换方案，再根据进展讨论实习是否合适？

## 没有实习时的替代路线

如果最终无法实习，你至少要做出这 4 个东西：

1. **项目 repo**
   - 代码、README、环境、脚本、结果。

2. **技术报告**
   - 类似 workshop paper / artifact report。

3. **开源或复现记录**
   - issue、PR、benchmark、复现文档。

4. **可讲的系统故事**
   - 从 LLM workload 到量化，再到 GEMM/硬件，再到评估。

目标是让简历上出现：

> 复现并扩展 LLM 低比特量化/近似 GEMM 评估框架，完成硬件友好 approximate multiplier 替换实验，评估 perplexity、latency、memory footprint、resource 和 energy trade-off。

## 顶会论文与实习的关系

顶会论文和实习都很有价值，但不是完全二选一。

- 有顶会：更容易进研究/核心系统优化岗位。
- 有强实习：更容易证明工程落地能力。
- 两者都没有：需要强项目 + 开源 + 可复现实验来补。
- 有一篇普通论文 + 强项目：也有机会。

硕士阶段最现实的目标：

1. 争取一段相关实习。
2. 同时把组内课题做成可投稿/可开源/可复现的项目。
3. 不要只追论文，也不要只追实习，二者要互相服务。

## 给你的建议

你现在不要马上焦虑“导师放不放实习”。更实际的是：

1. 先用 1-2 个月把老师给的任务做出阶段成果。
2. 把项目设计成 AI Infra 可识别的形式。
3. 同步时观察导师对实习的态度。
4. 研一上或研一下再正式谈相关实习。
5. 如果不放实习，就用开源贡献和高质量项目替代一部分信号。

你的项目如果做成完整链路，完全可以成为申请实习的核心材料。
