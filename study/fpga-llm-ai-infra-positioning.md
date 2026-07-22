# FPGA/LLM 项目与 AI Infra 路线的关系判断

更新时间：2026-06-24

## 核心判断

导师目前给的方向有前景，但不能把它理解成：

**“用 FPGA 替代 GPU，做通用大模型服务。”**

更合理的定位是：

**“面向 LLM 推理的硬件友好量化、近似 GEMM 和加速器原型研究。”**

它和 AI Infra 的关系不是云平台/SRE/集群调度那一层，而是底层 AI Systems：

- LLM inference optimization
- quantization
- GEMM / kernel / operator optimization
- accelerator-aware evaluation
- hardware-software co-design
- latency / throughput / energy / memory footprint trade-off

## 为什么小红书说法有一部分道理

如果讨论现实生产环境中的大规模 LLM serving，GPU/专用 AI 加速卡仍然是主流。

原因：

- GPU 软件生态成熟：CUDA、cuBLAS、Triton、TensorRT-LLM、vLLM 等。
- 大模型更新快，FPGA 每次适配新模型/新算子成本高。
- FPGA 通用算力密度和生态通常不如高端 GPU。
- 大规模部署更看端到端软件栈、调度、并行、运维和模型兼容性。

所以如果课题目标被表述成“FPGA 部署通用大模型并全面替代 GPU”，风险较高。

## 为什么这个方向仍然有价值

FPGA/硬件原型在研究中仍然有价值，尤其适合探索：

- 新数值格式
- 低比特量化
- 近似计算
- LUT-based multiplier
- GEMM 单元设计
- 硬件友好模型压缩
- 特定 workload 的能效优化

FlightLLM、CXL-SpecKV、Adana、AxCore 这类工作说明：学术界和体系结构/EDA/加速器社区仍然关注 LLM inference 的硬件优化。但它们通常不是在做“通用服务平台”，而是在研究某个明确瓶颈：

- compute density
- memory bandwidth
- KV Cache memory
- energy efficiency
- low-bit arithmetic
- model accuracy vs hardware cost

## 对你的项目最稳的定位

不要把课题标题写成：

**FPGA 上部署大模型**

更稳的标题是：

**面向 LLM 推理的硬件友好量化与近似 GEMM 加速**

或：

**Hardware-aware Quantization and Approximate GEMM Acceleration for LLM Inference**

这样它的核心不再是“FPGA 能不能跑大模型”，而是：

1. LLM 推理中 GEMM 是重要计算热点。
2. 低比特量化能降低存储和计算成本。
3. 近似乘法器可能进一步降低硬件资源和能耗。
4. 必须评估它对 perplexity / accuracy 的影响。
5. 必须评估 resource / delay / power / energy / throughput 的 trade-off。

## 与 AI Infra 的匹配程度

### 匹配的岗位方向

高度匹配：

- AI accelerator engineer
- inference optimization engineer
- hardware-aware ML systems
- model compression / quantization engineer
- AI compiler / kernel / operator optimization
- LLM inference systems, 偏底层优化

中等匹配：

- GPU inference performance engineer
- AI infra performance engineer
- serving framework optimization

需要额外补：

- vLLM / TensorRT-LLM
- CUDA / Triton
- profiling
- distributed serving
- batching / scheduling
- online serving metrics

匹配较弱：

- 云平台/SRE 型 AI Infra
- Kubernetes 资源调度
- 多租户服务治理
- 业务后端型 AI 平台

## 如何避免方向做窄

项目必须保留一条完整链路：

**LLM workload -> quantization -> GEMM bottleneck -> approximate multiplier -> hardware cost -> model accuracy -> inference efficiency**

如果只做到：

**LUT 近似乘法器资源更少**

那它更像传统 FPGA/数字电路项目。

如果能做到：

**近似乘法器替换后，LLM perplexity 损失可控，同时硬件资源/能耗/延迟改善**

那它就很接近底层 AI Infra / AI Systems。

## 建议加入的 GPU/真实推理连接

为了避免项目脱离主流 AI Infra，可以加入一个轻量桥接：

1. 用 Hugging Face 或 vLLM 跑相同模型的 FP16 / INT8 / INT4 baseline。
2. 记录 perplexity、tokens/s、memory footprint、latency。
3. 用 AxCore/Adana/近似乘法器结果与 baseline 对齐。
4. 把硬件收益转换成 AI Infra 语言：energy/token、memory footprint、throughput、accuracy loss。

不需要一开始深入做分布式 serving，但要能解释这个硬件优化在真实推理栈中的位置。

## 一句话结论

这个方向不是“FPGA 跑通用大模型服务”的路线；它更像是“以 FPGA/硬件原型研究 LLM 低比特计算和 GEMM 加速”的路线。  

如果你把它做成完整的 LLM 推理优化链路，它能支持你走底层 AI Infra / AI Systems；如果只停留在 FPGA 乘法器本身，它对 AI Infra 的支撑会明显变弱。
