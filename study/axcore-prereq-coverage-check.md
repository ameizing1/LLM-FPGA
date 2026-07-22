# AxCore 前置知识覆盖检查

更新时间：2026-06-30

检查对象：当前 `CS224N 与 CS336 学习计划`

## 结论

当前计划覆盖了 AxCore 所需的 LLM / Transformer / 推理系统基础，但没有完全覆盖 AxCore 论文最关键的硬件与数值表示基础。

如果按当前计划原样执行，即使完成第 8 周，也仍然会缺以下几块：

- FP16 / BF16 / FP32 / FP4 的二进制表示细节
- FP4 的 E3M0 / E2M1 / E1M2 格式
- normalized / subnormal number
- FPMA：floating-point multiplication approximation
- mixed-precision GEMM / direct mpGEMM
- systolic array / PE / weight-stationary dataflow
- RTL synthesis 参数如何进入 simulator
- CACTI 如何影响 SRAM / Buffer energy

因此，当前计划不是“不好”，而是偏 LLM Systems 主线；读 AxCore 还需要一个短的 AxCore 专项前置补丁。

## 覆盖情况表

| AxCore 所需前置 | 当前计划覆盖情况 | 说明 |
|---|---|---|
| LLM / Transformer 基本结构 | 覆盖充分 | CS224N P7/P8/P9 + CS336 architecture 足够 |
| Language model / perplexity | 覆盖充分 | CS224N LM/evaluation + CS336 evaluation 足够 |
| GEMM 在 LLM 中的位置 | 基本覆盖 | CS336 resource accounting / GEMM 笔记可覆盖 |
| Prefill / decode / KV cache | 基本覆盖 | CS336 inference 覆盖，足够理解系统背景 |
| Weight-only quantization | 部分覆盖 | 计划提到量化，但还需要专门补 scale、group size、direct mpGEMM |
| Low-bit floating point | 缺失 | AxCore 需要 FP4/E3M0/E2M1/E1M2、subnormal |
| FPMA | 缺失 | 这是 AxCore 的核心前置，CS224N/CS336 通常不会讲 |
| Approximate computing error | 部分覆盖 | evaluation 能帮你理解指标，但误差补偿机制需要另补 |
| Systolic array / PE / dataflow | 部分覆盖 | CS336 GPU/kernel 不等于专门的 systolic array |
| Hardware energy metrics | 部分覆盖 | CS336 resource accounting 有帮助，但 area/power/energy/synthesis/CACTI 需要另补 |
| Simulator 结果路径 | 缺失 | 需要结合 AxCore repo 读 `run_axcore.py`、`stats.py`、`simulator.py` |

## 学到哪里算覆盖 AxCore 前置

分三种目标：

### 目标 1：能读懂 AxCore 论文主线

需要完成：

- CS224N 第 1-3 周的必看内容
- CS336 第 4-5 周：tokenizer、Transformer、resource accounting
- 额外补丁 A：低比特浮点与 FPMA
- 额外补丁 B：systolic array 与硬件能耗指标

也就是说：不是等第 8 周，而是第 5 周后加 3-5 天 AxCore 专项补丁，就可以开始读论文主线。

### 目标 2：能解释 Figure 17 和 simulator 复现实验

需要完成：

- 上面目标 1 的全部内容
- CS336 第 6 周：GPU / kernel / memory hierarchy / tiling
- AxCore simulator 代码阅读：`run_axcore.py`、`EnergyAll.py`、`stats.py`、`simulator.py`

也就是说：学到第 6 周，再补 AxCore 专项前置和代码阅读，就基本覆盖 Figure 17 的前置。

### 目标 3：能设计 AM-LUT / AxCore 后续实验

需要完成：

- 上面目标 2 的全部内容
- CS336 第 7 周：inference、batching、KV cache、prefill/decode
- CS336 第 8 周：evaluation + 项目评价指标
- 额外补 AM-LUT/FPGA LUT multiplier 的误差、面积、延迟、功耗抽象

也就是说：完整第 8 周更适合做“项目设计”，但不是读 AxCore 论文主线的必要门槛。

## 建议加入的 AxCore 专项补丁

建议插在 CS336 第 5 周之后、第 6 周之前，或者和第 6 周并行。

### 补丁 1：低比特浮点数

要掌握：

- sign / exponent / mantissa
- exponent bias
- FP16 / BF16 / FP32 区别
- normalized vs subnormal
- FP4：E3M0、E2M1、E1M2

验收：

- 能解释为什么 FP4 的 subnormal 不能被忽略。
- 能说明 E3M0 / E2M1 / E1M2 的表达能力差异。

### 补丁 2：Weight-only quantization 与 direct mpGEMM

要掌握：

- W4A16 / W8A16
- scale / group size / dequantization
- indirect GEMM vs direct mpGEMM
- 为什么 direct mpGEMM 对硬件更友好

验收：

- 能解释 AxCore 为什么不先 dequantize 再做普通 FP GEMM。

### 补丁 3：FPMA

要掌握：

- `log2(1 + M) ~= M`
- 为什么 FP multiplication 可以近似成加法
- FPMA 为什么会有系统误差
- mixed precision 下为什么还要 alignment 和 bias correction

验收：

- 能用自己的话解释 AxCore 的 `R = A + Align(Wq) - B1 + C1` 在做什么。

### 补丁 4：Systolic array 与 PE

要掌握：

- PE
- systolic array
- weight-stationary dataflow
- accumulator
- buffer / tiling

验收：

- 能看懂 AxCore Figure 8 / 9 的数据流。

### 补丁 5：硬件能耗建模

要掌握：

- area
- leakage power
- dynamic power
- energy
- TOPS/W
- TOPS/mm2
- SRAM / DRAM / Buffer / Core energy
- synthesis CSV 与 CACTI 的作用

验收：

- 能解释 `axcore_res.csv` 里 `Static / Dram / Buffer / Core` 分别来自哪里。

