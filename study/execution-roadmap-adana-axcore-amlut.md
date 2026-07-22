# 执行路线：Adana -> AxCore -> AM-LUT -> 近似乘法器替换

更新时间：2026-06-26

## 总目标

把导师给的论文转化成一个可执行的小课题闭环：

**理解 LLM 低比特量化和 GEMM 计算瓶颈，跑通 AxCore 平台，选择 FPGA LUT-based 近似乘法器候选，并形成一个可汇报的替换方案。**

不要一开始追求完整创新。第一阶段的目标是：

1. 读懂主线论文。
2. 跑通可复现实验。
3. 找到可替换模块。
4. 形成可讨论方案。

## Step 0：建立工作目录和记录模板

时间：0.5 天

### 动作

建立三个文件：

1. `paper-notes.md`
2. `experiment-log.md`
3. `questions-for-advisor.md`

### 每篇论文笔记模板

```text
论文：
一句话问题：
核心观察：
方法：
硬件/系统设计：
评价指标：
关键实验结果：
我没看懂的问题：
和当前任务的关系：
```

### 实验记录模板

```text
日期：
目标：
环境：
命令：
结果：
报错：
解决过程：
下一步：
```

### 验收

- 后续所有阅读和实验都有记录。
- 不把问题散落在微信、PDF 批注和脑子里。

## Step 1：补 LLM 推理和量化最小背景

时间：2-3 天

### 学习内容

- Transformer decoder
- Attention / MLP / FFN
- prefill / decode
- GEMM 在 LLM 中的位置
- weight quantization vs activation quantization
- W4A16、W4A4、W3A3
- group-wise quantization
- perplexity / zero-shot accuracy

### 产出

一页笔记：

**《LLM 推理与低比特量化最小背景》**

必须回答：

1. LLM 推理中哪些部分主要是 GEMM？
2. 为什么 weight-only quantization 和 weight-activation quantization 不一样？
3. 为什么 activation quantization 需要在线处理？
4. perplexity 在量化论文里用来衡量什么？

### 验收

你能不用论文原句，口头解释：

- LLM 为什么适合低比特量化。
- 为什么低比特量化不只是省显存，也会影响硬件计算单元。

## Step 2：精读 Adana，学“量化论文怎么做”

时间：3-4 天

### 阅读重点

按顺序读：

1. Abstract
2. Introduction
3. Motivation
4. Adana Numeric Type
5. Quantization Framework
6. Microarchitecture
7. Evaluation

### 必须抓住的点

- small group 下数据分布为什么不再对称。
- nonuniform 和 asymmetric 分别解决什么问题。
- Adana numeric type 如何表示更密集的高分辨率区域。
- online activation quantization 的硬件代价是什么。
- QUANT engine 和 MM engine 分别做什么。
- 它如何同时评估 accuracy 和 hardware performance。

### 产出

一页图文笔记：

**《Adana：问题-观察-方法-硬件-评估》**

图至少包含：

```text
FP16 activation / weight
        ↓
group-wise statistics
        ↓
Adana numeric type selection
        ↓
low-bit quantization
        ↓
MM engine
        ↓
dequantization / accumulation
```

### 验收

能回答：

1. Adana 为什么不是普通 INT asymmetric quantization？
2. 它为什么强调 small group？
3. 它的硬件设计中最贵/最关键的部分可能是什么？
4. 它和 AxCore 的关系是什么？

## Step 3：精读 AxCore，定位代码和模块

时间：4-6 天

### 阅读重点

按顺序读：

1. Abstract
2. Introduction
3. Background
4. Method overview
5. PE / systolic array / simulator 部分
6. Artifact appendix / GitHub README

### 必须抓住的点

- AxCore 解决的是 FP-GEMM 硬件代价问题。
- 它采用 weight-only quantization。
- 它用 FPMA 思想把乘法替换为更便宜的操作。
- 它用 perplexity / zero-shot / compute density 做联合评估。
- 仓库分为 Hardware、Software、simulator、Profile 等部分。

### 产出

两张图：

1. **AxCore 论文方法图**

```text
activation
        +
low-bit quantized weight
        ↓
FPMA / approximate GEMM
        ↓
systolic array
        ↓
LLM accuracy + hardware efficiency
```

2. **AxCore 代码结构图**

```text
Hardware/AxCore
Software/AxCore
Software/axcore_simulator
Profile
```

### 验收

能回答：

1. AxCore 的乘法器/PE/GEMM 相关逻辑在哪里？
2. 软件侧 accuracy evaluation 和硬件侧 simulator 分别评估什么？
3. 如果要替换乘法器，最可能动哪一层？

## Step 4：跑通 AxCore 最小实验

时间：3-7 天，视环境而定

### 优先级

先跑软件侧，再跑 simulator，最后看硬件。

推荐顺序：

1. 克隆仓库。
2. 按 README 配环境。
3. 跑最小模型/最小数据集。
4. 跑 perplexity 或 simulator 中最轻量的一项。
5. 记录所有命令和报错。

### 产出

`experiment-log.md` 中至少包括：

- 环境版本
- 安装步骤
- 成功/失败命令
- 输出结果截图或文本
- 当前卡点

### 验收

最低标准：

- 明确 AxCore 平台结构。
- 至少跑通一个最小流程，或者形成清晰的环境卡点报告。

优秀标准：

- 跑出一个可复现的 perplexity/simulator 结果。
- 能改一个小参数并重新跑。

## Step 5：精读 AM-LUT / TCASI24，选择乘法器候选

时间：4-5 天

### 阅读重点

先读 TCASI24：

- LUT sharing
- carry switching
- 4x4 multiplier
- 8x8 multiplier construction
- approximate adders
- error metrics
- resource / delay / power
- open-source Verilog

再快速对比：

- ARITH24：incomplete sub-multipliers
- TC23：globally optimal multiplier design
- TCAD22：accurate/approximate softcore multiplier baseline
- ARITH15：Xilinx softcore multiplier 基础

### 产出

一张候选表：

```text
候选设计 | 位宽 | 是否近似 | LUT/FF/DSP | delay | power | 误差指标 | 有无代码 | 是否适合 INT8 GEMM | 备注
```

### 验收

选出 1-2 个候选乘法器，并说明：

- 为什么选它。
- 它的输入输出位宽是什么。
- 它可能带来什么硬件收益。
- 它可能给 LLM accuracy 带来什么风险。

## Step 6：做“替换路径”分析

时间：2-4 天

### 目标

先不急着写 Verilog 或改大工程，先把接口和评估路径想清楚。

### 要分析的问题

1. AxCore 当前计算单元是什么输入格式？
2. 乘法/近似乘法发生在哪一层？
3. 候选 INT8 近似乘法器的输入输出位宽是否匹配？
4. 如果不匹配，需要加什么 wrapper？
5. 替换后先做 functional correctness，还是先做 accuracy simulation？
6. 评价指标用哪些？

### 推荐两条路线

路线 A：软件/simulator 先行

```text
用 Python/C++/CUDA 模拟 approximate multiplier
        ↓
替换 GEMM 中乘法规则
        ↓
跑小模型/小数据集
        ↓
看 perplexity / zero-shot 变化
```

路线 B：硬件模块先行

```text
拿 TCASI24 Verilog
        ↓
改成目标 INT8 接口
        ↓
functional simulation
        ↓
综合看 LUT / delay / power
        ↓
再接 AxCore simulator
```

### 产出

一页接口分析：

```text
AxCore 原始计算单元：
候选乘法器：
输入格式：
输出格式：
需要新增 wrapper：
替换位置：
第一步验证：
最终评价指标：
```

### 验收

能向老师清楚说出：

- 我打算先在哪里替换。
- 为什么先走软件模拟或硬件模拟。
- 需要老师确认哪些技术选择。

## Step 7：做最小 baseline

时间：1-2 周

### 目标

做一个非常小但能闭环的 baseline。

### 最小 baseline 选项

选项 1：Python 乘法误差模拟

- 实现 exact INT8 multiplier。
- 实现一个 approximate INT8 multiplier。
- 随机输入测试 error distribution。
- 输出 MED/MRED/NMED/error rate。

选项 2：小 GEMM 模拟

- 生成小矩阵 A/B。
- exact GEMM vs approximate GEMM。
- 对比输出误差。
- 看误差随矩阵规模和数据分布变化。

选项 3：接入 AxCore simulator 的小改动

- 找到乘法/PE 模拟位置。
- 用近似乘法规则替换。
- 跑一个最小配置。

### 产出

至少一张图或表：

- error distribution
- GEMM output error
- resource/latency rough comparison
- perplexity 初步结果，如果能跑通

### 验收

哪怕结果很粗，也要能回答：

> 这个近似乘法器在 LLM/GEMM 场景下误差是否可能可控？

## Step 8：给老师做第一次阶段汇报

时间：完成 Step 1-7 后

### 汇报内容

建议 5-8 页，不要太长：

1. 任务理解
   - 当前主线：LLM 低比特量化 + approximate GEMM + FPGA LUT-based multiplier。

2. 文献关系图
   - Adana：量化思路。
   - AxCore：近似 GEMM 平台。
   - AM-LUT：候选乘法器。
   - FlightLLM：FPGA LLM 系统视角。

3. 已完成内容
   - 读了哪些论文。
   - 跑通了什么。
   - 发现了什么卡点。

4. 初步替换方案
   - 替换位置。
   - 候选乘法器。
   - 验证路线。

5. 需要老师确认的问题
   - 先做软件模拟还是硬件实现？
   - 指标优先级是什么？
   - 是否有指定平台或代码？
   - 是否有师兄/师姐可以对接？

### 验收

老师能基于你的汇报做出明确安排：

- 继续读哪篇。
- 跑哪个平台。
- 改哪个模块。
- 和谁对接。
- 下次汇报什么时候。

## 推荐时间表

如果每天 3-4 小时：

| 时间 | 任务 |
|---|---|
| 第 1-3 天 | Step 1：LLM 推理和量化最小背景 |
| 第 4-7 天 | Step 2：精读 Adana |
| 第 8-14 天 | Step 3-4：精读并跑 AxCore |
| 第 15-20 天 | Step 5：精读 AM-LUT / TCASI24 |
| 第 21-24 天 | Step 6：替换路径分析 |
| 第 25-35 天 | Step 7：最小 baseline |
| 第 36-40 天 | Step 8：阶段汇报 |

## 当前最该先做的三件事

1. **今天开始读 Adana**
   - 先写出“问题-观察-方法-硬件-评估”一页笔记。

2. **建立 AxCore 环境预研清单**
   - 先不急着安装，把 README、依赖、模型、数据集、硬件需求列出来。

3. **下载/定位 TCASI24 的开源 Verilog**
   - 先确认代码能不能拿到、模块是什么接口、有没有 8x8 乘法器。

## 不要踩的坑

- 不要一开始就完整复现 Adana。
- 不要一上来就改 AxCore 大工程。
- 不要只读乘法器，不懂 LLM 量化指标。
- 不要只看 perplexity，不看硬件资源。
- 不要只看 LUT savings，不看 LLM accuracy。
- 不要等完全懂了再汇报，阶段性卡点也值得同步。
