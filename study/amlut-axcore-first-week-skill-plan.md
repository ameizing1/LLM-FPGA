# AM-LUT + AxCore 第一周技能与任务清单

更新时间：2026-07-05

## 总原则

第一周不建议先系统学习完整 Python、CUDA、FPGA 综合或 LLM 量化课程。更合适的方式是项目驱动：围绕“实现 TCASI24 近似 INT8 乘法器行为模型，并做 GEMM 误差对比”补最小必要知识。

目标不是学完一门课，而是交付一个能运行、能解释、能扩展的小实验。

## 第一周必须掌握

### Python 基础实战

必须会：

- 写函数、循环、条件判断。
- 使用 list、dict、tuple。
- 读写 `.npy`、`.csv`、`.md`。
- 用 `argparse` 或简单配置切换实验参数。
- 把实验结果保存成表格。

暂时不需要：

- 系统学习面向对象。
- 学 Web、爬虫、数据库。
- 学复杂工程框架。

### NumPy

必须会：

- 创建数组：`np.array`、`np.zeros`、`np.arange`。
- 使用整数类型：`np.int8`、`np.int16`、`np.int32`。
- 做矩阵乘法：`@` 或 `np.matmul`。
- 做广播和索引。
- 保存/加载 lookup table：`np.save`、`np.load`。

重点风险：

- `int8` 乘法容易溢出，乘积和累加要转成 `int16` 或 `int32`。
- lookup table 的索引需要把 `[-128, 127]` 映射到 `[0, 255]`。

### 二进制与补码

必须会：

- 理解 signed int8 的范围：`-128` 到 `127`。
- 理解补码表示和符号扩展。
- 能把 `int8` 拆成高 4 bit 和低 4 bit。
- 理解 8x8 乘法如何由 4 个 4x4 子乘法组合。

暂时不需要：

- 完整学习计算机算术教材。
- 深入浮点标准。
- 手推复杂 Wallace tree / compressor tree。

### 近似乘法器误差指标

必须会：

- ED：error distance。
- ER：error rate。
- MED：mean error distance。
- NED：normalized error distance。
- MRED：mean relative error distance。
- signed error / bias。

重点：

不要只看平均误差。LLM GEMM 更关心误差是否有系统性偏置，以及在真实 operand distribution 下是否放大。

### GEMM 误差分析

必须会：

- exact GEMM 和 approximate GEMM 对比。
- 计算 MSE、RMSE、relative error、cosine similarity、max abs error。
- 区分 product-level error 和 GEMM-level error。
- 理解累加会放大或抵消单次乘法误差。

## 第一周边做边学

### TCASI24 乘法器结构

先只读懂两类：

- LSAM1：保守，误差小。
- CSAM2：激进，硬件省。

需要理解：

- LUT sharing 为什么能省 LUT。
- carry switching 为什么比直接截断 carry 更温和。
- 4x4 近似乘法器如何组合成 8x8。

暂时不需要：

- 一次性实现 140 种 8x8 组合。
- 完整复现论文所有 Vivado 表格。

### 实验组织

第一周实验文件建议分成：

- `approx_mul.py`：近似乘法器行为模型。
- `generate_lut.py`：生成 int8 lookup table。
- `test_gemm_error.py`：GEMM 误差测试。
- `summarize_results.py`：汇总结果。
- `results/`：保存 `.npy`、`.csv`、图表和报告。

## 第一周暂时不需要学

- 完整 CUDA 编程。
- 完整 PyTorch 模型部署。
- 完整 AxCore CUDA kernel 修改。
- 完整 FPGA/Verilog/Vivado 流程。
- 完整机器学习/深度学习课程。
- 完整 LLM inference serving。

这些都不是第一周 blocker。现在最大的 blocker 是把近似乘法器行为模型写对，并且把误差算清楚。

## 建议的 5 天安排

### Day 1：Python + NumPy 最小补齐

交付物：

- 一个 `exact_int8_mul_lut.npy`。
- 一个脚本能遍历所有 int8 输入组合并输出 exact product。

验收标准：

- 能解释为什么 lookup table 是 256x256。
- 能解释为什么乘积不能用 `int8` 保存。

### Day 2：实现 LSAM1 行为模型

交付物：

- `lsam1_int8_lut.npy`。
- product-level error 表格。

验收标准：

- 能说明 LSAM1 的近似点来自 LUT sharing。
- 能输出 ER/MED/NED/MRED。

### Day 3：实现 CSAM2 行为模型

交付物：

- `csam2_int8_lut.npy`。
- LSAM1 vs CSAM2 的 product-level error 对比。

验收标准：

- 能说明 CSAM2 的近似点来自 carry switching 和更激进的 LUT 删除。
- 能说明为什么 CSAM2 硬件更省但误差更大。

### Day 4：实现 approximate GEMM

交付物：

- `exact GEMM`、`LSAM1 GEMM`、`CSAM2 GEMM` 对比脚本。
- 随机分布下的 GEMM-level error 表格。

验收标准：

- 能解释 product-level error 和 GEMM-level error 为什么不同。
- 能说明误差在累加中可能抵消，也可能形成 bias。

### Day 5：做 distribution-sensitive error

交付物：

- uniform、normal-like、zero-heavy 三种输入分布下的误差对比。
- 一页中文实验小结。

验收标准：

- 能判断 LSAM1/CSAM2 是否值得进入 AxCore CUDA 或 LLM layer 级实验。

## 掌握度检查

第一周结束时，你应该能回答：

1. 为什么 INT8 乘法器需要用 `int16/int32` 保存乘积和累加？
2. lookup table 为什么是 256x256？
3. LSAM1 和 CSAM2 分别牺牲了什么精度，换来了什么硬件收益？
4. product-level error 为什么不能直接代表 GEMM-level error？
5. 如果某个近似乘法器 product-level MRED 很小，但 GEMM 输出 bias 很大，说明什么？
6. 哪种输入分布更接近 LLM quantized weight/activation？
7. 是否应该进入 AxCore CUDA 路线，判断依据是什么？

## 下一阶段入口

只有当第一周证明 LSAM1/CSAM2 在 GEMM-level error 上不是明显不可接受时，才进入下一阶段：

- AxCore CUDA Linear wrapper 替换；
- 小模型/单层 perplexity 或 layer output error；
- simulator 参数层硬件能耗对比；
- 最后再考虑 Verilog/SpinalHDL/FPGA 综合。
