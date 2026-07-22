# 正式推进项目方案：面向 LLM 推理的硬件友好量化与近似 GEMM 加速

更新时间：2026-06-26

## 0. 项目定位

项目主线：

**面向 LLM 推理的低比特量化、近似 GEMM 与 FPGA/LUT-based 乘法器软硬件协同优化。**

核心问题：

> 能否用 FPGA 友好的近似 INT8/低比特乘法器替换 LLM GEMM 中的部分乘法计算，在可接受的 perplexity / accuracy 损失下，降低硬件资源、延迟、功耗或能耗？

## 1. 第一阶段：明确研究问题和边界

目标：

- 不直接开始写代码，先把问题边界固定。

需要确定：

1. 目标模型：OPT / LLaMA / TinyLlama / Qwen 中选 1-2 个。
2. 目标任务：优先 perplexity，后续可扩展 zero-shot。
3. 目标数据集：优先 WikiText-2。
4. 目标计算：先聚焦 GEMM / Linear layer，不一开始碰完整 serving。
5. 目标量化：先选 W4A16 / W4A4 / INT8 中一种主配置。
6. 目标替换：先替换 simulator 或 software GEMM 中的 multiply，再考虑 Verilog/FPGA。

交付物：

- `problem_definition.md`
- 一张表：模型、数据集、量化配置、baseline、评价指标。

## 2. 第二阶段：建立 baseline

目标：

- 建立后续所有优化的对照组。

Baseline 至少包括：

1. FP16 baseline
2. 常规 INT8/INT4 quantization baseline
3. AxCore 原始实现或 simulator baseline

指标：

- perplexity
- GEMM output error / MSE
- latency 或模拟 cycle
- memory footprint
- 如果有硬件综合：LUT / FF / DSP / delay / power

交付物：

- `baseline_results.csv`
- `baseline_report.md`

验收标准：

- 任何后续修改都能和 baseline 做同表比较。

## 3. 第三阶段：复现 AxCore / Adana 关键路径

目标：

- 不要求完整复现论文所有实验，但要跑通与你项目相关的最小链路。

Adana 侧：

- 复述它的 quantization 思路。
- 理解 group-wise quantization、nonuniform/asymmetric numeric type、online activation quantization。
- 不一定先实现完整 Adana，先提炼可借鉴的量化评估方法。

AxCore 侧：

- 跑通软件评估或 simulator。
- 定位 approximate GEMM / PE / multiplier 相关模块。
- 明确替换接口。

交付物：

- `adana_notes.md`
- `axcore_code_map.md`
- `replaceable_module_analysis.md`

## 4. 第四阶段：设计近似乘法器替换方案

目标：

- 从 AM-LUT 文献中选出 1-2 个可实现的近似乘法器候选。

需要整理：

1. 输入位宽：INT8? INT4? signed/unsigned?
2. 输出位宽
3. 是否支持有符号乘法
4. 误差指标：MED、MRED、NMED、error rate
5. FPGA 资源：LUT、FF、DSP
6. 延迟和时序
7. 是否已有 Verilog
8. 如何接入 AxCore 的 GEMM/PE

交付物：

- `multiplier_candidate_table.csv`
- `multiplier_replacement_design.md`

验收标准：

- 能清楚说明为什么选这个乘法器，而不是随便替换。

## 5. 第五阶段：先做软件级替换验证

目标：

- 在硬件实现前，用软件模拟验证误差是否可控。

步骤：

1. 实现 exact multiply。
2. 实现 approximate multiply。
3. 接入 toy GEMM。
4. 接入 AxCore simulator 或简化 GEMM evaluation。
5. 比较 output error。
6. 如果可行，进一步跑 perplexity。

交付物：

- `approx_mul.py`
- `approx_gemm.py`
- `error_analysis.csv`
- `software_replacement_report.md`

验收标准：

- 能回答：乘法器误差是否会显著放大到 GEMM 输出或模型 perplexity。

## 6. 第六阶段：硬件级实现与综合

目标：

- 将候选近似乘法器转为 Verilog 可综合模块，并做资源/延迟评估。

步骤：

1. Verilog 实现或复用开源实现。
2. 写 testbench。
3. 与 exact multiplier 对比 functional output。
4. 进行综合。
5. 记录 LUT / FF / DSP / delay / power。

交付物：

- Verilog multiplier module
- testbench
- synthesis report
- `hardware_eval_report.md`

验收标准：

- 有准确的硬件收益数据，而不是只说理论上更省。

## 7. 第七阶段：统一评估和论文式整理

目标：

- 把软件效果和硬件收益放到同一个 trade-off 框架里。

最终表格应包含：

| 方法 | Perplexity | GEMM Error | LUT | DSP | Delay | Power | Energy | Speedup/Compute Density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | | | | | | | | |
| INT8/INT4 baseline | | | | | | | | |
| AxCore baseline | | | | | | | | |
| Proposed approx multiplier | | | | | | | | |

交付物：

- `evaluation_dashboard.csv`
- `figures/`
- `final_report.md`

## 8. 汇报节奏

建议每 2 周向导师同步一次。

每次汇报固定包含：

1. 本周期完成了什么。
2. 当前结果表格。
3. 遇到的阻塞。
4. 下周期计划。
5. 需要老师确认的问题。

不要只汇报“读了什么”，要尽量汇报：

- 跑通了什么；
- 对比了什么；
- 发现了什么；
- 下一步要验证什么。

## 9. 最小可行路径

如果要尽快做出第一个结果，优先走：

1. 读 Adana，整理量化思路。
2. 跑 AxCore 最小软件评估。
3. 实现一个 Python approximate INT8 multiply。
4. 接入 toy GEMM。
5. 比较 GEMM output error。
6. 选一个 AM-LUT 乘法器候选。
7. 写初步替换方案。

这个路径不依赖完整 FPGA 工具链，适合第一个月快速推进。

## 10. 关键风险

1. **只做硬件模块，缺少 LLM 评估。**
   - 规避：必须保留 perplexity / GEMM error baseline。

2. **只跑模型，缺少硬件收益。**
   - 规避：必须做 LUT / DSP / delay / power 对比。

3. **替换点不清楚。**
   - 规避：先做 AxCore code map。

4. **问题太大。**
   - 规避：第一阶段只做 GEMM/Linear，不碰完整 serving。

5. **实验不可复现。**
   - 规避：所有脚本、配置、结果表格固定保存。

## 11. 项目完成时应能回答的问题

1. LLM 推理中为什么 GEMM 是关键计算热点？
2. 低比特量化如何改变 GEMM 的数值表示和硬件实现？
3. 近似乘法器的误差如何传播到 GEMM 输出和模型 perplexity？
4. 替换近似乘法器后，硬件资源、延迟、功耗变化多少？
5. accuracy loss 和 hardware gain 是否值得？
6. 这个方法适合哪些模型、位宽和硬件场景？
