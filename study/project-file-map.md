# LLM-FPGA 项目文件导览

更新时间：2026-07-19

目的：用一份紧凑的文件地图说明当前项目各部分的功能，方便后续继续做 AxCore / AM-LUT 分析时快速找到入口。

## 1. 项目定位

当前工作区不是一个单一、干净的软件包，而是一个研究型工作目录，主要包含四层内容：

1. 参考论文和论文文本摘录。
2. 学习笔记、项目计划和阶段总结。
3. 一个小型 AM-LUT Python 行为模型原型。
4. 迁移过来的 AxCore 仓库、simulator 复现材料和实验脚本。

当前技术主线是：

```text
AM-LUT 行为模型
-> int8 LUT / GEMM 误差报告
-> AxCore Figure 17 simulator 分析
-> AM-LUT 参数层 sensitivity 实验
```

## 2. 顶层目录

| 路径 | 作用 | 什么时候优先看 |
|---|---|---|
| `am_lut_tcasi24/` | TCASI24 AM-LUT 近似乘法器的 Python 行为模型 | 想理解 LSAM1/CSAM2 的乘法行为或 GEMM 行为 |
| `scripts/` | LUT 生成和误差报告脚本 | 想生成 `.npy` LUT 或误差报告 |
| `tests/` | AM-LUT 模型的最小 smoke test | 想确认 Python 原型没有被改坏 |
| `study/` | 主要学习笔记、项目计划、AxCore 分析总结 | 想补背景、看之前结论 |
| `docs/` | 迁移总结和短期项目计划 | 想知道文件来源和迁移背景 |
| `outputs/` | 已生成产物：LUT、报告、论文页面渲染图、Figure 17 图片 | 想看结果，而不是看源码逻辑 |
| `references/` | 按主题分组的原始 PDF 论文 | 想查一手论文来源 |
| `work/` | 迁移过来的工作材料，包括 AxCore 源码和论文摘录 | 想看实际 simulator 代码或原始摘录文本 |
| `THREAD_INDEX.md` | 项目线程索引 | 想查看线程和主题的对应关系 |

## 3. AM-LUT Python 原型

### 源码包

`am_lut_tcasi24/`

- `tcasi24.py`：TCASI24 LSAM1 和 CSAM2 近似乘法器的行为模型。
  - 实现了近似 Xilinx LUT/CARRY4 结构的逻辑。
  - 构造 unsigned 4x4 近似乘法器。
  - 用四个 4x4 block 组合 unsigned 8x8 乘法。
  - 提供第一周版本的 signed int8 包装：对操作数取绝对值，用 unsigned 近似乘法器计算 magnitude，再恢复符号。
  - 重要限制：这个 signed int8 包装不是严格的 signed RTL 等价模型。
- `gemm.py`：exact 和 LUT-backed int8 GEMM 辅助函数。
  - `exact_gemm`：int8 输入，int32 累加。
  - `lut_gemm`：每个标量乘积从 256x256 LUT 查询。
- `__init__.py`：包的公开导出入口。

### 脚本

`scripts/`

- `generate_int8_luts.py`
  - 生成：
    - `outputs/luts/exact_int8_lut.npy`
    - `outputs/luts/lsam1_int8_lut.npy`
    - `outputs/luts/csam2_int8_lut.npy`
    - `outputs/luts/metadata.json`
- `run_error_report.py`
  - 加载 LUT。
  - 计算 product-level 误差。
  - 在多种操作数分布下计算 GEMM-level 误差。
  - 写出：
    - `outputs/reports/tcasi24_int8_error_report.json`
    - `outputs/reports/tcasi24_int8_error_report.md`

### 测试

`tests/smoke_test_tcasi24.py`

- 验证 4x4 近似乘法器输出在合法范围内。
- 验证 exact 8x8 乘法。
- 验证 exact int8 LUT 的 shape、dtype 和几个关键值。
- 验证使用 exact LUT 时，`exact_gemm` 和 `lut_gemm` 结果一致。

## 4. AxCore simulator 复现材料

主目录：

```text
work/axcore/Software/axcore_simulator/
```

重要文件：

| 路径 | 作用 |
|---|---|
| `README.md` | 原始 simulator 环境配置和 Figure 17 复现说明 |
| `scripts/fig_17.sh` | Figure 17 复现脚本：跑 6 个 synthesis CSV 配置，然后调用 `EnergyAll.py` |
| `run_axcore.py` | 主 simulation runner；把归一化时间和能耗拆分写入 `results/axcore_res.csv` |
| `EnergyAll.py` | 读取 `axcore_res.csv` 并生成 `results/fig_17.pdf` |
| `conf_axcore.ini` | AxCore 配置；`module = axcore` |
| `conf_figlut.ini` | FIGLUT 配置 |
| `conf_figna.ini` | FIGNA 配置 |
| `conf_fpma.ini` | FPMA 配置 |
| `conf_fpe.ini` | FPE baseline 配置 |
| `params/systolic_array_synth_*.csv` | W4/W8 和 FP16/BF16/FP32 的 core synthesis 功耗/面积参数 |
| `results/axcore_res.csv` | 已复现出的 CSV 结果 |
| `results/fig_17.pdf` | 已复现出的 Figure 17 图 |

关键 simulator 包：

```text
work/axcore/Software/axcore_simulator/AxCore/src/
```

关键概念：

- `simulator/simulator.py`
  - 读取 config 和 synthesis CSV。
  - 计算 cycles。
  - 根据 core power 和 SRAM Cacti 数据计算 energy cost。
  - 对 AM-LUT sensitivity 很关键，因为 `Dynamic Power (nW)` 和 `Leakage Power (nW)` 在这里进入能耗模型。
- `simulator/stats.py`
  - 把 cycles 和 memory traffic 转成能耗。
  - `get_energy_breakdown()` 返回 Static / Dram / Buffer / Core。
- `benchmarks/`
  - 包含 simulator 使用的 OPT-style layer workloads。

当前 Figure 17 复现结果的解释：

- Time 由 FPE cycles 归一化，当前结果里基本都是 `1.00`。
- 主要差异来自能耗，尤其是 `Core`。
- Core energy 主要由 synthesis CSV 里的功耗字段控制。
- Cacti 影响 SRAM buffer energy，不直接影响 core energy。

## 5. AxCore 内部的 AM-LUT sensitivity 实验

目录：

```text
work/axcore/Software/axcore_simulator/experiments/amlut_sensitivity/
```

文件：

- `README.md`：说明实验边界和运行命令。
- `amlut_sensitivity.py`：参数层实验辅助脚本。
- `params/`：生成的修改版 synthesis CSV。
- `results/`：归档的 sensitivity run 结果 CSV。
- `summary/`：汇总 CSV 输出。

实验思路：

1. 复制一个 baseline `params/systolic_array_synth_*.csv`。
2. 只修改 `Module == axcore` 这一行。
3. 缩放：
   - `Area (um^2)`
   - `Leakage Power (nW)`
   - `Dynamic Power (nW)`
4. 运行 `run_axcore.py --synth_csv <modified_csv>`。
5. 归档 `results/axcore_res.csv`。

这是参数层实验。它不能证明 RTL 正确性、FPGA 实现质量、latency speedup 或 LLM accuracy。

## 6. 学习笔记

项目的大部分记忆在 `study/`。最相关的分组如下。

### AxCore 复现和 simulator

- `axcore-experiment-log.md`：按时间记录的复现实验日志。
- `axcore-figure17-results-summary.md`：Figure 17 结果的紧凑总结。
- `axcore-figure17-analysis-report.md`：更详细的 Figure 17 结果解释。
- `axcore-simulator-code-reading-notes.md`：simulator 数据路径和代码路径笔记。
- `axcore-stage-handoff-to-fpga-llm-main.md`：继续推进项目用的阶段交接总结。

### AxCore 论文理解

- `axcore-paper-reading-guide.md`
- `axcore-paper-zh-translation.md`
- `axcore-section-4-1-mpfpma-notes.md`
- `axcore-prereq-coverage-check.md`
- `axcore-reading-and-ideas.md`

### AM-LUT 和项目规划

- `amlut-axcore-first-week-skill-plan.md`
- `axcore-amlut-parameter-sensitivity-plan.md`
- `execution-roadmap-adana-axcore-amlut.md`
- `formal-project-execution-plan.md`
- `project-integrated-ai-infra-roadmap.md`
- `reference-driven-next-steps.md`

### 更宽泛的学习/职业背景

- `llm-ai-systems-kickoff-plan.md`
- `llm-ai-systems-resource-map.md`
- `llm-inference-systems-course-map.md`
- `cs224n-cs336-study-plan.md`
- `mit-65940-quantization-watchlist.md`
- `fpga-llm-ai-infra-positioning.md`

## 7. 输出产物

`outputs/`

- `luts/`：生成的 int8 product LUT 和 metadata。
- `reports/`：生成的 AM-LUT 误差报告。
- `axcore-fig17-render/fig_17.png`：已复现 Figure 17 的图片版。
- `axcore-paper-pages/`：AxCore 论文页面渲染图。
- `axcore-micro25-extracted.txt`：AxCore 论文提取文本。
- 其他 markdown 文件：早期项目设置阶段产生的 AI hardware / career 相关笔记。

## 8. 参考资料和原始工作材料

`references/`

- `AM-LUT/`：AM-LUT / approximate multiplier 相关论文。
- `LLM-FPGA/`：AxCore、FlightLLM、CXL-SpeckKV、Adana 和其他 FPGA/LLM 相关论文。

`work/`

- `papers/`：本地论文 PDF，包括 AxCore。
- `paper_excerpts/`：从论文中提取出的文本文件。
- `micro58-axcore-main.zip`：AxCore 源码压缩包。
- `axcore/`：迁移过来的 AxCore 源码树。

## 9. 已知注意事项

1. 部分迁移来的中文 markdown 文件在 PowerShell 输出中会显示为乱码。把它们当作项目记忆文件使用，但编辑前要先确认编码。
2. 迁移时没有复制 AxCore simulator 的虚拟环境。完整重跑前应重新创建环境。
3. 当前 Windows Python 可以跑部分项目脚本，但完整 Figure 17 复现更适合在 WSL/Linux 中做，因为 Cacti 是 Linux 二进制。
4. simulator 路径里的 Cacti cache 有已知问题：类似 `and False` 的条件会阻止读取已有 CSV cache，导致重复运行时可能反复调用 Cacti。
5. 原始 `run_axcore.py` 对 `results/axcore_res.csv` 是 append 写入。新实验前要清空或归档旧结果。

## 10. 推荐阅读顺序

快速熟悉项目：

1. `study/project-file-map.md`
2. `docs/week1_am_lut_plan.md`
3. `study/axcore-stage-handoff-to-fpga-llm-main.md`
4. `study/axcore-figure17-analysis-report.md`
5. `study/axcore-simulator-code-reading-notes.md`
6. `work/axcore/Software/axcore_simulator/README.md`
7. `work/axcore/Software/axcore_simulator/experiments/amlut_sensitivity/README.md`
8. `am_lut_tcasi24/tcasi24.py`
9. `scripts/run_error_report.py`

下一步技术工作建议：

1. 重新运行 `tests/smoke_test_tcasi24.py`。
2. 确认 `outputs/reports/tcasi24_int8_error_report.md`。
3. 先跑一个 AM-LUT sensitivity dry-run。
4. dry-run 确认无误后，再跑一个完整 AxCore sensitivity point。
