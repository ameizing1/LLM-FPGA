# 旧 Codex 对话迁移摘要与关键记录索引

更新时间：2026-07-05

## 迁移结论

这次迁移把旧普通对话目录中的项目资料复制到了当前 `LLM-FPGA` 项目中，并为 5 个旧对话创建了对应的项目线程。

需要注意：新线程不是旧线程的完整克隆。它们继承的是迁移 prompt、旧线程摘要、当前项目文件和已迁入资料；不继承旧线程完整逐轮聊天历史、全部工具输出、旧线程内部隐式上下文。

## 已迁移文件

旧目录：

`C:\Users\LiuZhiWei\Documents\Codex\2026-06-17\ai`

当前项目：

`D:\LiuZhiWei\Documents\STUDTORWORK\研0\SJTU relative project\LLM-FPGA`

| 内容 | 旧位置 | 新位置 | 说明 |
| --- | --- | --- | --- |
| 学习记录 | `study` | `study` | 完整复制，28 个文件 |
| 输出材料 | `outputs` | `outputs` | 完整复制，18 个文件 |
| AxCore 工作目录 | `work` | `work` | 复制必要源码、论文、结果、实验脚本 |
| AxCore 虚拟环境 | `work\axcore\Software\axcore_simulator\.venv` | 未复制 | 约 227 MiB，可重建，不适合迁移 |

迁移后当前项目新增的重点材料包括：

- `study\axcore-experiment-log.md`
- `study\axcore-simulator-code-reading-notes.md`
- `study\axcore-amlut-parameter-sensitivity-plan.md`
- `study\axcore-stage-handoff-to-fpga-llm-main.md`
- `study\amlut-axcore-first-week-skill-plan.md`
- `outputs\axcore-fig17-render\fig_17.png`
- `outputs\axcore-paper-pages\*.png`
- `work\axcore\Software\axcore_simulator\results\axcore_res.csv`
- `work\axcore\Software\axcore_simulator\results\fig_17.pdf`
- `work\axcore\Software\axcore_simulator\experiments\amlut_sensitivity\amlut_sensitivity.py`
- `work\axcore\Software\axcore_simulator\experiments\amlut_sensitivity\README.md`

## 新旧线程对应关系

| 主题 | 旧线程 id | 新项目线程 id | 新线程状态 |
| --- | --- | --- | --- |
| 推进FPGA部署LLM-主线 | `019f028b-64be-7750-b9ea-e2bca6d82d40` | `019f311c-cf58-7b11-86dc-5cf767aee762` | 已创建 |
| 推进FPGA部署LLM-问题排查 | `019f0a5a-247a-7081-a0a9-e3423b07ab01` | `019f311c-d35a-7fb0-b8b1-219c237dbd7c` | 已创建 |
| 推进FPGA部署LLM-Axcore论文阅读 | `019f0e2f-265e-7ed2-bc65-c91792bba547` | `019f311c-d75e-7f23-8012-0f3ac31c8004` | 已创建 |
| 推进FPGA部署LLM-环境配置与Axcore复现 | `019f0750-0b21-7b02-a587-6b1e4d7323a8` | `019f311c-db96-7ad2-97db-bda35bf78cad` | 已创建 |
| 推进FPGA部署LLM-AM-LUT作Axcore替换 | `019f2e32-7da8-74d3-bbc8-af5a61117103` | `019f311c-df7d-7ab0-8e2c-4195e9dfff67` | 已创建 |

## 旧线程内容摘要

### 推进FPGA部署LLM-主线

定位：总项目路线、导师 README 解读、AxCore 与 AM-LUT 结合方式。

关键结论：

- 当前方向不是证明 AM-LUT 一定替换 AxCore 更好，而是比较 AxCore adder-based FPMA 与 FPGA LUT-based approximate multiplier 在低比特 LLM GEMM 中的精度和硬件折中。
- AxCore 公开硬件评估更偏 ASIC-oriented，使用 28nm TSMC、Synopsys Design Compiler、1GHz 目标频率；不能直接外推到 FPGA。
- AM-LUT 的价值在于 FPGA LUT、carry-chain、DSP 资源结构不同，值得做 FPGA-aware 对比。
- 建议路线：先软件数值层，再 simulator 参数层，再 CUDA/RTL。

### 推进FPGA部署LLM-问题排查

定位：WSL、网络、代理、工具链和系统风险排查。

关键结论：

- 曾发生严重卡顿和 Codex 重装风险，可能与 WSL 初始化残留进程、mirrored networking、Clash/代理、防火墙叠加有关。
- 后续涉及 `.wslconfig`、WSL 网络、TUN/虚拟网卡、防火墙和代理时必须非常保守。
- 环境配置阶段优先给用户具体步骤和判断标准，不默认全自动推进。
- 已知可用工具链：Ubuntu 22.04.5、Python 3.10.12、pip 22.0.2、gcc 11.4.0、make 4.3、git 2.34.1。

### 推进FPGA部署LLM-Axcore论文阅读

定位：AxCore 论文逐段理解、概念补课、公式推导。

关键结论：

- AxCore 面向 FP16 activation 与 FP4/低比特 weight 的 mixed-precision GEMM。
- FPMA/mpFPMA 用加法近似浮点乘法，但依赖 normalized number 的结构。
- FP4 subnormal 没有 implicit leading one，会破坏 FPMA 的默认假设，所以需要 SNC。
- `Align(Wq)` 用于低比特 weight 与高精度 activation 的 mantissa/fixed-point domain 对齐。
- `B1` 修正 exponent bias 差异，`C1` 补偿 FPMA 系统误差。
- 后续讲公式时使用块级 LaTeX，不使用行内 `$...$`。

### 推进FPGA部署LLM-环境配置与Axcore复现

定位：AxCore simulator 复现、Figure 17、Cacti、AM-LUT 参数层实验草案。

关键结论：

- AxCore simulator 忠实复现实验已经跑通。
- 完整 `scripts/fig_17.sh` 曾用时约 431 秒。
- 已生成 `results\axcore_res.csv` 和 `results\fig_17.pdf`。
- Figure 17 中 Time 基本为 1.00，主要差异来自能耗分解，尤其 Core。
- `Stats.get_energy_breakdown` 与 synthesis CSV 是理解 energy 结果的核心路径。
- Cacti cache 有明确问题：`if os.path.isfile(self.csv_file) and False` 导致不读取已有 cache，重复运行会反复调用 Cacti。
- 已有 AM-LUT 参数层 sensitivity 草案，默认 dry-run，不污染原始结果。

### 推进FPGA部署LLM-AM-LUT作Axcore替换

定位：TCASI24/AM-LUT 近似 INT8 乘法器行为模型、替换对比路线、第一周任务。

关键结论：

- 第一版只选 TCASI24 的 LSAM1 和 CSAM2：LSAM1 代表保守低误差，CSAM2 代表激进低硬件开销。
- 先生成 `exact_int8_lut.npy`、`lsam1_int8_lut.npy`、`csam2_int8_lut.npy`。
- 先做 approximate GEMM 小实验，再考虑真实 LLM operand distribution。
- 第一周重点技能是 Python、NumPy、补码/bit 操作、int8 溢出和误差指标。
- 暂时不需要完整 CUDA、Verilog/Vivado、完整 LLM 推理系统或完整机器学习课程。

## 关键文件索引

### 项目路线与学习计划

- `study\formal-project-execution-plan.md`
- `study\execution-roadmap-adana-axcore-amlut.md`
- `study\project-integrated-ai-infra-roadmap.md`
- `study\reference-driven-next-steps.md`
- `study\amlut-axcore-first-week-skill-plan.md`

### AxCore 论文与概念

- `study\axcore-paper-reading-guide.md`
- `study\axcore-paper-zh-translation.md`
- `study\axcore-section-4-1-mpfpma-notes.md`
- `study\axcore-prereq-coverage-check.md`

### AxCore 复现与 simulator

- `study\axcore-experiment-log.md`
- `study\axcore-figure17-results-summary.md`
- `study\axcore-figure17-analysis-report.md`
- `study\axcore-simulator-code-reading-notes.md`
- `study\axcore-stage-handoff-to-fpga-llm-main.md`
- `work\axcore\Software\axcore_simulator\results\axcore_res.csv`
- `work\axcore\Software\axcore_simulator\results\fig_17.pdf`
- `outputs\axcore-fig17-render\fig_17.png`

### AM-LUT 与替换实验

- `study\axcore-amlut-parameter-sensitivity-plan.md`
- `study\amlut-axcore-first-week-skill-plan.md`
- `work\axcore\Software\axcore_simulator\experiments\amlut_sensitivity\README.md`
- `work\axcore\Software\axcore_simulator\experiments\amlut_sensitivity\amlut_sensitivity.py`

### 输出和论文摘录

- `outputs\axcore-micro25-extracted.txt`
- `outputs\axcore-paper-pages\*.png`
- `work\paper_excerpts`
- `work\papers`

## 删除旧目录前检查清单

旧目录暂时不要直接删除。建议先完成以下检查：

- 当前项目中的新线程都能在侧边栏看到。
- 当前项目 `study` 中 28 个记录文件可打开。
- 当前项目 `outputs` 中 Figure 17 渲染图和 AxCore 论文页面图可打开。
- 当前项目 `work\axcore\Software\axcore_simulator\results` 中 `axcore_res.csv` 和 `fig_17.pdf` 存在。
- 当前项目 `work\axcore\Software\axcore_simulator\experiments\amlut_sensitivity` 中脚本和 README 存在。
- 如果需要重新跑 simulator，先重建 `.venv`，不要依赖旧目录里的虚拟环境。
- 确认旧目录里没有其他未迁移的大模型、临时文件或用户手动保存的重要文件。

完成这些检查后，可以先归档旧线程；如果还要释放磁盘空间，再删除旧目录中的可重建部分，例如旧 `.venv`。

## 已执行清理

执行时间：2026-07-05

- 旧线程已归档：
  - `019f028b-64be-7750-b9ea-e2bca6d82d40`
  - `019f0a5a-247a-7081-a0a9-e3423b07ab01`
  - `019f0e2f-265e-7ed2-bc65-c91792bba547`
  - `019f0750-0b21-7b02-a587-6b1e4d7323a8`
  - `019f2e32-7da8-74d3-bbc8-af5a61117103`
- 已删除旧目录中的可重建虚拟环境：
  - `C:\Users\LiuZhiWei\Documents\Codex\2026-06-17\ai\work\axcore\Software\axcore_simulator\.venv`
- 删除前 `.venv` 大小约 `227.3 MiB`。
- 删除后旧普通目录大小约 `44.2 MiB`，其中 `outputs` 仍存在。
