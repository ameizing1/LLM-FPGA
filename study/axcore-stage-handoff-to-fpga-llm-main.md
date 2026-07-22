# AxCore 复现阶段交接记录

更新时间：2026-06-28

用途：交接给“推进 FPGA 部署 LLM - 主线”继续讨论下一步。

## 1. 当前结论

AxCore simulator 的 Figure 17 忠实复现已经完成。当前最重要的下一步不是直接进入硬件替换，而是先把已经复现出的结果解释清楚，并把 simulator 的数据路径整理成后续可扩展的实验入口。

建议优先级仍然是：

1. A：检查 `fig_17.pdf`，提取 `axcore_res.csv` 关键结果。
2. B：读 `run_axcore.py` / `EnergyAll.py` / simulator 代码，解释 `Time` 和 energy breakdown 怎么来的。
3. C：Cacti cache miss 先记录为工程问题，不优先修。
4. F：在 A/B 基础上设计 Adana + AxCore + AM-LUT 的最小连接实验。
5. D/E：暂缓 accuracy/perplexity 路线和 SpinalHDL 硬件替换路线。

一句话判断：下一步优先 A+B，然后设计 F；不要现在直接改硬件。

## 2. 已完成复现状态

仓库位置：

`C:\Users\LiuZhiWei\Documents\Codex\2026-06-17\ai\work\axcore`

主要 simulator 路径：

`C:\Users\LiuZhiWei\Documents\Codex\2026-06-17\ai\work\axcore\Software\axcore_simulator`

环境：

- WSL Ubuntu 22.04.5 LTS
- Python 3.10.12
- gcc 11.4.0
- make 4.3
- git 2.34.1
- Python venv：`Software\axcore_simulator\.venv`
- Python 包：`numpy==2.0.2`、`pandas==2.2.3`、`matplotlib==3.10.9`、`graphviz==0.21`

Cacti 状态：

- 已 clone 并编译：`bitfusion\sram\cacti\cacti`
- 已复制到：`AxCore\sram\cacti\cacti`

已成功运行：

```bash
cd /mnt/c/Users/LiuZhiWei/Documents/Codex/2026-06-17/ai/work/axcore/Software/axcore_simulator
. .venv/bin/activate
sh scripts/fig_17.sh
```

完整 Figure 17 复现耗时约 431 秒。

输出文件：

- `work\axcore\Software\axcore_simulator\results\axcore_res.csv`
- `work\axcore\Software\axcore_simulator\results\fig_17.pdf`

PDF 核验：

- `fig_17.pdf` 是有效 1 页 PDF。
- 文件大小约 36 KB。
- 已渲染为 PNG：`outputs\axcore-fig17-render\fig_17.png`
- 视觉检查结果：图不是空白，包含 `W4-FP16`、`W4-BF16`、`W4-FP32`、`W8-FP16`、`W8-BF16`、`W8-FP32` 和 `Average`；上半部分为 normalized energy 堆叠柱，下半部分为 normalized TOPS/W；整体结构合理。
- Times New Roman 缺失只影响字体外观，不影响复现结论。

## 3. Figure 17 关键结果

以下表格来自 `results/axcore_res.csv` 的 Geomean 部分。`Total = Static + Dram + Buffer + Core`。`Core/FPE` 和 `Total/FPE` 用同配置下 FPE 作为 baseline。

| Config | Arch | Time | Static | Dram | Buffer | Core | Total | Core/FPE | Total/FPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W4-FP16 | AxCore | 1.00 | 0.22 | 0.12 | 0.04 | 0.10 | 0.48 | 0.16 | 0.46 |
| W4-FP16 | FGLUT | 1.00 | 0.22 | 0.12 | 0.04 | 0.20 | 0.58 | 0.32 | 0.56 |
| W4-FP16 | FIGNA | 1.00 | 0.22 | 0.12 | 0.04 | 0.14 | 0.52 | 0.22 | 0.50 |
| W4-FP16 | FPMA | 1.00 | 0.22 | 0.12 | 0.04 | 0.35 | 0.73 | 0.56 | 0.70 |
| W4-FP16 | FPE | 1.00 | 0.22 | 0.12 | 0.07 | 0.63 | 1.04 | 1.00 | 1.00 |
| W4-BF16 | AxCore | 1.00 | 0.26 | 0.14 | 0.04 | 0.12 | 0.56 | 0.21 | 0.53 |
| W4-BF16 | FGLUT | 1.00 | 0.26 | 0.14 | 0.04 | 0.14 | 0.58 | 0.25 | 0.55 |
| W4-BF16 | FIGNA | 1.00 | 0.26 | 0.14 | 0.04 | 0.15 | 0.59 | 0.27 | 0.56 |
| W4-BF16 | FPMA | 1.00 | 0.26 | 0.14 | 0.04 | 0.35 | 0.79 | 0.62 | 0.75 |
| W4-BF16 | FPE | 1.00 | 0.26 | 0.14 | 0.09 | 0.56 | 1.05 | 1.00 | 1.00 |
| W4-FP32 | AxCore | 1.00 | 0.09 | 0.05 | 0.02 | 0.09 | 0.25 | 0.11 | 0.25 |
| W4-FP32 | FGLUT | 1.00 | 0.09 | 0.05 | 0.02 | 0.14 | 0.30 | 0.17 | 0.30 |
| W4-FP32 | FIGNA | 1.00 | 0.09 | 0.05 | 0.02 | 0.10 | 0.26 | 0.12 | 0.26 |
| W4-FP32 | FPMA | 1.00 | 0.09 | 0.05 | 0.02 | 0.32 | 0.48 | 0.38 | 0.48 |
| W4-FP32 | FPE | 1.00 | 0.09 | 0.05 | 0.03 | 0.84 | 1.01 | 1.00 | 1.00 |
| W8-FP16 | AxCore | 1.00 | 0.23 | 0.15 | 0.03 | 0.11 | 0.52 | 0.19 | 0.50 |
| W8-FP16 | FGLUT | 1.00 | 0.23 | 0.15 | 0.03 | 0.40 | 0.81 | 0.68 | 0.78 |
| W8-FP16 | FIGNA | 1.00 | 0.23 | 0.15 | 0.03 | 0.20 | 0.61 | 0.34 | 0.59 |
| W8-FP16 | FPMA | 1.00 | 0.23 | 0.15 | 0.03 | 0.32 | 0.73 | 0.54 | 0.70 |
| W8-FP16 | FPE | 1.00 | 0.23 | 0.15 | 0.07 | 0.59 | 1.04 | 1.00 | 1.00 |
| W8-BF16 | AxCore | 1.00 | 0.27 | 0.18 | 0.04 | 0.11 | 0.60 | 0.22 | 0.58 |
| W8-BF16 | FGLUT | 1.00 | 0.27 | 0.18 | 0.04 | 0.27 | 0.76 | 0.53 | 0.73 |
| W8-BF16 | FIGNA | 1.00 | 0.27 | 0.18 | 0.04 | 0.21 | 0.70 | 0.41 | 0.67 |
| W8-BF16 | FPMA | 1.00 | 0.27 | 0.18 | 0.04 | 0.32 | 0.81 | 0.63 | 0.78 |
| W8-BF16 | FPE | 1.00 | 0.27 | 0.18 | 0.08 | 0.51 | 1.04 | 1.00 | 1.00 |
| W8-FP32 | AxCore | 1.00 | 0.11 | 0.08 | 0.02 | 0.10 | 0.31 | 0.12 | 0.30 |
| W8-FP32 | FGLUT | 1.00 | 0.11 | 0.08 | 0.02 | 0.26 | 0.47 | 0.33 | 0.46 |
| W8-FP32 | FIGNA | 1.00 | 0.11 | 0.08 | 0.02 | 0.14 | 0.35 | 0.18 | 0.34 |
| W8-FP32 | FPMA | 1.00 | 0.11 | 0.08 | 0.02 | 0.30 | 0.51 | 0.37 | 0.50 |
| W8-FP32 | FPE | 1.00 | 0.11 | 0.08 | 0.03 | 0.80 | 1.02 | 1.00 | 1.00 |

关键观察：

- 六组配置的 `Time` 全部是 `1.00`。这说明当前 Figure 17 复现实验中，几个架构在 cycle model 上没有体现时间差异；主要差异来自 energy，尤其是 `Core`。
- AxCore 在所有配置下的 `Core` 都明显低于 FPE。
- `Static / Dram / Buffer` 在同一配置内差异较小；因此第一阶段要研究 AM-LUT 对 Figure 17 的影响，最直接入口是 core power 参数，而不是 memory system。
- `W4-FP32` 和 `W8-FP32` 下 AxCore 的 `Total/FPE` 最低，说明 FPE 的 core energy 在这些配置下占比很高，低成本计算单元的节能效果被放大。

## 4. 代码路径解释

目前已经定位出的结果生成链路如下：

```text
benchmark layer definition
-> AxCoreSimulator runs each benchmark
-> simulator returns cycles and memory/core statistics
-> Stats.get_energy_breakdown computes Static/Dram/Buffer/Core
-> run_axcore.py normalizes cycles and energy against FPE
-> axcore_res.csv is written
-> EnergyAll.py plots fig_17.pdf
```

### 4.1 benchmark 定义

文件：

`AxCore\src\benchmarks\benchmarks.py`

其中：

```python
benchlist = ['opt_13b', 'opt_30b']
```

实际 OPT layer 配置在：

`AxCore\src\benchmarks\axcore_bench.py`

关键定义：

```python
opt_13b = generate_opt_layers(hidden_size=5120, ffn_dim=20480, num_layers=40, seq_len=1)
opt_30b = generate_opt_layers(hidden_size=7168, ffn_dim=28672, num_layers=48, seq_len=1)
```

当前 benchmark 主要覆盖：

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`
- `fc1`
- `fc2`

注意：attention score 和 value aggregation 相关层在代码里是注释掉的。因此 Figure 17 当前主要是 Linear/GEMM workload，不是完整 LLM inference serving simulator。

### 4.2 Time 归一化

文件：

`run_axcore.py`

逻辑：

- 分别创建 AxCore、FIGNA、FPE、FPMA、FGLUT simulator。
- 每个 simulator 跑 `benchmarks.benchlist`。
- `Time` 用对应架构 cycles 除以 FPE cycles。
- CSV 写入顺序是：`AxCore, FGLUT, FIGNA, FPMA, FPE`。

当前结果里 `Time` 全是 `1.00`，说明这些架构在这个实验路径下 cycles 相同或被建模为相同。对 AM-LUT 第一版实验来说，暂时不要宣称能改善 latency；更合理的说法是先评估 core energy / total energy sensitivity。

### 4.3 Energy breakdown

文件：

`AxCore\src\simulator\stats.py`

函数：

`Stats.get_energy_breakdown(...)`

CSV 中四项对应：

- `Static`：`total_cycles * dram_leak_energy`
- `Dram`：DRAM read/write traffic 乘以 DRAM energy cost
- `Buffer`：WBUF/IBUF/OBUF traffic 乘以 Cacti SRAM read/write cost
- `Core`：`total_cycles * core_leak_energy + active_cycles * core_dynamic_energy`

### 4.4 synthesis CSV 如何影响 Core

文件：

`AxCore\src\simulator\simulator.py`

函数：

`AxCoreSimulator.get_energy_cost(...)`

关键机制：

- 读取 `params\systolic_array_synth_*.csv`。
- 查找匹配的 module row。
- 如果没有完整 `N/M` row，会用 `N=1, M=1` fallback，再按 `N*M` 扩展。
- 使用 CSV 中的：
  - `Frequency`
  - `Area (um^2)`
  - `Leakage Power (nW)`
  - `Dynamic Power (nW)`
- 计算：
  - `core_dyn_energy = Dynamic Power / Frequency`
  - `core_leak_energy = Leakage Power / Frequency`

结论：AM-LUT 或其他 approximate multiplier 第一阶段最适合抽象为 synthesis CSV 中不同 module 的 leakage/dynamic/area 变化。

### 4.5 Cacti 如何影响 Buffer/SRAM

同样在：

`AxCore\src\simulator\simulator.py`

流程：

- 根据 SRAM bank size、block size、ports、technology node 等组成 `cfg_dict`。
- 通过 `self.sram_obj.get_data_clean(cfg_dict)` 查询或运行 Cacti。
- Cacti 输出 SRAM read/write/leak/area。
- 这些值再进入 WBUF/IBUF/OBUF energy cost。

当前观察到重复运行时有大量 `No entry found ... running cacti`，说明 cache 复用可能不充分。但它不影响当前复现正确性，只影响重复实验效率。

## 5. 当前工程问题

1. Cacti cache miss：重复运行会慢，先记录，不优先修。
2. Times New Roman 缺失：matplotlib fallback，影响外观，不影响结果。
3. pandas warning：目前不影响结果。
4. venv 位于 `/mnt/c`：可用但性能不是最优，后续高频实验可考虑迁移到 WSL ext4 路径。
5. 已有两份早先生成的 Markdown 可能存在中文乱码，建议后续优先参考本文件。

## 6. 下一步建议

### 下一步 1：完善 A/B 的可复用解释

建议继续把以下文件读透并做代码级注释笔记：

- `run_axcore.py`
- `EnergyAll.py`
- `AxCore\src\simulator\stats.py`
- `AxCore\src\simulator\simulator.py`
- `AxCore\src\benchmarks\benchmarks.py`
- `AxCore\src\benchmarks\axcore_bench.py`
- `params\systolic_array_synth_*.csv`

目标不是泛泛读代码，而是回答这几个问题：

- `Time` 为什么全是 `1.00`？
- `Core` 是由哪些 synthesis CSV 字段决定的？
- `Buffer` 为什么依赖 Cacti？
- OPT13B/OPT30B 是否只建模 Linear/GEMM？
- 如果加入 AM-LUT，第一阶段应该改 synthesis CSV 还是 simulator 内部 cycle model？

### 下一步 2：设计 AM-LUT 参数层 sensitivity study

第一版不要进 SpinalHDL，不要改硬件。建议从参数层做最小实验：

1. 复制某个 baseline synthesis CSV，例如 `systolic_array_synth_W4-FP16.csv`。
2. 构造 AM-LUT 假设版本，例如：
   - leakage power 降低 10/20/30%
   - dynamic power 降低 10/20/30%
   - area 降低 10/20/30%
3. 运行 AxCore simulator，观察 `Core` 和 `Total` 变化。
4. 和 FPE、FPMA、FIGNA、FGLUT、AxCore 原版对比。
5. 把结论限制在 energy sensitivity，不讨论 accuracy。

这个实验能快速回答一个关键问题：

如果 AM-LUT 的乘法器级节能成立，映射到 AxCore 的 system-level Figure 17 上，total energy 能下降多少？

### 下一步 3：再决定是否进入精度或硬件路线

只有当参数层 sensitivity study 说明收益值得进一步验证时，再进入：

- `Software/AxCore` accuracy/perplexity 路线：验证 approximate arithmetic 对模型精度的影响。
- `AxCore/src/simulator` 内部建模路线：如果 AM-LUT 会改变吞吐、packing、cycle 或 memory traffic。
- `Hardware/AxCore` SpinalHDL 路线：做硬件级实现与验证。

## 7. 给主线进程的建议问题

建议“推进 FPGA 部署 LLM - 主线”接下来重点讨论这几个问题：

1. AM-LUT 的最小可量化参数是什么：area、leakage、dynamic power、delay，还是 LUT utilization？
2. AM-LUT 的误差模型能否先抽象成 bit-width/operand distribution 相关的 error profile？
3. Adana 的量化思路能否给出更具体的 low-bit workload：W4/W8、FP16/BF16/FP32 accumulation 中应该优先哪一组？
4. AxCore 的 Figure 17 benchmark 主要是 GEMM，不是完整 LLM serving。后续是否需要补充 long-context/KV cache 相关 workload？
5. 第一篇阶段报告应定位为“复现 + simulator 数据路径解释 + AM-LUT 参数层实验计划”，而不是“已经完成硬件替换”。

