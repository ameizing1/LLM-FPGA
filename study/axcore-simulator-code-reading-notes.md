# AxCore simulator 代码阅读笔记

更新时间：2026-06-28

目标：解释 Figure 17 的结果来源，并为后续 AM-LUT / approximate multiplier 参数层实验确定安全入口。

## 1. 总体数据流

Figure 17 的复现路径可以概括为：

```text
scripts/fig_17.sh
-> run_axcore.py --synth_csv params/systolic_array_synth_*.csv
-> 5 个 config: axcore / figlut / figna / fpma / fpe
-> benchmark: opt_13b / opt_30b
-> AxCoreSimulator.get_cycles()
-> optimizer/get_stats_fast 得到 cycles 和 memory traffic
-> Stats.get_energy_breakdown()
-> results/axcore_res.csv
-> EnergyAll.py
-> results/fig_17.pdf
```

当前 Figure 17 不是完整 LLM serving simulator，更接近“OPT 线性层/GEMM workload 上的 accelerator energy model”。

## 2. run_axcore.py：结果组织与归一化

关键位置：

- `run_axcore.py:49`：`batch_size = 32`
- `run_axcore.py:56/73/89/105/121`：依次创建 `conf_axcore.ini`、`conf_figna.ini`、`conf_fpe.ini`、`conf_fpma.ini`、`conf_figlut.ini`
- `run_axcore.py:167`：用 append 模式写 `results/axcore_res.csv`
- `run_axcore.py:176`：AxCore time 用 AxCore cycles / FPE cycles
- `run_axcore.py:233`：energy 以 FPE total energy 为归一化基准

需要注意：

1. `results/axcore_res.csv` 是 append 写入，不是覆盖写入。  
   所以每次完整复现前最好删除旧的 `axcore_res.csv`，否则后续 `EnergyAll.py` 可能读到重复配置。

2. CSV 中 hardware 顺序写成：

```text
AxCore, FGLUT, FIGNA, FPMA, FPE
```

但 `EnergyAll.py` 画图时 categories 是：

```text
FPC, FPMA, FIGNA, FIGLUT, AxCore
```

因此 `EnergyAll.py` 里通过倒序索引重排。这里的 `FPC` 实际对应 CSV 中的 `FPE`。

3. `Time` 是 cycles 归一化值；`Static/Dram/Buffer/Core` 是 energy breakdown 各项分别除以同 benchmark 下 FPE total energy。

4. 当前复现中 `Time` 全部为 `1.00`。这不是说所有架构真实延迟一定相同，而是当前 simulator 路径下这些架构的 cycle model 没有体现差异。

## 3. config 文件：架构差异主要来自 module 名

五个 config 文件的主要差异：

```text
conf_axcore.ini -> [module] module = axcore
conf_figna.ini  -> [module] module = figna
conf_fpe.ini    -> [module] module = fpe
conf_fpma.ini   -> [module] module = fpma
conf_figlut.ini -> [module] module = figlut
```

它们的阵列和缓存配置基本相同：

```text
a = 64
c = 64
high_prec = 8
low_prec = 4
Wgt_SRAM = 131072
Act_SRAM = 131072
Out_SRAM = 131072
if_width = 1024
frequency = 1000000000
```

结论：Figure 17 中 AxCore/FGLUT/FIGNA/FPMA/FPE 的差异主要由 synthesis CSV 中对应 `Module` 行的功耗参数决定，而不是由不同阵列大小、SRAM 大小或 bandwidth 决定。

## 4. synthesis CSV：第一阶段最重要的实验入口

示例文件：

`params/systolic_array_synth_W4-FP16.csv`

表头：

```text
Module,Frequency,N,M,Weight (bits),Activation (bits),Area (um^2),Leakage Power (nW),Dynamic Power (nW),Total Power (nW)
```

示例行：

```text
axcore,1000000000,1,1,4,16,1,1056,112625,152352
figlut,1000000000,1,1,4,16,1,2101,217525,152352
figna,1000000000,1,1,4,16,1,1325,155806,152352
fpma,1000000000,1,1,4,16,1,2475,381832,152352
fpe,1000000000,1,1,4,16,1,5053,691006,152352
```

当前 CSV 中每个 module 都是 `N=1, M=1`。运行时会按 64x64 阵列扩展。

对 AM-LUT 来说，最小参数层实验就是构造新的 synthesis CSV，让 `axcore` 或新增 module 的：

- `Area (um^2)`
- `Leakage Power (nW)`
- `Dynamic Power (nW)`

按设定比例变化，然后观察 `Core` 和 `Total` 的变化。

## 5. AxCoreSimulator.get_energy_cost()

关键位置：

- `AxCore/src/simulator/simulator.py:223`：`get_energy_cost`
- `simulator.py:237-239`：WBUF/IBUF/OBUF bank 数在 energy path 中固定为 `1024/1024/16`
- `simulator.py:316-321`：先只用 `N/M` 查找完整 systolic array 行
- `simulator.py:323-327`：找不到完整 `N/M` 行时，fallback 到 `Module + N=1 + M=1`
- `simulator.py:331-336`：1x1 module 的 area/leak/dynamic power 按 `N*M` 放大
- `simulator.py:354`：返回 `EnergyTuple`

需要特别注意：

1. `get_energy_cost()` 中完整 `N/M` 查找没有启用 precision 过滤。  
   代码中 `Max Precision (bits)` / `Min Precision (bits)` 的 lookup 被注释掉了。

2. 如果没有完整 64x64 行，就会使用当前 config 的 `module_name` 查找 1x1 module。

3. `core_dyn_energy = Dynamic Power / Frequency`，`core_leak_energy = Leakage Power / Frequency`。  
   这两个值进入 `Stats.get_energy_breakdown()`。

4. Cacti 只影响 SRAM read/write energy；它不影响 core energy。

## 6. AxCoreSimulator.get_conv_cycles()

关键位置：

- `simulator.py:429`：`get_conv_cycles`
- `simulator.py:464-468`：根据 `Module` 行读取 `Weight (bits)` 和 `Activation (bits)`，然后覆盖 `iprec/wprec`
- `simulator.py:471`：把覆盖后的 precision 传入 optimizer

这点非常重要：benchmark layer 自身虽然有 precision 字段，但在 AxCore simulator 里会被 synthesis CSV 的当前 `Module` 行重新覆盖。

因此：

- `systolic_array_synth_W4-FP16.csv` 中 `axcore` 行的 `Weight (bits)=4`、`Activation (bits)=16` 会决定 AxCore 运行该配置时的实际精度。
- 如果要做 AM-LUT 参数实验，但希望保持 workload 和 precision 不变，就只改 area/leak/dynamic power，不改 weight/activation bits。
- 如果要研究 W4/W8 或 FP16/BF16/FP32 差异，应通过现有六个 synthesis CSV 变体控制，不要在第一版 AM-LUT 实验里混入新的 bit-width 变化。

## 7. Stats.get_energy_breakdown()

关键位置：

- `AxCore/src/simulator/stats.py:82`：`get_energy_breakdown`
- `stats.py:88-89`：Core energy
- `stats.py:91-98`：SRAM buffer energy
- `stats.py:102-103`：DRAM energy
- `stats.py:108`：Static energy
- `stats.py:110-113`：返回顺序

返回顺序为：

```text
[Static, Dram, Buffer, Core]
```

具体含义：

```text
Core   = total_cycles * core_leak_energy
       + (total_cycles - mem_stall_cycles) * core_dynamic_energy

Buffer = WBUF/IBUF/OBUF read/write traffic * Cacti-derived SRAM cost

Dram   = DRAM read/write traffic * fixed DRAM read/write cost

Static = total_cycles * fixed dram_leak_energy
```

注意：`sram_leak_energy` 在 breakdown 中被注释掉，没有计入 `Buffer`。

## 8. benchmark：OPT13B / OPT30B 的实际含义

关键位置：

- `AxCore/src/benchmarks/benchmarks.py`：`benchlist = ['opt_13b', 'opt_30b']`
- `AxCore/src/benchmarks/axcore_bench.py:114`：`generate_opt_layers`
- `axcore_bench.py:181`：`opt_13b`
- `axcore_bench.py:183`：`opt_30b`

定义：

```python
opt_13b = generate_opt_layers(hidden_size=5120, ffn_dim=20480, num_layers=40, seq_len=1)
opt_30b = generate_opt_layers(hidden_size=7168, ffn_dim=28672, num_layers=48, seq_len=1)
```

每层包含：

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`
- `fc1`
- `fc2`

最后还有：

- `lm_head`

被注释掉的部分：

- attention score: `QK^T`
- value aggregation: `SV`

结论：

当前 benchmark 对 decode-like 的 Linear/GEMM 很有参考价值，但不覆盖完整 attention/KV cache 行为。后续如果要贴近 long-context LLM inference，需要另外扩展 workload。

## 9. Cacti cache 问题

当前观察到：

- 运行时反复出现 `No entry found ... running cacti`
- `AxCore/sram/cacti_sweep.csv` 没有明显积累足够 cache

代码上也能看到一个类似问题：

- `AxCore/src/sweep/sweep.py` 中 simulator sweep 的 CSV 写回被注释掉。
- CactiSweep 是否写 cache 需要单独读 `AxCore/sram/cacti_sweep.py` 再确认。

目前判断：

- 这不影响结果正确性。
- 它会影响重复运行速度。
- 在开始大规模 sensitivity sweep 前，应当再检查是否值得修 cache。

## 10. 对 AM-LUT 实验的直接启发

第一阶段实验应该限定为：

```text
同 workload
同 W4/W8 与 FP16/BF16/FP32 precision
同 N/M/SRAM/bandwidth
只改变 core module 的 area/leakage/dynamic power
观察 Core 和 Total normalized energy 的变化
```

不要在第一阶段同时改变：

- attention workload
- accuracy/perplexity
- SpinalHDL RTL
- cycle model
- SRAM/Cacti
- batch size
- W4/W8 precision

原因：当前 simulator 的 Figure 17 差异主要经由 synthesis CSV 的 module power 进入。先在这个最短路径上验证 AM-LUT 的 system-level energy sensitivity，最容易得到可解释结论。

