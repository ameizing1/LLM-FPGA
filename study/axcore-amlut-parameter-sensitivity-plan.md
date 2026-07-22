# AM-LUT 参数层 sensitivity study 方案

更新时间：2026-06-28

目标：在不改 AxCore 原始硬件、不改 simulator cycle model、不改 benchmark workload 的前提下，先回答一个最小问题：

如果 AM-LUT / approximate multiplier 能让 core 的 area、leakage、dynamic power 降低，映射到 AxCore Figure 17 的 system-level energy 后，`Core` 和 `Total` 能下降多少？

## 1. 实验边界

第一版只做参数层实验。

保持不变：

- OPT13B / OPT30B benchmark
- batch size = 32
- 64x64 阵列
- WBUF/IBUF/OBUF 大小
- memory bandwidth
- W4/W8 和 FP16/BF16/FP32 配置
- cycle model
- Cacti SRAM 参数
- accuracy/perplexity 路线
- SpinalHDL/RTL 路线

只改变：

- synthesis CSV 中 `axcore` module 行的：
  - `Area (um^2)`
  - `Leakage Power (nW)`
  - `Dynamic Power (nW)`

第一版不新增 `amlut` module，而是生成“AxCore-AM-LUT 假设版 CSV”，让 `conf_axcore.ini` 仍然读取 `module = axcore`。这样不需要改 config 和 `run_axcore.py`，最小化扰动。

## 2. 为什么先改 axcore 行

`run_axcore.py` 会依次跑五个 config：

```text
conf_axcore.ini -> module = axcore
conf_figna.ini  -> module = figna
conf_fpe.ini    -> module = fpe
conf_fpma.ini   -> module = fpma
conf_figlut.ini -> module = figlut
```

如果我们只修改 CSV 里的 `axcore` 行：

- AxCore 结果会变成 AM-LUT 假设结果；
- FGLUT/FIGNA/FPMA/FPE baseline 保持不变；
- CSV/图表结构不需要改；
- 可以直接看 AxCore-AM-LUT 相对原 AxCore 和其他 baselines 的变化。

这比新增第六个架构更适合第一版，因为 `EnergyAll.py` 固定只画 5 个 category。

## 3. 实验变量

建议第一轮只扫 dynamic/leakage，不扫 area。

原因：

- Figure 17 的 energy breakdown 中 `Core` 直接由 `Leakage Power` 和 `Dynamic Power` 进入。
- `Area (um^2)` 当前主要影响 area report，不直接进入 Figure 17 的 energy breakdown。
- 如果第一轮目标是 energy sensitivity，优先扫 power 更直接。

建议变量：

```text
leakage_scale = [1.0, 0.9, 0.8, 0.7, 0.5]
dynamic_scale = [1.0, 0.9, 0.8, 0.7, 0.5]
```

第一阶段不要做 5x5 全组合，先做两条一维 sweep：

1. dynamic-only：

```text
leakage_scale = 1.0
dynamic_scale = 1.0 / 0.9 / 0.8 / 0.7 / 0.5
```

2. leakage-only：

```text
dynamic_scale = 1.0
leakage_scale = 1.0 / 0.9 / 0.8 / 0.7 / 0.5
```

如果这两条曲线有意义，再做组合 sweep：

```text
leakage_scale = dynamic_scale = 0.9 / 0.8 / 0.7 / 0.5
```

## 4. 推荐先跑的配置

不要一上来扫六个配置。建议先用一个代表配置验证脚本：

```text
W4-FP16
```

原因：

- 这是最容易解释的 low-bit weight + FP16 activation 场景。
- 单配置运行比完整 Figure 17 快。
- 可以先确认 CSV 生成、结果解析、归档方式都正确。

验证通过后再扩展到六个配置：

```text
W4-FP16
W4-BF16
W4-FP32
W8-FP16
W8-BF16
W8-FP32
```

## 5. 输出目录建议

不要复用 `results/` 作为唯一输出目录，因为 `run_axcore.py` 固定写：

```text
results/axcore_res.csv
```

建议每次实验完成后立刻归档：

```text
experiments/amlut_sensitivity/
  params/
    W4-FP16_dyn0.9_leak1.0.csv
    W4-FP16_dyn0.8_leak1.0.csv
  results/
    W4-FP16_dyn0.9_leak1.0_axcore_res.csv
    W4-FP16_dyn0.8_leak1.0_axcore_res.csv
  summary/
    amlut_sensitivity_summary.csv
    amlut_sensitivity_summary.md
```

注意：`run_axcore.py` 使用 append 模式写 `results/axcore_res.csv`，每次运行前必须清空或删除该文件。

## 6. 最小运行流程

对单个实验点：

1. 复制 baseline synthesis CSV。
2. 只修改 `Module == axcore` 行的 power 字段。
3. 清空 `results/axcore_res.csv`。
4. 运行：

```bash
python run_axcore.py --synth_csv experiments/amlut_sensitivity/params/W4-FP16_dyn0.8_leak1.0.csv
```

5. 复制：

```text
results/axcore_res.csv
-> experiments/amlut_sensitivity/results/W4-FP16_dyn0.8_leak1.0_axcore_res.csv
```

6. 解析 Geomean 中 AxCore 的：

```text
Time
Static
Dram
Buffer
Core
Total = Static + Dram + Buffer + Core
```

7. 和 baseline 原始 AxCore 对比：

```text
Core_delta = Core_new / Core_baseline
Total_delta = Total_new / Total_baseline
```

## 7. 预期现象

因为 Figure 17 中：

```text
Core = total_cycles * core_leak_energy
     + active_cycles * core_dynamic_energy
```

所以：

- 降低 dynamic power 会降低 active_cycles 对应部分；
- 降低 leakage power 会降低 total_cycles 对应部分；
- `Static/Dram/Buffer` 基本不会变；
- `Time` 预期仍为 1.00；
- `Total` 的下降幅度会小于 `Core` 的下降幅度，因为 non-core energy 不变。

## 8. 结果解释模板

第一轮结果应该用这种方式表述：

```text
在 AxCore simulator 的 Figure 17 workload 与 cycle model 不变的前提下，
将 axcore module 的 Dynamic Power 缩放为 baseline 的 X% 后，
W4-FP16 Geomean 的 normalized Core energy 从 A 降到 B，
normalized Total energy 从 C 降到 D。
这说明若 AM-LUT 的乘法器级 dynamic power 降低能够传导到 PE/module 级，
则 system-level total energy 的上限收益约为 ...
```

避免这样表述：

```text
AM-LUT 能让 LLM 推理加速 X 倍。
AM-LUT 能让模型精度保持不变。
AM-LUT 已经完成硬件替换。
```

第一版实验只能支持 energy sensitivity，不支持 latency、accuracy、RTL validity 结论。

## 9. 何时进入下一层

只有当参数层实验给出清晰收益，才进入下一层：

1. 如果 energy benefit 明显：
   - 继续做六配置完整 sweep；
   - 加入 area/power/delay 来自 AM-LUT 真实综合结果；
   - 考虑写技术报告。

2. 如果 benefit 很小：
   - 说明 core energy 不是当前瓶颈，或者 AM-LUT 降幅无法传导到 total energy；
   - 需要转向 workload、memory、precision mapping 或 attention/KV cache 问题。

3. 如果要讨论精度：
   - 进入 `Software/AxCore` 或自建 approximate arithmetic accuracy route。

4. 如果要讨论硬件真实性：
   - 进入 SpinalHDL/RTL 或 FPGA LUT-level implementation route。

