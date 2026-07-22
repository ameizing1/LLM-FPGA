# AxCore 最小实验推进日志

更新时间：2026-06-27 12:06 +08:00

## 背景

当前目标：逐步探索 AxCore 是否能在本地跑通最小实验，不一开始就直接投入完整的软件、仿真器或硬件复现实验。

当前研究主线：

LLM 低比特量化 -> 近似/低成本 GEMM -> FPGA LUT-based approximate multiplier -> 软硬件协同评估。

本地 AxCore 路径：

`C:\Users\LiuZhiWei\Documents\Codex\2026-06-17\ai\work\axcore`

仓库获取方式：

- PowerShell 中 `git` 不可用。
- 因此使用 GitHub zip 下载，并解压到 `work\axcore`。

## 仓库结构

- `Profile`：轻量级 Figure 2 workload profile 脚本。
- `Software\AxCore`：基于 PyTorch 的困惑度和 zero-shot 评估框架。
- `Software\axcore_simulator`：性能/能耗仿真器。
- `Hardware\AxCore`：SpinalHDL 硬件设计与测试。

## 本地环境观察

- 本地 Python：3.13.9。
- AxCore README 预期环境：Ubuntu 22.04.5、Conda 25.1.1、Python 3.9、gcc 11.4。
- 首次运行前检查到的 Python 包状态：
  - `numpy`：已安装
  - `matplotlib`：已安装
  - `pandas`：已安装
  - `graphviz`：未安装

## 路线判断

### Profile 路线

风险低。它不需要 GPU、Hugging Face 模型下载、CUDA 或 Linux-only 构建工具。

预期入口：

- `Profile\fig2_cal.py`：计算 OPT-175B 和 LLaMA-3.1-405B 在不同序列长度下 Linear 与 Attention 的 OP 占比。
- `Profile\fig2.py`：使用固定数据绘制 Figure 2。

### Software evaluation 路线

环境和资源成本高。README 要求 PyTorch 2.5.1 + CUDA 12.4。脚本会自动下载 Hugging Face 模型和数据集。即使较小的 perplexity 脚本，README 也标注约 30 分钟和 6 GB 显存；更大的模型需要更多 VRAM/RAM。

当前决定：不在未讨论前运行这条路线。

### Simulator 路线

中等风险。它看起来偏 CPU，但 README 预期 Ubuntu/gcc/make，并要求从源码构建 Cacti。在 Windows 上可能不顺。

当前决定：先静态检查；在安装/编译 Cacti 或改变环境前先讨论。

### Hardware 路线

环境成本高。README 推荐 Docker image，并需要 SBT/Iverilog。这更适合作为后续步骤。

当前决定：不在未讨论前运行。

## 2026-06-27 Profile 运行结果

命令：

- `python fig2_cal.py`
- `$env:MPLBACKEND='Agg'; python fig2.py`

结果：

- `fig2_cal.py` 成功。
- `fig2.py` 使用非交互后端成功。第一次直接运行时因为 `plt.show()` 等待交互式窗口而超时；设置 `MPLBACKEND=Agg` 后解决。
- 生成/更新文件：`work\axcore\Profile\figure2.pdf`。

`fig2_cal.py` 的关键输出：

- OPT-175B 的 Linear OPs 从 1k 序列长度下的 98.63% 降到 32k 下的 69.23%。
- OPT-175B 的 Attention OPs 从 1k 下的 1.37% 升到 32k 下的 30.77%。
- LLaMA-3.1-405B 的 Linear OPs 从 1k 下的 99.10% 降到 32k 下的 77.46%。
- LLaMA-3.1-405B 的 Attention OPs 从 1k 下的 0.90% 升到 32k 下的 22.54%。

解释：

这复现了 AxCore 中较轻量的 workload motivation 部分。结果支持这样一个判断：即使上下文长度增长，Linear/GEMM 仍然是重要的计算目标。

## 2026-06-27 Simulator 静态检查

发现：

- `Software\axcore_simulator\requirements.txt` 列出了 `pandas`、`graphviz`、`matplotlib`。
- 本地 Python 已有 `pandas` 和 `matplotlib`，但缺少 Python `graphviz` 包。
- `python run_axcore.py --help` 最初失败，因为在参数解析前会导入 `dnnweaver2.graph`，而该模块需要 `graphviz.Digraph`。
- 仓库中已有 `AxCore\sram\cacti_sweep.csv` 和 `bitfusion\sram\cacti_sweep.csv`。
- `CactiSweep.__init__` 在缺少 Cacti 二进制时会打印警告；如果所需 SRAM 配置已经在 CSV 缓存中，理论上可以不调用 Cacti。

临时处理：

- 将缺失的 `graphviz` Python 包安装到本地依赖目录 `work\axcore\.pydeps`。
- 从 `Software\axcore_simulator` 使用临时 `PYTHONPATH=..\..\.pydeps` 加载该依赖。
- 这样没有污染全局 Python 环境。

结果：

- `python run_axcore.py --help` 成功。
- 创建 `AxCoreSimulator` 对象成功：
  - N = 64
  - M = 64
  - high precision = 8
  - low precision = 4
- simulator 会提示 Cacti 二进制缺失，但对象创建本身不会失败。

## 2026-06-27 单配置 simulator 尝试

命令：

`python run_axcore.py --synth_csv ./params/systolic_array_synth_W4-FP16.csv`

结果：

- 失败位置：SRAM energy lookup。
- 代码尝试调用 Cacti，因为在 `AxCore\sram\cacti_sweep.csv` 中没有找到匹配条目。
- 由于本地没有 Cacti 二进制，最终报错：

`AssertionError: Can't run cacti, no binary found. Please clone and compile cacti first.`

重要细节：

- `CactiSweep.__init__` 里目前写着 `if os.path.isfile(self.csv_file) and False`，所以它实际上故意不读取已有 CSV 缓存。
- 即使把这个判断改成读取缓存，当前 `AxCore\sram\cacti_sweep.csv` 也没有命中第一次需要的 AxCore SRAM 配置：
  - `size (bytes) = 2048`
  - `block size (bytes) = 128`
  - `read-write port = 0`
  - `technology (u) = 0.028`

解释：

Simulator 路线目前已经不再卡在 Python import。真正的阻塞点是 memory-model data generation。要继续严谨推进，需要二选一：

1. 按 AxCore README 的方式构建并使用 Cacti。
2. 做一个本地近似/兼容补丁，为 SRAM energy 提供近似值，并明确标注这是 bring-up hack，不是论文级忠实复现。

当前决策点：

先停在这里讨论。不要在未确认前修改 simulator 行为，也不要直接安装/编译 Cacti。

## 2026-06-27 WSL 检查

用户希望先尝试“忠实复现实验”，并提到之前可能配置过 WSL。

已执行检查：

- `wsl.exe -l -v`
- `wsl.exe --status`
- `wsl.exe --list --quiet`
- 检查注册表：`HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss`
- 检查本地应用包目录中是否存在 Ubuntu/Debian/Canonical 相关包

结果：

- Windows 上存在 `wsl.exe`。
- WSL 默认版本显示为 2。
- 当前没有已注册的 Linux 发行版。
- 注册表中没有发现已注册的 WSL 发行版。
- `LOCALAPPDATA\Packages` 中没有发现 Ubuntu/Debian/Canonical 相关发行版包。

解释：

本机 WSL 功能可用，但目前没有可直接进入的 Ubuntu/Debian 环境。之前可能配置过 WSL，但当前系统状态下发行版已经不存在、未注册，或被迁移/卸载。

忠实复现实验的下一步：

安装 `Ubuntu-22.04` WSL 发行版。AxCore README 预期 Ubuntu 22.04.5，因此 Ubuntu 22.04 是最接近的本地复现环境。

注意：

安装发行版会改变系统状态、占用磁盘空间，并且首次启动需要创建 Linux 用户名和密码。因此这里先停下确认，不擅自安装。

## 2026-06-27 Ubuntu-22.04 WSL 安装与网络状态

用户确认继续推进忠实复现实验，并询问磁盘空间是否足够。

磁盘空间：

- C 盘可用约 144 GB。
- D 盘可用约 160 GB。
- Ubuntu WSL 根文件系统显示约 955 GB 可用。

结论：安装 Ubuntu-22.04、Cacti 和 AxCore simulator 依赖的空间足够。后续如果下载大模型、CUDA 镜像或大量数据集，则需要单独评估。

已执行：

- `wsl.exe --install Ubuntu-22.04`

过程：

- 安装命令第一次超过 2 分钟未返回，但随后检查发现 `Ubuntu-22.04` 已注册。
- 期间残留了多个 `wsl --install` 和 `wsl -d Ubuntu-22.04 ...` 客户端进程，导致 `wsl --shutdown` 和若干检查命令卡住。
- 已手动结束这些卡住的 `wsl.exe` 客户端进程。
- 之后 Ubuntu 可以正常非交互进入。

当前 WSL 发行版：

- 名称：`Ubuntu-22.04`
- WSL 版本：2
- 系统版本：Ubuntu 22.04.5 LTS

基础工具链状态：

- `python3`：已存在，版本 3.10.12。
- `git`：已存在，版本 2.34.1。
- `pip3`：缺失。
- `gcc`：缺失。
- `make`：缺失。

当前阻塞点：

WSL 内部网络存在问题：

- 可以 ping 通 `8.8.8.8`，说明基础 ICMP 网络存在。
- DNS 可以解析域名。
- 但 WSL 内部访问 HTTP/HTTPS TCP 连接超时：
  - Ubuntu 官方源超时。
  - 清华源超时。
  - 阿里源/中科大源超时。
  - 百度 HTTP 也超时。
- 因此当前无法通过 `apt-get update` 和 `apt-get install` 安装 `build-essential`、`python3-pip`、`python3-venv` 等。

代理检查：

- Windows 上存在 Clash Verge/Mihomo 相关进程。
- Windows 本机 `127.0.0.1:7897` 可连通。
- `netstat` 显示代理只监听 `127.0.0.1:7897`。
- WSL 无法访问 `127.0.0.1:7897`，因为这指向 WSL 自己。
- WSL 也无法访问 Windows WSL 网关 `172.26.0.1:7897`，因为代理没有监听局域网地址。

建议下一步：

在 Clash Verge/Mihomo 中开启类似“Allow LAN / 允许局域网连接”的设置，使代理监听 `0.0.0.0:7897`，然后在 WSL 中临时设置：

`http_proxy=http://172.26.0.1:7897`

`https_proxy=http://172.26.0.1:7897`

之后再重新执行 `apt-get update` 和依赖安装。

## 2026-06-28 WSL 网络与工具链补充

用户补充要求：后续凡是可能影响 Codex 网络的操作必须谨慎处理。

当前处理原则：

- 不再随意修改 Windows/WSL 网络配置。
- 不在未确认前启用或关闭代理/TUN/防火墙等全局网络功能。
- 后续只使用当前已经可用的网络状态下载项目依赖。

已知网络相关变更：

- 曾新增 `C:\Users\LiuZhiWei\.wslconfig`，内容为：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
firewall=true
```

作用：

- 使 WSL 从 NAT 模式切到 mirrored networking。
- WSL 之后可以访问 Windows 本机的 `127.0.0.1:7897` 代理端口。
- 解决了此前 WSL 内部 HTTP/HTTPS 连接超时的问题。

注意：

- 这是全局 WSL 配置，会影响 WSL 网络行为。
- 当前不继续改动该配置。

当前 WSL 工具链状态：

- Ubuntu：22.04.5 LTS
- Python：3.10.12
- pip：22.0.2
- gcc：11.4.0
- make：4.3
- git：2.34.1

## 2026-06-28 Cacti 构建

工作目录：

`/mnt/c/Users/LiuZhiWei/Documents/Codex/2026-06-17/ai/work/axcore/Software/axcore_simulator`

执行步骤：

1. 确认 `bitfusion/sram/cacti` 和 `AxCore/sram/cacti` 均不存在。
2. 执行：

`git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti`

3. 执行：

`make -C ./bitfusion/sram/cacti`

4. 编译完成后复制到 AxCore SRAM 目录：

`cp -r ./bitfusion/sram/cacti ./AxCore/sram/`

结果：

- `bitfusion/sram/cacti/cacti` 已生成。
- `AxCore/sram/cacti/cacti` 已生成。
- 二进制大小约 3.5 MB。
- `./bitfusion/sram/cacti/cacti -help` 可以执行并输出 CACTI 参数说明。

编译输出中存在 C++ warning，但没有导致构建失败。

## 2026-06-28 Python 环境

为 simulator 创建了隔离虚拟环境：

`Software/axcore_simulator/.venv`

安装依赖时遇到的问题：

- 第一次 `pip install -r requirements.txt` 超时。
- 中断后 `pandas` 处于不完整状态，`import pandas` 报错：

`ImportError: C extension: pandas.util not built.`

处理方式：

- 清理残留 pip 进程。
- 删除损坏的 pandas/numpy 包目录。
- 使用清华 PyPI 镜像安装固定版本：

`numpy==2.0.2`

`pandas==2.2.3`

最终 Python 依赖状态：

- `numpy 2.0.2`
- `pandas 2.2.3`
- `graphviz 0.21`
- `matplotlib 3.10.9`

验证结果：

- `import numpy/pandas/graphviz/matplotlib` 均成功。
- `.venv/bin/python run_axcore.py --help` 成功。

## 2026-06-28 单配置 simulator 验证

命令：

`mkdir -p results && : > results/axcore_res.csv && .venv/bin/python run_axcore.py --synth_csv ./params/systolic_array_synth_W4-FP16.csv`

结果：

- 成功运行。
- Cacti 被实际调用。
- 生成 `results/axcore_res.csv`。
- 单配置运行耗时约 91 秒。

观察：

- 运行过程中多次出现 `No entry found ... running cacti`。
- 这说明 simulator 能正确调用 Cacti。
- 但 `AxCore/sram/cacti_sweep.csv` 没有明显保留所有新增查询项，后续重复运行时仍可能重复调用 Cacti。

## 2026-06-28 完整 Figure 17 复现实验

命令：

`cd /mnt/c/Users/LiuZhiWei/Documents/Codex/2026-06-17/ai/work/axcore/Software/axcore_simulator && . .venv/bin/activate && sh scripts/fig_17.sh`

结果：

- 成功运行完成。
- 总耗时约 431 秒。
- 生成文件：
  - `results/axcore_res.csv`，约 4.9 KB
  - `results/fig_17.pdf`，约 36 KB

`results/axcore_res.csv` 包含 6 组配置：

- `systolic_array_synth_W4-FP16`
- `systolic_array_synth_W4-BF16`
- `systolic_array_synth_W4-FP32`
- `systolic_array_synth_W8-FP16`
- `systolic_array_synth_W8-BF16`
- `systolic_array_synth_W8-FP32`

每组配置包含：

- `Time`
- `Static`
- `Dram`
- `Buffer`
- `Core`

并覆盖：

- `Opt13B`
- `Opt30B`
- `Geomean`

运行过程中的非致命信息：

- 多次 Cacti cache miss，导致重复调用 Cacti。
- `matplotlib` 提示缺少 `Times New Roman` 字体，因此绘图时发生字体 fallback。
- `pandas` 有 FutureWarning，不影响本次运行结果。

当前状态：

- AxCore simulator 的 Figure 17 路线已经跑通。
- 当前结果属于按 README 方式编译 Cacti 后的较忠实复现。
- 仍需进一步检查 `fig_17.pdf` 的视觉效果以及是否需要安装 Times New Roman 或替代字体来减少字体警告。

建议下一步：

1. 打开/渲染 `results/fig_17.pdf`，检查图是否完整。
2. 记录 Figure 17 的结果结构和关键数值。
3. 阅读 `run_axcore.py` 和 `EnergyAll.py`，明确 Time、Static、Dram、Buffer、Core 每项的计算路径。
4. 再考虑是否修复 Cacti cache 读取逻辑，减少后续重复调用。
## 2026-06-28 继续推进：代码路径解释与 AM-LUT 参数层实验草案

本轮没有修改网络、代理、WSL 配置，也没有直接运行新的 simulator 实验。

完成内容：

- 新增代码阅读笔记：`study/axcore-simulator-code-reading-notes.md`
- 新增 AM-LUT 参数层实验方案：`study/axcore-amlut-parameter-sensitivity-plan.md`
- 新增可执行脚本草案：`work/axcore/Software/axcore_simulator/experiments/amlut_sensitivity/amlut_sensitivity.py`
- 新增脚本说明：`work/axcore/Software/axcore_simulator/experiments/amlut_sensitivity/README.md`

关键修正认识：

- Figure 17 中五个架构 config 的主要差异是 `[module] module = ...`。
- `get_energy_cost()` 在没有完整 `N/M` 行时，会用 `Module + N=1 + M=1` 查 synthesis CSV，然后按 `N*M` 放大 area/leakage/dynamic power。
- `get_conv_cycles()` 会读取当前 module 行的 `Weight (bits)` 和 `Activation (bits)`，并覆盖 layer 原始 precision。
- `Stats.get_energy_breakdown()` 中 `Buffer` 只计 SRAM read/write dynamic energy，`sram_leak_energy` 被注释掉。
- Cacti cache 反复 miss 的直接原因之一是 `CactiSweep.__init__` 中 `if os.path.isfile(self.csv_file) and False` 导致不会读取已有 CSV cache。

AM-LUT 第一版实验边界：

- 只复制 synthesis CSV，并修改 `axcore` 行的 `Area (um^2)`、`Leakage Power (nW)`、`Dynamic Power (nW)`。
- 不新增第六个绘图架构，不改 `EnergyAll.py`。
- 不改 cycle model，不改 benchmark，不改 RTL，不做 accuracy/perplexity。
- 第一轮建议只跑 `W4-FP16` 单配置、dynamic-only 或 leakage-only sensitivity。

脚本验证：

```bash
cd /mnt/c/Users/LiuZhiWei/Documents/Codex/2026-06-17/ai/work/axcore/Software/axcore_simulator
. .venv/bin/activate
python -m py_compile experiments/amlut_sensitivity/amlut_sensitivity.py
python experiments/amlut_sensitivity/amlut_sensitivity.py --config W4-FP16 --dynamic-scale 0.8 --leakage-scale 1.0 --area-scale 1.0
```

验证结果：

- dry-run 成功。
- 已生成参数文件：`experiments/amlut_sensitivity/params/W4-FP16_dyn0p8_leak1_area1.csv`
- diff 显示只改变了 `axcore` 行的 dynamic/total power 数值和格式化空白。
- 未执行 `--run`，所以没有启动新 simulator 实验。

下一步建议：

1. 用户阅读 AxCore 和背景知识时，可重点对照 `axcore-simulator-code-reading-notes.md`。
2. 如果决定跑第一组 AM-LUT 参数层实验，先运行单点：

```bash
python experiments/amlut_sensitivity/amlut_sensitivity.py --config W4-FP16 --dynamic-scale 0.8 --leakage-scale 1.0 --area-scale 1.0 --run
```

3. 单点结果合理后，再扫 dynamic-only：

```text
dynamic_scale = 1.0 / 0.9 / 0.8 / 0.7 / 0.5
leakage_scale = 1.0
area_scale = 1.0
config = W4-FP16
```
