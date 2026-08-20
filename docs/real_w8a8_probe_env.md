# real W8A8 distribution probe 环境说明

更新时间：2026-07-24

## 环境位置

由于项目路径较深，直接在项目根目录创建 venv 安装 PyTorch 会触发 Windows 路径过长错误：

```text
WinError 206: 文件名或扩展名太长
```

因此当前已创建短路径虚拟环境：

```text
C:\venvs\llmfpga-probe
```

Python 版本：

```text
Python 3.12.13
```

## 已安装关键依赖

已验证可 import：

```text
numpy==2.5.1
torch==2.13.0+cpu
transformers==5.14.1
datasets==5.0.0
```

`torch.cuda.is_available()` 当前为 `False`，即该环境是 CPU 运行环境。

## 运行脚本

在项目根目录执行：

```powershell
& 'C:\venvs\llmfpga-probe\Scripts\python.exe' scripts\run_real_w8a8_distribution_probe.py
```

建议第一次先跑小规模 smoke：

```powershell
& 'C:\venvs\llmfpga-probe\Scripts\python.exe' scripts\run_real_w8a8_distribution_probe.py `
  --dataset Salesforce/wikitext `
  --text-samples 1 `
  --max-seq-len 32 `
  --max-linear-layers 1 `
  --max-rows-per-layer 32 `
  --max-cols-per-layer 32 `
  --product-pairs-per-layer 1000 `
  --out-dir outputs\reports\real_w8a8_env_smoke
```

## 已验证内容

1. Python 包 import 通过：
   - `numpy`
   - `torch`
   - `transformers`
   - `datasets`
2. 脚本参数解析通过：
   - `scripts\run_real_w8a8_distribution_probe.py --help`
3. 所需 LUT 文件存在并可读取：
   - `outputs/luts/lsam1_int8_lut.npy`
   - `outputs/fpga_luts/fpga_cand17_signed_wrapper_int8_lut.npy`
   - `outputs/hybrid_luts/cand17_exact_if_min_abs_le_32.npy`
   - `outputs/hybrid_luts/cand17_lsam1_if_min_abs_le_32.npy`

## 当前剩余阻塞点

代理开启后，小规模 smoke 已经跑通，并生成：

```text
outputs\reports\real_w8a8_env_smoke\real_w8a8_distribution_probe.json
outputs\reports\real_w8a8_env_smoke\real_w8a8_distribution_probe.md
```

注意：当前 `datasets==5.0.0` 下，短数据集名 `wikitext` 会报 HF URI 解析错误。运行时请显式使用：

```text
--dataset Salesforce/wikitext
```

如果后续要使用默认参数运行，建议先把脚本默认 `--dataset` 从 `wikitext` 改成 `Salesforce/wikitext`，或每次命令行显式传入该参数。

## 不建议自动改动的内容

不要为了这一步直接自动改：

- WSL 网络配置
- Windows 代理配置
- Clash/TUN 配置
- 防火墙规则

更稳妥的下一步是二选一：

1. 手动确认 Hugging Face 访问可用后，直接用上面的命令重跑。
2. 准备本地模型目录和本地数据集，然后用 `--model <local_path>` 等参数绕开在线下载。
