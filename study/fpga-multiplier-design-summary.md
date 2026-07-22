# FPGA_multiplier design summary

更新时间：2026-07-22

## 1. 目录内容

`FPGA_multiplier/` 里有三类设计：

| 目录 | 内容 | 作用 |
|---|---|---|
| `accurate_unsigned8x8/` | 两套 unsigned 8x8 精确乘法器结构 | unsigned 近似乘法器的结构/资源基线 |
| `accurate_signed8x8/` | 两套 signed 8x8 精确乘法器结构 | signed 扩展和二补码处理参考 |
| `approx_unsigned8x8/` | 21 个 unsigned 8x8 近似乘法器候选 | 当前最重要的候选设计池 |

`approx_unsigned8x8` 中每个编号目录都包含最终候选：

```text
final_best_approx88_cascade.v
final_best_approx88_cascade_inits.json
pipeline.log
```

其中 Verilog 顶层模块统一为：

```verilog
module approx88_cascade(input wire [7:0] a, input wire [7:0] b, output wire [15:0] prod);
```

## 2. 近似乘法器结构

以候选 17 为例，设计采用 6+2 位分解：

```text
a = {ah[1:0], al[5:0]}
b = {bh[1:0], bl[5:0]}

ll = approx66_cross_low(al, bl)
hl = approx62_hl(bl, ah)
lh = approx62_lh(al, bh)
hh = LUT-based 2x2 high product

prod = comp88_cascade(hh, hl, lh, ll)
```

也就是说，它不是传统 array multiplier，而是：

1. 用多个 LUT6/LUT6_2 近似生成局部乘积；
2. 用 LUT compressor 近似合并 partial products；
3. 最后用少量 CARRY4 做级联加法/压缩。

候选 17 的 Verilog 注释给出的 unsigned 8x8 全输入误差为：

```text
MRED = 0.0260110648
MED  = 241.1925964355
ER   = 0.7973175049
WCE  = 4034
```

## 3. 21 个候选的最终指标

以下指标来自各目录 `pipeline.log` 的最后记录。

| id | MRED | MED | ER | WCE | err/65536 | bits |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.0260186669 | 241.3112 | 0.7974 | 4034 | 52257 | 1136 |
| 8 | 0.0271779703 | 244.7915 | 0.8192 | 4034 | 53685 | 1112 |
| 12 | 0.0272202312 | 251.0897 | 0.7898 | 4034 | 51762 | 1156 |
| 3 | 0.0276599765 | 264.5730 | 0.7511 | 4290 | 49221 | 1016 |
| 20 | 0.0281388163 | 267.4724 | 0.7866 | 4290 | 51551 | 984 |
| 14 | 0.0281690565 | 266.2338 | 0.7811 | 4034 | 51187 | 1098 |
| 10 | 0.0283623508 | 266.4849 | 0.7588 | 4034 | 49731 | 1068 |
| 11 | 0.0284454762 | 262.1297 | 0.7920 | 4034 | 51904 | 1026 |
| 19 | 0.0284777787 | 257.8294 | 0.7912 | 4034 | 51854 | 1050 |
| 21 | 0.0296552847 | 269.5532 | 0.7891 | 4034 | 51714 | 1082 |
| 9 | 0.0297206482 | 272.0694 | 0.7999 | 4290 | 52423 | 1064 |
| 4 | 0.0299753156 | 260.3384 | 0.8251 | 4034 | 54075 | 1056 |
| 6 | 0.0299753156 | 260.3384 | 0.8251 | 4034 | 54075 | 1056 |
| 15 | 0.0302424623 | 262.5477 | 0.8330 | 4034 | 54594 | 1070 |
| 5 | 0.0311184748 | 274.9394 | 0.7777 | 4034 | 50970 | 1052 |
| 7 | 0.0311184748 | 274.9394 | 0.7777 | 4034 | 50970 | 1052 |
| 2 | 0.0311391529 | 257.0252 | 0.8317 | 4290 | 54507 | 1014 |
| 18 | 0.0315587428 | 266.8071 | 0.8252 | 4034 | 54078 | 1074 |
| 16 | 0.0317362830 | 265.2734 | 0.8784 | 4034 | 57567 | 1054 |
| 1 | 0.0317619759 | 276.9301 | 0.8417 | 4290 | 55164 | 1136 |
| 13 | 0.0327730925 | 278.0700 | 0.8373 | 4034 | 54876 | 1050 |

## 4. 候选选择建议

当前不建议只选一个候选。建议保留三个代表点：

| 用途 | 推荐候选 | 原因 |
|---|---:|---|
| 主候选，误差最优 | 17 | MRED 和 MED 最低，适合作为论文主线候选 |
| 低资源候选 | 20 | `bits=984` 最低，但 WCE=4290，适合做硬件开销优先对照 |
| 低错误率候选 | 3 或 10 | 3 的 ER 最低但 WCE=4290；10 的 ER 较低且 WCE=4034 |

第一轮 W8A8 实验建议至少测：

```text
exact int8
candidate 17
candidate 20
candidate 10
TCASI24 LSAM1
TCASI24 CSAM2
```

这样能覆盖：

1. 最低平均相对误差；
2. 最低结构 bit 数；
3. 较低错误率；
4. TCASI24 保守/激进对照。

## 5. 结构资源粗估

直接统计 Verilog 原语实例，得到：

| 设计 | LUT6_2 | LUT6 | CARRY4 |
|---|---:|---:|---:|
| approx candidate 17 | 32 | 5 | 3 |
| approx candidate 20 | 32 | 5 | 3 |
| approx candidate 3 | 32 | 5 | 3 |
| accurate unsigned8x8/1 | 17 | 0 | 6 |
| accurate unsigned8x8/2 | 16 | 4 | 5 |
| accurate signed8x8/1 | 30 | 0 | 14 |
| accurate signed8x8/2 | 25 | 4 | 12 |

注意：这个统计只是结构粗估，不等于 Vivado 综合后的 LUT 数。最终论文表格需要用同一 FPGA、同一综合约束重新综合，避免被优化策略影响公平性。

## 6. Signed 扩展建议

当前 `approx_unsigned8x8` 是 unsigned 8x8。若用于标准 W8A8，需要扩展到二补码 signed int8。

推荐第一版：

```text
sign = a[7] xor b[7]
mag_a = abs(a)
mag_b = abs(b)
mag_p = approx88_cascade(mag_a, mag_b)
prod = sign ? -mag_p : mag_p
```

原因：

1. 标准 W8A8 的 int8 通常是二补码，不是 sign-magnitude；
2. 不能简单使用 `a[6:0]` 和 `b[6:0]` 当幅值；
3. `abs(-128)=128`，所以仍需要 8-bit magnitude，不建议第一版改成 7x7；
4. signed wrapper 的硬件开销要单独统计，包括两个 abs、一个 sign xor、一个 conditional negate。

`accurate_signed8x8/` 里的 signed 精确设计可以作为二补码 signed 结构参考，但对近似版本，第一阶段建议先用 wrapper 复用 unsigned core，降低风险。

## 7. 下一步实验

建议按以下顺序推进：

1. 从候选 17/20/10 生成 unsigned lookup table；
2. 加 signed wrapper，生成 signed int8 lookup table；
3. 跑 product-level error，对比 exact、TCASI24 LSAM1、TCASI24 CSAM2；
4. 跑 W8A8 GEMM-level error；
5. 若 GEMM 误差可接受，再接小模型或 calibration set 做 W8A8 推理误差；
6. 最后再做统一 Vivado 综合，报告 LUT/CARRY/Delay。

当前最重要的判断不是继续盲目优化乘法器，而是先验证：

```text
候选 17/20/10 在 signed W8A8 GEMM 中是否仍然保持可接受误差。
```
