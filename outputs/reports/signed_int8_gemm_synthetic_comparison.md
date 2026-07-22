# Signed INT8 Synthetic GEMM 统一对比报告

## 实验口径

- GEMM shape：`M=64, K=128, N=64`。
- 随机种子：`20260705`。
- exact baseline：标准 signed int8 乘法 + int32 累加。
- approximate GEMM：每个标量乘法从对应 signed int8 product LUT 查表，再做 int32 累加。
- 本报告使用 synthetic distribution，只用于快速诊断误差累加趋势；正式 LLM 结论还需要真实 activation/weight 分布与端到端评估。

## 指标说明

记 exact 输出为 `y_i`，approx 输出为 `yhat_i`，误差为 `e_i`：

```text
e_i = yhat_i - y_i

error_rate = count(yhat_i != y_i) / N
MAE        = (sum |e_i|) / N
RMSE       = sqrt((sum e_i^2) / N)
p99_abs    = percentile(|e_i|, 99)
max_abs    = max |e_i|
rel_l2     = sqrt(sum e_i^2) / sqrt(sum y_i^2)
```

- `RMSE` 不是平均相对均方误差；它是均方根误差，会更重视少数大错误。
- `rel_l2` 才是相对量，用来衡量整体输出向量或矩阵相对 exact 的偏离比例。

## 输入分布说明

- `uniform_int8`：`A` 和 `B` 都从 `-128` 到 `127` 均匀随机采样。这个分布覆盖全输入空间，偏向硬件压力测试，不代表真实 LLM 分布。
- `small_normal`：`A` 和 `B` 都从均值为 0、标准差为 16 的正态分布采样，四舍五入并裁剪到 int8。它模拟大量值集中在 0 附近的量化张量。
- `sparse_small`：先从均值为 0、标准差为 12 的正态分布采样，再随机把约 70% 元素置 0。它用来观察零值/小值很多时误差是否会被放大。
- `outlier_channels`：主体来自均值为 0、标准差为 10 的正态分布，再注入约 2% 的大幅值 outlier，幅值在 `96` 到 `127`。它模拟 LLM activation outlier 对近似乘法的压力。
- `nonnegative_activation_x_weight`：`A` 从 `0` 到 `127` 非负均匀采样，`B` 使用 `small_normal`。它模拟非负 activation 与 signed weight 相乘的情况，但不一定代表所有 LLM linear 层。

## 对比结果

### uniform_int8

| 设计 | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 2155.164 | 2850.676 | 8169 | 10880 | 0.045850 |
| TCASI24 CSAM2 | 7678.795 | 9683.115 | 25083 | 40224 | 0.155742 |
| FPGA cand17 | 1324.707 | 1669.031 | 4388 | 5850 | 0.026845 |
| FPGA cand20 | 1842.482 | 2316.229 | 5950 | 8996 | 0.037254 |
| FPGA cand10 | 1834.648 | 2305.316 | 5905 | 9768 | 0.037078 |

### small_normal

| 设计 | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 9.055 | 12.013 | 32 | 48 | 0.004185 |
| TCASI24 CSAM2 | 111.746 | 143.486 | 392 | 592 | 0.049992 |
| FPGA cand17 | 340.906 | 435.666 | 1160 | 1909 | 0.151790 |
| FPGA cand20 | 359.870 | 457.002 | 1216 | 1836 | 0.159224 |
| FPGA cand10 | 358.754 | 454.706 | 1206 | 1960 | 0.158424 |

### sparse_small

| 设计 | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 1.223 | 3.274 | 8 | 24 | 0.006783 |
| TCASI24 CSAM2 | 23.258 | 36.782 | 120 | 168 | 0.076200 |
| FPGA cand17 | 46.475 | 65.799 | 220 | 504 | 0.136314 |
| FPGA cand20 | 47.208 | 69.834 | 244 | 468 | 0.144673 |
| FPGA cand10 | 52.839 | 76.444 | 250 | 532 | 0.158365 |

### outlier_channels

| 设计 | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 57.496 | 224.716 | 1947 | 2192 | 0.056692 |
| TCASI24 CSAM2 | 268.947 | 597.204 | 3796 | 6408 | 0.150663 |
| FPGA cand17 | 173.059 | 222.439 | 636 | 975 | 0.056117 |
| FPGA cand20 | 173.562 | 229.326 | 680 | 1264 | 0.057855 |
| FPGA cand10 | 188.125 | 249.550 | 776 | 1352 | 0.062957 |

### nonnegative_activation_x_weight

| 设计 | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 130.234 | 173.544 | 496 | 896 | 0.012572 |
| TCASI24 CSAM2 | 506.912 | 638.022 | 1664 | 2688 | 0.046221 |
| FPGA cand17 | 631.281 | 786.211 | 2001 | 3083 | 0.056957 |
| FPGA cand20 | 720.457 | 900.337 | 2293 | 3106 | 0.065224 |
| FPGA cand10 | 878.936 | 1105.210 | 2887 | 4612 | 0.080066 |

## 当前观察方式

- 如果某个设计 product-level 看起来不错，但 GEMM `rel_l2` 明显放大，说明误差在累加中没有抵消。
- 如果 `max_abs` 很大但 `rel_l2` 不高，说明主要风险可能集中在少数输出元素。
- 如果 small/sparse 分布下误差明显小于 uniform 分布，说明真实 LLM 分布可能比全范围随机测试更温和，但这需要真实数据验证。

## 为什么不同分布下结果差异明显

- `uniform_int8` 里大幅值输入很多，FPGA cand17 的 product-level `max_abs` 和 `RMSE` 比 LSAM1 更小，所以 GEMM 累加后 `rel_l2` 看起来可以略优。
- `small_normal` 和 `sparse_small` 里大量乘法发生在小幅值区域。LSAM1 的 product-level `error_rate` 很低，很多小乘法能保持 exact；FPGA cand17 的错误更频繁，小误差在 GEMM 里大量累加，所以劣化明显。
- `nonnegative_activation_x_weight` 会让符号组合更偏向固定模式，误差不一定像正负随机输入那样互相抵消，因此 cand17/20/10 的累加误差更容易显现。
- `outlier_channels` 中少量大值主导整体 L2 范数，cand17 的大误差更受控，所以 `rel_l2` 接近甚至略低于 LSAM1；但它的 `MAE` 仍更高，说明普通位置的小误差更多。
- 这说明当前 FPGA 候选的优势更像是“限制最坏误差”，而 LSAM1 的优势更像是“保护小值/稀疏区域”。真实 LLM 里哪种更重要，需要下一步用真实 activation/weight 分布验证。
