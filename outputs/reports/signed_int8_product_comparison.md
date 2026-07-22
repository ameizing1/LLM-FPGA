# Signed INT8 Product-Level 统一对比报告

## 实验口径

- 输入范围：`a,b in [-128,127]`，共 65536 个 signed int8 输入组合。
- exact baseline：标准 signed int8 乘法，输出按 int32 统计。
- TCASI24 LSAM1/CSAM2：当前项目里的 unsigned TCASI24 8x8 行为模型 + signed wrapper。
- FPGA cand17/20/10：直接仿真组里 Verilog unsigned core 生成 LUT，再使用 signed wrapper。
- 本报告只比较 product-level 精度，不包含硬件资源、时序或端到端 LLM 精度。

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

## 对比结果

| 设计 | error_rate | MAE | RMSE | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| TCASI24 LSAM1 | 0.076416 | 38.188 | 260.543 | 2048 | 2312 | 0.047705 |
| TCASI24 CSAM2 | 0.434753 | 304.938 | 849.350 | 4096 | 4624 | 0.155516 |
| FPGA cand17 | 0.745605 | 87.916 | 148.601 | 535 | 880 | 0.027209 |
| FPGA cand20 | 0.740479 | 116.821 | 206.425 | 708 | 930 | 0.037796 |
| FPGA cand10 | 0.690918 | 116.681 | 205.509 | 697 | 930 | 0.037629 |

## 当前观察

- TCASI24 LSAM1 的 `error_rate` 和 `MAE` 最低，说明它更少出错，平均绝对偏差也更小。
- FPGA cand17 的 `RMSE`、`max_abs` 和 `rel_l2` 更低，说明虽然出错更频繁，但大误差更受控。
- TCASI24 CSAM2 在当前 signed int8 口径下误差明显更大，后续可作为激进近似对照点。
- 因为各指标指向不同，不能只凭 product-level 宣称某个设计全面更好，需要继续看 GEMM 累加后的表现。
