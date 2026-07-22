# TCASI24 int8 AM-LUT error report

- GEMM shape: M=64, K=128, N=64
- RNG seed: 20260705
- Signed behavior: sign-magnitude wrapper around unsigned 8x8 TCASI24 blocks.
- Interpretation guardrail: this report compares error behavior only; it does not claim AM-LUT is better than AxCore.

## Product-level error

| mode | error_rate | mae | rmse | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 0.076416 | 38.188 | 260.543 | 2312 | 0.047705 |
| csam2 | 0.434753 | 304.938 | 849.350 | 4624 | 0.155516 |

## GEMM-level and distribution-sensitive error

### uniform_int8

| mode | mae | rmse | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 2155.164 | 2850.676 | 8169 | 10880 | 0.045850 |
| csam2 | 7678.795 | 9683.115 | 25083 | 40224 | 0.155742 |

### small_normal

| mode | mae | rmse | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 9.055 | 12.013 | 32 | 48 | 0.004185 |
| csam2 | 111.746 | 143.486 | 392 | 592 | 0.049992 |

### sparse_small

| mode | mae | rmse | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 1.223 | 3.274 | 8 | 24 | 0.006783 |
| csam2 | 23.258 | 36.782 | 120 | 168 | 0.076200 |

### outlier_channels

| mode | mae | rmse | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 57.496 | 224.716 | 1947 | 2192 | 0.056692 |
| csam2 | 268.947 | 597.204 | 3796 | 6408 | 0.150663 |

### nonnegative_activation_x_weight

| mode | mae | rmse | p99_abs | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| lsam1 | 130.234 | 173.544 | 496 | 896 | 0.012572 |
| csam2 | 506.912 | 638.022 | 1664 | 2688 | 0.046221 |
