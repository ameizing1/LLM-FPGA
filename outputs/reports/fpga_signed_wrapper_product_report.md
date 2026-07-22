# FPGA signed-wrapper int8 product-level report

- Source: Verilog-simulated unsigned `approx88_cascade` LUTs.
- Signed behavior: `abs(a), abs(b) -> unsigned core -> restore sign`.
- Scope: behavior-model precision only; this is not a signed RTL area/timing report.

## Overall Product-Level Error

| candidate | error_rate | mae | rmse | max_abs | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| cand10 | 0.690918 | 116.681 | 205.509 | 930 | 0.037629 |
| cand17 | 0.745605 | 87.916 | 148.601 | 880 | 0.027209 |
| cand20 | 0.740479 | 116.821 | 206.425 | 930 | 0.037796 |

## Bucketed Error

### nonnegative_x_nonnegative

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 16384 | 116.681 | 930 | 0.038075 |
| cand17 | 16384 | 87.916 | 880 | 0.027531 |
| cand20 | 16384 | 116.821 | 930 | 0.038245 |

### nonnegative_x_negative

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 16384 | 116.681 | 930 | 0.037631 |
| cand17 | 16384 | 87.916 | 880 | 0.027211 |
| cand20 | 16384 | 116.821 | 930 | 0.037799 |

### negative_x_nonnegative

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 16384 | 116.681 | 930 | 0.037631 |
| cand17 | 16384 | 87.916 | 880 | 0.027211 |
| cand20 | 16384 | 116.821 | 930 | 0.037799 |

### negative_x_negative

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 16384 | 116.681 | 930 | 0.037193 |
| cand17 | 16384 | 87.916 | 880 | 0.026894 |
| cand20 | 16384 | 116.821 | 930 | 0.037359 |

### has_minus128

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 511 | 0.000 | 0 | 0.000000 |
| cand17 | 511 | 0.000 | 0 | 0.000000 |
| cand20 | 511 | 0.000 | 0 | 0.000000 |

### small_magnitude_le_16

| candidate | cases | mae | max_abs | rel_l2 |
| --- | --- | --- | --- | --- |
| cand10 | 1089 | 7.258 | 98 | 0.192588 |
| cand17 | 1089 | 6.810 | 80 | 0.156754 |
| cand20 | 1089 | 6.024 | 48 | 0.127756 |
