# Next Experiment Roadmap: TCASI24, FPGA Candidates, and AxCore

Updated: 2026-07-22

## What Can Be Compared Now

We can already compare product-level signed int8 behavior between:

- exact signed int8 multiplication;
- TCASI24 LSAM1 / CSAM2 signed-wrapper behavior model;
- FPGA candidate 17 / 20 / 10 signed-wrapper behavior model.

This comparison is fair at the behavior-model level because all methods are evaluated with the same signed int8 input space:

```text
a, b in [-128, 127]
total cases = 65536
```

Current product-level results:

| design | error_rate | MAE | RMSE | max_abs | relative L2 |
|---|---:|---:|---:|---:|---:|
| TCASI24 LSAM1 | 0.076416 | 38.188 | 260.543 | 2312 | 0.047705 |
| TCASI24 CSAM2 | 0.434753 | 304.938 | 849.350 | 4624 | 0.155516 |
| FPGA cand17 | 0.745605 | 87.916 | 148.601 | 880 | 0.027209 |
| FPGA cand20 | 0.740479 | 116.821 | 206.425 | 930 | 0.037796 |
| FPGA cand10 | 0.690918 | 116.681 | 205.509 | 930 | 0.037629 |

Initial interpretation:

- LSAM1 has much lower error rate and lower MAE.
- FPGA cand17 has higher error rate but lower RMSE, lower max absolute error, and lower relative L2 error than LSAM1.
- CSAM2 is clearly the aggressive TCASI24 point and currently looks much worse than cand17/20/10 under this signed int8 product test.

The useful story is not "one design wins on every metric." It is:

```text
LSAM1 makes fewer product mistakes, while cand17 makes more frequent but more bounded mistakes.
```

This is exactly why GEMM/layer tests matter next.

## What Should Not Be Claimed Yet

Do not directly claim:

```text
our FPGA candidate is better than TCASI24
```

A safer claim is:

```text
under the current signed-wrapper product-level test, FPGA cand17 shows lower RMSE/WCE/relative-L2 than TCASI24 LSAM1, but LSAM1 has lower error-rate/MAE.
```

Also do not compare AxCore at product-level. AxCore is not just a standalone signed int8 multiplier design; it is a quantization-aware approximate GEMM unit and software/hardware evaluation framework.

## Comparison Roles

| Level | Main comparison | Why |
|---|---|---|
| product-level | TCASI24 vs FPGA candidates | Both are multiplier-level approximate arithmetic designs. |
| GEMM/layer-level | exact W8A8 vs FPGA candidates, optionally TCASI24 | This tests whether product errors accumulate acceptably. |
| end-to-end | AxCore baseline vs our inserted multiplier method | AxCore reports PPL/zero-shot accuracy and performance/energy at system level. |

## Concrete Next Steps

### Step 1: Unified Product Report

Create one report that combines TCASI24 and FPGA candidates under the same signed int8 metrics.

Deliverable:

```text
outputs/reports/signed_int8_product_comparison.md
```

Expected table:

```text
exact, TCASI24-LSAM1, TCASI24-CSAM2, FPGA-cand17, FPGA-cand20, FPGA-cand10
```

### Step 2: Unified Synthetic GEMM Test

Run the same synthetic GEMM tests for FPGA candidates that already exist for TCASI24.

Deliverable:

```text
outputs/reports/signed_int8_gemm_synthetic_comparison.md
```

Purpose:

- not final LLM evidence;
- quick check whether product-level bounded errors help after accumulation;
- select 1-2 candidates for real-data testing.

### Step 3: Real W8A8 Distribution Collection

Use AxCore-style evaluation inputs rather than only random distributions.

Target:

- WikiText2 samples;
- one small OPT model first, preferably OPT-125M or OPT-350M for local feasibility;
- collect real activation tensors and weights from linear layers;
- quantize to W8A8;
- record `int8 activation x int8 weight` pair distribution and layer output error.

Deliverables:

```text
outputs/reports/w8a8_real_distribution_stats.md
outputs/reports/w8a8_layer_error_comparison.md
```

### Step 4: AxCore Framework Integration

After layer-level results look promising, add our method into AxCore's software evaluation path.

Target files to study next:

```text
work/axcore/Software/AxCore/evaluation/wikitext/evaluate_hf.py
work/axcore/Software/AxCore/evaluation/lmeval/lmeval.py
work/axcore/Software/AxCore/approximation_computation/
```

Deliverables:

```text
WikiText perplexity for baseline AxCore path vs our candidate path
selected zero-shot tasks if resources allow
```

### Step 5: Hardware Cost

Only after precision looks acceptable:

- synthesize unsigned core;
- synthesize signed wrapper + unsigned core;
- compare against exact signed multiplier and AxCore-relevant unit cost if possible.

Deliverables:

```text
LUT / FF / CARRY / DSP / timing table
```

