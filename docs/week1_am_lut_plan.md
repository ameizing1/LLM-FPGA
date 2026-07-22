# Week 1 AM-LUT experiment plan

Goal: compare TCASI24 LUT-based approximate multipliers with the exact int8 product path before making any AxCore replacement claim.

## Scope

- Implement TCASI24 LSAM1 and CSAM2 behavior models from the authors' Verilog.
- Generate `exact_int8_lut.npy`, `lsam1_int8_lut.npy`, and `csam2_int8_lut.npy`.
- Run product-level, GEMM-level, and distribution-sensitive error reports.
- Use results to decide whether an AxCore simulator or CUDA Linear wrapper is worth building next.

## First simplification

The current int8 model uses a sign-magnitude wrapper:

1. take `abs(a)` and `abs(b)`;
2. run the unsigned 8x8 approximate multiplier;
3. restore the sign.

This is a practical first-week behavior model for LLM int8 experiments, not signed RTL. If the error looks promising, the next hardware step is to replace this wrapper with a signed RTL-consistent model.

## Commands

```powershell
& 'C:\Users\LiuZhiWei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\generate_int8_luts.py
& 'C:\Users\LiuZhiWei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\run_error_report.py
& 'C:\Users\LiuZhiWei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tests\smoke_test_tcasi24.py
```

## Reading targets

- `references/AM-LUT/Hardware-Efficient_Multipliers_With_FPGA-Based_Approximation_for_Error-Resilient_Applications - TCASI24.pdf`
- `references/LLM-FPGA/AxCore A qantization-Aware Approximate GEMM Unit for LLM - Micro25.pdf`
- Verilog source used for this model: https://github.com/YnuGuoLab/DATE_FPGA_Approx_Mul
