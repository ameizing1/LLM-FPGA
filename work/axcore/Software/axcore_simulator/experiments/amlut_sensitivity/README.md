# AM-LUT Sensitivity Experiment

This folder contains a parameter-layer sensitivity helper for AxCore Figure 17 experiments.

## Purpose

The first experiment does not replace RTL or modify the AxCore simulator cycle model. It only changes the `axcore` row in a copied synthesis CSV:

- `Area (um^2)`
- `Leakage Power (nW)`
- `Dynamic Power (nW)`

This keeps FGLUT/FIGNA/FPMA/FPE baselines unchanged and treats AxCore as a hypothetical AxCore-AM-LUT variant.

## Dry Run

Generate a modified synthesis CSV without running the simulator:

```bash
cd /mnt/c/Users/LiuZhiWei/Documents/Codex/2026-06-17/ai/work/axcore/Software/axcore_simulator
. .venv/bin/activate
python experiments/amlut_sensitivity/amlut_sensitivity.py \
  --config W4-FP16 \
  --dynamic-scale 0.8 \
  --leakage-scale 1.0 \
  --area-scale 1.0
```

Output:

```text
experiments/amlut_sensitivity/params/W4-FP16_dyn0p8_leak1_area1.csv
```

## Run One Experiment

Run one simulator point and archive `results/axcore_res.csv`:

```bash
python experiments/amlut_sensitivity/amlut_sensitivity.py \
  --config W4-FP16 \
  --dynamic-scale 0.8 \
  --leakage-scale 1.0 \
  --area-scale 1.0 \
  --run
```

Archived output:

```text
experiments/amlut_sensitivity/results/
experiments/amlut_sensitivity/summary/amlut_sensitivity_summary.csv
```

## Recommended First Sweep

Start with dynamic-only:

```text
dynamic_scale = 1.0 / 0.9 / 0.8 / 0.7 / 0.5
leakage_scale = 1.0
area_scale = 1.0
config = W4-FP16
```

Then leakage-only:

```text
dynamic_scale = 1.0
leakage_scale = 1.0 / 0.9 / 0.8 / 0.7 / 0.5
area_scale = 1.0
config = W4-FP16
```

Only after single-config results are interpretable should the sweep be expanded to all six Figure 17 configs.

## Interpretation Boundary

This experiment can support:

- Core energy sensitivity
- Total energy sensitivity
- Whether a lower-power approximate multiplier is worth deeper validation

It cannot support:

- LLM accuracy claims
- latency speedup claims
- RTL correctness claims
- FPGA implementation claims

