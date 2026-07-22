# LLM-FPGA

Research workspace for exploring FPGA-oriented LLM arithmetic, AM-LUT multiplier experiments, AxCore notes, and related reproducibility scripts.

## Layout

- `scripts/`: report generation, LUT generation, and RTL verification helpers.
- `tests/`: smoke tests for signed wrapper and TCASI24-related behavior.
- `am_lut_tcasi24/`: TCASI24 and AM-LUT Python model code.
- `multiplier_models/`: multiplier model helpers.
- `FPGA_multiplier/`: FPGA multiplier design artifacts.
- `docs/`: project migration notes and planning docs.
- `study/`: reading notes, experiment logs, and project roadmaps.
- `outputs/reports/`: lightweight generated reports intended for version control.

Large rendered outputs, archives, PDFs, caches, and temporary build artifacts are intentionally ignored by Git.
