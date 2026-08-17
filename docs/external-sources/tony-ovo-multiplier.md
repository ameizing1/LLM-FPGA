# Tony-ovo Multiplier Import

## Purpose

This project imports the signed 8x8 LUT trainer needed to improve the
S88-6x2 Quality and Balanced multiplier designs for the LLM-FPGA workflow.

## Upstream Source

- Repository: `https://github.com/Tony-ovo/Multiplier`
- Imported commit: `9e79c363b83cde9340941962e4e46101ed16de84`
- Import date: 2026-08-17
- Upstream path: `approximate/signed_8x8/`

## Imported Scope

- Core trainer package: `multiplier_models/signed88/`
- Training, refinement, verification, summary, and calibration scripts:
  `scripts/*signed88*.py`
- Baseline test fixture and focused Quality/Balanced tests: `tests/`
- RTL templates: `FPGA_multiplier/signed8x8_6x2/Quality/` and
  `FPGA_multiplier/signed8x8_6x2/Balanced/`

The upstream Bash batch scripts and the other design families were not
imported as active project components.

## Local Adaptations

- Scripts resolve the repository root and import from `multiplier_models/`.
- Default calibration input is the committed smoke fixture under `tests/data/`.
- Default training outputs are under `tmp/`, which is local-only.
- Command-line design choices are restricted to `quality` and `balanced`.

## Attribution and License

The upstream repository did not contain a license file when this import was
made. Treat this code as internal collaboration material. Confirm permission
and agree on a license before making a public redistribution.
