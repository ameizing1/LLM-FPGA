# Project Rules for AI Work

This repository is an FPGA-oriented LLM arithmetic research workspace. Keep source, notes, reproducible reports, and generated bulk artifacts clearly separated.

## File Placement

- Put reusable Python scripts in `scripts/`.
- Put tests and smoke checks in `tests/`.
- Put reusable Python package/model code in `am_lut_tcasi24/` or `multiplier_models/`.
- Put project documentation, environment notes, reproducibility notes, and external source indexes in `docs/`.
- Put study notes, paper reading notes, experiment logs, handoff notes, and learning-oriented annotated snippets in `study/`.
- Put generated reports in `outputs/reports/`.
- Put temporary extraction files, scratch files, and one-off conversion intermediates in `tmp/`.
- Put external repositories, copied third-party code, downloaded papers, and local reproduction sandboxes in `work/`.
- Put FPGA multiplier source/design assets in `FPGA_multiplier/`, but do not place large unfiltered candidate sweeps there unless they are intentionally kept as local-only artifacts.

## Git Tracking Policy

Commit these by default:

- Source code in `scripts/`, `tests/`, `am_lut_tcasi24/`, and `multiplier_models/`.
- Human-written notes and summaries in `study/` and `docs/`.
- Small configuration files such as requirements files.
- Small, decision-supporting reports in `outputs/reports/`, especially `.md`, `.json`, and `.csv` files that support project conclusions.
- Final selected RTL/design files when they are needed for reuse or reproduction.

Do not commit these by default:

- Virtual environments such as `.venv/` and `.venv-probe/`.
- `tmp/`.
- External source trees in `work/` unless explicitly promoted and license/source are documented.
- Large generated candidate sweeps, logs, arrays, model weights, caches, rendered images, archives, PDFs, and document export folders.
- Word lock files such as `~$*.docx`.

## Before Creating Files

Before adding a new file, classify it as one of:

- source
- test
- documentation
- study note
- key report
- temporary artifact
- external reference
- bulk generated artifact

Choose the directory from that classification. Do not add new project files to the repository root unless they are standard root-level project files such as `README.md`, `.gitignore`, `AGENTS.md`, requirements files, or thread/index files.

## Generated Experiment Outputs

For experiments that generate many files:

- Keep the generator script and exact command/config needed to reproduce the run.
- Keep a compact summary report under `outputs/reports/`.
- Keep only the final selected small subset of generated RTL/JSON artifacts if needed.
- Leave bulk candidates local-only unless the user explicitly asks to preserve them in Git.

## GitHub Desktop Hygiene

Before committing through GitHub Desktop:

- Review the changed file list.
- Do not commit thousands of generated files as one broad commit.
- Commit logically grouped changes, for example `scripts + matching study note + key report`.
- If `Changes` is dominated by generated files, update `.gitignore` before committing.

