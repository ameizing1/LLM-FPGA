#!/usr/bin/env python3
"""
AM-LUT parameter-layer sensitivity helper for AxCore simulator.

Default mode is dry-run: generate modified synthesis CSV files and print the
commands needed to run experiments. Use --run to execute run_axcore.py.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "amlut_sensitivity"
PARAM_DIR = EXP_DIR / "params"
RESULT_DIR = EXP_DIR / "results"
SUMMARY_DIR = EXP_DIR / "summary"


CONFIG_TO_BASELINE = {
    "W4-FP16": ROOT / "params" / "systolic_array_synth_W4-FP16.csv",
    "W4-BF16": ROOT / "params" / "systolic_array_synth_W4-BF16.csv",
    "W4-FP32": ROOT / "params" / "systolic_array_synth_W4-FP32.csv",
    "W8-FP16": ROOT / "params" / "systolic_array_synth_W8-FP16.csv",
    "W8-BF16": ROOT / "params" / "systolic_array_synth_W8-BF16.csv",
    "W8-FP32": ROOT / "params" / "systolic_array_synth_W8-FP32.csv",
}


def ensure_dirs() -> None:
    for path in (PARAM_DIR, RESULT_DIR, SUMMARY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def scale_axcore_row(src: Path, dst: Path, dynamic_scale: float, leakage_scale: float, area_scale: float) -> None:
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"No CSV header found in {src}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {src}")

    touched = False
    for row in rows:
        if row["Module"].strip() == "axcore":
            row["Area (um^2)"] = format_float(float(row["Area (um^2)"]) * area_scale)
            row["Leakage Power (nW)"] = format_float(float(row["Leakage Power (nW)"]) * leakage_scale)
            row["Dynamic Power (nW)"] = format_float(float(row["Dynamic Power (nW)"]) * dynamic_scale)
            try:
                row["Total Power (nW)"] = format_float(float(row["Total Power (nW)"]) * dynamic_scale)
            except ValueError:
                pass
            touched = True

    if not touched:
        raise ValueError(f"Could not find Module=axcore in {src}")

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def run_one(config: str, synth_csv: Path, tag: str) -> Path:
    result_csv = ROOT / "results" / "axcore_res.csv"
    result_csv.parent.mkdir(exist_ok=True)
    result_csv.write_text("", encoding="utf-8")

    cmd = ["python", "run_axcore.py", "--synth_csv", str(synth_csv)]
    subprocess.run(cmd, cwd=ROOT, check=True)

    archived = RESULT_DIR / f"{config}_{tag}_axcore_res.csv"
    shutil.copy2(result_csv, archived)
    return archived


def parse_geomean_axcore(result_csv: Path) -> dict[str, float]:
    lines = result_csv.read_text(encoding="utf-8").splitlines()
    metrics: dict[str, list[float]] = {}
    for line in lines:
        if line.startswith(("Time", "Static", "Dram", "Buffer", "Core")):
            cells = [cell.strip() for cell in line.split(",")]
            metric = cells[0]
            values = [float(cell) for cell in cells[1:] if cell]
            metrics[metric] = values

    # Geomean AxCore is the 11th numeric value: Opt13B[0:5], Opt30B[5:10], Geomean[10:15].
    out = {metric: values[10] for metric, values in metrics.items()}
    out["Total"] = out["Static"] + out["Dram"] + out["Buffer"] + out["Core"]
    return out


def append_summary(row: dict[str, str | float]) -> None:
    summary_csv = SUMMARY_DIR / "amlut_sensitivity_summary.csv"
    headers = [
        "config",
        "tag",
        "dynamic_scale",
        "leakage_scale",
        "area_scale",
        "Time",
        "Static",
        "Dram",
        "Buffer",
        "Core",
        "Total",
        "result_csv",
        "synth_csv",
    ]
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_tag(dynamic_scale: float, leakage_scale: float, area_scale: float) -> str:
    return (
        f"dyn{dynamic_scale:g}_"
        f"leak{leakage_scale:g}_"
        f"area{area_scale:g}"
    ).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIG_TO_BASELINE), default="W4-FP16")
    parser.add_argument("--dynamic-scale", type=float, default=1.0)
    parser.add_argument("--leakage-scale", type=float, default=1.0)
    parser.add_argument("--area-scale", type=float, default=1.0)
    parser.add_argument("--run", action="store_true", help="Execute run_axcore.py after generating the CSV.")
    args = parser.parse_args()

    ensure_dirs()

    tag = make_tag(args.dynamic_scale, args.leakage_scale, args.area_scale)
    baseline = CONFIG_TO_BASELINE[args.config]
    synth_csv = PARAM_DIR / f"{args.config}_{tag}.csv"

    scale_axcore_row(
        baseline,
        synth_csv,
        dynamic_scale=args.dynamic_scale,
        leakage_scale=args.leakage_scale,
        area_scale=args.area_scale,
    )

    print(f"Generated: {synth_csv}")
    print("Equivalent manual command:")
    print(f"  python run_axcore.py --synth_csv {synth_csv}")

    if not args.run:
        print("Dry-run only. Add --run to execute the simulator and archive results.")
        return

    archived = run_one(args.config, synth_csv, tag)
    parsed = parse_geomean_axcore(archived)
    append_summary(
        {
            "config": args.config,
            "tag": tag,
            "dynamic_scale": args.dynamic_scale,
            "leakage_scale": args.leakage_scale,
            "area_scale": args.area_scale,
            **parsed,
            "result_csv": str(archived),
            "synth_csv": str(synth_csv),
        }
    )
    print(f"Archived result: {archived}")
    print(f"Summary updated: {SUMMARY_DIR / 'amlut_sensitivity_summary.csv'}")
    print(
        "Geomean AxCore: "
        f"Time={parsed['Time']:.2f}, Core={parsed['Core']:.2f}, Total={parsed['Total']:.2f}"
    )


if __name__ == "__main__":
    main()
