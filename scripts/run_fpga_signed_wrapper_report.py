"""Report product-level error for FPGA signed-wrapper int8 LUTs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import INT8_VALUES, exact_int8_lut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-dir", default="outputs/fpga_luts", help="directory containing signed-wrapper LUTs")
    parser.add_argument("--out-dir", default="outputs/reports", help="directory for report artifacts")
    return parser.parse_args()


def _metrics(exact: np.ndarray, approx: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    exact64 = exact.astype(np.int64)
    approx64 = approx.astype(np.int64)
    if mask is not None:
        exact64 = exact64[mask]
        approx64 = approx64[mask]
    err = approx64 - exact64
    abs_err = np.abs(err)
    nonzero = np.abs(exact64) > 0
    rel = abs_err[nonzero] / np.abs(exact64[nonzero]) if np.any(nonzero) else np.array([0.0])
    denom = max(float(np.linalg.norm(exact64.ravel())), 1.0)
    return {
        "cases": float(exact64.size),
        "error_rate": float(np.mean(err != 0)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "max_abs_error": float(np.max(abs_err)),
        "mean_relative_error_nonzero": float(np.mean(rel)),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
    }


def _load_luts(lut_dir: Path) -> dict[str, np.ndarray]:
    luts: dict[str, np.ndarray] = {}
    for path in sorted(lut_dir.glob("fpga_cand*_signed_wrapper_int8_lut.npy")):
        match = re.search(r"fpga_cand(\d+)_signed_wrapper_int8_lut", path.name)
        if not match:
            continue
        luts[f"cand{match.group(1)}"] = np.load(path)
    if not luts:
        raise FileNotFoundError(f"no signed-wrapper LUTs found in {lut_dir}")
    return luts


def _bucket_masks() -> dict[str, np.ndarray]:
    a = INT8_VALUES[:, None]
    b = INT8_VALUES[None, :]
    return {
        "all": np.ones((256, 256), dtype=bool),
        "nonnegative_x_nonnegative": (a >= 0) & (b >= 0),
        "nonnegative_x_negative": (a >= 0) & (b < 0),
        "negative_x_nonnegative": (a < 0) & (b >= 0),
        "negative_x_negative": (a < 0) & (b < 0),
        "has_minus128": (a == -128) | (b == -128),
        "small_magnitude_le_16": (np.abs(a) <= 16) & (np.abs(b) <= 16),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for name, metrics in data["product_level"]["all"].items():
        rows.append(
            {
                "candidate": name,
                "error_rate": f"{metrics['error_rate']:.6f}",
                "mae": f"{metrics['mae']:.3f}",
                "rmse": f"{metrics['rmse']:.3f}",
                "max_abs": f"{metrics['max_abs_error']:.0f}",
                "rel_l2": f"{metrics['relative_l2_error']:.6f}",
            }
        )

    lines = [
        "# FPGA signed-wrapper int8 product-level report",
        "",
        "- Source: Verilog-simulated unsigned `approx88_cascade` LUTs.",
        "- Signed behavior: `abs(a), abs(b) -> unsigned core -> restore sign`.",
        "- Scope: behavior-model precision only; this is not a signed RTL area/timing report.",
        "",
        "## Overall Product-Level Error",
        "",
        _markdown_table(rows, ["candidate", "error_rate", "mae", "rmse", "max_abs", "rel_l2"]),
        "",
        "## Bucketed Error",
        "",
    ]

    for bucket, metrics_by_candidate in data["product_level"].items():
        if bucket == "all":
            continue
        bucket_rows = []
        for name, metrics in metrics_by_candidate.items():
            bucket_rows.append(
                {
                    "candidate": name,
                    "cases": f"{metrics['cases']:.0f}",
                    "mae": f"{metrics['mae']:.3f}",
                    "max_abs": f"{metrics['max_abs_error']:.0f}",
                    "rel_l2": f"{metrics['relative_l2_error']:.6f}",
                }
            )
        lines.extend(
            [
                f"### {bucket}",
                "",
                _markdown_table(bucket_rows, ["candidate", "cases", "mae", "max_abs", "rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exact = exact_int8_lut()
    luts = _load_luts(lut_dir)
    masks = _bucket_masks()
    product_level = {
        bucket: {name: _metrics(exact, lut, mask) for name, lut in luts.items()} for bucket, mask in masks.items()
    }

    data = {
        "product_level": product_level,
        "interpretation": "Signed-wrapper precision baseline, not final signed RTL resource/timing result.",
    }
    json_path = out_dir / "fpga_signed_wrapper_product_report.json"
    md_path = out_dir / "fpga_signed_wrapper_product_report.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
