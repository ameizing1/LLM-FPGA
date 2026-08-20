"""Score LUT-backed int8 multipliers with a calibration pair histogram.

This script turns a saved signed-int8 pair histogram into a distribution-aware
loss report.  The default loss is weighted absolute product error:

    L_dist = sum_{a,b} P_calib(a,b) * |p_hat(a,b) - a*b|

Here P_calib is estimated from the histogram counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DESIGN_LABELS = {
    "exact": "Exact int8 product",
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histogram-npy", required=True, help="path to a 256x256 signed-int8 pair histogram")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="distribution_weighted_error_report")
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ["numpy"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _load_luts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tcasi_dir = Path(args.tcasi_lut_dir)
    fpga_dir = Path(args.fpga_lut_dir)
    return {
        "exact": np.fromfunction(lambda i, j: (i.astype(np.int16) - 128) * (j.astype(np.int16) - 128), (256, 256), dtype=np.int32).astype(np.int32),
        "tcasi24_lsam1": np.load(tcasi_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }


def _weighted_metrics(hist: np.ndarray, exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    counts = np.asarray(hist, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("histogram is empty")

    err = approx.astype(np.float64) - exact.astype(np.float64)
    abs_err = np.abs(err)
    sq_err = err**2
    return {
        "weighted_mae": float(np.sum(counts * abs_err) / total),
        "weighted_rmse": float(np.sqrt(np.sum(counts * sq_err) / total)),
        "weighted_bias": float(np.sum(counts * err) / total),
        "weighted_max_abs": float(np.max(abs_err)),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    hist = data["histogram"]
    rows = [
        {"design": DESIGN_LABELS.get(name, name), **metrics}
        for name, metrics in data["scores"].items()
    ]
    lines = [
        "# Distribution-Weighted Error Report",
        "",
        "## Definition",
        "",
        "$$",
        r"\mathcal{L}_{dist} = \sum_{a,b} P_{\mathrm{calib}}(a,b)\cdot |\hat p(a,b)-ab|",
        "$$",
        "",
        "## Histogram Summary",
        "",
        f"- histogram path: `{data['histogram_path']}`",
        f"- total sampled pairs: `{hist['total_pairs']}`",
        f"- nonzero bins: `{hist['nonzero_bins']}`",
        f"- top bin: `{hist['top_bin_name']}`",
        "",
        "## Weighted Scores",
        "",
        _markdown_table(
            rows,
            ["design", "weighted_mae", "weighted_rmse", "weighted_bias", "weighted_max_abs"],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _require_runtime()

    hist_path = Path(args.histogram_npy)
    hist = np.load(hist_path)
    if hist.shape != (256, 256):
        raise ValueError(f"histogram must have shape (256, 256), got {hist.shape}")
    hist = hist.astype(np.int64, copy=False)

    luts = _load_luts(args)
    exact = luts["exact"]
    scores = {
        name: _weighted_metrics(hist, exact, lut)
        for name, lut in luts.items()
        if name != "exact"
    }

    if np.count_nonzero(hist):
        flat_idx = int(np.argmax(hist))
        top_idx = np.unravel_index(flat_idx, hist.shape)
        top_bin_name = f"({top_idx[0] - 128}, {top_idx[1] - 128})"
    else:
        top_bin_name = "empty"

    data = {
        "histogram_path": str(hist_path),
        "histogram": {
            "total_pairs": int(hist.sum()),
            "nonzero_bins": int(np.count_nonzero(hist)),
            "top_bin_name": top_bin_name,
        },
        "scores": scores,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
