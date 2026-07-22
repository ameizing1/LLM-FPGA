"""Generate product-level and GEMM-level error reports for TCASI24 int8 LUTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm
from am_lut_tcasi24.tcasi24 import INT8_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-dir", default="outputs/luts", help="directory containing generated .npy LUTs")
    parser.add_argument("--out-dir", default="outputs/reports", help="directory for report artifacts")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def _metrics(exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    exact64 = exact.astype(np.int64)
    approx64 = approx.astype(np.int64)
    err = approx64 - exact64
    abs_err = np.abs(err)
    nonzero = np.abs(exact64) > 0
    rel = abs_err[nonzero] / np.abs(exact64[nonzero]) if np.any(nonzero) else np.array([0.0])
    denom = max(float(np.linalg.norm(exact64.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "max_abs_error": float(np.max(abs_err)),
        "mean_relative_error_nonzero": float(np.mean(rel)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
    }


def _sample_int8(rng: np.random.Generator, shape: tuple[int, int], kind: str) -> np.ndarray:
    if kind == "uniform":
        return rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
    if kind == "small_normal":
        return np.clip(np.rint(rng.normal(0, 16, size=shape)), -128, 127).astype(np.int8)
    if kind == "sparse_small":
        values = np.clip(np.rint(rng.normal(0, 12, size=shape)), -128, 127)
        mask = rng.random(shape) < 0.7
        values[mask] = 0
        return values.astype(np.int8)
    if kind == "outlier_channels":
        values = np.clip(np.rint(rng.normal(0, 10, size=shape)), -128, 127)
        outliers = rng.random(shape) < 0.02
        signs = rng.choice([-1, 1], size=shape)
        mags = rng.integers(96, 128, size=shape)
        values[outliers] = signs[outliers] * mags[outliers]
        return values.astype(np.int8)
    if kind == "nonnegative_activation":
        return rng.integers(0, 128, size=shape, dtype=np.int16).astype(np.int8)
    raise ValueError(f"unknown distribution: {kind}")


def _load_luts(lut_dir: Path) -> dict[str, np.ndarray]:
    return {
        "exact": np.load(lut_dir / "exact_int8_lut.npy"),
        "lsam1": np.load(lut_dir / "lsam1_int8_lut.npy"),
        "csam2": np.load(lut_dir / "csam2_int8_lut.npy"),
    }


def product_report(luts: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    exact = luts["exact"]
    expected = (INT8_VALUES[:, None].astype(np.int32) * INT8_VALUES[None, :].astype(np.int32)).astype(np.int16)
    if not np.array_equal(exact, expected):
        raise AssertionError("exact_int8_lut.npy does not match numpy int8 product reference")
    return {mode: _metrics(exact, lut) for mode, lut in luts.items() if mode != "exact"}


def gemm_report(
    luts: dict[str, np.ndarray],
    *,
    shape: tuple[int, int, int],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m, k, n = shape
    cases = {
        "uniform_int8": ("uniform", "uniform"),
        "small_normal": ("small_normal", "small_normal"),
        "sparse_small": ("sparse_small", "sparse_small"),
        "outlier_channels": ("outlier_channels", "outlier_channels"),
        "nonnegative_activation_x_weight": ("nonnegative_activation", "small_normal"),
    }

    report: dict[str, Any] = {}
    for case_name, (a_kind, b_kind) in cases.items():
        a = _sample_int8(rng, (m, k), a_kind)
        b = _sample_int8(rng, (k, n), b_kind)
        exact = exact_gemm(a, b)
        report[case_name] = {
            mode: _metrics(exact, lut_gemm(a, b, lut))
            for mode, lut in luts.items()
            if mode != "exact"
        }
    return report


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(path: Path, data: dict[str, Any], *, shape: tuple[int, int, int], seed: int) -> None:
    product_rows = []
    for mode, metrics in data["product_level"].items():
        product_rows.append(
            {
                "mode": mode,
                "error_rate": f"{metrics['error_rate']:.6f}",
                "mae": f"{metrics['mae']:.3f}",
                "rmse": f"{metrics['rmse']:.3f}",
                "max_abs": f"{metrics['max_abs_error']:.0f}",
                "rel_l2": f"{metrics['relative_l2_error']:.6f}",
            }
        )

    lines = [
        "# TCASI24 int8 AM-LUT error report",
        "",
        f"- GEMM shape: M={shape[0]}, K={shape[1]}, N={shape[2]}",
        f"- RNG seed: {seed}",
        "- Signed behavior: sign-magnitude wrapper around unsigned 8x8 TCASI24 blocks.",
        "- Interpretation guardrail: this report compares error behavior only; it does not claim AM-LUT is better than AxCore.",
        "",
        "## Product-level error",
        "",
        _markdown_table(product_rows, ["mode", "error_rate", "mae", "rmse", "max_abs", "rel_l2"]),
        "",
        "## GEMM-level and distribution-sensitive error",
        "",
    ]

    for case_name, case_metrics in data["gemm_level"].items():
        rows = []
        for mode, metrics in case_metrics.items():
            rows.append(
                {
                    "mode": mode,
                    "mae": f"{metrics['mae']:.3f}",
                    "rmse": f"{metrics['rmse']:.3f}",
                    "p99_abs": f"{metrics['p99_abs_error']:.0f}",
                    "max_abs": f"{metrics['max_abs_error']:.0f}",
                    "rel_l2": f"{metrics['relative_l2_error']:.6f}",
                }
            )
        lines.extend(
            [
                f"### {case_name}",
                "",
                _markdown_table(rows, ["mode", "mae", "rmse", "p99_abs", "max_abs", "rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    luts = _load_luts(lut_dir)
    shape = (args.m, args.k, args.n)
    data = {
        "product_level": product_report(luts),
        "gemm_level": gemm_report(luts, shape=shape, seed=args.seed),
    }

    json_path = out_dir / "tcasi24_int8_error_report.json"
    md_path = out_dir / "tcasi24_int8_error_report.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(md_path, data, shape=shape, seed=args.seed)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
