"""Run behavior-level protection experiments for FPGA cand17.

This script is diagnostic only.  The hybrid LUTs generated here are upper-bound
behavior models used to identify which input regions are most important for
GEMM accuracy.  They are not proposed as the final hardware implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm
from multiplier_models.signed_wrapper import INT8_VALUES, exact_int8_lut


@dataclass(frozen=True)
class DesignSpec:
    key: str
    label: str
    lut: np.ndarray
    mask: np.ndarray | None = None
    family: str = "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts", help="TCASI24 LUT directory")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts", help="FPGA signed-wrapper LUT directory")
    parser.add_argument("--out-dir", default="outputs/reports", help="report output directory")
    parser.add_argument("--hybrid-lut-dir", default="outputs/hybrid_luts", help="generated hybrid LUT directory")
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
    denom = max(float(np.linalg.norm(exact64.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "max_abs_error": float(np.max(abs_err)),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
    }


def _sample_int8(rng: np.random.Generator, shape: tuple[int, int], kind: str) -> np.ndarray:
    if kind == "uniform":
        return rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
    if kind == "small_normal":
        return np.clip(np.rint(rng.normal(0, 16, size=shape)), -128, 127).astype(np.int8)
    if kind == "sparse_small":
        values = np.clip(np.rint(rng.normal(0, 12, size=shape)), -128, 127)
        values[rng.random(shape) < 0.7] = 0
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
    raise ValueError(f"unknown distribution kind: {kind}")


def _small_operand_mask(threshold: int) -> np.ndarray:
    mags = np.abs(INT8_VALUES.astype(np.int32))
    return (mags[:, None] <= threshold) | (mags[None, :] <= threshold)


def _both_small_mask(threshold: int) -> np.ndarray:
    mags = np.abs(INT8_VALUES.astype(np.int32))
    return (mags[:, None] <= threshold) & (mags[None, :] <= threshold)


def _product_magnitude_mask(threshold: int) -> np.ndarray:
    values = INT8_VALUES.astype(np.int32)
    products = values[:, None] * values[None, :]
    return np.abs(products) <= threshold


def _build_hybrid_lut(base: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.astype(np.int32).copy()
    out[mask] = replacement.astype(np.int32)[mask]
    return out


def _load_design_luts(tcasi_dir: Path, fpga_dir: Path) -> dict[str, np.ndarray]:
    exact = np.load(tcasi_dir / "exact_int8_lut.npy").astype(np.int32)
    if not np.array_equal(exact, exact_int8_lut()):
        raise AssertionError("exact_int8_lut.npy does not match signed int8 reference")
    return {
        "exact": exact,
        "tcasi24_lsam1": np.load(tcasi_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
    }


def _append_hybrid(
    designs: list[DesignSpec],
    *,
    key: str,
    label: str,
    base: np.ndarray,
    replacement: np.ndarray,
    mask: np.ndarray,
    family: str,
) -> None:
    designs.append(
        DesignSpec(
            key=key,
            label=label,
            lut=_build_hybrid_lut(base, replacement, mask),
            mask=mask,
            family=family,
        )
    )


def _make_designs(luts: dict[str, np.ndarray]) -> list[DesignSpec]:
    exact = luts["exact"]
    lsam1 = luts["tcasi24_lsam1"]
    cand17 = luts["fpga_cand17"]

    designs = [
        DesignSpec("tcasi24_lsam1", "TCASI24 LSAM1", lsam1),
        DesignSpec("fpga_cand17", "FPGA cand17", cand17),
    ]

    for threshold in [16, 32]:
        mask = _small_operand_mask(threshold)
        _append_hybrid(
            designs,
            key=f"cand17_exact_if_min_abs_le_{threshold}",
            label=f"cand17 + exact(min<={threshold})",
            base=cand17,
            replacement=exact,
            mask=mask,
            family="min_abs",
        )
        _append_hybrid(
            designs,
            key=f"cand17_lsam1_if_min_abs_le_{threshold}",
            label=f"cand17 + LSAM1(min<={threshold})",
            base=cand17,
            replacement=lsam1,
            mask=mask,
            family="min_abs",
        )

    _append_hybrid(
        designs,
        key="cand17_exact_if_both_abs_le_16",
        label="cand17 + exact(both<=16)",
        base=cand17,
        replacement=exact,
        mask=_both_small_mask(16),
        family="both_abs",
    )

    for threshold in [256, 512, 1024]:
        mask = _product_magnitude_mask(threshold)
        _append_hybrid(
            designs,
            key=f"cand17_exact_if_abs_product_le_{threshold}",
            label=f"cand17 + exact(|ab|<={threshold})",
            base=cand17,
            replacement=exact,
            mask=mask,
            family="product_abs",
        )
        _append_hybrid(
            designs,
            key=f"cand17_lsam1_if_abs_product_le_{threshold}",
            label=f"cand17 + LSAM1(|ab|<={threshold})",
            base=cand17,
            replacement=lsam1,
            mask=mask,
            family="product_abs",
        )

    return designs


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _product_report(exact: np.ndarray, designs: list[DesignSpec]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for design in designs:
        metrics = _metrics(exact, design.lut)
        metrics["replacement_coverage"] = float(np.mean(design.mask)) if design.mask is not None else 0.0
        report[design.key] = metrics
    return report


def _gemm_report(
    designs: list[DesignSpec],
    *,
    shape: tuple[int, int, int],
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    m, k, n = shape
    cases = {
        "uniform_int8": ("uniform", "uniform"),
        "small_normal": ("small_normal", "small_normal"),
        "sparse_small": ("sparse_small", "sparse_small"),
        "outlier_channels": ("outlier_channels", "outlier_channels"),
        "nonnegative_activation_x_weight": ("nonnegative_activation", "small_normal"),
    }

    report: dict[str, dict[str, dict[str, float]]] = {}
    for case_name, (a_kind, b_kind) in cases.items():
        a = _sample_int8(rng, (m, k), a_kind)
        b = _sample_int8(rng, (k, n), b_kind)
        exact = exact_gemm(a, b)
        report[case_name] = {
            design.key: _metrics(exact, lut_gemm(a, b, design.lut)) for design in designs
        }
    return report


def _write_luts(path: Path, designs: list[DesignSpec]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for design in designs:
        if design.key.startswith("cand17_"):
            np.save(path / f"{design.key}.npy", design.lut.astype(np.int32))


def _format_metrics(metrics: dict[str, float], include_coverage: bool = False) -> dict[str, str]:
    formatted = {
        "error_rate": f"{metrics['error_rate']:.6f}",
        "MAE": f"{metrics['mae']:.3f}",
        "RMSE": f"{metrics['rmse']:.3f}",
        "p99_abs": f"{metrics['p99_abs_error']:.0f}",
        "max_abs": f"{metrics['max_abs_error']:.0f}",
        "rel_l2": f"{metrics['relative_l2_error']:.6f}",
    }
    if include_coverage:
        formatted["coverage"] = f"{100.0 * metrics.get('replacement_coverage', 0.0):.2f}%"
    return formatted


def _improvement_rows(
    baseline: dict[str, dict[str, float]],
    designs: list[DesignSpec],
    metric_key: str = "relative_l2_error",
) -> list[dict[str, str]]:
    rows = []
    base_value = baseline["fpga_cand17"][metric_key]
    for design in designs:
        value = baseline[design.key][metric_key]
        improvement = 0.0 if base_value == 0 else (base_value - value) / base_value
        rows.append(
            {
                "设计": design.label,
                "rel_l2": f"{value:.6f}",
                "相对 cand17 改善": f"{100.0 * improvement:.2f}%",
            }
        )
    return rows


def _write_markdown(
    path: Path,
    *,
    product: dict[str, dict[str, float]],
    gemm: dict[str, dict[str, dict[str, float]]],
    designs: list[DesignSpec],
    shape: tuple[int, int, int],
    seed: int,
) -> None:
    design_by_key = {design.key: design for design in designs}

    product_rows = []
    for design in designs:
        row = {"设计": design.label}
        row.update(_format_metrics(product[design.key], include_coverage=True))
        product_rows.append(row)

    m, k, n = shape
    lines = [
        "# cand17 小值保护与乘积幅度保护实验报告",
        "",
        "## 实验目的",
        "",
        "这个实验不是最终硬件方案，而是 behavior-level upper-bound 诊断。它回答的问题是：如果 cand17 在某些敏感输入区域被更准确的乘法行为替代，GEMM 误差能恢复多少。",
        "",
        "本次分别测试两类方案：",
        "",
        "- small-value 保护：当至少一个操作数幅度较小时切换 replacement。",
        "",
        "$$",
        "\\min(|a|, |b|) \\le T",
        "$$",
        "",
        "- product-magnitude 保护：当真实乘积幅度较小时切换 replacement。",
        "",
        "$$",
        "|a\\cdot b| \\le P",
        "$$",
        "",
        "这里的 `exact(...)` 是理论上界；`LSAM1(...)` 是低成本近似保护的参考，不表示最终一定要并联多个完整乘法器。",
        "",
        "## 实验设置",
        "",
        f"- GEMM shape：`M={m}, K={k}, N={n}`。",
        f"- 随机种子：`{seed}`。",
        "- exact baseline：标准 signed int8 乘法 + int32 累加。",
        "- approximate GEMM：每个标量乘法从 signed int8 product LUT 查表，再做 int32 累加。",
        "- `coverage` 表示被 replacement 覆盖的 signed int8 输入组合比例，可作为粗略硬件复杂度/保护范围信号。",
        "",
        "## Product-Level 结果",
        "",
        _markdown_table(
            product_rows,
            ["设计", "coverage", "error_rate", "MAE", "RMSE", "p99_abs", "max_abs", "rel_l2"],
        ),
        "",
        "## Synthetic GEMM 结果",
        "",
    ]

    for case_name, metrics_by_design in gemm.items():
        rows = []
        for design in designs:
            row = {"设计": design_by_key[design.key].label}
            row.update(_format_metrics(metrics_by_design[design.key]))
            rows.append(row)
        lines.extend(
            [
                f"### {case_name}",
                "",
                _markdown_table(rows, ["设计", "MAE", "RMSE", "p99_abs", "max_abs", "rel_l2"]),
                "",
                "相对原始 cand17 的 `rel_l2` 改善：",
                "",
                _markdown_table(_improvement_rows(metrics_by_design, designs), ["设计", "rel_l2", "相对 cand17 改善"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 初步解读",
            "",
            "- `min<=32` 比 `min<=16` 保护范围更大，如果 small/sparse 场景明显改善，说明退化确实与小幅值操作数区域有关。",
            "- `|ab|<=P` 是更直接的误差敏感区域定位：它覆盖的是小乘积，而不是只看单个操作数是否小。若它优于 `min<=T`，后续乘法器训练可以把 loss 权重更多放在小乘积区域。",
            "- `exact(...)` 和 `LSAM1(...)` 的差距可以帮助判断：如果 LSAM1 replacement 已经接近 exact replacement，说明不一定需要完整精确乘法器，可能通过轻量约束或重新训练近似 LUT 达到类似方向。",
            "- 这些结果只说明应该优先优化哪些输入区域；最终仍应追求单个有符号近似核或轻量局部逻辑，而不是简单并联多套完整乘法器。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    luts = _load_design_luts(Path(args.tcasi_lut_dir), Path(args.fpga_lut_dir))
    designs = _make_designs(luts)
    _write_luts(Path(args.hybrid_lut_dir), designs)

    shape = (args.m, args.k, args.n)
    product = _product_report(luts["exact"], designs)
    gemm = _gemm_report(designs, shape=shape, seed=args.seed)
    data = {
        "product_level": product,
        "synthetic_gemm": gemm,
        "designs": [
            {
                "key": design.key,
                "label": design.label,
                "family": design.family,
                "replacement_coverage": float(np.mean(design.mask)) if design.mask is not None else 0.0,
            }
            for design in designs
        ],
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "seed": args.seed,
        "note": "Behavior-level upper-bound experiment, not final hardware implementation.",
    }

    json_path = out_dir / "small_value_protection_experiment.json"
    md_path = out_dir / "small_value_protection_experiment.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, product=product, gemm=gemm, designs=designs, shape=shape, seed=args.seed)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
