"""生成 signed int8 product-level 与 synthetic GEMM 统一对比报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm
from multiplier_models.signed_wrapper import INT8_VALUES, exact_int8_lut


DESIGN_ORDER = [
    "tcasi24_lsam1",
    "tcasi24_csam2",
    "fpga_cand17",
    "fpga_cand20",
    "fpga_cand10",
]

DESIGN_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts", help="TCASI24 LUT 目录")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts", help="FPGA signed-wrapper LUT 目录")
    parser.add_argument("--out-dir", default="outputs/reports", help="报告输出目录")
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
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "max_abs_error": float(np.max(abs_err)),
        "mean_relative_error_nonzero": float(np.mean(rel)),
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
    raise ValueError(f"未知分布类型: {kind}")


def _load_luts(tcasi_lut_dir: Path, fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    luts = {
        "exact": np.load(tcasi_lut_dir / "exact_int8_lut.npy").astype(np.int32),
        "tcasi24_lsam1": np.load(tcasi_lut_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_lut_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_lut_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_lut_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_lut_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }
    if not np.array_equal(luts["exact"], exact_int8_lut()):
        raise AssertionError("exact_int8_lut.npy 与标准 signed int8 乘法不一致")
    return luts


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def product_report(luts: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    exact = luts["exact"]
    return {name: _metrics(exact, luts[name]) for name in DESIGN_ORDER}


def gemm_report(
    luts: dict[str, np.ndarray],
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
        report[case_name] = {name: _metrics(exact, lut_gemm(a, b, luts[name])) for name in DESIGN_ORDER}
    return report


def _metric_explanations() -> list[str]:
    return [
        "记 exact 输出为 `y_i`，approx 输出为 `yhat_i`，误差为 `e_i`：",
        "",
        "```text",
        "e_i = yhat_i - y_i",
        "",
        "error_rate = count(yhat_i != y_i) / N",
        "MAE        = (sum |e_i|) / N",
        "RMSE       = sqrt((sum e_i^2) / N)",
        "p99_abs    = percentile(|e_i|, 99)",
        "max_abs    = max |e_i|",
        "rel_l2     = sqrt(sum e_i^2) / sqrt(sum y_i^2)",
        "```",
        "",
        "- `RMSE` 不是平均相对均方误差；它是均方根误差，会更重视少数大错误。",
        "- `rel_l2` 才是相对量，用来衡量整体输出向量或矩阵相对 exact 的偏离比例。",
    ]


def _distribution_explanations() -> list[str]:
    return [
        "- `uniform_int8`：`A` 和 `B` 都从 `-128` 到 `127` 均匀随机采样。这个分布覆盖全输入空间，偏向硬件压力测试，不代表真实 LLM 分布。",
        "- `small_normal`：`A` 和 `B` 都从均值为 0、标准差为 16 的正态分布采样，四舍五入并裁剪到 int8。它模拟大量值集中在 0 附近的量化张量。",
        "- `sparse_small`：先从均值为 0、标准差为 12 的正态分布采样，再随机把约 70% 元素置 0。它用来观察零值/小值很多时误差是否会被放大。",
        "- `outlier_channels`：主体来自均值为 0、标准差为 10 的正态分布，再注入约 2% 的大幅值 outlier，幅值在 `96` 到 `127`。它模拟 LLM activation outlier 对近似乘法的压力。",
        "- `nonnegative_activation_x_weight`：`A` 从 `0` 到 `127` 非负均匀采样，`B` 使用 `small_normal`。它模拟非负 activation 与 signed weight 相乘的情况，但不一定代表所有 LLM linear 层。",
    ]


def write_product_markdown(path: Path, data: dict[str, dict[str, float]]) -> None:
    rows = []
    for name in DESIGN_ORDER:
        metrics = data[name]
        rows.append(
            {
                "设计": DESIGN_LABELS[name],
                "error_rate": f"{metrics['error_rate']:.6f}",
                "MAE": f"{metrics['mae']:.3f}",
                "RMSE": f"{metrics['rmse']:.3f}",
                "p99_abs": f"{metrics['p99_abs_error']:.0f}",
                "max_abs": f"{metrics['max_abs_error']:.0f}",
                "rel_l2": f"{metrics['relative_l2_error']:.6f}",
            }
        )

    lines = [
        "# Signed INT8 Product-Level 统一对比报告",
        "",
        "## 实验口径",
        "",
        "- 输入范围：`a,b in [-128,127]`，共 65536 个 signed int8 输入组合。",
        "- exact baseline：标准 signed int8 乘法，输出按 int32 统计。",
        "- TCASI24 LSAM1/CSAM2：当前项目里的 unsigned TCASI24 8x8 行为模型 + signed wrapper。",
        "- FPGA cand17/20/10：直接仿真组里 Verilog unsigned core 生成 LUT，再使用 signed wrapper。",
        "- 本报告只比较 product-level 精度，不包含硬件资源、时序或端到端 LLM 精度。",
        "",
        "## 指标说明",
        "",
        *_metric_explanations(),
        "",
        "## 对比结果",
        "",
        _markdown_table(rows, ["设计", "error_rate", "MAE", "RMSE", "p99_abs", "max_abs", "rel_l2"]),
        "",
        "## 当前观察",
        "",
        "- TCASI24 LSAM1 的 `error_rate` 和 `MAE` 最低，说明它更少出错，平均绝对偏差也更小。",
        "- FPGA cand17 的 `RMSE`、`max_abs` 和 `rel_l2` 更低，说明虽然出错更频繁，但大误差更受控。",
        "- TCASI24 CSAM2 在当前 signed int8 口径下误差明显更大，后续可作为激进近似对照点。",
        "- 因为各指标指向不同，不能只凭 product-level 宣称某个设计全面更好，需要继续看 GEMM 累加后的表现。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_gemm_markdown(
    path: Path,
    data: dict[str, dict[str, dict[str, float]]],
    *,
    shape: tuple[int, int, int],
    seed: int,
) -> None:
    m, k, n = shape
    lines = [
        "# Signed INT8 Synthetic GEMM 统一对比报告",
        "",
        "## 实验口径",
        "",
        f"- GEMM shape：`M={m}, K={k}, N={n}`。",
        f"- 随机种子：`{seed}`。",
        "- exact baseline：标准 signed int8 乘法 + int32 累加。",
        "- approximate GEMM：每个标量乘法从对应 signed int8 product LUT 查表，再做 int32 累加。",
        "- 本报告使用 synthetic distribution，只用于快速诊断误差累加趋势；正式 LLM 结论还需要真实 activation/weight 分布与端到端评估。",
        "",
        "## 指标说明",
        "",
        *_metric_explanations(),
        "",
        "## 输入分布说明",
        "",
        *_distribution_explanations(),
        "",
        "## 对比结果",
        "",
    ]

    for case_name, metrics_by_design in data.items():
        rows = []
        for name in DESIGN_ORDER:
            metrics = metrics_by_design[name]
            rows.append(
                {
                    "设计": DESIGN_LABELS[name],
                    "MAE": f"{metrics['mae']:.3f}",
                    "RMSE": f"{metrics['rmse']:.3f}",
                    "p99_abs": f"{metrics['p99_abs_error']:.0f}",
                    "max_abs": f"{metrics['max_abs_error']:.0f}",
                    "rel_l2": f"{metrics['relative_l2_error']:.6f}",
                }
            )
        lines.extend(
            [
                f"### {case_name}",
                "",
                _markdown_table(rows, ["设计", "MAE", "RMSE", "p99_abs", "max_abs", "rel_l2"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 当前观察方式",
            "",
            "- 如果某个设计 product-level 看起来不错，但 GEMM `rel_l2` 明显放大，说明误差在累加中没有抵消。",
            "- 如果 `max_abs` 很大但 `rel_l2` 不高，说明主要风险可能集中在少数输出元素。",
            "- 如果 small/sparse 分布下误差明显小于 uniform 分布，说明真实 LLM 分布可能比全范围随机测试更温和，但这需要真实数据验证。",
            "",
            "## 为什么不同分布下结果差异明显",
            "",
            "- `uniform_int8` 里大幅值输入很多，FPGA cand17 的 product-level `max_abs` 和 `RMSE` 比 LSAM1 更小，所以 GEMM 累加后 `rel_l2` 看起来可以略优。",
            "- `small_normal` 和 `sparse_small` 里大量乘法发生在小幅值区域。LSAM1 的 product-level `error_rate` 很低，很多小乘法能保持 exact；FPGA cand17 的错误更频繁，小误差在 GEMM 里大量累加，所以劣化明显。",
            "- `nonnegative_activation_x_weight` 会让符号组合更偏向固定模式，误差不一定像正负随机输入那样互相抵消，因此 cand17/20/10 的累加误差更容易显现。",
            "- `outlier_channels` 中少量大值主导整体 L2 范数，cand17 的大误差更受控，所以 `rel_l2` 接近甚至略低于 LSAM1；但它的 `MAE` 仍更高，说明普通位置的小误差更多。",
            "- 这说明当前 FPGA 候选的优势更像是“限制最坏误差”，而 LSAM1 的优势更像是“保护小值/稀疏区域”。真实 LLM 里哪种更重要，需要下一步用真实 activation/weight 分布验证。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    luts = _load_luts(Path(args.tcasi_lut_dir), Path(args.fpga_lut_dir))
    shape = (args.m, args.k, args.n)

    product = product_report(luts)
    gemm = gemm_report(luts, shape=shape, seed=args.seed)
    all_data = {
        "product_level": product,
        "synthetic_gemm": gemm,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "seed": args.seed,
    }

    product_json = out_dir / "signed_int8_product_comparison.json"
    product_md = out_dir / "signed_int8_product_comparison.md"
    gemm_json = out_dir / "signed_int8_gemm_synthetic_comparison.json"
    gemm_md = out_dir / "signed_int8_gemm_synthetic_comparison.md"

    product_json.write_text(json.dumps(product, indent=2), encoding="utf-8")
    gemm_json.write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    write_product_markdown(product_md, product)
    write_gemm_markdown(gemm_md, gemm, shape=shape, seed=args.seed)

    print(f"wrote {product_json}")
    print(f"wrote {product_md}")
    print(f"wrote {gemm_json}")
    print(f"wrote {gemm_md}")


if __name__ == "__main__":
    main()
