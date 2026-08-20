"""Build a focused report for reconsidering near-exact multiplier candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"


CANDIDATES = [
    {
        "id": "exact_w8a8",
        "name": "Exact signed W8A8",
        "kind": "exact signed baseline",
        "resource": "47 LUT6_2 + 10 CARRY4",
        "resource_source": "FPGA_multiplier/accurate_signed8x8/1/signed88.v 注释",
        "lut_proxy": 47,
        "carry4": 10,
        "metric_key": None,
        "ppl_key": "exact_w8a8",
        "note": "精确有符号 8x8 基准。",
    },
    {
        "id": "manualu88_approx2",
        "name": "Manual approx2",
        "kind": "unsigned core + signed wrapper",
        "resource": "23 LUT6_2 + 16 LUT6 + 6 CARRY4",
        "resource_source": "未找到 README，暂用 Verilog primitive count",
        "lut_proxy": 39,
        "carry4": 6,
        "metric_key": "manualu88_approx2",
        "ppl_key": None,
        "note": "当前 signed-wrapper 行为下输出等于 exact；若实际用于 signed W8A8，还需计入 wrapper 额外逻辑。",
    },
    {
        "id": "manualu88_approx5_1",
        "name": "Manual approx5_1",
        "kind": "unsigned core + signed wrapper",
        "resource": "37 LUT6_2 + 1 LUT6 + 9 CARRY4",
        "resource_source": "未找到 README，暂用 Verilog primitive count",
        "lut_proxy": 38,
        "carry4": 9,
        "metric_key": "manualu88_approx5_1",
        "ppl_key": "manualu88_approx5_1",
        "note": "精度很好，但学弟反馈其接近 accurate 改表；应作为高精度/低风险候选而不是激进近似候选。",
    },
    {
        "id": "manualu88_approx5_2",
        "name": "Manual approx5_2",
        "kind": "unsigned core + signed wrapper",
        "resource": "35 LUT6_2 + 1 LUT6 + 8 CARRY4",
        "resource_source": "未找到 README，暂用 Verilog primitive count",
        "lut_proxy": 36,
        "carry4": 8,
        "metric_key": "manualu88_approx5_2",
        "ppl_key": None,
        "note": "资源略低于 approx5_1，精度仍明显优于 Balanced 系列。",
    },
    {
        "id": "s8862_quality",
        "name": "S88-6x2 Quality",
        "kind": "native signed 8x8",
        "resource": "40 LUT6_2 + 8 CARRY4",
        "resource_source": "FPGA_multiplier/signed8x8_6x2/Quality/README.md",
        "lut_proxy": 40,
        "carry4": 8,
        "metric_key": "s8862_quality",
        "ppl_key": "s8862_quality",
        "note": "只近似最低权重 6x2 子块；相比 exact signed8x8 有资源节省，且精度/PPL 很稳。",
    },
    {
        "id": "s8862_balanced",
        "name": "S88-6x2 Balanced",
        "kind": "native signed 8x8",
        "resource": "39 LUT6_2 + 7 CARRY4",
        "resource_source": "FPGA_multiplier/signed8x8_6x2/Balanced/README.md",
        "lut_proxy": 39,
        "carry4": 7,
        "metric_key": "s8862_balanced",
        "ppl_key": "s8862_balanced",
        "note": "近似低、中两个 6x2 子块；资源更省，但 PPL 明显劣于 Quality。",
    },
    {
        "id": "s8889_balanced_run_00_seed_100000_best_rtl",
        "name": "S88-202689 Balanced best",
        "kind": "native signed 8x8",
        "resource": "39 LUT6_2 + 7 CARRY4",
        "resource_source": "best_rtl/README.md 与 trained_artifact",
        "lut_proxy": 39,
        "carry4": 7,
        "metric_key": "s8889_balanced_run_00_seed_100000_best_rtl",
        "ppl_key": "s8888_balanced_run_00_seed_100000_best_rtl",
        "note": "GEMM 比旧 Balanced 略好，但 512-token PPL 没有同步变好。",
    },
    {
        "id": "tcasi24_lsam1",
        "name": "TCASI24 LSAM1 simplified",
        "kind": "TCASI behavior baseline",
        "resource": "简化 4x4 行为模型，非完整 8x8 RTL资源",
        "resource_source": "当前本地简化模型；不可与 native 8x8 资源直接比较",
        "lut_proxy": None,
        "carry4": None,
        "metric_key": "tcasi24_lsam1",
        "ppl_key": "tcasi24_lsam1",
        "note": "精度基线，资源口径暂不和本地 native signed 8x8 混比。",
    },
    {
        "id": "tcasi24_csam2",
        "name": "TCASI24 CSAM2 simplified",
        "kind": "TCASI behavior baseline",
        "resource": "简化 4x4 行为模型，非完整 8x8 RTL资源",
        "resource_source": "当前本地简化模型；不可与 native 8x8 资源直接比较",
        "lut_proxy": None,
        "carry4": None,
        "metric_key": "tcasi24_csam2",
        "ppl_key": "tcasi24_csam2",
        "note": "低精度 TCASI 参考点。",
    },
]


def load_json(name: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_sources() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    weighted: dict[str, dict[str, Any]] = {}
    gemm: dict[str, float] = {}
    for name in ["mixed_signed_quant_latest_full.json", "mixed_signed_quant_202689_1800_full.json"]:
        data = load_json(name)
        weighted.update(data.get("weighted_scores", {}))
        gemm.update(data.get("gemm", {}).get("summary", {}).get("mean_relative_l2_error", {}))
    return weighted, gemm


def ppl_sources() -> dict[str, float]:
    ppl: dict[str, float] = {}
    for name in [
        "signed_w8a8_ppl_dist2055_512tok_key.json",
        "signed_w8a8_ppl_native_s8862_top2_512tok_key.json",
        "signed_w8a8_ppl_latest_top2_512tok_key.json",
    ]:
        data = load_json(name)
        for row in data.get("results", []):
            ppl[row["design"]] = float(row["metrics"]["ppl"])
    return ppl


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row.get(col, "") for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    weighted, gemm = metric_sources()
    ppl = ppl_sources()
    exact_lut = 47
    exact_carry = 10

    rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        key = cand["metric_key"]
        row = {
            **cand,
            "weighted_mae": weighted.get(key, {}).get("weighted_mae") if key else 0.0,
            "weighted_rmse": weighted.get(key, {}).get("weighted_rmse") if key else 0.0,
            "gemm_rel_l2": gemm.get(key) if key else 0.0,
            "ppl": ppl.get(cand["ppl_key"]) if cand.get("ppl_key") else None,
        }
        if cand["lut_proxy"] is not None:
            row["lut_saving_vs_exact_signed"] = (exact_lut - cand["lut_proxy"]) / exact_lut * 100.0
            row["carry_saving_vs_exact_signed"] = (exact_carry - cand["carry4"]) / exact_carry * 100.0
        else:
            row["lut_saving_vs_exact_signed"] = None
            row["carry_saving_vs_exact_signed"] = None
        rows.append(row)

    table_rows = []
    for row in sorted(rows, key=lambda item: (item["weighted_mae"] is None, item["weighted_mae"] or 0.0)):
        table_rows.append(
            {
                "设计": row["name"],
                "类型": row["kind"],
                "资源": row["resource"],
                "资源来源": row["resource_source"],
                "LUT节省": fmt(row["lut_saving_vs_exact_signed"], 1) + "%" if row["lut_saving_vs_exact_signed"] is not None else "",
                "CARRY4节省": fmt(row["carry_saving_vs_exact_signed"], 1) + "%" if row["carry_saving_vs_exact_signed"] is not None else "",
                "weighted MAE": fmt(row["weighted_mae"], 6),
                "weighted RMSE": fmt(row["weighted_rmse"], 6),
                "GEMM rel L2": fmt(row["gemm_rel_l2"], 6),
                "PPL(512tok)": fmt(row["ppl"], 4),
            }
        )

    lines = [
        "# 重新纳入近精确乘法器后的性能-资源对比",
        "",
        "## 口径说明",
        "",
        "- 性能使用当前已经测过的 signed W8A8 数据：真实 calibration 加权 product error、synthetic/real-layer GEMM mean relative L2、512-token PPL smoke。",
        "- 资源优先采用设计目录中的 README、trained_artifact 或 RTL 注释；没有文档说明的 manual unsigned 设计，暂时退回到 Verilog primitive count。",
        "- manual unsigned-wrapper 的资源只覆盖 unsigned core 本体。若最终用于 signed W8A8，需要额外加入取绝对值、符号处理和补码恢复逻辑，因此和 native signed 8x8 的资源比较要保守看。",
        "- `Exact signed W8A8` 的资源基准取 `47 LUT6_2 + 10 CARRY4`，节省比例仅作为结构级初步估计，不等价于 Vivado post-synthesis utilization。",
        "",
        "## 总表",
        "",
        table(
            table_rows,
            [
                "设计",
                "类型",
                "资源",
                "资源来源",
                "LUT节省",
                "CARRY4节省",
                "weighted MAE",
                "weighted RMSE",
                "GEMM rel L2",
                "PPL(512tok)",
            ],
        ),
        "",
        "## 判断",
        "",
        "1. `S88-6x2 Quality` 不能简单剔除。它相对 exact signed8x8 的标称资源从 `47/10` 降到 `40/8`，同时 weighted MAE、GEMM rel L2、PPL 都很稳，是当前最像“可写进主线”的 native signed 候选。",
        "2. `Manual approx5_1` 的精度甚至略好于 Quality，但它是 unsigned core + signed wrapper，而且资源没有设计文档背书。它适合作为高精度参考候选，暂时不适合直接宣称硬件收益。",
        "3. `Manual approx2` 在当前 signed-wrapper 行为下等价 exact，因此更像“结构实现/验证参考”，不是近似精度-资源 trade-off 的有效论据。",
        "4. `S88-202689 Balanced best` 和 `S88-6x2 Balanced` 资源相同，但 measured weighted MAE 更低；不过 PPL 反而更差，说明只优化 product/GEMM 指标还不足以保证端到端收益。",
        "5. 如果近期要向老师汇报，建议把 `S88-6x2 Quality`、`Manual approx5_1`、`S88-202689 Balanced best` 都放回候选池，但主结论写成“Quality 是当前最稳的 native signed 候选，manual approx5_1 是高精度参考，Balanced 系列提供更激进资源点但端到端仍需优化”。",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "reconsidered_near_exact_multiplier_tradeoff.json"
    md_path = REPORT_DIR / "reconsidered_near_exact_multiplier_tradeoff.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
