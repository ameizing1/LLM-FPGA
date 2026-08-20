"""Build one table comparing TCASI24 8x8 RTL and current multiplier candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"


LOCAL_ROWS = [
    {
        "id": "exact_w8a8",
        "name": "Exact signed W8A8",
        "group": "local exact baseline",
        "resource": "47 LUT6_2 + 10 CARRY4",
        "resource_source": "accurate_signed8x8/1/signed88.v 注释",
        "metric_key": None,
        "ppl512_key": "exact_w8a8",
        "note": "本地 exact signed 8x8 基准。",
    },
    {
        "id": "manualu88_approx2",
        "name": "Manual approx2",
        "group": "local unsigned core + signed wrapper",
        "resource": "23 LUT6_2 + 16 LUT6 + 6 CARRY4",
        "resource_source": "无 README，暂用 Verilog primitive count",
        "metric_key": "manualu88_approx2",
        "ppl512_key": "manualu88_approx2",
        "note": "当前 wrapper 行为等价 exact；作为验证参考更合适。",
    },
    {
        "id": "manualu88_approx5_1",
        "name": "Manual approx5_1",
        "group": "local unsigned core + signed wrapper",
        "resource": "37 LUT6_2 + 1 LUT6 + 9 CARRY4",
        "resource_source": "无 README，暂用 Verilog primitive count",
        "metric_key": "manualu88_approx5_1",
        "ppl512_key": "manualu88_approx5_1",
        "note": "精度很好；但 signed wrapper 额外逻辑尚未计入最终综合。",
    },
    {
        "id": "manualu88_approx5_2",
        "name": "Manual approx5_2",
        "group": "local unsigned core + signed wrapper",
        "resource": "35 LUT6_2 + 1 LUT6 + 8 CARRY4",
        "resource_source": "无 README，暂用 Verilog primitive count",
        "metric_key": "manualu88_approx5_2",
        "ppl512_key": "manualu88_approx5_2",
        "note": "比 approx5_1 更省一些，精度仍较好。",
    },
    {
        "id": "s8862_quality",
        "name": "S88-6x2 Quality",
        "group": "local native signed 8x8",
        "resource": "40 LUT6_2 + 8 CARRY4",
        "resource_source": "signed8x8_6x2/Quality/README.md",
        "metric_key": "s8862_quality",
        "ppl512_key": "s8862_quality",
        "note": "只近似最低权重 6x2 子块；当前最稳的 native signed 候选。",
    },
    {
        "id": "s8862_balanced",
        "name": "S88-6x2 Balanced",
        "group": "local native signed 8x8",
        "resource": "39 LUT6_2 + 7 CARRY4",
        "resource_source": "signed8x8_6x2/Balanced/README.md",
        "metric_key": "s8862_balanced",
        "ppl512_key": "s8862_balanced",
        "note": "比 Quality 更省，但 PPL 劣化明显。",
    },
    {
        "id": "s8889_balanced_run_00_seed_100000_best_rtl",
        "name": "S88-202689 Balanced best",
        "group": "local native signed 8x8",
        "resource": "39 LUT6_2 + 7 CARRY4",
        "resource_source": "best_rtl/README.md 与 trained_artifact",
        "metric_key": "s8889_balanced_run_00_seed_100000_best_rtl",
        "ppl512_key": "s8888_balanced_run_00_seed_100000_best_rtl",
        "note": "GEMM 比旧 Balanced 略好，但 PPL 未同步变好。",
    },
]


TCASI_ROWS = [
    {
        "id": "tcasi8x8_acca_1111",
        "name": "TCASI24 ACCA_1111 RTL",
        "resource": "57 LUT + 7 CARRY4",
        "resource_source": "TCASI24 Table VIII；本地 RTL proxy 为 51 LUT proxy + 6 CARRY4",
        "paper_mred_pct": 0.3,
        "paper_latency_ns": 5.215,
        "paper_power_mw": 408.208,
        "note": "高精度 TCASI24 8x8 RTL；signed-wrapper 行为等同当前 LSAM1 简化模型。",
    },
    {
        "id": "tcasi8x8_moda_1334",
        "name": "TCASI24 MODA_1334 RTL",
        "resource": "50 LUT + 3 CARRY4",
        "resource_source": "TCASI24 Table VIII；本地 RTL proxy 为 46 LUT proxy + 3 CARRY4",
        "paper_mred_pct": 2.0,
        "paper_latency_ns": 5.019,
        "paper_power_mw": 363.987,
        "note": "CARRY4 明显更少，但 signed W8A8 精度下降较明显。",
    },
    {
        "id": "tcasi8x8_hslp_1134",
        "name": "TCASI24 HSLP_1134 RTL",
        "resource": "52 LUT + 3 CARRY4",
        "resource_source": "TCASI24 Table VIII；本地 RTL proxy 为 52 LUT proxy + 3 CARRY4",
        "paper_mred_pct": 5.5,
        "paper_latency_ns": 4.751,
        "paper_power_mw": 314.450,
        "note": "paper power 更低，但当前 signed W8A8 GEMM/PPL 劣化较大。",
    },
    {
        "id": "tcasi8x8_ncca_1134",
        "name": "TCASI24 NCCA_1134 RTL",
        "resource": "51 LUT + 2 CARRY4",
        "resource_source": "TCASI24 Table VIII；本地 RTL proxy 为 51 LUT proxy + 2 CARRY4",
        "paper_mred_pct": 6.0,
        "paper_latency_ns": 4.387,
        "paper_power_mw": 309.179,
        "note": "paper CARRY4/power 最低；signed-wrapper LUT 与 HSLP 当前等价。",
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
    for name in [
        "mixed_signed_quant_latest_full.json",
        "mixed_signed_quant_202689_1800_full.json",
        "tcasi24_8x8_rtl_signed_w8a8_gemm.json",
    ]:
        data = load_json(name)
        weighted.update(data.get("weighted_scores", {}))
        gemm.update(data.get("gemm", {}).get("summary", {}).get("mean_relative_l2_error", {}))
    return weighted, gemm


def ppl_sources(names: list[str]) -> dict[str, float]:
    ppl: dict[str, float] = {}
    for name in names:
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
    ppl512 = ppl_sources(
        [
            "signed_w8a8_ppl_dist2055_512tok_key.json",
            "signed_w8a8_ppl_native_s8862_top2_512tok_key.json",
            "signed_w8a8_ppl_latest_top2_512tok_key.json",
            "signed_w8a8_ppl_tcasi8x8_rtl_part1_seqlen512_512tok_key.json",
            "signed_w8a8_ppl_tcasi8x8_rtl_part2_seqlen512_512tok_key.json",
            "signed_w8a8_ppl_manual_extra_seqlen512_512tok_key.json",
        ]
    )
    exact_ppl = ppl512.get("exact_w8a8")

    rows: list[dict[str, Any]] = []
    for cand in LOCAL_ROWS:
        key = cand["metric_key"]
        rows.append(
            {
                **cand,
                "paper_mred_pct": None,
                "paper_latency_ns": None,
                "paper_power_mw": None,
                "weighted_mae": weighted.get(key, {}).get("weighted_mae") if key else 0.0,
                "weighted_rmse": weighted.get(key, {}).get("weighted_rmse") if key else 0.0,
                "gemm_rel_l2": gemm.get(key) if key else 0.0,
                "ppl512": ppl512.get(cand["ppl512_key"]) if cand.get("ppl512_key") else None,
            }
        )
    for cand in TCASI_ROWS:
        key = cand["id"]
        rows.append(
            {
                **cand,
                "group": "TCASI24 full 8x8 RTL",
                "weighted_mae": weighted.get(key, {}).get("weighted_mae"),
                "weighted_rmse": weighted.get(key, {}).get("weighted_rmse"),
                "gemm_rel_l2": gemm.get(key),
                "ppl512": ppl512.get(key),
            }
        )

    table_rows = []
    for row in rows:
        ppl = row.get("ppl512")
        ppl_delta = (ppl / exact_ppl - 1.0) * 100.0 if ppl is not None and exact_ppl else None
        table_rows.append(
            {
                "设计": row["name"],
                "类别": row["group"],
                "资源": row["resource"],
                "资源来源": row["resource_source"],
                "paper MRED": fmt(row.get("paper_mred_pct"), 1) + "%" if row.get("paper_mred_pct") is not None else "",
                "weighted MAE": fmt(row.get("weighted_mae"), 6),
                "weighted RMSE": fmt(row.get("weighted_rmse"), 6),
                "GEMM rel L2": fmt(row.get("gemm_rel_l2"), 6),
                "PPL 512tok": fmt(ppl, 4),
                "PPL变化": fmt(ppl_delta, 2) + "%" if ppl_delta is not None else "",
                "备注": row["note"],
            }
        )

    lines = [
        "# TCASI24 与当前候选乘法器性能-资源合并对比",
        "",
        "## 口径说明",
        "",
        "- 本地候选性能使用当前 signed W8A8 已测数据；TCASI24 `ACCA/MODA/HSLP/NCCA` 使用本地 RTL 生成 LUT 后得到的 signed-wrapper product/GEMM 数据。",
        "- TCASI24 资源优先引用论文 Table VIII 的 `LUT/CARRY4/latency/power`；本地候选资源优先引用设计 README 或 RTL 注释。两者还没有经过同一台 Vivado、同一器件、同一约束重综合，因此只能做阶段性横向判断。",
        "- `PPL 512tok` 统一采用 `seq_len=512, max_eval_tokens=512, eval_style=axcore`；这一版不再混用 128-token smoke 结果。",
        "- manual unsigned-wrapper 的资源只统计 unsigned core，本表仍未计入 signed wrapper 真实硬件开销。",
        "",
        "## 合并总表",
        "",
        table(
            table_rows,
            [
                "设计",
                "类别",
                "资源",
                "资源来源",
                "paper MRED",
                "weighted MAE",
                "weighted RMSE",
                "GEMM rel L2",
                "PPL 512tok",
                "PPL变化",
                "备注",
            ],
        ),
        "",
        "## 阶段性判断",
        "",
        "1. 如果优先看 native signed 8x8，`S88-6x2 Quality` 是当前最稳的本地候选：资源 `40 LUT6_2 + 8 CARRY4`，weighted MAE 和 GEMM rel L2 接近 `Manual approx5_1`，512-token PPL 也接近 exact W8A8。",
        "2. 如果和 TCASI24 高精度点比较，`TCASI24 ACCA_1111` 的 weighted MAE 最低，但 paper 资源为 `57 LUT + 7 CARRY4`；`S88-6x2 Quality` 的 LUT 更少、CARRY4 多 1 个，且 GEMM/PPL 表现更好，需要后续统一 Vivado 后才能下硬件结论。",
        "3. `TCASI24 MODA/HSLP/NCCA` 确实降低 CARRY4 和 paper power，但在当前 signed W8A8 输入分布下 weighted MAE/GEMM/PPL 劣化明显，不适合作为当前精度主线。",
        "4. `Manual approx5_1` 是强参考点，但由于是 unsigned core + signed wrapper，且资源缺少设计文档说明，暂时不应直接作为最终硬件优选结论。",
        "",
        "## LUT 与 CARRY4 怎么取舍",
        "",
        "- `LUT` 是 FPGA 里通用查找表资源，主要承载任意组合逻辑。LUT 用多了，通常会挤占普通控制逻辑、选择逻辑、路由资源，也可能增加布线压力。",
        "- `CARRY4` 是 Xilinx FPGA 里专门给加法/进位链准备的快速硬核资源。它特别适合加法器、累加器、比较器这类 carry propagation 逻辑，速度通常比纯 LUT 搭出来的进位链更好。",
        "- 对乘法器来说，`CARRY4` 往往对应部分积压缩、加法树、最终加法器。减少 CARRY4 可能降低面积和进位链功耗，但如果为了减少 CARRY4 增加很多 LUT 或绕远路，可能反而让延迟/路由变差。",
        "- 取舍上不能简单说 `1 个 CARRY4 等于多少个 LUT`。第一阶段可以分别列 `LUT` 和 `CARRY4`；第二阶段用 Vivado 同器件综合后同时看 `LUT utilization`、`CARRY4 utilization`、`timing slack/critical path` 和 `power`。",
        "- 对我们当前项目，建议排序优先级是：先满足 PPL/精度，再比较 LUT 与 CARRY4；在精度相近时，优先选 LUT/CARRY4 都不极端、timing 风险更小的设计。",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "tcasi_and_reconsidered_multiplier_tradeoff.json"
    md_path = REPORT_DIR / "tcasi_and_reconsidered_multiplier_tradeoff.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
