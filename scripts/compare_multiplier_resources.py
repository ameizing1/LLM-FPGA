"""Create a lightweight resource-vs-accuracy report for multiplier candidates.

The resource numbers in this script are structural primitive counts from the
checked-in Verilog, not post-place Vivado utilization. They are still useful as
a first-pass comparison because most candidate RTL explicitly instantiates
LUT6_2, LUT6, and CARRY4 primitives.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"


PRIMITIVES = {"LUT6_2", "LUT6", "CARRY4"}


CANDIDATES = [
    {
        "name": "Exact unsigned8x8",
        "id": "exact_unsigned88_1",
        "path": "FPGA_multiplier/accurate_unsigned8x8/1",
        "top": "accurate88",
        "kind": "exact unsigned",
        "lut_key": None,
        "ppl_key": None,
    },
    {
        "name": "Exact signed8x8",
        "id": "exact_signed88_1",
        "path": "FPGA_multiplier/accurate_signed8x8/1",
        "top": "signed88",
        "kind": "exact signed",
        "lut_key": None,
        "ppl_key": None,
        "declared_lut6_2": 47,
        "declared_lut6": 0,
        "declared_carry4": 10,
        "resource_source": "RTL comment",
    },
    {
        "name": "Manual-Exact-ish approx2",
        "id": "manualu88_approx2",
        "path": "FPGA_multiplier/unsigned8x8_approx_manual/approx2",
        "top": "approx88",
        "kind": "manual unsigned-wrapper",
        "lut_key": "manualu88_approx2",
        "ppl_key": None,
    },
    {
        "name": "Manual-Comp66-Accurate approx5_1",
        "id": "manualu88_approx5_1",
        "path": "FPGA_multiplier/unsigned8x8_approx_manual/approx5/1",
        "top": "approx88",
        "kind": "manual unsigned-wrapper",
        "lut_key": "manualu88_approx5_1",
        "ppl_key": "manualu88_approx5_1",
    },
    {
        "name": "Manual-Comp66-Accurate approx5_2",
        "id": "manualu88_approx5_2",
        "path": "FPGA_multiplier/unsigned8x8_approx_manual/approx5/2",
        "top": "approx88",
        "kind": "manual unsigned-wrapper",
        "lut_key": "manualu88_approx5_2",
        "ppl_key": None,
    },
    {
        "name": "signed8x8_6x2 Quality",
        "id": "s8862_quality",
        "path": "FPGA_multiplier/signed8x8_6x2/Quality",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8862_quality",
        "ppl_key": "s8862_quality",
        "declared_lut6_2": 40,
        "declared_lut6": 0,
        "declared_carry4": 8,
        "resource_source": "README",
    },
    {
        "name": "signed8x8_6x2 Balanced",
        "id": "s8862_balanced",
        "path": "FPGA_multiplier/signed8x8_6x2/Balanced",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8862_balanced",
        "ppl_key": "s8862_balanced",
        "declared_lut6_2": 39,
        "declared_lut6": 0,
        "declared_carry4": 7,
        "resource_source": "README",
    },
    {
        "name": "signed8x8_202689 Balanced best",
        "id": "s8889_balanced_run_00_seed_100000_best_rtl",
        "path": "FPGA_multiplier/signed8x8_202689_1800/balanced/run_00_seed_100000/best_rtl",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8889_balanced_run_00_seed_100000_best_rtl",
        "ppl_key": "s8888_balanced_run_00_seed_100000_best_rtl",
        "declared_lut6_2": 39,
        "declared_lut6": 0,
        "declared_carry4": 7,
        "resource_source": "trained_artifact/README",
    },
    {
        "name": "signed8x8_202689 Fast best",
        "id": "s8889_fast_run_00_seed_0_best_rtl",
        "path": "FPGA_multiplier/signed8x8_202689_1800/fast/run_00_seed_0/best_rtl",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8889_fast_run_00_seed_0_best_rtl",
        "ppl_key": None,
        "declared_lut6_2": 24,
        "declared_lut6": 0,
        "declared_carry4": 6,
        "resource_source": "trained_artifact/README",
    },
    {
        "name": "signed8x8_202689 Area best",
        "id": "s8889_area_run_02_seed_302000_best_rtl",
        "path": "FPGA_multiplier/signed8x8_202689_1800/area/run_02_seed_302000/best_rtl",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8889_area_run_02_seed_302000_best_rtl",
        "ppl_key": None,
        "declared_lut6_2": 16,
        "declared_lut6": 0,
        "declared_carry4": 5,
        "resource_source": "trained_artifact/README",
    },
    {
        "name": "signed8x8_202689 Aggressive best",
        "id": "s8889_aggressive_run_06_seed_206000_best_rtl",
        "path": "FPGA_multiplier/signed8x8_202689_1800/aggressive/run_06_seed_206000/best_rtl",
        "top": "s88_top",
        "kind": "native signed",
        "lut_key": "s8889_aggressive_run_06_seed_206000_best_rtl",
        "ppl_key": None,
        "declared_lut6_2": 22,
        "declared_lut6": 4,
        "declared_carry4": 4,
        "resource_source": "trained_artifact/README",
    },
]


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def _load_modules(folder: Path) -> dict[str, str]:
    modules: dict[str, str] = {}
    for path in sorted(folder.glob("*.v")):
        text = _strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        for match in re.finditer(r"\bmodule\s+(\w+)\b(.*?)(?=\bendmodule\b)", text, flags=re.S):
            modules[match.group(1)] = match.group(2)
    return modules


def _count_type_instances(body: str, type_name: str) -> int:
    pattern = re.compile(rf"\b{re.escape(type_name)}\b\s*(?:#\s*\(|[A-Za-z_][\w$]*\s*\()", re.S)
    return len(pattern.findall(body))


def _expand_counts(modules: dict[str, str], top: str, stack: tuple[str, ...] = ()) -> Counter[str]:
    if top in stack:
        raise ValueError(f"recursive module instantiation detected: {' -> '.join((*stack, top))}")
    if top not in modules:
        raise KeyError(f"top/module {top!r} not found; available: {sorted(modules)}")

    body = modules[top]
    counts: Counter[str] = Counter({primitive: _count_type_instances(body, primitive) for primitive in PRIMITIVES})

    for module_name in sorted(modules, key=len, reverse=True):
        if module_name == top:
            continue
        inst_count = _count_type_instances(body, module_name)
        if inst_count:
            sub_counts = _expand_counts(modules, module_name, (*stack, top))
            for key, value in sub_counts.items():
                counts[key] += inst_count * value

    return counts


def _load_json(name: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_sources() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    weighted: dict[str, dict[str, Any]] = {}
    gemm: dict[str, float] = {}

    for name in ["mixed_signed_quant_latest_full.json", "mixed_signed_quant_202689_1800_full.json"]:
        data = _load_json(name)
        weighted.update(data.get("weighted_scores", {}))
        gemm.update(data.get("gemm", {}).get("summary", {}).get("mean_relative_l2_error", {}))

    return weighted, gemm


def _ppl_sources() -> dict[str, float]:
    ppl: dict[str, float] = {}
    for name in [
        "signed_w8a8_ppl_dist2055_512tok_key.json",
        "signed_w8a8_ppl_native_s8862_top2_512tok_key.json",
        "signed_w8a8_ppl_latest_top2_512tok_key.json",
    ]:
        data = _load_json(name)
        for row in data.get("results", []):
            ppl[row["design"]] = float(row["metrics"]["ppl"])
    return ppl


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row.get(col, "") for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    weighted, gemm = _metric_sources()
    ppl = _ppl_sources()
    rows: list[dict[str, Any]] = []

    for cand in CANDIDATES:
        folder = ROOT / cand["path"]
        modules = _load_modules(folder)
        counts = _expand_counts(modules, cand["top"])
        auto_lut6_2 = counts["LUT6_2"]
        auto_lut6 = counts["LUT6"]
        auto_carry4 = counts["CARRY4"]
        lut6_2 = cand.get("declared_lut6_2", auto_lut6_2)
        lut6 = cand.get("declared_lut6", auto_lut6)
        carry4 = cand.get("declared_carry4", auto_carry4)
        lut_site_proxy = lut6_2 + lut6
        key = cand["lut_key"]
        rows.append(
            {
                **cand,
                "lut6_2": lut6_2,
                "lut6": lut6,
                "carry4": carry4,
                "lut_site_proxy": lut_site_proxy,
                "auto_lut6_2": auto_lut6_2,
                "auto_lut6": auto_lut6,
                "auto_carry4": auto_carry4,
                "resource_source": cand.get("resource_source", "auto primitive count"),
                "weighted_mae": weighted.get(key, {}).get("weighted_mae") if key else None,
                "gemm_rel_l2": gemm.get(key) if key else None,
                "ppl": ppl.get(cand["ppl_key"]) if cand.get("ppl_key") else None,
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "multiplier_resource_accuracy_comparison.json"
    md_path = REPORT_DIR / "multiplier_resource_accuracy_comparison.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "设计": row["name"],
                "类型": row["kind"],
                "LUT6_2": str(row["lut6_2"]),
                "LUT6": str(row["lut6"]),
                "LUT proxy": str(row["lut_site_proxy"]),
                "CARRY4": str(row["carry4"]),
                "资源来源": row["resource_source"],
                "weighted MAE": _fmt(row["weighted_mae"], 6),
                "GEMM rel L2": _fmt(row["gemm_rel_l2"], 6),
                "PPL": _fmt(row["ppl"], 4),
            }
        )

    lines = [
        "# 乘法器资源-精度对比",
        "",
        "说明：若候选目录 README / trained_artifact 明确给出资源，本表优先采用该标称结构资源；否则使用从 Verilog 递归展开得到的 primitive count，统计显式例化的 `LUT6_2`、`LUT6`、`CARRY4`。它们都不是 Vivado post-synthesis utilization，但可作为第一版横向资源趋势对比。",
        "",
        "`LUT proxy = LUT6_2 + LUT6`，只是粗略 LUT primitive 数；真实 FPGA 中 `LUT6_2` 的 packing、综合优化、unused output 修剪都会影响最终 LUT utilization。",
        "",
        "注意：`manual unsigned-wrapper` 行只统计 unsigned core 本体；如果实际用于 signed W8A8，还需要额外符号处理/取绝对值/符号恢复逻辑，最终资源应以后续综合结果为准。",
        "",
        _table(table_rows, ["设计", "类型", "LUT6_2", "LUT6", "LUT proxy", "CARRY4", "资源来源", "weighted MAE", "GEMM rel L2", "PPL"]),
        "",
        "## 当前观察",
        "",
        "1. `Manual-Comp66-Accurate approx5_1` 精度很好，unsigned core 本体资源与 native signed Balanced 接近；考虑 signed-wrapper 额外逻辑后，它未必有明显资源优势。",
        "2. `signed8x8_6x2 Quality` 的标称结构资源是 `40 LUT6_2 + 8 CARRY4`，相比 `Exact signed8x8` 注释中的 `47 LUT6_2 + 10 CARRY4` 有一定节省，但仍需 Vivado 综合确认真实 utilization。",
        "3. `signed8x8_202689 Balanced best` 与 `signed8x8_6x2 Balanced` 的标称结构资源相同，GEMM 略好，但 PPL 没有同步变好。",
        "4. `Area` 和 `Aggressive` 的 LUT/CARRY4 更省，但 weighted MAE 明显变大，当前不适合作为主精度候选。",
        "5. 后续若要做论文级资源结论，需要补 Vivado 同板卡、同约束、同 top、同综合策略下的 utilization/timing/power。",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
