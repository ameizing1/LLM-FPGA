"""Build a clean 8x8 multiplier precision/resource comparison report."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from multiplier_models.signed_wrapper import exact_int8_lut


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
LUT_DIR = ROOT / "outputs" / "fpga_luts"
TCASI_LUT_DIR = ROOT / "outputs" / "luts"
HIST_PATH = REPORT_DIR / "w8a8_calibration_hist_smoke_pair_histogram.npy"

PRIMITIVES = {"LUT6_2", "LUT6", "CARRY4"}


SIMULATED_DESIGNS = [
    {
        "id": "tcasi24_lsam1",
        "name": "TCASI24 LSAM1 simplified-8x8",
        "kind": "TCASI behavior model",
        "lut": TCASI_LUT_DIR / "lsam1_int8_lut.npy",
        "resource_note": "4x LSAM1 4x4 core only; excludes exact 8x8 accumulation",
        "paper_lut": None,
        "paper_carry4": None,
        "paper_mred_pct": None,
    },
    {
        "id": "tcasi24_csam2",
        "name": "TCASI24 CSAM2 simplified-8x8",
        "kind": "TCASI behavior model",
        "lut": TCASI_LUT_DIR / "csam2_int8_lut.npy",
        "resource_note": "4x CSAM2 4x4 core only; excludes exact 8x8 accumulation",
        "paper_lut": None,
        "paper_carry4": None,
        "paper_mred_pct": None,
    },
    {
        "id": "fpga_cand17",
        "name": "Old FPGA cand17 signed-wrapper",
        "kind": "unsigned 8x8 + signed wrapper",
        "lut": LUT_DIR / "fpga_cand17_signed_wrapper_int8_lut.npy",
        "resource_path": ROOT / "FPGA_multiplier" / "approx_unsigned8x8" / "17",
        "top": "approx88_cascade",
    },
    {
        "id": "manualu88_approx5_1",
        "name": "Manual approx5_1: approx66 + accurate62",
        "kind": "unsigned 8x8 + signed wrapper",
        "lut": LUT_DIR / "manualu88_approx5_1_signed_wrapper_int8_lut.npy",
        "resource_path": ROOT / "FPGA_multiplier" / "unsigned8x8_approx_manual" / "approx5" / "1",
        "top": "approx88",
    },
    {
        "id": "manualu88_approx5_2",
        "name": "Manual approx5_2: approx66 + accurate62 variant",
        "kind": "unsigned 8x8 + signed wrapper",
        "lut": LUT_DIR / "manualu88_approx5_2_signed_wrapper_int8_lut.npy",
        "resource_path": ROOT / "FPGA_multiplier" / "unsigned8x8_approx_manual" / "approx5" / "2",
        "top": "approx88",
    },
    {
        "id": "s8862_balanced",
        "name": "S88-6x2 Balanced",
        "kind": "native signed 8x8",
        "lut": LUT_DIR / "s8862_balanced_signed_int8_lut.npy",
        "resource_path": ROOT / "FPGA_multiplier" / "signed8x8_6x2" / "Balanced",
        "top": "s88_top",
    },
    {
        "id": "s8889_balanced_run_00_seed_100000_best_rtl",
        "name": "S88-202689 Balanced best",
        "kind": "native signed 8x8",
        "lut": LUT_DIR / "s8889_balanced_run_00_seed_100000_best_rtl_signed_int8_lut.npy",
        "resource_path": ROOT
        / "FPGA_multiplier"
        / "signed8x8_202689_1800"
        / "balanced"
        / "run_00_seed_100000"
        / "best_rtl",
        "top": "s88_top",
        "ppl_alias": "s8888_balanced_run_00_seed_100000_best_rtl",
    },
    {
        "id": "s8889_fast_run_00_seed_0_best_rtl",
        "name": "S88-202689 Fast best",
        "kind": "native signed 8x8",
        "lut": LUT_DIR / "s8889_fast_run_00_seed_0_best_rtl_signed_int8_lut.npy",
        "resource_path": ROOT
        / "FPGA_multiplier"
        / "signed8x8_202689_1800"
        / "fast"
        / "run_00_seed_0"
        / "best_rtl",
        "top": "s88_top",
    },
    {
        "id": "s8889_area_run_02_seed_302000_best_rtl",
        "name": "S88-202689 Area best",
        "kind": "native signed 8x8",
        "lut": LUT_DIR / "s8889_area_run_02_seed_302000_best_rtl_signed_int8_lut.npy",
        "resource_path": ROOT
        / "FPGA_multiplier"
        / "signed8x8_202689_1800"
        / "area"
        / "run_02_seed_302000"
        / "best_rtl",
        "top": "s88_top",
    },
    {
        "id": "s8889_aggressive_run_06_seed_206000_best_rtl",
        "name": "S88-202689 Aggressive best",
        "kind": "native signed 8x8",
        "lut": LUT_DIR / "s8889_aggressive_run_06_seed_206000_best_rtl_signed_int8_lut.npy",
        "resource_path": ROOT
        / "FPGA_multiplier"
        / "signed8x8_202689_1800"
        / "aggressive"
        / "run_06_seed_206000"
        / "best_rtl",
        "top": "s88_top",
    },
]


TCASI24_TABLE_VIII = [
    {
        "name": "TCASI24 Proposed Acc",
        "paper_mred_pct": 0.0,
        "paper_power_mw": 424.639,
        "paper_latency_ns": 5.219,
        "paper_lut": 61,
        "paper_carry4": 7,
    },
    {
        "name": "TCASI24 ACCA_1111",
        "paper_mred_pct": 0.3,
        "paper_power_mw": 408.208,
        "paper_latency_ns": 5.215,
        "paper_lut": 57,
        "paper_carry4": 7,
    },
    {
        "name": "TCASI24 MODA_1334",
        "paper_mred_pct": 2.0,
        "paper_power_mw": 363.987,
        "paper_latency_ns": 5.019,
        "paper_lut": 50,
        "paper_carry4": 3,
    },
    {
        "name": "TCASI24 HSLP_1134",
        "paper_mred_pct": 5.5,
        "paper_power_mw": 314.450,
        "paper_latency_ns": 4.751,
        "paper_lut": 52,
        "paper_carry4": 3,
    },
    {
        "name": "TCASI24 NCCA_1134",
        "paper_mred_pct": 6.0,
        "paper_power_mw": 309.179,
        "paper_latency_ns": 4.387,
        "paper_lut": 51,
        "paper_carry4": 2,
    },
]


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


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
        raise KeyError(f"top/module {top!r} not found under available modules {sorted(modules)}")

    body = modules[top]
    counts: Counter[str] = Counter({prim: _count_type_instances(body, prim) for prim in PRIMITIVES})
    for module_name in sorted(modules, key=len, reverse=True):
        if module_name == top:
            continue
        inst_count = _count_type_instances(body, module_name)
        if inst_count:
            sub_counts = _expand_counts(modules, module_name, (*stack, top))
            for key, value in sub_counts.items():
                counts[key] += inst_count * value
    return counts


def _resource_counts(design: dict[str, Any]) -> dict[str, int | None]:
    folder = design.get("resource_path")
    top = design.get("top")
    if not folder or not top:
        return {"lut6_2": None, "lut6": None, "lut_proxy": None, "carry4": None}
    modules = _load_modules(Path(folder))
    counts = _expand_counts(modules, str(top))
    lut6_2 = int(counts["LUT6_2"])
    lut6 = int(counts["LUT6"])
    return {"lut6_2": lut6_2, "lut6": lut6, "lut_proxy": lut6_2 + lut6, "carry4": int(counts["CARRY4"])}


def _product_metrics(lut: np.ndarray) -> dict[str, float]:
    exact = exact_int8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    denom = max(float(np.linalg.norm(exact.ravel())), 1.0)
    return {
        "er": float(np.mean(err != 0)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "bias": float(np.mean(err)),
        "max_abs": float(np.max(abs_err)),
        "mred_nonzero_exact_pct": float(np.mean(abs_err[nonzero] / np.abs(exact[nonzero])) * 100.0),
        "relative_l2": float(np.linalg.norm(err.ravel()) / denom),
    }


def _weighted_metrics(hist: np.ndarray, lut: np.ndarray) -> dict[str, float]:
    counts = hist.astype(np.float64, copy=False)
    total = float(np.sum(counts))
    exact = exact_int8_lut().astype(np.float64)
    err = lut.astype(np.float64) - exact
    abs_err = np.abs(err)
    return {
        "weighted_mae": float(np.sum(counts * abs_err) / total),
        "weighted_rmse": float(np.sqrt(np.sum(counts * err**2) / total)),
        "weighted_bias": float(np.sum(counts * err) / total),
        "weighted_max_abs": float(np.max(abs_err)),
    }


def _load_json(name: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gemm_sources() -> dict[str, float]:
    out: dict[str, float] = {}
    for name in [
        "mixed_signed_quant_latest_full.json",
        "mixed_signed_quant_202689_1800_full.json",
    ]:
        data = _load_json(name)
        out.update(data.get("gemm", {}).get("summary", {}).get("mean_relative_l2_error", {}))
    return out


def _ppl_sources() -> dict[str, float]:
    out: dict[str, float] = {}
    for name in [
        "signed_w8a8_ppl_dist2055_512tok_key.json",
        "signed_w8a8_ppl_native_s8862_top2_512tok_key.json",
        "signed_w8a8_ppl_latest_top2_512tok_key.json",
        "signed_w8a8_ppl_202689_1800_top_seq64_mismatch.json",
    ]:
        data = _load_json(name)
        for row in data.get("results", []):
            out[row["design"]] = float(row["metrics"]["ppl"])
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[dict[str, str]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row.get(col, "") for col in cols) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    hist = np.load(HIST_PATH).astype(np.int64)
    gemm = _gemm_sources()
    ppl = _ppl_sources()
    rows: list[dict[str, Any]] = []

    for design in SIMULATED_DESIGNS:
        lut_path = Path(design["lut"])
        if not lut_path.exists():
            continue
        lut = np.load(lut_path).astype(np.int32)
        product = _product_metrics(lut)
        weighted = _weighted_metrics(hist, lut)
        res = _resource_counts(design)
        design_id = design["id"]
        rows.append(
            {
                "id": design_id,
                "name": design["name"],
                "kind": design["kind"],
                "lut_path": str(lut_path.relative_to(ROOT)),
                **res,
                **product,
                **weighted,
                "gemm_rel_l2": gemm.get(design_id),
                "ppl": ppl.get(design.get("ppl_alias", design_id)),
                "resource_note": design.get("resource_note", ""),
            }
        )

    json_path = REPORT_DIR / "tcasi8x8_and_our8x8_precision_resource_comparison.json"
    md_path = REPORT_DIR / "tcasi8x8_and_our8x8_precision_resource_comparison.md"
    json_path.write_text(
        json.dumps({"simulated_rows": rows, "tcasi24_table_viii": TCASI24_TABLE_VIII}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    precision_rows = []
    for row in sorted(rows, key=lambda item: item["weighted_mae"]):
        precision_rows.append(
            {
                "设计": row["name"],
                "类型": row["kind"],
                "MAE": _fmt(row["mae"], 3),
                "weighted MAE": _fmt(row["weighted_mae"], 6),
                "weighted RMSE": _fmt(row["weighted_rmse"], 6),
                "GEMM rel L2": _fmt(row["gemm_rel_l2"], 6),
                "PPL": _fmt(row["ppl"], 4),
            }
        )

    resource_rows = []
    for row in rows:
        resource_rows.append(
            {
                "设计": row["name"],
                "类型": row["kind"],
                "本地 LUT proxy": _fmt(row["lut_proxy"], 0),
                "本地 CARRY4": _fmt(row["carry4"], 0),
                "资源口径": row.get("resource_note") or "Verilog structural primitive count",
            }
        )
    for row in TCASI24_TABLE_VIII:
        resource_rows.append(
            {
                "设计": row["name"],
                "类型": "TCASI24 paper 8x8",
                "本地 LUT proxy": "",
                "本地 CARRY4": "",
                "资源口径": f"paper: {row['paper_lut']} LUT / {row['paper_carry4']} CARRY4, "
                f"MRED {row['paper_mred_pct']}%, {row['paper_latency_ns']} ns, {row['paper_power_mw']} mW",
            }
        )

    tcasi_paper_rows = [
        {
            "设计": row["name"],
            "paper MRED": f"{row['paper_mred_pct']:.1f}%",
            "paper LUT": str(row["paper_lut"]),
            "paper CARRY4": str(row["paper_carry4"]),
            "latency(ns)": f"{row['paper_latency_ns']:.3f}",
            "power(mW)": f"{row['paper_power_mw']:.3f}",
        }
        for row in TCASI24_TABLE_VIII
    ]

    lines = [
        "# TCASI24 8x8 与当前 8x8 乘法器精度-资源对比",
        "",
        "## 结论先看",
        "",
        "1. 目前能直接跑 signed W8A8 行为精度的是本地已有 `.npy` LUT：简化 TCASI LSAM1/CSAM2、旧 cand17、manual 系列、native signed S88 系列。",
        "2. TCASI24 论文里的 `ACCA_1111/MODA_1334/HSLP_1134/NCCA_1134` 暂时没有在本地找到 RTL，因此这里只能引用论文 Table VIII 的 8x8 product-level `MRED` 和资源，不能声称已经跑过 signed W8A8/GEMM/PPL。",
        "3. 当前最好的本地候选仍是 `Manual approx5_1`：真实 signed calibration 下 weighted MAE 接近 TCASI 简化 LSAM1，GEMM rel L2 明显更低；但它是 unsigned core + signed wrapper，最终硬件资源还要把符号处理计入。",
        "",
        "## 本地可仿真的 signed W8A8 精度",
        "",
        _table(precision_rows, ["设计", "类型", "MAE", "weighted MAE", "weighted RMSE", "GEMM rel L2", "PPL"]),
        "",
        "说明：`weighted MAE/RMSE` 使用当前 OPT-125M signed W8A8 calibration pair histogram；`GEMM rel L2` 和 `PPL` 若为空，表示当前还没有跑对应端到端/矩阵乘 smoke 测试。",
        "",
        "## TCASI24 论文 8x8 资源与 paper 精度",
        "",
        _table(tcasi_paper_rows, ["设计", "paper MRED", "paper LUT", "paper CARRY4", "latency(ns)", "power(mW)"]),
        "",
        "说明：这张表来自 TCASI24 Table VIII，口径是论文中的 unsigned 8x8 乘法器 product-level 结果和 FPGA synthesis 资源；它不是我们当前 signed W8A8 输入分布下的测试结果。",
        "",
        "## 资源消耗对比口径",
        "",
        _table(resource_rows, ["设计", "类型", "本地 LUT proxy", "本地 CARRY4", "资源口径"]),
        "",
        "## 需要注意",
        "",
        "- `本地 LUT proxy = LUT6_2 + LUT6`，只是从 Verilog 显式 primitive 递归展开得到的结构计数，不等价于 Vivado post-synthesis utilization。",
        "- manual unsigned-wrapper 的资源行只统计 unsigned core；真实 signed W8A8 硬件还需要取绝对值、符号异或、结果取反/补码等逻辑。",
        "- TCASI 简化 LSAM1/CSAM2 是四个同类 4x4 近似乘法块加 exact Python accumulation 的行为模型，不是论文 Table VIII 的 ACCA/MODA/HSLP/NCCA 完整 8x8 RTL。",
        "- 如果后续要写进论文/汇报成硬件结论，建议补 Vivado 同器件、同 top、同约束下的 LUT/CARRY4/timing/power。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
