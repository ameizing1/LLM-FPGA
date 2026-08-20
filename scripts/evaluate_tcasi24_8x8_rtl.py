"""Evaluate TCASI24 open-source 8x8 RTL under signed W8A8-style tests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_int8_lut, exact_unsigned8_lut
from scripts.generate_fpga_signed_wrapper_luts import PRIMITIVES_VERILOG as BASE_PRIMITIVES_VERILOG
from scripts.run_real_w8a8_distribution_probe import (
    _evaluate_layer,
    _require_runtime,
    _run_model_and_collect,
    _summarize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "work" / "Approx_Mul_FPGA"

DESIGNS = {
    "ACCA_1111": {
        "src": "ACCA_1111.v",
        "top": "ac_1111",
        "paper_mred_pct": 0.3,
        "paper_lut": 57,
        "paper_carry4": 7,
        "paper_latency_ns": 5.215,
        "paper_power_mw": 408.208,
    },
    "MODA_1334": {
        "src": "MODA_1334.v",
        "top": "inexact_1334",
        "paper_mred_pct": 2.0,
        "paper_lut": 50,
        "paper_carry4": 3,
        "paper_latency_ns": 5.019,
        "paper_power_mw": 363.987,
    },
    "HSLP_1134": {
        "src": "HSLP_1134.v",
        "top": "HSLP_1134",
        "paper_mred_pct": 5.5,
        "paper_lut": 52,
        "paper_carry4": 3,
        "paper_latency_ns": 4.751,
        "paper_power_mw": 314.450,
    },
    "NCCA_1134": {
        "src": "NCCA_1134.v",
        "top": "LUT2_1134",
        "paper_mred_pct": 6.0,
        "paper_lut": 51,
        "paper_carry4": 2,
        "paper_latency_ns": 4.387,
        "paper_power_mw": 309.179,
    },
}

BASELINE_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1 simplified-8x8",
    "tcasi24_csam2": "TCASI24 CSAM2 simplified-8x8",
    "manualu88_approx5_1": "Manual approx5_1",
    "s8889_balanced_run_00_seed_100000_best_rtl": "S88-202689 Balanced best",
}

UNSIGNED_TB = r"""
module tb;
    reg [7:0] a;
    reg [7:0] b;
    wire [15:0] prod8;
    integer i;
    integer j;

    TOP_MODULE dut(.a(a), .b(b), .prod8(prod8));

    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                a = i[7:0];
                b = j[7:0];
                #1;
                $display("%0d,%0d,%0d", i, j, prod8);
            end
        end
        $finish;
    end
endmodule
"""

EXTRA_PRIMITIVES_VERILOG = r"""
module LUT5 #(parameter [31:0] INIT = 32'h0)
(
    input I0, input I1, input I2, input I3, input I4,
    output O
);
    assign O = INIT[{I4, I3, I2, I1, I0}];
endmodule
"""

PRIMITIVES_VERILOG = BASE_PRIMITIVES_VERILOG + EXTRA_PRIMITIVES_VERILOG

PRIMITIVES = {"LUT6_2", "LUT6", "LUT5", "CARRY4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--designs", nargs="+", default=list(DESIGNS))
    parser.add_argument("--lut-dir", default="outputs/tcasi24_8x8_luts")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--histogram-npy", default="outputs/reports/w8a8_calibration_hist_smoke_pair_histogram.npy")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="tcasi24_8x8_rtl_signed_w8a8")
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--top-k-gemm", type=int, default=4)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")

    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-offset", type=int, default=0)
    parser.add_argument("--text-samples", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-linear-layers", type=int, default=12)
    parser.add_argument("--max-rows-per-layer", type=int, default=256)
    parser.add_argument("--max-cols-per-layer", type=int, default=256)
    parser.add_argument("--product-pairs-per-layer", type=int, default=200_000)
    parser.add_argument("--activation-scale", choices=["per_tensor", "per_token"], default="per_token")
    parser.add_argument("--weight-scale", choices=["per_tensor", "per_channel"], default="per_channel")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _rtl_files(repo: Path, design: str) -> list[Path]:
    info = DESIGNS[design]
    four_src = repo / "TCAS" / "4x4" / "src"
    four = [four_src / name for name in ["LSAM1.v", "LSAM2.v", "CSAM1.v", "CSAM2.v"]]
    adders = sorted((repo / "TCAS" / "8x8" / "adder").glob("*.v"))
    src = repo / "TCAS" / "8x8" / "src" / info["src"]
    if not src.exists():
        raise FileNotFoundError(src)
    return [*four, *adders, src]


def _simulate_unsigned_lut(repo: Path, design: str, *, iverilog: str, vvp: str) -> np.ndarray:
    files = _rtl_files(repo, design)
    top = DESIGNS[design]["top"]
    with tempfile.TemporaryDirectory(prefix="tcasi8x8_") as tmp:
        tmp_dir = Path(tmp)
        primitives = tmp_dir / "xilinx_primitives_sim.v"
        testbench = tmp_dir / "tb_dump_unsigned_lut.v"
        sim_out = tmp_dir / "sim.vvp"
        primitives.write_text(PRIMITIVES_VERILOG, encoding="utf-8")
        testbench.write_text(UNSIGNED_TB.replace("TOP_MODULE", top), encoding="utf-8")

        cmd = [iverilog, "-g2012", "-o", str(sim_out), str(primitives), *map(str, files), str(testbench)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            run = subprocess.run([vvp, str(sim_out)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print("command failed:", " ".join(map(str, exc.cmd)), file=sys.stderr)
            if exc.stdout:
                print("stdout:", exc.stdout, file=sys.stderr)
            if exc.stderr:
                print("stderr:", exc.stderr, file=sys.stderr)
            raise

    lut = np.empty((256, 256), dtype=np.uint32)
    rows = 0
    for line in run.stdout.splitlines():
        if not re.fullmatch(r"\d+,\d+,\d+", line.strip()):
            continue
        a_s, b_s, p_s = line.strip().split(",")
        lut[int(a_s), int(b_s)] = int(p_s)
        rows += 1
    if rows != 256 * 256:
        raise AssertionError(f"{design}: expected 65536 rows, got {rows}")
    return lut


def _unsigned_metrics(lut: np.ndarray) -> dict[str, float]:
    exact = exact_unsigned8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    return {
        "er": float(np.mean(err != 0)),
        "med": float(np.mean(abs_err)),
        "ned_pct": float(np.mean(abs_err) / ((2**8 - 1) ** 2) * 100.0),
        "mred_pct": float(np.mean(abs_err[nonzero] / exact[nonzero]) * 100.0),
        "wce": float(np.max(abs_err)),
    }


def _signed_product_metrics(lut: np.ndarray) -> dict[str, float]:
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


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _load_modules(files: list[Path]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for path in files:
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
    body = modules[top]
    counts: Counter[str] = Counter({prim: _count_type_instances(body, prim) for prim in PRIMITIVES})
    for module_name in sorted(modules, key=len, reverse=True):
        if module_name == top:
            continue
        inst_count = _count_type_instances(body, module_name)
        if inst_count:
            sub = _expand_counts(modules, module_name, (*stack, top))
            for key, value in sub.items():
                counts[key] += inst_count * value
    return counts


def _resource_proxy(repo: Path, design: str) -> dict[str, int]:
    modules = _load_modules(_rtl_files(repo, design))
    counts = _expand_counts(modules, DESIGNS[design]["top"])
    return {
        "lut6_2": int(counts["LUT6_2"]),
        "lut6": int(counts["LUT6"]),
        "lut5": int(counts["LUT5"]),
        "lut_proxy": int(counts["LUT6_2"] + counts["LUT6"] + counts["LUT5"]),
        "carry4": int(counts["CARRY4"]),
    }


def _baseline_luts(tcasi_lut_dir: Path, fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    luts: dict[str, np.ndarray] = {}
    for name, path in {
        "tcasi24_lsam1": tcasi_lut_dir / "lsam1_int8_lut.npy",
        "tcasi24_csam2": tcasi_lut_dir / "csam2_int8_lut.npy",
        "manualu88_approx5_1": fpga_lut_dir / "manualu88_approx5_1_signed_wrapper_int8_lut.npy",
        "s8889_balanced_run_00_seed_100000_best_rtl": fpga_lut_dir
        / "s8889_balanced_run_00_seed_100000_best_rtl_signed_int8_lut.npy",
    }.items():
        if path.exists():
            luts[name] = np.load(path).astype(np.int32)
    return luts


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


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    product_rows = []
    for name, metrics in sorted(data["signed_product_metrics"].items(), key=lambda item: item[1]["mae"]):
        product_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "MAE": _fmt(metrics["mae"], 3),
                "RMSE": _fmt(metrics["rmse"], 3),
                "bias": _fmt(metrics["bias"], 3),
                "max_abs": _fmt(metrics["max_abs"], 0),
                "rel_l2": _fmt(metrics["relative_l2"], 6),
            }
        )

    weighted_rows = []
    for name, metrics in sorted(data["weighted_scores"].items(), key=lambda item: item[1]["weighted_mae"]):
        weighted_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "weighted MAE": _fmt(metrics["weighted_mae"], 6),
                "weighted RMSE": _fmt(metrics["weighted_rmse"], 6),
                "weighted bias": _fmt(metrics["weighted_bias"], 6),
            }
        )

    unsigned_rows = []
    for name, metrics in sorted(data["unsigned_metrics"].items(), key=lambda item: item[1]["mred_pct"]):
        paper = data["paper"].get(name, {})
        unsigned_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "RTL MRED": f"{metrics['mred_pct']:.3f}%",
                "paper MRED": "" if "paper_mred_pct" not in paper else f"{paper['paper_mred_pct']:.1f}%",
                "RTL MED": _fmt(metrics["med"], 3),
                "WCE": _fmt(metrics["wce"], 0),
            }
        )

    resource_rows = []
    for name, res in data["resource_proxy"].items():
        paper = data["paper"].get(name, {})
        resource_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "struct LUT proxy": str(res["lut_proxy"]),
                "struct CARRY4": str(res["carry4"]),
                "paper LUT": "" if "paper_lut" not in paper else str(paper["paper_lut"]),
                "paper CARRY4": "" if "paper_carry4" not in paper else str(paper["paper_carry4"]),
            }
        )

    lines = [
        "# TCASI24 8x8 开源 RTL signed W8A8 测试",
        "",
        "## 说明",
        "",
        f"- RTL repo: `{data['repo']}`",
        f"- commit: `{data['repo_commit']}`",
        f"- designs: `{', '.join(data['designs'])}`",
        f"- signed calibration histogram: `{data['histogram_path']}`",
        "",
        r"这里先把 TCASI24 unsigned 8x8 RTL 仿真为 \(256\times256\) product LUT，再用 signed-wrapper 方式接到 signed W8A8 行为测试中。这个 signed-wrapper 口径用于和我们当前 unsigned-core 候选保持一致，不代表 TCASI 原文已经提供 signed 8x8 版本。",
        "",
        "## Unsigned product-level：RTL 与论文口径核对",
        "",
        _table(unsigned_rows, ["设计", "RTL MRED", "paper MRED", "RTL MED", "WCE"]),
        "",
        "## Signed-wrapper product-level",
        "",
        _table(product_rows, ["设计", "MAE", "RMSE", "bias", "max_abs", "rel_l2"]),
        "",
        "## Signed calibration 分布加权误差",
        "",
        _table(weighted_rows, ["设计", "weighted MAE", "weighted RMSE", "weighted bias"]),
        "",
        "## 资源 proxy",
        "",
        _table(resource_rows, ["设计", "struct LUT proxy", "struct CARRY4", "paper LUT", "paper CARRY4"]),
        "",
        "说明：`struct LUT proxy` 是从 RTL 显式 primitive 递归展开得到的 `LUT6_2 + LUT6`，不是 Vivado post-synthesis utilization；paper 资源来自 TCASI24 Table VIII。",
        "",
    ]

    gemm = data.get("gemm")
    if gemm:
        gemm_rows = []
        for name, value in sorted(gemm["summary"]["mean_relative_l2_error"].items(), key=lambda item: item[1]):
            gemm_rows.append({"设计": data["labels"].get(name, name), "mean rel_l2": _fmt(value, 6)})
        lines.extend(
            [
                "## Signed W8A8 GEMM smoke",
                "",
                f"- sampled Linear layers: `{gemm['summary']['layers']}`",
                f"- selected TCASI RTL designs: `{', '.join(gemm['selected_designs'])}`",
                "",
                _table(gemm_rows, ["设计", "mean rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _repo_commit(repo: Path) -> str:
    try:
        run = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return run.stdout.strip()
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = np.load(args.histogram_npy).astype(np.int64)
    labels = dict(BASELINE_LABELS)
    unsigned_metrics: dict[str, dict[str, float]] = {}
    signed_product_metrics: dict[str, dict[str, float]] = {}
    resource_proxy: dict[str, dict[str, int]] = {}
    paper: dict[str, dict[str, Any]] = {}
    design_luts: dict[str, np.ndarray] = {}

    for design in args.designs:
        if design not in DESIGNS:
            raise ValueError(f"unknown design {design}; choices: {sorted(DESIGNS)}")
        design_id = f"tcasi8x8_{design.lower()}"
        labels[design_id] = f"TCASI24 {design} RTL"
        unsigned_path = lut_dir / f"{design_id}_unsigned8_lut.npy"
        signed_path = lut_dir / f"{design_id}_signed_wrapper_int8_lut.npy"
        if signed_path.exists() and unsigned_path.exists():
            unsigned_lut = np.load(unsigned_path).astype(np.uint32)
            signed_lut = np.load(signed_path).astype(np.int32)
        else:
            unsigned_lut = _simulate_unsigned_lut(repo, design, iverilog=args.iverilog, vvp=args.vvp)
            signed_lut = build_signed_wrapper_lut(unsigned_lut).astype(np.int32)
            np.save(unsigned_path, unsigned_lut)
            np.save(signed_path, signed_lut)
        unsigned_metrics[design_id] = _unsigned_metrics(unsigned_lut)
        signed_product_metrics[design_id] = _signed_product_metrics(signed_lut)
        resource_proxy[design_id] = _resource_proxy(repo, design)
        paper[design_id] = {k: v for k, v in DESIGNS[design].items() if k.startswith("paper_")}
        design_luts[design_id] = signed_lut
        print(
            f"{design_id}: unsigned MRED={unsigned_metrics[design_id]['mred_pct']:.3f}%, "
            f"signed weighted metrics pending",
            flush=True,
        )

    baseline_luts = _baseline_luts(Path(args.tcasi_lut_dir), Path(args.fpga_lut_dir))
    all_luts = {**baseline_luts, **design_luts}
    weighted_scores = {name: _weighted_metrics(hist, lut) for name, lut in all_luts.items()}

    data: dict[str, Any] = {
        "repo": str(repo),
        "repo_commit": _repo_commit(repo),
        "designs": args.designs,
        "histogram_path": str(args.histogram_npy),
        "labels": labels,
        "unsigned_metrics": unsigned_metrics,
        "signed_product_metrics": signed_product_metrics,
        "weighted_scores": weighted_scores,
        "resource_proxy": resource_proxy,
        "paper": paper,
        "lut_dir": str(lut_dir),
    }

    selected = [
        name
        for name, _ in sorted(
            ((name, weighted_scores[name]) for name in design_luts),
            key=lambda item: item[1]["weighted_mae"],
        )[: args.top_k_gemm]
    ]

    if not args.skip_gemm and selected:
        _require_runtime()
        rng = np.random.default_rng(args.seed)
        setattr(args, "save_pair_histogram", False)
        gemm_luts = {**baseline_luts, **{name: design_luts[name] for name in selected}}
        samples = _run_model_and_collect(args)
        layer_reports = [_evaluate_layer(sample, gemm_luts, rng=rng, args=args) for sample in samples]
        data["gemm"] = {
            "selected_designs": selected,
            "summary": _summarize(layer_reports, list(gemm_luts)),
            "layers": layer_reports,
        }

    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
