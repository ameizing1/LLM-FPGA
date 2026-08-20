"""Evaluate native-signed and unsigned-core multiplier folders under signed W8A8.

Native signed candidates are simulated through:

    module s88_top(input signed [7:0] a, input signed [7:0] b,
                   output signed [15:0] prod);

Unsigned candidates are simulated through:

    module approx88(input [7:0] a, input [7:0] b, output [15:0] prod);

and then converted to signed int8 behavior with the existing signed-wrapper:

    abs(a), abs(b) -> unsigned core -> restore sign
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_int8_lut
from scripts.generate_fpga_signed_wrapper_luts import PRIMITIVES_VERILOG
from scripts.run_real_w8a8_distribution_probe import (
    _evaluate_layer,
    _require_runtime,
    _run_model_and_collect,
    _summarize,
)


SIGNED_TESTBENCH = r"""
module tb;
    reg signed [7:0] a;
    reg signed [7:0] b;
    wire signed [15:0] prod;
    integer i;
    integer j;

    s88_top dut(.a(a), .b(b), .prod(prod));

    initial begin
        for (i = -128; i < 128; i = i + 1) begin
            for (j = -128; j < 128; j = j + 1) begin
                a = i[7:0];
                b = j[7:0];
                #1;
                $display("%0d,%0d,%0d", i, j, prod);
            end
        end
        $finish;
    end
endmodule
"""


UNSIGNED_TESTBENCH = r"""
module tb;
    reg [7:0] a;
    reg [7:0] b;
    wire [15:0] prod;
    integer i;
    integer j;

    approx88 dut(.a(a), .b(b), .prod(prod));

    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                a = i[7:0];
                b = j[7:0];
                #1;
                $display("%0d,%0d,%0d", i, j, prod);
            end
        end
        $finish;
    end
endmodule
"""


BASELINE_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17 signed-wrapper",
    "fpga_cand20": "FPGA cand20 signed-wrapper",
    "fpga_cand10": "FPGA cand10 signed-wrapper",
    "s8862_balanced": "signed8x8_6x2 balanced",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-signed-root", default="FPGA_multiplier/signed8x8_202688_1000")
    parser.add_argument("--unsigned-root", default="FPGA_multiplier/unsigned8x8_approx_manual")
    parser.add_argument("--native-prefix", default="s8888")
    parser.add_argument("--unsigned-prefix", default="manualu88")
    parser.add_argument("--lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--histogram-npy", default="outputs/reports/w8a8_calibration_hist_smoke_pair_histogram.npy")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="mixed_signed_quant_candidates")
    parser.add_argument("--top-k-gemm", type=int, default=8)
    parser.add_argument("--skip-gemm", action="store_true")
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
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _safe_name(path: Path, root: Path) -> str:
    return "_".join(path.relative_to(root).parts).lower().replace("-", "_")


def _native_method_label(variant_id: str) -> str:
    if variant_id.startswith("balanced_"):
        return "S88-Balanced: approx low/mid 6x2, high exact"
    if variant_id.startswith("fast_"):
        return "S88-Fast: all 6x2 use no-CARRY4 local carry prediction"
    if variant_id.startswith("area_"):
        return "S88-Area: AL quantized to multiples of 16, truncated LL low carry"
    if variant_id.startswith("aggressive_"):
        return "S88-Aggressive: LL LUT-only, CARRY4 only from exact fused MACs"
    return f"S88-{variant_id}"


def _unsigned_method_label(variant_id: str) -> str:
    labels = {
        "approx1": "Manual-1: approx66 + approx62 cross terms + LUT HH",
        "approx2": "Manual-Exact-ish: accurate66 + accurate22, signed-wrapper exact",
        "approx3": "Manual-3: approx66 + approx62 cross terms + LUT HH",
        "approx4": "Manual-4: approx66 + approx62 cross terms + LUT HH",
        "approx5_1": "Manual-Comp66-Accurate: approx66 + accurate62 cross terms",
        "approx5_2": "Manual-Comp66-Accurate variant: approx66 + accurate62 cross terms",
        "approx5_3": "Manual-Comp66-Accurate variant",
        "approx5_4": "Manual-Comp66-Accurate variant",
        "approx5_5": "Manual-Comp66-Accurate variant",
        "approx5_6": "Manual-Comp66-Accurate variant",
        "approx5_7": "Manual-LowOR-HighCarry comp66",
        "approx5_8": "Manual-LowOR-HighCarry comp66 optimized",
        "approx5_9": "Manual-LowOR-HighCarry comp66 optimized+",
        "approx5_10": "Manual-AllOR comp66",
        "approx5_11": "Manual-AllOR comp66 optimized INIT",
        "approx5_12": "Manual-LUTBudget6 comp66 remap",
        "approx5_13": "Manual-LUTBudget6 comp66 remap variant",
        "approx5_14": "Manual-LUTBudget6 comp66 remap variant",
        "approx5_15": "Manual-AllOR comp66 variant",
        "approx5_16": "Manual-AllOR comp66 variant",
    }
    return labels.get(variant_id, f"Manual-{variant_id}")


def _discover_native_signed(root: Path) -> list[tuple[str, list[Path]]]:
    variants: list[tuple[str, list[Path]]] = []
    for top in sorted(root.rglob("signed88_approx_top.v")):
        folder = top.parent
        variants.append((_safe_name(folder, root), sorted(folder.glob("*.v"))))
    return variants


def _discover_unsigned(root: Path) -> list[tuple[str, list[Path]]]:
    variants: list[tuple[str, list[Path]]] = []
    for top in sorted(root.rglob("approx88.v")):
        folder = top.parent
        variants.append((_safe_name(folder, root), sorted(folder.glob("*.v"))))
    return variants


def _simulate(verilog_files: list[Path], testbench_text: str, pattern: str, *, iverilog: str, vvp: str) -> list[tuple[int, int, int]]:
    with tempfile.TemporaryDirectory(prefix="mixed_lut_") as tmp:
        tmp_dir = Path(tmp)
        primitives = tmp_dir / "xilinx_primitives_sim.v"
        testbench = tmp_dir / "tb_dump_lut.v"
        sim_out = tmp_dir / "sim.vvp"
        primitives.write_text(PRIMITIVES_VERILOG, encoding="utf-8")
        testbench.write_text(testbench_text, encoding="utf-8")

        cmd = [iverilog, "-g2012", "-o", str(sim_out), str(primitives), *map(str, verilog_files), str(testbench)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        run = subprocess.run([vvp, str(sim_out)], check=True, capture_output=True, text=True)

    rows: list[tuple[int, int, int]] = []
    for line in run.stdout.splitlines():
        if re.fullmatch(pattern, line.strip()):
            a_s, b_s, p_s = line.strip().split(",")
            rows.append((int(a_s), int(b_s), int(p_s)))
    if len(rows) != 256 * 256:
        raise AssertionError(f"expected 65536 simulation rows, got {len(rows)}")
    return rows


def _simulate_signed_lut(verilog_files: list[Path], *, iverilog: str, vvp: str) -> np.ndarray:
    lut = np.empty((256, 256), dtype=np.int32)
    rows = _simulate(verilog_files, SIGNED_TESTBENCH, r"-?\d+,-?\d+,-?\d+", iverilog=iverilog, vvp=vvp)
    for a, b, prod in rows:
        lut[a + 128, b + 128] = prod
    return lut


def _simulate_unsigned_lut(verilog_files: list[Path], *, iverilog: str, vvp: str) -> np.ndarray:
    lut = np.empty((256, 256), dtype=np.uint32)
    rows = _simulate(verilog_files, UNSIGNED_TESTBENCH, r"\d+,\d+,\d+", iverilog=iverilog, vvp=vvp)
    for a, b, prod in rows:
        lut[a, b] = prod
    return lut


def _product_metrics(lut: np.ndarray) -> dict[str, float]:
    exact = exact_int8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    denom = max(float(np.linalg.norm(exact.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "bias": float(np.mean(err)),
        "max_abs": float(np.max(abs_err)),
        "mred_nonzero_exact": float(np.mean(abs_err[nonzero] / np.abs(exact[nonzero]))),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
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


def _load_baseline_luts(tcasi_lut_dir: Path, fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    luts = {
        "tcasi24_lsam1": np.load(tcasi_lut_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_lut_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_lut_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_lut_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_lut_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }
    for name in ["s8862_balanced"]:
        path = fpga_lut_dir / f"{name}_signed_int8_lut.npy"
        if path.exists():
            luts[name] = np.load(path).astype(np.int32)
    return luts


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    product_rows = []
    for name, metrics in sorted(data["candidate_product_metrics"].items(), key=lambda item: item[1]["mae"]):
        product_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "类型": data["candidate_types"][name],
                "MAE": f"{metrics['mae']:.3f}",
                "RMSE": f"{metrics['rmse']:.3f}",
                "bias": f"{metrics['bias']:.3f}",
                "max_abs": f"{metrics['max_abs']:.0f}",
                "rel_l2": f"{metrics['relative_l2_error']:.6f}",
            }
        )

    weighted_rows = []
    for name, metrics in sorted(data["weighted_scores"].items(), key=lambda item: item[1]["weighted_mae"]):
        weighted_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "weighted_MAE": f"{metrics['weighted_mae']:.6f}",
                "weighted_RMSE": f"{metrics['weighted_rmse']:.6f}",
                "weighted_bias": f"{metrics['weighted_bias']:.6f}",
                "weighted_max_abs": f"{metrics['weighted_max_abs']:.0f}",
            }
        )

    lines = [
        "# 最新乘法器 signed W8A8 行为测试",
        "",
        "## 测试对象",
        "",
        f"- native signed root: `{data['native_signed_root']}`",
        f"- unsigned root: `{data['unsigned_root']}`",
        f"- native signed 候选数: `{data['native_signed_count']}`",
        f"- unsigned-core signed-wrapper 候选数: `{data['unsigned_count']}`",
        f"- signed calibration histogram: `{data['histogram_path']}`",
        f"- sampled pairs: `{data['histogram_summary']['total_pairs']}`",
        "",
        "说明：`native_signed` 直接使用 Verilog 中的 signed 8x8 top；`unsigned_wrapper` 是先仿真 uint8 `approx88`，再按 signed-wrapper 方式用于 signed W8A8。",
        "",
        "## Product-Level signed int8 全空间误差",
        "",
        _markdown_table(product_rows, ["设计", "类型", "MAE", "RMSE", "bias", "max_abs", "rel_l2"]),
        "",
        "## Signed calibration 分布加权误差",
        "",
        _markdown_table(weighted_rows, ["设计", "weighted_MAE", "weighted_RMSE", "weighted_bias", "weighted_max_abs"]),
        "",
    ]

    gemm = data.get("gemm")
    if gemm:
        rows = []
        for name, value in sorted(gemm["summary"]["mean_relative_l2_error"].items(), key=lambda item: item[1]):
            rows.append({"设计": data["labels"].get(name, name), "mean_rel_l2": f"{value:.6f}"})
        lines.extend(
            [
                "## Signed W8A8 GEMM smoke 结果",
                "",
                f"- model: `{gemm['config']['model']}`",
                f"- activation: `signed int8 symmetric {gemm['config']['activation_scale']}`",
                f"- weight: `signed int8 symmetric {gemm['config']['weight_scale']}`",
                f"- sampled Linear layers: `{gemm['summary']['layers']}`",
                f"- 新候选选择: `{', '.join(gemm['selected_candidates'])}`",
                "",
                _markdown_table(rows, ["设计", "mean_rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    native_root = Path(args.native_signed_root)
    unsigned_root = Path(args.unsigned_root)
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered_native = _discover_native_signed(native_root)
    discovered_unsigned = _discover_unsigned(unsigned_root)
    if not discovered_native and not discovered_unsigned:
        raise FileNotFoundError("no native signed or unsigned candidates found")

    labels = {**BASELINE_LABELS}
    candidate_types: dict[str, str] = {}
    candidate_luts: dict[str, np.ndarray] = {}
    candidate_product_metrics: dict[str, dict[str, float]] = {}
    source_verilog: dict[str, list[str]] = {}

    for variant_id, verilog_files in discovered_native:
        name = f"{args.native_prefix}_{variant_id}"
        labels[name] = f"{_native_method_label(variant_id)} ({variant_id})"
        candidate_types[name] = "native_signed"
        lut_path = lut_dir / f"{name}_signed_int8_lut.npy"
        if lut_path.exists():
            lut = np.load(lut_path).astype(np.int32)
        else:
            lut = _simulate_signed_lut(verilog_files, iverilog=args.iverilog, vvp=args.vvp)
            np.save(lut_path, lut.astype(np.int32))
        candidate_luts[name] = lut
        candidate_product_metrics[name] = _product_metrics(lut)
        source_verilog[name] = [str(path) for path in verilog_files]
        print(f"{name}: native signed LUT ready, MAE={candidate_product_metrics[name]['mae']:.3f}", flush=True)

    for variant_id, verilog_files in discovered_unsigned:
        name = f"{args.unsigned_prefix}_{variant_id}"
        labels[name] = f"{_unsigned_method_label(variant_id)} ({variant_id})"
        candidate_types[name] = "unsigned_wrapper"
        unsigned_lut_path = lut_dir / f"{name}_unsigned8_lut.npy"
        signed_lut_path = lut_dir / f"{name}_signed_wrapper_int8_lut.npy"
        if signed_lut_path.exists():
            signed_lut = np.load(signed_lut_path).astype(np.int32)
        else:
            if unsigned_lut_path.exists():
                unsigned_lut = np.load(unsigned_lut_path).astype(np.uint32)
            else:
                unsigned_lut = _simulate_unsigned_lut(verilog_files, iverilog=args.iverilog, vvp=args.vvp)
                np.save(unsigned_lut_path, unsigned_lut)
            signed_lut = build_signed_wrapper_lut(unsigned_lut).astype(np.int32)
            np.save(signed_lut_path, signed_lut)
        candidate_luts[name] = signed_lut
        candidate_product_metrics[name] = _product_metrics(signed_lut)
        source_verilog[name] = [str(path) for path in verilog_files]
        print(f"{name}: unsigned-wrapper signed LUT ready, MAE={candidate_product_metrics[name]['mae']:.3f}", flush=True)

    hist_path = Path(args.histogram_npy)
    hist = np.load(hist_path)
    if hist.shape != (256, 256):
        raise ValueError(f"histogram must have shape (256, 256), got {hist.shape}")
    hist = hist.astype(np.int64, copy=False)

    baseline_luts = _load_baseline_luts(Path(args.tcasi_lut_dir), lut_dir)
    all_luts = {**baseline_luts, **candidate_luts}
    weighted_scores = {name: _weighted_metrics(hist, lut) for name, lut in all_luts.items()}
    selected = [
        name
        for name, _ in sorted(
            ((name, weighted_scores[name]) for name in candidate_luts),
            key=lambda item: item[1]["weighted_mae"],
        )[: args.top_k_gemm]
    ]
    top_idx = tuple(int(x) for x in np.unravel_index(int(np.argmax(hist)), hist.shape))

    data: dict[str, Any] = {
        "native_signed_root": str(native_root),
        "unsigned_root": str(unsigned_root),
        "native_signed_count": len(discovered_native),
        "unsigned_count": len(discovered_unsigned),
        "candidate_names": list(candidate_luts),
        "candidate_types": candidate_types,
        "source_verilog": source_verilog,
        "lut_dir": str(lut_dir),
        "histogram_path": str(hist_path),
        "histogram_summary": {
            "total_pairs": int(hist.sum()),
            "nonzero_bins": int(np.count_nonzero(hist)),
            "top_signed_bin": str((top_idx[0] - 128, top_idx[1] - 128)),
        },
        "labels": labels,
        "candidate_product_metrics": candidate_product_metrics,
        "weighted_scores": weighted_scores,
    }

    if not args.skip_gemm and args.top_k_gemm > 0:
        _require_runtime()
        rng = np.random.default_rng(args.seed)
        setattr(args, "save_pair_histogram", False)
        gemm_luts = {**baseline_luts, **{name: candidate_luts[name] for name in selected}}
        samples = _run_model_and_collect(args)
        layer_reports = [_evaluate_layer(sample, gemm_luts, rng=rng, args=args) for sample in samples]
        data["gemm"] = {
            "selected_candidates": selected,
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "text_offset": args.text_offset,
                "text_samples": args.text_samples,
                "max_seq_len": args.max_seq_len,
                "max_linear_layers": args.max_linear_layers,
                "max_rows_per_layer": args.max_rows_per_layer,
                "max_cols_per_layer": args.max_cols_per_layer,
                "activation_scale": args.activation_scale,
                "weight_scale": args.weight_scale,
                "device": args.device,
                "seed": args.seed,
                "local_files_only": args.local_files_only,
            },
            "summary": _summarize(layer_reports, list(gemm_luts)),
            "layers": layer_reports,
        }

    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
