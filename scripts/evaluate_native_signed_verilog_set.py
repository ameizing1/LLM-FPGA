"""Evaluate native signed 8x8 Verilog multiplier variants.

Each variant directory is expected to contain a standard top module:

    module s88_top(input signed [7:0] a, input signed [7:0] b,
                   output signed [15:0] prod);

The script simulates all 256x256 signed int8 input pairs into a LUT, then runs
product-level, signed calibration-weighted, and optional signed W8A8 GEMM tests.
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

from multiplier_models.signed_wrapper import exact_int8_lut
from scripts.generate_fpga_signed_wrapper_luts import PRIMITIVES_VERILOG
from scripts.run_real_w8a8_distribution_probe import (
    _evaluate_layer,
    _require_runtime,
    _run_model_and_collect,
    _summarize,
)


BASELINE_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17 signed-wrapper",
    "fpga_cand20": "FPGA cand20 signed-wrapper",
    "fpga_cand10": "FPGA cand10 signed-wrapper",
}


TESTBENCH_VERILOG = r"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", default="FPGA_multiplier/signed8x8_6x2")
    parser.add_argument("--variant-prefix", default="s8862")
    parser.add_argument("--lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument(
        "--histogram-npy",
        default="outputs/reports/w8a8_calibration_hist_smoke_pair_histogram.npy",
        help="256x256 signed-int8 pair histogram, indexed by value + 128",
    )
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="native_signed8x8_6x2_report")
    parser.add_argument("--top-k-gemm", type=int, default=6)
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
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _discover_variants(root: Path) -> list[tuple[str, list[Path]]]:
    variants: list[tuple[str, list[Path]]] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(child.glob("*.v"))
        if (child / "signed88_approx_top.v").exists() and files:
            variants.append((child.name.lower(), files))
    if not variants:
        raise FileNotFoundError(f"no signed variants with signed88_approx_top.v found under {root}")
    return variants


def _simulate_signed_lut(verilog_files: list[Path], *, iverilog: str, vvp: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="s88_lut_") as tmp:
        tmp_dir = Path(tmp)
        primitives = tmp_dir / "xilinx_primitives_sim.v"
        testbench = tmp_dir / "tb_dump_signed_lut.v"
        sim_out = tmp_dir / "sim.vvp"
        primitives.write_text(PRIMITIVES_VERILOG, encoding="utf-8")
        testbench.write_text(TESTBENCH_VERILOG, encoding="utf-8")

        cmd = [iverilog, "-g2012", "-o", str(sim_out), str(primitives), *map(str, verilog_files), str(testbench)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        run = subprocess.run([vvp, str(sim_out)], check=True, capture_output=True, text=True)

    lut = np.empty((256, 256), dtype=np.int32)
    rows = 0
    for line in run.stdout.splitlines():
        if not re.fullmatch(r"-?\d+,-?\d+,-?\d+", line.strip()):
            continue
        a_s, b_s, p_s = line.strip().split(",")
        lut[int(a_s) + 128, int(b_s) + 128] = int(p_s)
        rows += 1
    if rows != 256 * 256:
        raise AssertionError(f"expected 65536 simulation rows, got {rows}")
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
    if total <= 0.0:
        raise ValueError("histogram is empty")
    exact = exact_int8_lut().astype(np.float64)
    approx = lut.astype(np.float64)
    err = approx - exact
    abs_err = np.abs(err)
    return {
        "weighted_mae": float(np.sum(counts * abs_err) / total),
        "weighted_rmse": float(np.sqrt(np.sum(counts * err**2) / total)),
        "weighted_bias": float(np.sum(counts * err) / total),
        "weighted_max_abs": float(np.max(abs_err)),
    }


def _load_baseline_luts(tcasi_lut_dir: Path, fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    return {
        "tcasi24_lsam1": np.load(tcasi_lut_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_lut_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_lut_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_lut_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_lut_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    product_rows = []
    for name, metrics in sorted(data["variant_product_metrics"].items(), key=lambda item: item[1]["mae"]):
        product_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "ER": f"{metrics['error_rate']:.6f}",
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
        "# native signed 8x8 乘法器评测报告",
        "",
        "## 评测对象",
        "",
        f"- variant root: `{data['variant_root']}`",
        f"- 有效设计数: `{len(data['variant_names'])}`",
        f"- signed calibration histogram: `{data['histogram_path']}`",
        f"- calibration sampled pairs: `{data['histogram_summary']['total_pairs']}`",
        f"- calibration nonzero bins: `{data['histogram_summary']['nonzero_bins']}`",
        f"- top signed bin: `{data['histogram_summary']['top_signed_bin']}`",
        "",
        "## Product-Level 误差",
        "",
        _markdown_table(product_rows, ["设计", "ER", "MAE", "RMSE", "bias", "max_abs", "rel_l2"]),
        "",
        "## Signed Calibration 分布加权误差",
        "",
        _markdown_table(weighted_rows, ["设计", "weighted_MAE", "weighted_RMSE", "weighted_bias", "weighted_max_abs"]),
        "",
    ]

    gemm = data.get("gemm")
    if gemm is not None:
        rows = []
        for name, value in sorted(gemm["summary"]["mean_relative_l2_error"].items(), key=lambda item: item[1]):
            rows.append({"设计": data["labels"].get(name, name), "mean_rel_l2": f"{value:.6f}"})
        lines.extend(
            [
                "## Signed W8A8 GEMM Smoke 结果",
                "",
                f"- model: `{gemm['config']['model']}`",
                f"- dataset: `{gemm['config']['dataset']}` / `{gemm['config']['dataset_config']}` / `{gemm['config']['split']}`",
                f"- activation quantization: `signed int8 symmetric {gemm['config']['activation_scale']}`",
                f"- weight quantization: `signed int8 symmetric {gemm['config']['weight_scale']}`",
                f"- sampled Linear layers: `{gemm['summary']['layers']}`",
                f"- GEMM 设计选择: `{', '.join(gemm['selected_variants'])}`",
                "",
                _markdown_table(rows, ["设计", "mean_rel_l2"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    variant_root = Path(args.variant_root)
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _discover_variants(variant_root)
    labels = {**BASELINE_LABELS}
    variant_luts: dict[str, np.ndarray] = {}
    variant_product_metrics: dict[str, dict[str, float]] = {}
    source_verilog: dict[str, list[str]] = {}
    for variant_name, verilog_files in variants:
        name = f"{args.variant_prefix}_{variant_name}"
        labels[name] = f"signed8x8_6x2 {variant_name}"
        lut_path = lut_dir / f"{name}_signed_int8_lut.npy"
        if lut_path.exists():
            lut = np.load(lut_path).astype(np.int32)
        else:
            lut = _simulate_signed_lut(verilog_files, iverilog=args.iverilog, vvp=args.vvp)
            np.save(lut_path, lut.astype(np.int32))
        if lut.shape != (256, 256):
            raise ValueError(f"{name} LUT shape must be (256, 256), got {lut.shape}")
        variant_luts[name] = lut
        variant_product_metrics[name] = _product_metrics(lut)
        source_verilog[name] = [str(path) for path in verilog_files]
        print(f"{name}: LUT ready, MAE={variant_product_metrics[name]['mae']:.3f}")

    hist_path = Path(args.histogram_npy)
    hist = np.load(hist_path)
    if hist.shape != (256, 256):
        raise ValueError(f"histogram must have shape (256, 256), got {hist.shape}")
    hist = hist.astype(np.int64, copy=False)
    baseline_luts = _load_baseline_luts(Path(args.tcasi_lut_dir), lut_dir)
    all_luts = {**baseline_luts, **variant_luts}
    weighted_scores = {name: _weighted_metrics(hist, lut) for name, lut in all_luts.items()}
    selected = [
        name
        for name, _ in sorted(
            ((name, weighted_scores[name]) for name in variant_luts),
            key=lambda item: item[1]["weighted_mae"],
        )[: args.top_k_gemm]
    ]
    top_idx = tuple(int(x) for x in np.unravel_index(int(np.argmax(hist)), hist.shape))

    data: dict[str, Any] = {
        "variant_root": str(variant_root),
        "variant_names": list(variant_luts),
        "source_verilog": source_verilog,
        "lut_dir": str(lut_dir),
        "histogram_path": str(hist_path),
        "histogram_summary": {
            "total_pairs": int(hist.sum()),
            "nonzero_bins": int(np.count_nonzero(hist)),
            "top_signed_bin": str((top_idx[0] - 128, top_idx[1] - 128)),
        },
        "labels": labels,
        "variant_product_metrics": variant_product_metrics,
        "weighted_scores": weighted_scores,
    }

    if not args.skip_gemm and args.top_k_gemm > 0:
        _require_runtime()
        rng = np.random.default_rng(args.seed)
        setattr(args, "save_pair_histogram", False)
        gemm_luts = {**baseline_luts, **{name: variant_luts[name] for name in selected}}
        samples = _run_model_and_collect(args)
        layer_reports = [_evaluate_layer(sample, gemm_luts, rng=rng, args=args) for sample in samples]
        data["gemm"] = {
            "selected_variants": selected,
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
