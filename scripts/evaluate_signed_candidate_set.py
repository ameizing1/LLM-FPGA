"""Evaluate signed-wrapper int8 behavior for a candidate multiplier set.

This is the signed counterpart of evaluate_unsigned_candidate_set.py.  It uses
Verilog-simulated unsigned LUTs through the current signed-wrapper behavior:

    abs(a), abs(b) -> unsigned core -> restore sign

The result is a precision proxy for signed W8A8 evaluation, not a final signed
RTL area/timing result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_int8_lut
from scripts.evaluate_unsigned_candidate_set import _candidate_sort_key, _discover_candidates, _markdown_table
from scripts.generate_fpga_signed_wrapper_luts import _simulate_unsigned_lut
from scripts.run_real_w8a8_distribution_probe import (
    _evaluate_layer,
    _require_runtime,
    _run_model_and_collect,
    _summarize,
)


BASELINE_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default="FPGA_multiplier/approx_unsigned8x8_202684_2055")
    parser.add_argument("--candidate-prefix", default="fpga_dist2055_cand")
    parser.add_argument("--verilog-name", default="final_best_approx88_cascade.v")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument(
        "--histogram-npy",
        default="outputs/reports/w8a8_calibration_hist_smoke_pair_histogram.npy",
        help="256x256 signed-int8 pair histogram, indexed by value + 128",
    )
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="signed_candidate_set_2055_report")
    parser.add_argument("--top-k-gemm", type=int, default=5)
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


def _raw_signed_product_metrics(lut: np.ndarray) -> dict[str, float]:
    exact = exact_int8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    rel = abs_err[nonzero] / np.abs(exact[nonzero])
    denom = max(float(np.linalg.norm(exact.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "bias": float(np.mean(err)),
        "max_abs": float(np.max(abs_err)),
        "mred_nonzero_exact": float(np.mean(rel)),
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


def _format_weighted_rows(scores: dict[str, dict[str, float]], labels: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for name, metrics in sorted(scores.items(), key=lambda item: item[1]["weighted_mae"]):
        rows.append(
            {
                "设计": labels.get(name, name),
                "weighted_MAE": f"{metrics['weighted_mae']:.6f}",
                "weighted_RMSE": f"{metrics['weighted_rmse']:.6f}",
                "weighted_bias": f"{metrics['weighted_bias']:.6f}",
                "weighted_max_abs": f"{metrics['weighted_max_abs']:.0f}",
            }
        )
    return rows


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    product_rows = []
    for name, metrics in sorted(data["candidate_product_metrics"].items(), key=lambda item: item[1]["mae"]):
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

    lines = [
        "# signed W8A8 候选乘法器评测报告",
        "",
        "## 评测对象",
        "",
        f"- candidate root: `{data['candidate_root']}`",
        f"- 有效候选数: `{len(data['candidate_names'])}`",
        f"- signed calibration histogram: `{data['histogram_path']}`",
        f"- calibration sampled pairs: `{data['histogram_summary']['total_pairs']}`",
        f"- calibration nonzero bins: `{data['histogram_summary']['nonzero_bins']}`",
        f"- top signed bin: `{data['histogram_summary']['top_signed_bin']}`",
        "",
        "当前候选仍然是 unsigned core 的 signed-wrapper 行为模型：",
        "",
        "$$",
        r"\hat p_{\mathrm{signed}}(a,b)=\operatorname{sign}(a)\operatorname{sign}(b)\cdot M_{\mathrm{unsigned}}(|a|,|b|)",
        "$$",
        "",
        "分布加权误差定义为：",
        "",
        "$$",
        r"\mathcal{L}_{signed}=\sum_{a,b}P_{\mathrm{calib}}(a,b)\cdot|\hat p_{\mathrm{signed}}(a,b)-ab|",
        "$$",
        "",
        "## Product-Level 候选自身误差",
        "",
        _markdown_table(product_rows, ["设计", "ER", "MAE", "RMSE", "bias", "max_abs", "rel_l2"]),
        "",
        "## Signed Calibration 分布加权误差",
        "",
        _markdown_table(
            _format_weighted_rows(data["weighted_scores"], data["labels"]),
            ["设计", "weighted_MAE", "weighted_RMSE", "weighted_bias", "weighted_max_abs"],
        ),
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
                f"- GEMM 新候选选择: `{', '.join(gemm['selected_candidates'])}`",
                "",
                _markdown_table(rows, ["设计", "mean_rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    tcasi_lut_dir = Path(args.tcasi_lut_dir)
    fpga_lut_dir = Path(args.fpga_lut_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpga_lut_dir.mkdir(parents=True, exist_ok=True)

    candidates = _discover_candidates(candidate_root, args.verilog_name)
    baseline_luts = _load_baseline_luts(tcasi_lut_dir, fpga_lut_dir)
    labels = {**BASELINE_LABELS}
    candidate_luts: dict[str, np.ndarray] = {}
    candidate_product_metrics: dict[str, dict[str, float]] = {}
    source_verilog: dict[str, str] = {}

    for candidate_id, verilog_path in candidates:
        name = f"{args.candidate_prefix}{candidate_id}"
        labels[name] = f"Dist2055 cand{candidate_id}"
        unsigned_path = fpga_lut_dir / f"{name}_unsigned8_lut.npy"
        signed_path = fpga_lut_dir / f"{name}_signed_wrapper_int8_lut.npy"
        if signed_path.exists():
            signed_lut = np.load(signed_path).astype(np.int32)
        else:
            if unsigned_path.exists():
                unsigned_lut = np.load(unsigned_path).astype(np.uint32)
            else:
                unsigned_lut = _simulate_unsigned_lut(verilog_path, iverilog=args.iverilog, vvp=args.vvp)
                np.save(unsigned_path, unsigned_lut)
            signed_lut = build_signed_wrapper_lut(unsigned_lut).astype(np.int32)
            np.save(signed_path, signed_lut)
        if signed_lut.shape != (256, 256):
            raise ValueError(f"{name} signed LUT shape must be (256, 256), got {signed_lut.shape}")
        candidate_luts[name] = signed_lut
        candidate_product_metrics[name] = _raw_signed_product_metrics(signed_lut)
        source_verilog[name] = str(verilog_path)
        print(f"{name}: signed LUT ready, MAE={candidate_product_metrics[name]['mae']:.3f}")

    hist_path = Path(args.histogram_npy)
    hist = np.load(hist_path)
    if hist.shape != (256, 256):
        raise ValueError(f"histogram must have shape (256, 256), got {hist.shape}")
    hist = hist.astype(np.int64, copy=False)
    all_luts = {**baseline_luts, **candidate_luts}
    weighted_scores = {name: _weighted_metrics(hist, lut) for name, lut in all_luts.items()}
    top_candidates = [
        name
        for name, _ in sorted(
            ((name, weighted_scores[name]) for name in candidate_luts),
            key=lambda item: item[1]["weighted_mae"],
        )[: args.top_k_gemm]
    ]

    top_idx = tuple(int(x) for x in np.unravel_index(int(np.argmax(hist)), hist.shape))
    data: dict[str, Any] = {
        "candidate_root": str(candidate_root),
        "candidate_names": list(candidate_luts),
        "source_verilog": source_verilog,
        "lut_dir": str(fpga_lut_dir),
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
        gemm_luts = {**baseline_luts, **{name: candidate_luts[name] for name in top_candidates}}
        samples = _run_model_and_collect(args)
        layer_reports = [_evaluate_layer(sample, gemm_luts, rng=rng, args=args) for sample in samples]
        data["gemm"] = {
            "selected_candidates": top_candidates,
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
