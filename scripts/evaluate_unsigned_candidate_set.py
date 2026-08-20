"""Evaluate a folder of unsigned 8x8 approximate multiplier candidates.

The script follows the current AM-LUT workflow:

1. simulate each Verilog candidate into a raw uint8 x uint8 LUT;
2. report product-level error against exact uint8 multiplication;
3. report distribution-weighted product error using a saved calibration
   histogram;
4. optionally run the unsigned W8A8 zero-point GEMM probe on the best
   candidates selected by weighted MAE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.tcasi24 import mul8_unsigned
from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_unsigned8_lut
from scripts.generate_fpga_signed_wrapper_luts import _simulate_unsigned_lut
from scripts.run_unsigned_w8a8_zero_point_probe import (
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
    parser.add_argument(
        "--candidate-root",
        default="FPGA_multiplier/approx_unsigned8x8_202684_2055",
        help="directory containing numeric candidate subdirectories",
    )
    parser.add_argument("--candidate-prefix", default="fpga_dist2055_cand")
    parser.add_argument("--candidate-label-prefix", default="Dist2055 cand")
    parser.add_argument("--verilog-name", default="final_best_approx88_cascade.v")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument(
        "--histogram-npy",
        default="outputs/reports/unsigned_w8a8_calibration_hist_smoke_uint8_activation_uint8_weight_pair_histogram.npy",
        help="256x256 raw uint8 pair histogram used for distribution-weighted error",
    )
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="unsigned_candidate_set_2055_report")
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
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _candidate_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**9, path.name)


def _discover_candidates(root: Path, verilog_name: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for child in sorted((p for p in root.iterdir() if p.is_dir()), key=_candidate_sort_key):
        verilog = child / verilog_name
        if verilog.exists():
            candidates.append((child.name, verilog))
    if not candidates:
        raise FileNotFoundError(f"no candidates with {verilog_name} found under {root}")
    return candidates


def _raw_uint8_product_metrics(lut: np.ndarray) -> dict[str, float]:
    exact = exact_unsigned8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    rel = abs_err[nonzero] / exact[nonzero]
    return {
        "error_rate": float(np.mean(err != 0)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "bias": float(np.mean(err)),
        "max_abs": float(np.max(abs_err)),
        "mred_nonzero_exact": float(np.mean(rel)),
    }


def _weighted_metrics(hist: np.ndarray, lut: np.ndarray) -> dict[str, float]:
    counts = hist.astype(np.float64, copy=False)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("histogram is empty")

    exact = exact_unsigned8_lut().astype(np.float64)
    approx = lut.astype(np.float64)
    err = approx - exact
    abs_err = np.abs(err)
    return {
        "weighted_mae": float(np.sum(counts * abs_err) / total),
        "weighted_rmse": float(np.sqrt(np.sum(counts * err**2) / total)),
        "weighted_bias": float(np.sum(counts * err) / total),
        "weighted_max_abs": float(np.max(abs_err)),
    }


def _build_tcasi_unsigned_lut(mode: str) -> np.ndarray:
    lut = np.empty((256, 256), dtype=np.uint32)
    for a in range(256):
        for b in range(256):
            lut[a, b] = mul8_unsigned(a, b, mode)
    return lut


def _load_baseline_luts(fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    return {
        "tcasi24_lsam1": _build_tcasi_unsigned_lut("lsam1"),
        "tcasi24_csam2": _build_tcasi_unsigned_lut("csam2"),
        "fpga_cand17": np.load(fpga_lut_dir / "fpga_cand17_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand20": np.load(fpga_lut_dir / "fpga_cand20_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand10": np.load(fpga_lut_dir / "fpga_cand10_unsigned8_lut.npy").astype(np.uint32),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _format_score_rows(scores: dict[str, dict[str, float]], labels: dict[str, str]) -> list[dict[str, str]]:
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
    candidate_rows = []
    for name, metrics in sorted(data["candidate_product_metrics"].items(), key=lambda item: item[1]["mae"]):
        candidate_rows.append(
            {
                "设计": data["labels"].get(name, name),
                "ER": f"{metrics['error_rate']:.6f}",
                "MAE": f"{metrics['mae']:.3f}",
                "RMSE": f"{metrics['rmse']:.3f}",
                "bias": f"{metrics['bias']:.3f}",
                "max_abs": f"{metrics['max_abs']:.0f}",
                "MRED": f"{metrics['mred_nonzero_exact']:.6f}",
            }
        )

    lines = [
        "# unsigned 8x8 分布感知训练候选评测报告",
        "",
        "## 评测对象",
        "",
        f"- candidate root: `{data['candidate_root']}`",
        f"- 有效候选数: `{len(data['candidate_names'])}`",
        f"- calibration histogram: `{data['histogram_path']}`",
        f"- calibration sampled pairs: `{data['histogram_summary']['total_pairs']}`",
        f"- calibration nonzero bins: `{data['histogram_summary']['nonzero_bins']}`",
        f"- top raw uint8 bin: `{data['histogram_summary']['top_bin']}`",
        "",
        "这里的分布加权误差使用真实 unsigned W8A8 zero-point 路线采样到的 raw uint8 输入对：",
        "",
        "$$",
        r"\mathcal{L}_{uint8}=\sum_{q_a,q_b}P_{\mathrm{calib}}(q_a,q_b)\cdot|\hat p(q_a,q_b)-q_aq_b|",
        "$$",
        "",
        "## Product-Level 候选自身误差",
        "",
        _markdown_table(candidate_rows, ["设计", "ER", "MAE", "RMSE", "bias", "max_abs", "MRED"]),
        "",
        "## Calibration 分布加权误差",
        "",
        _markdown_table(
            _format_score_rows(data["weighted_scores"], data["labels"]),
            ["设计", "weighted_MAE", "weighted_RMSE", "weighted_bias", "weighted_max_abs"],
        ),
        "",
    ]

    gemm = data.get("gemm")
    if gemm is not None:
        lines.extend(
            [
                "## 真实输入 GEMM Smoke 结果",
                "",
                f"- model: `{gemm['config']['model']}`",
                f"- dataset: `{gemm['config']['dataset']}` / `{gemm['config']['dataset_config']}` / `{gemm['config']['split']}`",
                f"- activation quantization: `uint8 asymmetric {gemm['config']['activation_scale']}`",
                f"- weight quantization: `uint8 asymmetric {gemm['config']['weight_scale']}`",
                f"- sampled Linear layers: `{gemm['summary']['layers']}`",
                f"- GEMM 新候选选择: `{', '.join(gemm['selected_candidates'])}`",
                "",
            ]
        )
        for route_name, route_summary in gemm["summary"]["routes"].items():
            rows = []
            for name, value in sorted(
                route_summary["mean_dequant_relative_l2_error"].items(),
                key=lambda item: item[1],
            ):
                rows.append(
                    {
                        "设计": data["labels"].get(name, name),
                        "int_rel_l2": f"{route_summary['mean_int_relative_l2_error'][name]:.6f}",
                        "dequant_rel_l2": f"{value:.6f}",
                        "dequant_RMSE": f"{route_summary['mean_dequant_rmse'][name]:.6e}",
                    }
                )
            lines.extend(
                [
                    f"### {gemm['route_labels'][route_name]}",
                    "",
                    _markdown_table(rows, ["设计", "int_rel_l2", "dequant_rel_l2", "dequant_RMSE"]),
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    fpga_lut_dir = Path(args.fpga_lut_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpga_lut_dir.mkdir(parents=True, exist_ok=True)

    candidates = _discover_candidates(candidate_root, args.verilog_name)
    baseline_luts = _load_baseline_luts(fpga_lut_dir)
    labels = {**BASELINE_LABELS}
    candidate_luts: dict[str, np.ndarray] = {}
    candidate_product_metrics: dict[str, dict[str, float]] = {}
    source_verilog: dict[str, str] = {}

    for candidate_id, verilog_path in candidates:
        name = f"{args.candidate_prefix}{candidate_id}"
        labels[name] = f"{args.candidate_label_prefix}{candidate_id}"
        unsigned_path = fpga_lut_dir / f"{name}_unsigned8_lut.npy"
        signed_path = fpga_lut_dir / f"{name}_signed_wrapper_int8_lut.npy"
        if unsigned_path.exists():
            lut = np.load(unsigned_path).astype(np.uint32)
        else:
            lut = _simulate_unsigned_lut(verilog_path, iverilog=args.iverilog, vvp=args.vvp)
            np.save(unsigned_path, lut)
            np.save(signed_path, build_signed_wrapper_lut(lut).astype(np.int32))
        if lut.shape != (256, 256):
            raise ValueError(f"{name} LUT shape must be (256, 256), got {lut.shape}")
        candidate_luts[name] = lut
        candidate_product_metrics[name] = _raw_uint8_product_metrics(lut)
        source_verilog[name] = str(verilog_path)
        print(f"{name}: LUT ready, MAE={candidate_product_metrics[name]['mae']:.3f}")

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

    data: dict[str, Any] = {
        "candidate_root": str(candidate_root),
        "candidate_names": list(candidate_luts),
        "source_verilog": source_verilog,
        "lut_dir": str(fpga_lut_dir),
        "histogram_path": str(hist_path),
        "histogram_summary": {
            "total_pairs": int(hist.sum()),
            "nonzero_bins": int(np.count_nonzero(hist)),
            "top_bin": str(tuple(int(x) for x in np.unravel_index(int(np.argmax(hist)), hist.shape))),
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
            "route_labels": {
                "uint8_activation_int8_weight": "activation uint8 asymmetric, weight int8 symmetric",
                "uint8_activation_uint8_weight": "activation uint8 asymmetric, weight uint8 asymmetric",
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
