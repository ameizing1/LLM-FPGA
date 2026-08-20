"""Probe real W8A8 Linear-layer distributions and LUT-backed GEMM errors.

This script intentionally runs a small diagnostic workload instead of full
perplexity evaluation.  It samples inputs to Hugging Face Linear layers,
quantizes activations/weights to signed int8, then evaluates how TCASI24 LSAM1
and FPGA candidate LUTs behave on the sampled GEMM fragments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm


DESIGN_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
    "cand17_exact_if_min_abs_le_32": "cand17 + exact(min<=32)",
    "cand17_lsam1_if_min_abs_le_32": "cand17 + LSAM1(min<=32)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m", help="Hugging Face model id or local path")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-offset", type=int, default=0, help="skip this many non-empty text samples before probing")
    parser.add_argument("--text-samples", type=int, default=8, help="number of non-empty text samples")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-linear-layers", type=int, default=12)
    parser.add_argument("--max-rows-per-layer", type=int, default=256)
    parser.add_argument("--max-cols-per-layer", type=int, default=256)
    parser.add_argument("--product-pairs-per-layer", type=int, default=200_000)
    parser.add_argument("--activation-scale", choices=["per_tensor", "per_token"], default="per_token")
    parser.add_argument("--weight-scale", choices=["per_tensor", "per_channel"], default="per_channel")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--local-files-only", action="store_true", help="load Hugging Face model files from local cache")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--hybrid-lut-dir", default="outputs/hybrid_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="real_w8a8_distribution_probe")
    parser.add_argument("--save-pair-histogram", action="store_true", help="save sampled signed-int8 pair histogram as .npy")
    parser.add_argument(
        "--pair-histogram-path",
        default="",
        help="optional explicit path for the saved pair histogram .npy file",
    )
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ["torch", "transformers", "datasets"] if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing Python packages: {joined}. Install the AxCore runtime first, "
            "for example torch + transformers + datasets, then rerun this script."
        )


def _metrics(exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    exact64 = exact.astype(np.int64)
    approx64 = approx.astype(np.int64)
    err = approx64 - exact64
    abs_err = np.abs(err)
    denom = max(float(np.linalg.norm(exact64.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "max_abs_error": float(np.max(abs_err)),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
    }


def _symmetric_quant_np(
    x: np.ndarray,
    *,
    axis: int | None,
    qmax: int = 127,
) -> np.ndarray:
    arr = x.astype(np.float32, copy=False)
    if axis is None:
        scale = np.max(np.abs(arr)) / qmax
        if not np.isfinite(scale) or scale == 0.0:
            return np.zeros_like(arr, dtype=np.int8)
        return np.clip(np.rint(arr / scale), -128, 127).astype(np.int8)

    max_abs = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = max_abs / qmax
    scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
    return np.clip(np.rint(arr / scale), -128, 127).astype(np.int8)


def _quantize_activation(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "per_tensor":
        return _symmetric_quant_np(x, axis=None)
    if mode == "per_token":
        return _symmetric_quant_np(x, axis=1)
    raise ValueError(f"unknown activation scale mode: {mode}")


def _quantize_weight(weight_out_in: np.ndarray, mode: str) -> np.ndarray:
    if mode == "per_tensor":
        return _symmetric_quant_np(weight_out_in, axis=None)
    if mode == "per_channel":
        return _symmetric_quant_np(weight_out_in, axis=1)
    raise ValueError(f"unknown weight scale mode: {mode}")


def _pair_distribution(
    a_q: np.ndarray,
    b_q: np.ndarray,
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[dict[str, float], np.ndarray | None]:
    m, k = a_q.shape
    k2, n = b_q.shape
    if k != k2:
        raise ValueError(f"incompatible product sampling shapes: {a_q.shape}, {b_q.shape}")
    total_pairs = m * k * n
    sample_count = min(max_pairs, total_pairs)

    row_idx = rng.integers(0, m, size=sample_count)
    k_idx = rng.integers(0, k, size=sample_count)
    col_idx = rng.integers(0, n, size=sample_count)
    a = a_q[row_idx, k_idx].astype(np.int32)
    b = b_q[k_idx, col_idx].astype(np.int32)
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    abs_product = np.abs(a * b)
    hist = np.bincount((a.astype(np.int32) + 128) * 256 + (b.astype(np.int32) + 128), minlength=256 * 256)

    return {
        "sampled_pairs": float(sample_count),
        "pct_a_zero": float(np.mean(a == 0)),
        "pct_b_zero": float(np.mean(b == 0)),
        "pct_any_zero": float(np.mean((a == 0) | (b == 0))),
        "pct_min_abs_le_16": float(np.mean(np.minimum(abs_a, abs_b) <= 16)),
        "pct_min_abs_le_32": float(np.mean(np.minimum(abs_a, abs_b) <= 32)),
        "pct_both_abs_le_16": float(np.mean((abs_a <= 16) & (abs_b <= 16))),
        "pct_abs_product_le_512": float(np.mean(abs_product <= 512)),
        "pct_abs_product_le_1024": float(np.mean(abs_product <= 1024)),
        "pct_same_sign_nonzero": float(np.mean((a * b) > 0)),
        "pct_opposite_sign_nonzero": float(np.mean((a * b) < 0)),
        "mean_abs_a": float(np.mean(abs_a)),
        "mean_abs_b": float(np.mean(abs_b)),
        "mean_abs_product": float(np.mean(abs_product)),
    }, hist.astype(np.int64, copy=False)


def _load_luts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tcasi_dir = Path(args.tcasi_lut_dir)
    fpga_dir = Path(args.fpga_lut_dir)
    hybrid_dir = Path(args.hybrid_lut_dir)
    luts = {
        "tcasi24_lsam1": np.load(tcasi_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }
    optional = {
        "cand17_exact_if_min_abs_le_32": hybrid_dir / "cand17_exact_if_min_abs_le_32.npy",
        "cand17_lsam1_if_min_abs_le_32": hybrid_dir / "cand17_lsam1_if_min_abs_le_32.npy",
    }
    for name, path in optional.items():
        if path.exists():
            luts[name] = np.load(path).astype(np.int32)
    return luts


@dataclass
class LayerSample:
    name: str
    a_q: np.ndarray
    b_q: np.ndarray


class LinearCollector:
    def __init__(self, args: argparse.Namespace, torch_module: Any):
        self.args = args
        self.torch_module = torch_module
        self.samples: dict[str, LayerSample] = {}
        self.handles: list[Any] = []

    def attach(self, model: Any) -> None:
        linear_names = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, self.torch_module.nn.Linear)
        ][: self.args.max_linear_layers]

        for name, module in linear_names:
            handle = module.register_forward_pre_hook(self._make_hook(name, module))
            self.handles.append(handle)

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str, module: Any) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            x = inputs[0].detach().float().cpu()
            if x.ndim < 2:
                return

            existing = self.samples.get(name)
            if existing is not None and existing.a_q.shape[0] >= self.args.max_rows_per_layer:
                return

            x_np = x.reshape(-1, x.shape[-1]).numpy()
            remaining = self.args.max_rows_per_layer
            if existing is not None:
                remaining -= existing.a_q.shape[0]
            x_np = x_np[:remaining]
            if x_np.shape[0] == 0:
                return

            weight_np = module.weight.detach().float().cpu().numpy()
            a_q_new = _quantize_activation(x_np, self.args.activation_scale)
            w_q = _quantize_weight(weight_np, self.args.weight_scale)
            if existing is None:
                self.samples[name] = LayerSample(name=name, a_q=a_q_new, b_q=w_q.T.copy())
            else:
                existing.a_q = np.concatenate([existing.a_q, a_q_new], axis=0)

        return hook

    def is_full(self) -> bool:
        if len(self.samples) < self.args.max_linear_layers:
            return False
        return all(sample.a_q.shape[0] >= self.args.max_rows_per_layer for sample in self.samples.values())


def _load_texts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = []
    seen_non_empty = 0
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        if seen_non_empty < args.text_offset:
            seen_non_empty += 1
            continue
        texts.append(text)
        if len(texts) >= args.text_samples:
            break
    if not texts:
        raise RuntimeError("dataset did not provide any non-empty text samples")
    return texts


def _run_model_and_collect(args: argparse.Namespace) -> list[LayerSample]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.eval().to(args.device)

    collector = LinearCollector(args, torch)
    collector.attach(model)
    texts = _load_texts(args)

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            _ = model(**encoded)
            if collector.is_full():
                break

    collector.detach()
    return list(collector.samples.values())


def _evaluate_layer(
    sample: LayerSample,
    luts: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    b_q = sample.b_q
    if b_q.shape[1] > args.max_cols_per_layer:
        cols = np.sort(rng.choice(b_q.shape[1], size=args.max_cols_per_layer, replace=False))
        b_q = b_q[:, cols]

    exact = exact_gemm(sample.a_q, b_q)
    gemm_metrics = {name: _metrics(exact, lut_gemm(sample.a_q, b_q, lut)) for name, lut in luts.items()}
    dist_metrics, hist = _pair_distribution(
        sample.a_q,
        b_q,
        rng=rng,
        max_pairs=args.product_pairs_per_layer,
    )
    report = {
        "name": sample.name,
        "a_shape": list(sample.a_q.shape),
        "b_shape": list(b_q.shape),
        "distribution": dist_metrics,
        "gemm_metrics": gemm_metrics,
    }
    if args.save_pair_histogram:
        report["pair_histogram"] = hist.tolist() if hist is not None else None
    return report


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize(layer_reports: list[dict[str, Any]], lut_names: list[str]) -> dict[str, Any]:
    distributions = [layer["distribution"] for layer in layer_reports]
    summary: dict[str, Any] = {
        "layers": len(layer_reports),
        "mean_distribution": {
            key: _mean([float(dist[key]) for dist in distributions])
            for key in distributions[0]
            if key != "sampled_pairs"
        }
        if distributions
        else {},
        "mean_relative_l2_error": {},
    }
    for name in lut_names:
        summary["mean_relative_l2_error"][name] = _mean(
            [float(layer["gemm_metrics"][name]["relative_l2_error"]) for layer in layer_reports]
        )
    return summary


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    dist = data["summary"]["mean_distribution"]
    dist_rows = [
        {"指标": key, "均值": f"{value:.6f}"}
        for key, value in dist.items()
    ]
    rel_rows = [
        {"设计": DESIGN_LABELS.get(key, key), "mean_rel_l2": f"{value:.6f}"}
        for key, value in data["summary"]["mean_relative_l2_error"].items()
    ]

    lines = [
        "# 真实 W8A8 Linear 输入分布与 TCASI/FPGA 对比报告",
        "",
        "## 实验目的",
        "",
        "这个实验用于判断 synthetic GEMM 中暴露的 cand17 退化，在真实模型的 W8A8 Linear 输入里是否也会出现。",
        "",
        "重点观察真实量化后的乘法输入对是否大量落在：",
        "",
        "$$",
        "\\min(|a|, |b|) \\le 32",
        "$$",
        "",
        "以及 TCASI24 LSAM1/CSAM2 与 FPGA cand17/20/10 在真实 Linear GEMM 片段上的相对误差差异。",
        "",
        "## 实验设置",
        "",
        f"- model：`{data['config']['model']}`",
        f"- dataset：`{data['config']['dataset']}` / `{data['config']['dataset_config']}` / `{data['config']['split']}`",
        f"- activation quantization：`{data['config']['activation_scale']}`",
        f"- weight quantization：`{data['config']['weight_scale']}`",
        f"- sampled Linear layers：`{data['summary']['layers']}`",
        f"- compared designs：`{len(data['design_labels'])}`",
        "",
        "## 平均输入分布",
        "",
        _markdown_table(dist_rows, ["指标", "均值"]),
        "",
        "## 平均 GEMM 相对误差",
        "",
        _markdown_table(rel_rows, ["设计", "mean_rel_l2"]),
        "",
        "## 每层结果",
        "",
    ]

    for layer in data["layers"]:
        dist = layer["distribution"]
        rows = []
        for name, metrics in layer["gemm_metrics"].items():
            rows.append(
                {
                    "设计": DESIGN_LABELS.get(name, name),
                    "MAE": f"{metrics['mae']:.3f}",
                    "RMSE": f"{metrics['rmse']:.3f}",
                    "max_abs": f"{metrics['max_abs_error']:.0f}",
                    "rel_l2": f"{metrics['relative_l2_error']:.6f}",
                }
            )
        lines.extend(
            [
                f"### {layer['name']}",
                "",
                f"- A shape：`{layer['a_shape']}`，B shape：`{layer['b_shape']}`",
                f"- `pct_min_abs_le_32`：`{dist['pct_min_abs_le_32']:.6f}`",
                f"- `pct_abs_product_le_1024`：`{dist['pct_abs_product_le_1024']:.6f}`",
                "",
                _markdown_table(rows, ["设计", "MAE", "RMSE", "max_abs", "rel_l2"]),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        _require_runtime()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    luts = _load_luts(args)
    samples = _run_model_and_collect(args)
    layer_reports = [_evaluate_layer(sample, luts, rng=rng, args=args) for sample in samples]

    pair_histogram = None
    if args.save_pair_histogram:
        pair_histogram = np.zeros(256 * 256, dtype=np.int64)
        for layer in layer_reports:
            layer_hist = layer.pop("pair_histogram", None)
            if layer_hist is None:
                continue
            pair_histogram += np.asarray(layer_hist, dtype=np.int64)
        pair_histogram = pair_histogram.reshape(256, 256)
        hist_path = Path(args.pair_histogram_path) if args.pair_histogram_path else out_dir / f"{args.report_name}_pair_histogram.npy"
        np.save(hist_path, pair_histogram)
    else:
        hist_path = None

    data = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "text_offset": args.text_offset,
            "text_samples": args.text_samples,
            "max_seq_len": args.max_seq_len,
            "max_linear_layers": args.max_linear_layers,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "device": args.device,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "save_pair_histogram": args.save_pair_histogram,
        },
        "design_labels": {name: DESIGN_LABELS.get(name, name) for name in luts},
        "summary": _summarize(layer_reports, list(luts)),
        "layers": layer_reports,
    }
    if hist_path is not None:
        data["summary"]["pair_histogram_path"] = str(hist_path)

    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
