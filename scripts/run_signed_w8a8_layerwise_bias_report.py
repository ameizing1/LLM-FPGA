"""Diagnose layer-wise signed W8A8 GEMM error bias for selected LUT multipliers.

The model is first converted to Exact signed W8A8 Linear layers.  Hooks then
capture the inputs actually seen by each Linear layer in that common reference
forward pass.  Every approximate LUT is evaluated against the same quantized
activation/weight fragments, so this report isolates local multiplier error
from error propagation between transformer blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm
import run_signed_w8a8_ppl_probe as signed_probe


DEFAULT_DESIGNS = (
    "s8862_balanced",
    "s88ref_balanced_topology_best_rtl",
    "s8862_quality",
)


@dataclass
class LayerSample:
    name: str
    activation: np.ndarray
    activation_scale: np.ndarray
    weight: np.ndarray
    weight_scale: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-eval-tokens", type=int, default=512)
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="skip this many tokens before capturing a common Exact W8A8 forward pass",
    )
    parser.add_argument("--eval-style", choices=["current", "axcore"], default="axcore")
    parser.add_argument("--activation-scale", choices=["per_tensor", "per_token"], default="per_token")
    parser.add_argument("--weight-scale", choices=["per_tensor", "per_channel"], default="per_channel")
    parser.add_argument("--designs", nargs="+", default=list(DEFAULT_DESIGNS))
    parser.add_argument("--max-linear-layers", type=int, default=0)
    parser.add_argument("--max-rows-per-layer", type=int, default=32)
    parser.add_argument("--max-cols-per-layer", type=int, default=128)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--tcasi8x8-lut-dir", default="outputs/tcasi24_8x8_luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="signed88_balanced_layerwise_bias")
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ("torch", "transformers", "datasets") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _stable_columns(name: str, count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=limit, replace=False))


def _even_rows(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


class ExactW8A8Collector:
    def __init__(self, args: argparse.Namespace, torch_module: Any, replaced: list[str]):
        self.args = args
        self.torch = torch_module
        self.replaced = set(replaced)
        self.samples: dict[str, LayerSample] = {}
        self.handles: list[Any] = []

    def attach(self, model: Any) -> None:
        modules = dict(model.named_modules())
        for name in sorted(self.replaced):
            self.handles.append(modules[name].register_forward_pre_hook(self._make_hook(name)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            if name in self.samples or not inputs:
                return
            source = inputs[0].detach().float()
            if source.ndim < 2:
                return
            flat = source.reshape(-1, source.shape[-1])
            q_activation, scale = signed_probe._quantize_activation(flat, self.args.activation_scale)
            rows = _even_rows(q_activation.shape[0], self.args.max_rows_per_layer)
            row_index = self.torch.as_tensor(rows, dtype=self.torch.long, device=q_activation.device)
            q_selected = q_activation[row_index].cpu().numpy().astype(np.int8, copy=False)
            if scale.numel() == 1:
                scale_selected = scale.reshape(1, 1).cpu().numpy().astype(np.float32, copy=False)
            else:
                scale_selected = scale[row_index].cpu().numpy().astype(np.float32, copy=False)

            columns = _stable_columns(name, module.out_features, self.args.max_cols_per_layer)
            column_index = self.torch.as_tensor(columns, dtype=self.torch.long, device=module.q_weight.device)
            q_weight = module.q_weight[column_index].cpu().numpy().astype(np.int8, copy=False)
            w_scale = module.w_scale[column_index].cpu().numpy().astype(np.float32, copy=False)
            self.samples[name] = LayerSample(
                name=name,
                activation=q_selected.copy(),
                activation_scale=scale_selected.copy(),
                weight=q_weight.T.copy(),
                weight_scale=w_scale.copy(),
            )

        return hook


def _error_metrics(error: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    err = error.astype(np.float64, copy=False)
    ref = exact.astype(np.float64, copy=False)
    abs_err = np.abs(err)
    count = int(err.size)
    abs_sum = float(abs_err.sum())
    signed_sum = float(err.sum())
    return {
        "count": count,
        "sum_error": signed_sum,
        "sum_abs_error": abs_sum,
        "sum_sq_error": float(np.square(err).sum()),
        "sum_sq_exact": float(np.square(ref).sum()),
        "error_count": int(np.count_nonzero(err)),
        "positive_error_count": int(np.count_nonzero(err > 0)),
        "negative_error_count": int(np.count_nonzero(err < 0)),
        "mean_error": signed_sum / max(count, 1),
        "mae": abs_sum / max(count, 1),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / max(float(np.linalg.norm(ref.ravel())), 1.0)),
        "signed_to_abs_ratio": abs(signed_sum) / max(abs_sum, 1e-12),
        "positive_error_rate": float(np.mean(err > 0)),
        "negative_error_rate": float(np.mean(err < 0)),
    }


def _aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    count = sum(int(item["count"]) for item in metrics)
    sum_error = sum(float(item["sum_error"]) for item in metrics)
    sum_abs = sum(float(item["sum_abs_error"]) for item in metrics)
    sum_sq_error = sum(float(item["sum_sq_error"]) for item in metrics)
    sum_sq_exact = sum(float(item["sum_sq_exact"]) for item in metrics)
    return {
        "count": count,
        "mean_error": sum_error / max(count, 1),
        "mae": sum_abs / max(count, 1),
        "rmse": float(np.sqrt(sum_sq_error / max(count, 1))),
        "relative_l2_error": float(np.sqrt(sum_sq_error) / max(np.sqrt(sum_sq_exact), 1.0)),
        "signed_to_abs_ratio": abs(sum_error) / max(sum_abs, 1e-12),
        "positive_error_rate": sum(int(item["positive_error_count"]) for item in metrics) / max(count, 1),
        "negative_error_rate": sum(int(item["negative_error_count"]) for item in metrics) / max(count, 1),
    }


def _capture_exact_w8a8_samples(args: argparse.Namespace) -> list[LayerSample]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    replaced = signed_probe._replace_linears(model, args, "exact_w8a8", None)
    model.eval().to(args.device)

    collector = ExactW8A8Collector(args, torch, replaced)
    collector.attach(model)
    try:
        token_offset = max(0, int(getattr(args, "token_offset", 0)))
        load_args = argparse.Namespace(**vars(args))
        load_args.max_eval_tokens = token_offset + max(2, args.max_eval_tokens)
        input_ids = signed_probe._load_eval_token_ids(load_args, tokenizer)
        input_ids = input_ids[
            :,
            token_offset : token_offset + max(2, args.max_eval_tokens),
        ].to(args.device)
        if input_ids.shape[1] < 2:
            raise RuntimeError(
                f"token offset {token_offset} leaves fewer than two tokens for capture"
            )
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        collector.detach()
    return [collector.samples[name] for name in sorted(collector.samples)]


def _evaluate_layer(sample: LayerSample, luts: dict[str, np.ndarray]) -> dict[str, Any]:
    exact_int = exact_gemm(sample.activation, sample.weight)
    scale = sample.activation_scale * sample.weight_scale.reshape(1, -1)
    exact_real = exact_int.astype(np.float64) * scale
    designs: dict[str, Any] = {}
    for name, lut in luts.items():
        approx_int = lut_gemm(sample.activation, sample.weight, lut)
        int_error = approx_int - exact_int
        real_error = int_error.astype(np.float64) * scale
        designs[name] = {
            "int_accumulator": _error_metrics(int_error, exact_int),
            "dequantized_output": _error_metrics(real_error, exact_real),
        }
    return {
        "name": sample.name,
        "activation_shape": list(sample.activation.shape),
        "weight_shape": list(sample.weight.shape),
        "designs": designs,
    }


def _format_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row[col] for col in columns) + " |" for row in rows]
    return "\n".join((header, sep, *body))


def _label(name: str) -> str:
    if name == "s8862_balanced":
        return "S88-6x2 Balanced baseline"
    if name == "s88ref_balanced_topology_best_rtl":
        return "Balanced topology-refined"
    if name == "s8862_quality":
        return "S88-6x2 Quality"
    return signed_probe.DESIGN_LABELS.get(name, name)


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    aggregate_rows = []
    for name, metrics in data["summary"]["aggregate_int_accumulator"].items():
        aggregate_rows.append(
            {
                "设计": _label(name),
                "mean signed error": f"{metrics['mean_error']:.4f}",
                "MAE": f"{metrics['mae']:.4f}",
                "rel L2": f"{metrics['relative_l2_error']:.6f}",
                "directionality": f"{metrics['signed_to_abs_ratio']:.4f}",
                "positive / negative": f"{metrics['positive_error_rate']:.3f} / {metrics['negative_error_rate']:.3f}",
            }
        )

    comparison_rows = []
    for row in data["summary"]["refined_vs_baseline_layers"]:
        comparison_rows.append(
            {
                "Linear layer": row["name"],
                "baseline mean err": f"{row['baseline_mean_error']:.3f}",
                "refined mean err": f"{row['refined_mean_error']:.3f}",
                "bias change": f"{row['mean_error_delta']:+.3f}",
                "refined directionality": f"{row['refined_signed_to_abs_ratio']:.3f}",
                "refined rel L2": f"{row['refined_relative_l2_error']:.5f}",
            }
        )

    lines = [
        "# S88 Balanced 逐层 GEMM 偏置诊断",
        "",
        "## 目的",
        "",
        "本报告比较 `S88-6x2 Balanced baseline` 与 `Balanced topology-refined` 在同一批真实 signed W8A8 Linear 输入上的局部 GEMM 误差。",
        "",
        "输入来自 Exact signed W8A8 模型的共同前向过程，因此每个设计看到的量化 activation、量化 weight 与采样输出通道完全一致。该口径用于隔离局部乘法器误差，不把前层误差传播混入本轮比较。",
        "",
        "## 配置",
        "",
        f"- model: `{data['config']['model']}`",
        f"- dataset: `{data['config']['dataset']}` / `{data['config']['dataset_config']}` / `{data['config']['split']}`",
        f"- captured Linear layers: `{data['summary']['captured_layers']}`",
        f"- activation quantization: `{data['config']['activation_scale']}`",
        f"- weight quantization: `{data['config']['weight_scale']}`",
        f"- sampled rows / layer: at most `{data['config']['max_rows_per_layer']}`",
        f"- sampled output channels / layer: at most `{data['config']['max_cols_per_layer']}`",
        "",
        "## 指标含义",
        "",
        "每层先计算 int32 累加器误差：",
        "",
        "$$",
        "E = \\hat{Y}_{\\mathrm{int32}} - Y_{\\mathrm{int32}}",
        "$$",
        "",
        "- `mean signed error`：\\(\\operatorname{mean}(E)\\)。正值代表整体高估，负值代表整体低估。",
        "- `directionality`：\\(\\frac{|\\sum E|}{\\sum |E|}\\)。越接近 1，说明正负误差越不抵消、越接近单向漂移；接近 0 则说明有较多抵消。",
        "- `rel L2`：\\(\\frac{\\|E\\|_2}{\\|Y_{\\mathrm{int32}}\\|_2}\\)。",
        "",
        "## 汇总",
        "",
        _format_table(
            aggregate_rows,
            ["设计", "mean signed error", "MAE", "rel L2", "directionality", "positive / negative"],
        ),
        "",
        "## Baseline 与 Refined 的逐层偏置变化",
        "",
        "下表按 `|bias change|` 从大到小排列；正的 `bias change` 表示 refined 相比 baseline 更偏高。",
        "",
        _format_table(
            comparison_rows,
            ["Linear layer", "baseline mean err", "refined mean err", "bias change", "refined directionality", "refined rel L2"],
        ),
        "",
        "## 使用边界",
        "",
        "该报告用于解释局部 GEMM 误差结构，不能单独替代端到端 PPL。后续候选仍需要先通过本报告的 bias 约束，再运行相同长度的 PPL smoke test。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _require_runtime()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_luts = signed_probe._load_luts(args)
    missing = [name for name in args.designs if name not in all_luts]
    if missing:
        raise ValueError(f"unknown LUT designs: {', '.join(missing)}")
    luts = {name: all_luts[name] for name in args.designs}

    print("capturing common Exact signed W8A8 layer inputs ...", flush=True)
    samples = _capture_exact_w8a8_samples(args)
    print(f"captured {len(samples)} Linear layers", flush=True)
    layers = []
    for index, sample in enumerate(samples, start=1):
        print(f"evaluating {index}/{len(samples)}: {sample.name}", flush=True)
        layers.append(_evaluate_layer(sample, luts))

    aggregate_int = {
        name: _aggregate([layer["designs"][name]["int_accumulator"] for layer in layers])
        for name in luts
    }
    aggregate_real = {
        name: _aggregate([layer["designs"][name]["dequantized_output"] for layer in layers])
        for name in luts
    }

    comparisons = []
    baseline = "s8862_balanced"
    refined = "s88ref_balanced_topology_best_rtl"
    if baseline in luts and refined in luts:
        for layer in layers:
            base_metrics = layer["designs"][baseline]["int_accumulator"]
            refined_metrics = layer["designs"][refined]["int_accumulator"]
            comparisons.append(
                {
                    "name": layer["name"],
                    "baseline_mean_error": base_metrics["mean_error"],
                    "refined_mean_error": refined_metrics["mean_error"],
                    "mean_error_delta": refined_metrics["mean_error"] - base_metrics["mean_error"],
                    "baseline_signed_to_abs_ratio": base_metrics["signed_to_abs_ratio"],
                    "refined_signed_to_abs_ratio": refined_metrics["signed_to_abs_ratio"],
                    "refined_relative_l2_error": refined_metrics["relative_l2_error"],
                }
            )
        comparisons.sort(key=lambda row: abs(float(row["mean_error_delta"])), reverse=True)

    data = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "max_eval_tokens": args.max_eval_tokens,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "max_rows_per_layer": args.max_rows_per_layer,
            "max_cols_per_layer": args.max_cols_per_layer,
            "include_lm_head": args.include_lm_head,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "designs": args.designs,
        },
        "summary": {
            "captured_layers": len(layers),
            "aggregate_int_accumulator": aggregate_int,
            "aggregate_dequantized_output": aggregate_real,
            "refined_vs_baseline_layers": comparisons,
        },
        "layers": layers,
    }
    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
