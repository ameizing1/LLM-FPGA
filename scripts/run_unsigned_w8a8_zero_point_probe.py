"""Evaluate unsigned W8A8 zero-point GEMM with LUT-backed multipliers."""

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

from am_lut_tcasi24.tcasi24 import mul8_unsigned


DESIGN_LABELS = {
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}

ROUTE_LABELS = {
    "uint8_activation_int8_weight": "activation uint8 asymmetric, weight int8 symmetric",
    "uint8_activation_uint8_weight": "activation uint8 asymmetric, weight uint8 asymmetric",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-offset", type=int, default=0, help="skip this many non-empty text samples before probing")
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
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="real_w8a8_unsigned_zero_point_probe")
    parser.add_argument("--save-pair-histogram", action="store_true", help="save sampled unsigned LUT input histograms as .npy")
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ["torch", "transformers", "datasets"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _metrics(exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    exact64 = exact.astype(np.float64)
    approx64 = approx.astype(np.float64)
    err = approx64 - exact64
    abs_err = np.abs(err)
    denom = max(float(np.linalg.norm(exact64.ravel())), 1.0)
    return {
        "error_rate": float(np.mean(err != 0)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "max_abs_error": float(np.max(abs_err)),
        "relative_l2_error": float(np.linalg.norm(err.ravel()) / denom),
    }


def _asym_uint8_quant_np(x: np.ndarray, *, axis: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = x.astype(np.float32, copy=False)
    qmin = 0
    qmax = 255
    if axis is None:
        x_min = float(np.min(arr))
        x_max = float(np.max(arr))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            q = np.zeros_like(arr, dtype=np.uint8)
            return q, np.array(1.0, dtype=np.float32), np.array(0, dtype=np.int32)
        scale = (x_max - x_min) / float(qmax - qmin)
        zp = int(np.rint(qmin - x_min / scale))
        zp = int(np.clip(zp, qmin, qmax))
        q = np.clip(np.rint(arr / scale + zp), qmin, qmax).astype(np.uint8)
        return q, np.array(scale, dtype=np.float32), np.array(zp, dtype=np.int32)

    x_min = np.min(arr, axis=axis, keepdims=True)
    x_max = np.max(arr, axis=axis, keepdims=True)
    scale = (x_max - x_min) / float(qmax - qmin)
    invalid = (scale == 0.0) | ~np.isfinite(scale)
    scale = np.where(invalid, 1.0, scale).astype(np.float32)
    zp = np.rint(qmin - x_min / scale)
    zp = np.clip(zp, qmin, qmax).astype(np.int32)
    q = np.clip(np.rint(arr / scale + zp), qmin, qmax).astype(np.uint8)
    q = np.where(invalid, 0, q).astype(np.uint8)
    return q, scale, zp


def _sym_int8_quant_np(x: np.ndarray, *, axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    arr = x.astype(np.float32, copy=False)
    qmax = 127
    if axis is None:
        scale = float(np.max(np.abs(arr))) / qmax
        if not np.isfinite(scale) or scale == 0.0:
            return np.zeros_like(arr, dtype=np.int8), np.array(1.0, dtype=np.float32)
        q = np.clip(np.rint(arr / scale), -128, 127).astype(np.int8)
        return q, np.array(scale, dtype=np.float32)

    max_abs = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = max_abs / qmax
    scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(arr / scale), -128, 127).astype(np.int8)
    return q, scale


def _activation_uint8(x: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "per_tensor":
        return _asym_uint8_quant_np(x, axis=None)
    if mode == "per_token":
        return _asym_uint8_quant_np(x, axis=1)
    raise ValueError(f"unknown activation scale mode: {mode}")


def _weight_int8(weight_out_in: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "per_tensor":
        q, scale = _sym_int8_quant_np(weight_out_in, axis=None)
    elif mode == "per_channel":
        q, scale = _sym_int8_quant_np(weight_out_in, axis=1)
    else:
        raise ValueError(f"unknown weight scale mode: {mode}")
    return q.T.copy(), np.asarray(scale).T.copy()


def _weight_uint8(weight_out_in: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "per_tensor":
        q, scale, zp = _asym_uint8_quant_np(weight_out_in, axis=None)
    elif mode == "per_channel":
        q, scale, zp = _asym_uint8_quant_np(weight_out_in, axis=1)
    else:
        raise ValueError(f"unknown weight scale mode: {mode}")
    return q.T.copy(), np.asarray(scale).T.copy(), np.asarray(zp).T.copy()


def _build_tcasi_unsigned_lut(mode: str) -> np.ndarray:
    lut = np.empty((256, 256), dtype=np.uint32)
    for a in range(256):
        for b in range(256):
            lut[a, b] = mul8_unsigned(a, b, mode)
    return lut


def _load_unsigned_luts(fpga_lut_dir: Path) -> dict[str, np.ndarray]:
    luts = {
        "tcasi24_lsam1": _build_tcasi_unsigned_lut("lsam1"),
        "tcasi24_csam2": _build_tcasi_unsigned_lut("csam2"),
        "fpga_cand17": np.load(fpga_lut_dir / "fpga_cand17_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand20": np.load(fpga_lut_dir / "fpga_cand20_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand10": np.load(fpga_lut_dir / "fpga_cand10_unsigned8_lut.npy").astype(np.uint32),
    }
    for name, lut in luts.items():
        if lut.shape != (256, 256):
            raise ValueError(f"{name} LUT shape must be (256, 256), got {lut.shape}")
    return luts


def _uint8_product_sum_lut(a_q: np.ndarray, b_q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if a_q.dtype != np.uint8:
        a_q = a_q.astype(np.uint8)
    if b_q.dtype != np.uint8:
        b_q = b_q.astype(np.uint8)
    if a_q.shape[1] != b_q.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {a_q.shape}, {b_q.shape}")
    acc = np.zeros((a_q.shape[0], b_q.shape[1]), dtype=np.int64)
    for k in range(a_q.shape[1]):
        acc += lut[np.ix_(a_q[:, k], b_q[k, :])].astype(np.int64)
    return acc


def _uint8_int8_product_sum_lut(a_q: np.ndarray, b_q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if a_q.dtype != np.uint8:
        a_q = a_q.astype(np.uint8)
    b_i = b_q.astype(np.int16, copy=False)
    if a_q.shape[1] != b_i.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {a_q.shape}, {b_i.shape}")
    acc = np.zeros((a_q.shape[0], b_i.shape[1]), dtype=np.int64)
    for k in range(a_q.shape[1]):
        b_vals = b_i[k, :]
        b_mag = np.abs(b_vals).astype(np.uint8)
        signs = np.where(b_vals < 0, -1, 1).astype(np.int64)
        products = lut[np.ix_(a_q[:, k], b_mag)].astype(np.int64)
        acc += products * signs[None, :]
    return acc


def _center_uint8(q: np.ndarray, zp: np.ndarray) -> np.ndarray:
    return q.astype(np.int32) - np.asarray(zp, dtype=np.int32)


def _scale_outer(a_scale: np.ndarray, b_scale: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(a_scale, dtype=np.float64)
    b = np.asarray(b_scale, dtype=np.float64)
    if a.ndim == 0:
        a = np.full((shape[0], 1), float(a), dtype=np.float64)
    if b.ndim == 0:
        b = np.full((1, shape[1]), float(b), dtype=np.float64)
    return a.reshape(shape[0], 1) * b.reshape(1, shape[1])


def _evaluate_uint8_uint8(
    a_q: np.ndarray,
    a_scale: np.ndarray,
    a_zp: np.ndarray,
    b_q: np.ndarray,
    b_scale: np.ndarray,
    b_zp: np.ndarray,
    luts: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    a_center = _center_uint8(a_q, a_zp)
    b_center = _center_uint8(b_q, b_zp)
    exact_int = a_center.astype(np.int64) @ b_center.astype(np.int64)
    out_scale = _scale_outer(a_scale, b_scale, exact_int.shape)
    exact_real = exact_int.astype(np.float64) * out_scale

    sum_a = np.sum(a_q.astype(np.int64), axis=1, keepdims=True)
    sum_b = np.sum(b_q.astype(np.int64), axis=0, keepdims=True)
    za = np.asarray(a_zp, dtype=np.int64)
    zb = np.asarray(b_zp, dtype=np.int64)
    if za.ndim == 0:
        za = np.full((a_q.shape[0], 1), int(za), dtype=np.int64)
    if zb.ndim == 0:
        zb = np.full((1, b_q.shape[1]), int(zb), dtype=np.int64)
    za = za.reshape(a_q.shape[0], 1)
    zb = zb.reshape(1, b_q.shape[1])
    k_dim = a_q.shape[1]

    metrics: dict[str, dict[str, float]] = {}
    for name, lut in luts.items():
        approx_products = _uint8_product_sum_lut(a_q, b_q, lut)
        approx_int = approx_products - (sum_a * zb) - (za * sum_b) + (k_dim * za * zb)
        approx_real = approx_int.astype(np.float64) * out_scale
        int_metrics = _metrics(exact_int, approx_int)
        real_metrics = _metrics(exact_real, approx_real)
        metrics[name] = {
            **{f"int_{key}": value for key, value in int_metrics.items()},
            **{f"dequant_{key}": value for key, value in real_metrics.items()},
        }
    return metrics, exact_int


def _evaluate_uint8_int8(
    a_q: np.ndarray,
    a_scale: np.ndarray,
    a_zp: np.ndarray,
    b_q: np.ndarray,
    b_scale: np.ndarray,
    luts: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    a_center = _center_uint8(a_q, a_zp)
    b_i = b_q.astype(np.int32)
    exact_int = a_center.astype(np.int64) @ b_i.astype(np.int64)
    out_scale = _scale_outer(a_scale, b_scale, exact_int.shape)
    exact_real = exact_int.astype(np.float64) * out_scale

    za = np.asarray(a_zp, dtype=np.int64)
    if za.ndim == 0:
        za = np.full((a_q.shape[0], 1), int(za), dtype=np.int64)
    za = za.reshape(a_q.shape[0], 1)
    sum_b = np.sum(b_i.astype(np.int64), axis=0, keepdims=True)

    metrics: dict[str, dict[str, float]] = {}
    for name, lut in luts.items():
        approx_products = _uint8_int8_product_sum_lut(a_q, b_q, lut)
        approx_int = approx_products - za * sum_b
        approx_real = approx_int.astype(np.float64) * out_scale
        int_metrics = _metrics(exact_int, approx_int)
        real_metrics = _metrics(exact_real, approx_real)
        metrics[name] = {
            **{f"int_{key}": value for key, value in int_metrics.items()},
            **{f"dequant_{key}": value for key, value in real_metrics.items()},
        }
    return metrics, exact_int


def _pair_distribution(
    a_q: np.ndarray,
    a_zp: np.ndarray,
    b_q: np.ndarray,
    b_zp_or_none: np.ndarray | None,
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[dict[str, float], np.ndarray]:
    m, k = a_q.shape
    k2, n = b_q.shape
    if k != k2:
        raise ValueError(f"incompatible product sampling shapes: {a_q.shape}, {b_q.shape}")
    total_pairs = m * k * n
    sample_count = min(max_pairs, total_pairs)
    row_idx = rng.integers(0, m, size=sample_count)
    k_idx = rng.integers(0, k, size=sample_count)
    col_idx = rng.integers(0, n, size=sample_count)

    a_raw = a_q[row_idx, k_idx].astype(np.int32)
    a_zp_arr = np.asarray(a_zp, dtype=np.int32)
    if a_zp_arr.ndim == 0:
        za = np.full(sample_count, int(a_zp_arr), dtype=np.int32)
    else:
        za = a_zp_arr.reshape(m)[row_idx]
    a_center = a_raw - za

    b_raw = b_q[k_idx, col_idx].astype(np.int32)
    if b_zp_or_none is None:
        b_center = b_raw
        zb = np.zeros(sample_count, dtype=np.int32)
        b_lut_input = np.abs(b_raw)
    else:
        b_zp_arr = np.asarray(b_zp_or_none, dtype=np.int32)
        if b_zp_arr.ndim == 0:
            zb = np.full(sample_count, int(b_zp_arr), dtype=np.int32)
        else:
            zb = b_zp_arr.reshape(n)[col_idx]
        b_center = b_raw - zb
        b_lut_input = b_raw

    abs_a = np.abs(a_center)
    abs_b = np.abs(b_center)
    abs_product = np.abs(a_center * b_center)
    hist = np.bincount(a_raw * 256 + b_lut_input, minlength=256 * 256)
    return {
        "sampled_pairs": float(sample_count),
        "pct_a_center_zero": float(np.mean(a_center == 0)),
        "pct_b_center_zero": float(np.mean(b_center == 0)),
        "pct_any_center_zero": float(np.mean((a_center == 0) | (b_center == 0))),
        "pct_min_abs_center_le_16": float(np.mean(np.minimum(abs_a, abs_b) <= 16)),
        "pct_min_abs_center_le_32": float(np.mean(np.minimum(abs_a, abs_b) <= 32)),
        "pct_abs_center_product_le_1024": float(np.mean(abs_product <= 1024)),
        "mean_abs_center_a": float(np.mean(abs_a)),
        "mean_abs_center_b": float(np.mean(abs_b)),
        "mean_abs_center_product": float(np.mean(abs_product)),
        "mean_raw_a": float(np.mean(a_raw)),
        "mean_raw_b": float(np.mean(b_raw)),
        "mean_zp_a": float(np.mean(za)),
        "mean_zp_b": float(np.mean(zb)),
        "pct_raw_a_within_16_of_zp": float(np.mean(np.abs(a_raw - za) <= 16)),
        "pct_raw_b_within_16_of_zp": float(np.mean(np.abs(b_raw - zb) <= 16)),
    }, hist.astype(np.int64, copy=False)


@dataclass
class LayerFloatSample:
    name: str
    a_float: np.ndarray
    weight_out_in: np.ndarray


class LinearCollector:
    def __init__(self, args: argparse.Namespace, torch_module: Any):
        self.args = args
        self.torch_module = torch_module
        self.samples: dict[str, LayerFloatSample] = {}
        self.handles: list[Any] = []

    def attach(self, model: Any) -> None:
        linear_names = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, self.torch_module.nn.Linear)
        ][: self.args.max_linear_layers]
        for name, module in linear_names:
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name, module)))

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
            if existing is not None and existing.a_float.shape[0] >= self.args.max_rows_per_layer:
                return

            x_np = x.reshape(-1, x.shape[-1]).numpy()
            remaining = self.args.max_rows_per_layer
            if existing is not None:
                remaining -= existing.a_float.shape[0]
            x_np = x_np[:remaining]
            if x_np.shape[0] == 0:
                return

            weight_np = module.weight.detach().float().cpu().numpy()
            if existing is None:
                self.samples[name] = LayerFloatSample(name=name, a_float=x_np.copy(), weight_out_in=weight_np.copy())
            else:
                existing.a_float = np.concatenate([existing.a_float, x_np], axis=0)

        return hook

    def is_full(self) -> bool:
        if len(self.samples) < self.args.max_linear_layers:
            return False
        return all(sample.a_float.shape[0] >= self.args.max_rows_per_layer for sample in self.samples.values())


def _load_texts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts: list[str] = []
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


def _run_model_and_collect(args: argparse.Namespace) -> list[LayerFloatSample]:
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
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            _ = model(**encoded)
            if collector.is_full():
                break
    collector.detach()
    return list(collector.samples.values())


def _evaluate_layer(
    sample: LayerFloatSample,
    luts: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    weight = sample.weight_out_in
    if weight.shape[0] > args.max_cols_per_layer:
        cols = np.sort(rng.choice(weight.shape[0], size=args.max_cols_per_layer, replace=False))
        weight = weight[cols, :]

    a_q, a_scale, a_zp = _activation_uint8(sample.a_float, args.activation_scale)

    b_i8, b_i8_scale = _weight_int8(weight, args.weight_scale)
    mixed_metrics, _ = _evaluate_uint8_int8(a_q, a_scale, a_zp, b_i8, b_i8_scale, luts)
    mixed_dist, mixed_hist = _pair_distribution(
        a_q,
        a_zp,
        b_i8,
        None,
        rng=rng,
        max_pairs=args.product_pairs_per_layer,
    )

    b_u8, b_u8_scale, b_u8_zp = _weight_uint8(weight, args.weight_scale)
    both_metrics, _ = _evaluate_uint8_uint8(a_q, a_scale, a_zp, b_u8, b_u8_scale, b_u8_zp, luts)
    both_dist, both_hist = _pair_distribution(
        a_q,
        a_zp,
        b_u8,
        b_u8_zp,
        rng=rng,
        max_pairs=args.product_pairs_per_layer,
    )

    mixed_route: dict[str, Any] = {
        "distribution": mixed_dist,
        "gemm_metrics": mixed_metrics,
    }
    both_route: dict[str, Any] = {
        "distribution": both_dist,
        "gemm_metrics": both_metrics,
    }
    if args.save_pair_histogram:
        mixed_route["pair_histogram"] = mixed_hist.tolist()
        both_route["pair_histogram"] = both_hist.tolist()

    return {
        "name": sample.name,
        "a_shape": list(a_q.shape),
        "b_shape": list(b_i8.shape),
        "routes": {
            "uint8_activation_int8_weight": mixed_route,
            "uint8_activation_uint8_weight": both_route,
        },
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize(layer_reports: list[dict[str, Any]], lut_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"layers": len(layer_reports), "routes": {}}
    for route_name in ROUTE_LABELS:
        route_layers = [layer["routes"][route_name] for layer in layer_reports]
        dists = [route["distribution"] for route in route_layers]
        route_summary: dict[str, Any] = {
            "mean_distribution": {
                key: _mean([float(dist[key]) for dist in dists])
                for key in dists[0]
                if key != "sampled_pairs"
            }
            if dists
            else {},
            "mean_int_relative_l2_error": {},
            "mean_dequant_relative_l2_error": {},
            "mean_dequant_rmse": {},
        }
        for name in lut_names:
            route_summary["mean_int_relative_l2_error"][name] = _mean(
                [float(route["gemm_metrics"][name]["int_relative_l2_error"]) for route in route_layers]
            )
            route_summary["mean_dequant_relative_l2_error"][name] = _mean(
                [float(route["gemm_metrics"][name]["dequant_relative_l2_error"]) for route in route_layers]
            )
            route_summary["mean_dequant_rmse"][name] = _mean(
                [float(route["gemm_metrics"][name]["dequant_rmse"]) for route in route_layers]
            )
        summary["routes"][route_name] = route_summary
    return summary


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Unsigned W8A8 Zero-Point 真实输入测试报告",
        "",
        "## 实验目的",
        "",
        "本实验用于回答：如果采用 unsigned 8-bit 量化并引入 zero point，当前 TCASI24 与 FPGA unsigned 8x8 近似乘法器在真实 LLM Linear 输入分布下是否会比 signed wrapper 方案更稳。",
        "",
        "本报告里的 approximate multiplier 只替换原始乘法项，zero-point correction 暂时按精确整数加减法处理：",
        "",
        "$$",
        "\\sum_k (q_{a,k}-z_a)(q_{b,k}-z_b)",
        "= \\sum_k q_{a,k}q_{b,k} - z_b\\sum_k q_{a,k} - z_a\\sum_k q_{b,k} + Kz_az_b",
        "$$",
        "",
        "这是一版行为级上界/可行性测试，不代表最终 RTL 的面积和时序。",
        "",
        "## 实验设置",
        "",
        f"- model: `{data['config']['model']}`",
        f"- dataset: `{data['config']['dataset']}` / `{data['config']['dataset_config']}` / `{data['config']['split']}`",
        f"- activation quantization: `uint8 asymmetric {data['config']['activation_scale']}`",
        f"- weight quantization: `{data['config']['weight_scale']}`",
        f"- sampled Linear layers: `{data['summary']['layers']}`",
        "",
    ]

    for route_name, route_summary in data["summary"]["routes"].items():
        lines.extend(
            [
                f"## {ROUTE_LABELS[route_name]}",
                "",
                "### 平均输入分布",
                "",
            ]
        )
        dist_rows = [
            {"指标": key, "均值": f"{value:.6f}"}
            for key, value in route_summary["mean_distribution"].items()
        ]
        lines.extend([_markdown_table(dist_rows, ["指标", "均值"]), ""])

        rel_rows = []
        for name in data["design_labels"]:
            rel_rows.append(
                {
                    "设计": DESIGN_LABELS.get(name, name),
                    "int_rel_l2": f"{route_summary['mean_int_relative_l2_error'][name]:.6f}",
                    "dequant_rel_l2": f"{route_summary['mean_dequant_relative_l2_error'][name]:.6f}",
                    "dequant_RMSE": f"{route_summary['mean_dequant_rmse'][name]:.6e}",
                }
            )
        lines.extend(
            [
                "### 平均 GEMM 误差",
                "",
                _markdown_table(rel_rows, ["设计", "int_rel_l2", "dequant_rel_l2", "dequant_RMSE"]),
                "",
            ]
        )

    lines.extend(["## 每层结果", ""])
    for layer in data["layers"]:
        lines.extend([f"### {layer['name']}", "", f"- A shape: `{layer['a_shape']}`, B shape: `{layer['b_shape']}`", ""])
        for route_name in ROUTE_LABELS:
            route = layer["routes"][route_name]
            rows = []
            for name, metrics in route["gemm_metrics"].items():
                rows.append(
                    {
                        "设计": DESIGN_LABELS.get(name, name),
                        "int_rel_l2": f"{metrics['int_relative_l2_error']:.6f}",
                        "dequant_rel_l2": f"{metrics['dequant_relative_l2_error']:.6f}",
                        "dequant_MAE": f"{metrics['dequant_mae']:.6e}",
                        "dequant_RMSE": f"{metrics['dequant_rmse']:.6e}",
                    }
                )
            lines.extend(
                [
                    f"#### {ROUTE_LABELS[route_name]}",
                    "",
                    f"- `pct_min_abs_center_le_32`: `{route['distribution']['pct_min_abs_center_le_32']:.6f}`",
                    f"- `mean_raw_a`: `{route['distribution']['mean_raw_a']:.3f}`, `mean_zp_a`: `{route['distribution']['mean_zp_a']:.3f}`",
                    "",
                    _markdown_table(rows, ["设计", "int_rel_l2", "dequant_rel_l2", "dequant_MAE", "dequant_RMSE"]),
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

    luts = _load_unsigned_luts(Path(args.fpga_lut_dir))
    samples = _run_model_and_collect(args)
    layer_reports = [_evaluate_layer(sample, luts, rng=rng, args=args) for sample in samples]

    pair_histogram_paths: dict[str, str] = {}
    if args.save_pair_histogram:
        for route_name in ROUTE_LABELS:
            route_hist = np.zeros(256 * 256, dtype=np.int64)
            for layer in layer_reports:
                hist = layer["routes"][route_name].pop("pair_histogram", None)
                if hist is not None:
                    route_hist += np.asarray(hist, dtype=np.int64)
            route_hist = route_hist.reshape(256, 256)
            hist_path = out_dir / f"{args.report_name}_{route_name}_pair_histogram.npy"
            np.save(hist_path, route_hist)
            pair_histogram_paths[route_name] = str(hist_path)

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
            "max_rows_per_layer": args.max_rows_per_layer,
            "max_cols_per_layer": args.max_cols_per_layer,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "device": args.device,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "save_pair_histogram": args.save_pair_histogram,
        },
        "route_labels": ROUTE_LABELS,
        "design_labels": {name: DESIGN_LABELS.get(name, name) for name in luts},
        "summary": _summarize(layer_reports, list(luts)),
        "layers": layer_reports,
    }
    if pair_histogram_paths:
        data["summary"]["pair_histogram_paths"] = pair_histogram_paths

    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
