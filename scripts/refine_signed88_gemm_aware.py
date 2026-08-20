"""Refine a signed8x8 LUT against real model Linear/GEMM samples.

This is a small, topology-preserving discrete search.  It keeps the hardware
resource class fixed and changes only the mutable INIT bits of an existing
design.  The search objective is evaluated on quantized activation/weight
matrices captured from an Exact signed W8A8 model, rather than only on a
product-level calibration histogram.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "multiplier_models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from signed88.common import ObjectiveWeights, hex_to_int, int_to_hex, read_json, write_json
from signed88.data import load_calibration_csv
from signed88.hardware import get_design
from signed88.metrics import evaluate_design

import run_signed_w8a8_layerwise_bias_report as layerwise


DEFAULT_BASE = PROJECT_ROOT / "multiplier_models" / "signed88" / "hardware"
DEFAULT_CALIBRATION = PROJECT_ROOT / "tests" / "data" / "w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
DEFAULT_OUT = PROJECT_ROOT / "tmp" / "signed88_gemm_aware"
DEFAULT_RTL_ROOT = PROJECT_ROOT / "FPGA_multiplier" / "signed8x8_6x2"


@dataclass
class GemmSample:
    name: str
    activation: np.ndarray
    weight: np.ndarray
    scale: np.ndarray
    exact_int: np.ndarray
    exact_real: np.ndarray
    state_index: np.ndarray
    exact_low: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", default="balanced", choices=("balanced", "quality"))
    p.add_argument("--base-inits-json", default=None)
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--max-eval-tokens", type=int, default=512)
    p.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="skip this many WikiText tokens before the capture forward pass",
    )
    p.add_argument("--activation-scale", choices=("per_tensor", "per_token"), default="per_token")
    p.add_argument("--weight-scale", choices=("per_tensor", "per_channel"), default="per_channel")
    p.add_argument("--max-linear-layers", type=int, default=12)
    p.add_argument("--max-rows-per-layer", type=int, default=8)
    p.add_argument("--max-cols-per-layer", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--samples", default=None, help="reuse a previously saved .npz sample set")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--rtl-template-root", default=str(DEFAULT_RTL_ROOT))
    p.add_argument("--calibration-csv", default=str(DEFAULT_CALIBRATION))
    p.add_argument("--calibration-weight-column", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bit-rounds", type=int, default=4)
    p.add_argument("--pair-rounds", type=int, default=0)
    p.add_argument("--pair-candidate-bits", type=int, default=24)
    p.add_argument("--max-wce", type=int, default=80)
    p.add_argument("--max-product-workload-mred", type=float, default=-1.0)
    p.add_argument("--gemm-rel-l2-weight", type=float, default=1.0)
    p.add_argument("--gemm-nmae-weight", type=float, default=0.25)
    p.add_argument("--gemm-bias-weight", type=float, default=0.25)
    p.add_argument("--gemm-directionality-weight", type=float, default=0.05)
    return p.parse_args()


def _require_runtime() -> None:
    missing = [
        name
        for name in ("torch", "transformers", "datasets")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _collector_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        model=args.model,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_eval_tokens=args.max_eval_tokens,
        token_offset=args.token_offset,
        eval_style="axcore",
        activation_scale=args.activation_scale,
        weight_scale=args.weight_scale,
        max_linear_layers=args.max_linear_layers,
        max_rows_per_layer=args.max_rows_per_layer,
        max_cols_per_layer=args.max_cols_per_layer,
        include_lm_head=False,
        device=args.device,
        local_files_only=args.local_files_only,
    )


def _sample_path(args: argparse.Namespace) -> Path:
    return Path(args.samples) if args.samples else Path(args.out_dir) / "gemm_samples.npz"


def _save_samples(path: Path, samples: list[layerwise.LayerSample], config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        prefix = f"l{index:03d}"
        arrays[f"{prefix}_activation"] = sample.activation.astype(np.int8, copy=False)
        arrays[f"{prefix}_activation_scale"] = sample.activation_scale.astype(np.float32, copy=False)
        arrays[f"{prefix}_weight"] = sample.weight.astype(np.int8, copy=False)
        arrays[f"{prefix}_weight_scale"] = sample.weight_scale.astype(np.float32, copy=False)
        metadata.append({"name": sample.name, "prefix": prefix})
    arrays["metadata_json"] = np.asarray(json.dumps({"config": dict(config), "layers": metadata}))
    np.savez_compressed(path, **arrays)


def _load_or_capture_samples(args: argparse.Namespace) -> tuple[list[layerwise.LayerSample], Path]:
    path = _sample_path(args)
    if path.exists():
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata_json"].item()))
        samples = []
        for item in metadata["layers"]:
            prefix = item["prefix"]
            samples.append(
                layerwise.LayerSample(
                    name=item["name"],
                    activation=archive[f"{prefix}_activation"],
                    activation_scale=archive[f"{prefix}_activation_scale"],
                    weight=archive[f"{prefix}_weight"],
                    weight_scale=archive[f"{prefix}_weight_scale"],
                )
            )
        return samples, path

    captured = layerwise._capture_exact_w8a8_samples(_collector_args(args))
    _save_samples(
        path,
        captured,
        {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "max_eval_tokens": args.max_eval_tokens,
            "token_offset": args.token_offset,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "max_linear_layers": args.max_linear_layers,
            "max_rows_per_layer": args.max_rows_per_layer,
            "max_cols_per_layer": args.max_cols_per_layer,
        },
    )
    return captured, path


def _prepare_samples(samples: list[layerwise.LayerSample]) -> list[GemmSample]:
    prepared: list[GemmSample] = []
    for sample in samples:
        activation = sample.activation.astype(np.int8, copy=False)
        weight = sample.weight.astype(np.int8, copy=False)
        a_low = activation.astype(np.int32) & 63
        b_low = weight.astype(np.int32) & 63
        state_index = (
            a_low[:, :, None] * 64 + b_low[None, :, :]
        ).astype(np.int16, copy=False)
        exact_low = (state_index // 64) * (state_index % 64)
        exact_int = layerwise.exact_gemm(activation, weight)
        scale = sample.activation_scale * sample.weight_scale.reshape(1, -1)
        exact_real = exact_int.astype(np.float64) * scale
        prepared.append(
            GemmSample(
                name=sample.name,
                activation=activation,
                weight=weight,
                scale=scale.astype(np.float64, copy=False),
                exact_int=exact_int,
                exact_real=exact_real,
                state_index=state_index,
                exact_low=exact_low.astype(np.int32, copy=False),
            )
        )
    return prepared


def _all_bits(design: Any) -> list[tuple[str, int]]:
    return [
        (name, int(bit))
        for name in design.spec.train_names
        for bit in design.spec.search_bits[name]
    ]


def _flip(inits: Mapping[str, str], flips: list[tuple[str, int]]) -> dict[str, str]:
    values = {name: hex_to_int(value) for name, value in inits.items()}
    for name, bit in flips:
        values[name] ^= 1 << bit
    return {name: int_to_hex(value) for name, value in values.items()}


def _sample_metrics(samples: list[GemmSample], low: np.ndarray) -> dict[str, Any]:
    layer_rows: list[dict[str, Any]] = []
    sq_error = sq_exact = abs_error = abs_exact = 0.0
    signed_error = 0.0
    directionality_sum = 0.0
    for sample in samples:
        delta = low[sample.state_index] - sample.exact_low
        error_int = np.sum(delta, axis=1, dtype=np.int32)
        error_real = error_int.astype(np.float64) * sample.scale
        sum_sq_error = float(np.square(error_real).sum())
        sum_sq_exact = float(np.square(sample.exact_real).sum())
        sum_abs_error = float(np.abs(error_real).sum())
        sum_abs_exact = float(np.abs(sample.exact_real).sum())
        sum_error = float(error_real.sum())
        sq_error += sum_sq_error
        sq_exact += sum_sq_exact
        abs_error += sum_abs_error
        abs_exact += sum_abs_exact
        signed_error += sum_error
        directionality_sum += abs(sum_error) / max(sum_abs_error, 1e-12)
        layer_rows.append(
            {
                "name": sample.name,
                "mean_error": sum_error / max(error_real.size, 1),
                "mae": sum_abs_error / max(error_real.size, 1),
                "relative_l2": float(np.sqrt(sum_sq_error) / max(np.sqrt(sum_sq_exact), 1e-12)),
                "directionality": abs(sum_error) / max(sum_abs_error, 1e-12),
                "sum_error": sum_error,
                "sum_abs_error": sum_abs_error,
            }
        )

    rel_l2 = float(np.sqrt(sq_error) / max(np.sqrt(sq_exact), 1e-12))
    nmae = float(abs_error / max(abs_exact, 1e-12))
    bias_ratio = float(abs(signed_error) / max(abs_exact, 1e-12))
    directionality = float(directionality_sum / max(len(samples), 1))
    return {
        "relative_l2": rel_l2,
        "normalized_mae": nmae,
        "bias_ratio": bias_ratio,
        "directionality": directionality,
        "sum_error": signed_error,
        "sum_abs_error": abs_error,
        "layers": layer_rows,
    }


def _score(metrics: Mapping[str, float], args: argparse.Namespace) -> float:
    return (
        args.gemm_rel_l2_weight * float(metrics["relative_l2"])
        + args.gemm_nmae_weight * float(metrics["normalized_mae"])
        + args.gemm_bias_weight * float(metrics["bias_ratio"])
        + args.gemm_directionality_weight * float(metrics["directionality"])
    )


def _signed_lut_from_low(design: Any, inits: Mapping[str, str]) -> np.ndarray:
    low = design.hard_low_numpy(inits).astype(np.int32).reshape(64, 64)
    lut = np.empty((256, 256), dtype=np.int32)
    for a in range(-128, 128):
        al = a & 63
        for b in range(-128, 128):
            bl = b & 63
            # lut_gemm indexes signed int8 operands as value + 128.
            lut[a + 128, b + 128] = a * b + int(low[al, bl]) - al * bl
    return lut


def _verify_signed_lut(design: Any, inits: Mapping[str, str], lut: np.ndarray) -> None:
    expected = np.empty((256, 256), dtype=np.int32)
    low = design.hard_low_numpy(inits).astype(np.int32).reshape(64, 64)
    for a in range(-128, 128):
        for b in range(-128, 128):
            expected[a + 128, b + 128] = (
                a * b + int(low[a & 63, b & 63]) - (a & 63) * (b & 63)
            )
    if not np.array_equal(expected, lut):
        index = np.argwhere(expected != lut)[0]
        raise AssertionError(
            "generated signed LUT disagrees with the design at "
            f"a={int(index[0]) - 128}, b={int(index[1]) - 128}"
        )


def _label_bit(bit: tuple[str, int]) -> str:
    return f"{bit[0]}[{bit[1]}]"


def _write_report(path: Path, data: Mapping[str, Any]) -> None:
    base = data["summary"]["baseline"]
    final = data["summary"]["final"]
    rows = [
        ("Balanced baseline", base),
        ("GEMM-aware refined", final),
    ]
    lines = [
        "# Signed88 GEMM-aware INIT refinement",
        "",
        "## 实验目的",
        "",
        "本实验固定 Balanced 的 RTL 拓扑与资源类别，只搜索可修改的 LUT INIT bit。候选的主要目标直接在真实模型 Linear 层采样得到的 signed W8A8 矩阵上计算，而不是只依赖 product-level calibration histogram。",
        "",
        "每个采样层使用相同的量化 activation、weight 与 scale。对每个候选乘法器，先得到乘积误差，再沿 GEMM 的 reduction 维累加：",
        "",
        "$$",
        "E_{ij}=\\sum_k\\left(\\hat{p}(A_{ik},B_{kj})-A_{ik}B_{kj}\\right).",
        "$$",
        "",
        "目标函数为：",
        "",
        "$$",
        "\\mathcal{J}_{\\mathrm{GEMM}}="
        "\\lambda_1\\,\\mathrm{relL2}"
        "+\\lambda_2\\,\\mathrm{nMAE}"
        "+\\lambda_3\\,\\mathrm{bias}"
        "+\\lambda_4\\,\\mathrm{directionality}.",
        "$$",
        "",
        "- `relL2`：所有采样层反量化输出误差的整体相对 \\(L_2\\) 误差。",
        "- `nMAE`：反量化输出绝对误差总和除以精确输出绝对值总和。",
        "- `bias`：所有采样层 signed error 总和的绝对值，再按精确输出绝对值总和归一化。",
        "- `directionality`：逐层计算 \\(\\lvert\\sum E\\rvert/\\sum\\lvert E\\rvert\\) 后取平均。",
        "",
        "## 配置",
        "",
        f"- model: `{data['config']['model']}`",
        f"- sampled Linear layers: `{data['config']['sampled_layers']}`",
        f"- rows / layer: at most `{data['config']['max_rows_per_layer']}`",
        f"- output channels / layer: at most `{data['config']['max_cols_per_layer']}`",
        f"- activation quantization: `{data['config']['activation_scale']}`",
        f"- weight quantization: `{data['config']['weight_scale']}`",
        f"- max WCE constraint: `{data['config']['max_wce']}`",
        "",
        "## 结果",
        "",
        "| 设计 | GEMM score | rel L2 | nMAE | bias ratio | directionality | product WCE | product workload MRED |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{item['gemm']['score']:.8f}",
                    f"{item['gemm']['relative_l2']:.8f}",
                    f"{item['gemm']['normalized_mae']:.8f}",
                    f"{item['gemm']['bias_ratio']:.8f}",
                    f"{item['gemm']['directionality']:.8f}",
                    str(item["product"]["WCE"]),
                    f"{item['product']['workload_MRED']:.8f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 接受的 bit flip",
            "",
            "| round | changed bits | old score | new score |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for row in data["summary"]["accepted_steps"]:
        lines.append(
            f"| {row['round']} | {', '.join(row['flips'])} | "
            f"{row['old_score']:.8f} | {row['new_score']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "这版搜索只说明 GEMM-aware objective 是否能在固定资源类别内找到更合适的 INIT。最终候选仍需用统一长度的端到端 PPL、完整 product-level 报告和 RTL 综合结果复核。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    _require_runtime()
    rng = random.Random(args.seed)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = get_design(args.design)

    if args.base_inits_json:
        source = read_json(Path(args.base_inits_json))
        base_inits = design.normalize_inits(source.get("inits", source))
    else:
        base_inits = design.normalize_inits(design.spec.base_inits)

    print("loading or capturing real signed W8A8 Linear samples ...", flush=True)
    raw_samples, sample_path = _load_or_capture_samples(args)
    samples = _prepare_samples(raw_samples)
    print(f"prepared {len(samples)} layers from {sample_path}", flush=True)

    profile = load_calibration_csv(Path(args.calibration_csv), args.calibration_weight_column)
    product_objective = ObjectiveWeights()
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    bits = _all_bits(design)

    def evaluate(inits: Mapping[str, str]) -> dict[str, Any]:
        key = tuple(hex_to_int(inits[name]) for name in design.spec.train_names)
        if key in cache:
            return cache[key]
        product = evaluate_design(design, inits, profile, product_objective).to_dict()
        gemm = _sample_metrics(samples, design.hard_low_numpy(inits))
        gemm["score"] = _score(gemm, args)
        result = {"inits": dict(inits), "product": product, "gemm": gemm}
        cache[key] = result
        return result

    def valid(result: Mapping[str, Any]) -> bool:
        product = result["product"]
        if args.max_wce > 0 and int(product["WCE"]) > args.max_wce:
            return False
        if (
            args.max_product_workload_mred >= 0
            and float(product["workload_MRED"]) > args.max_product_workload_mred
        ):
            return False
        return True

    current = evaluate(base_inits)
    accepted: list[dict[str, Any]] = []
    print(
        f"[baseline] score={current['gemm']['score']:.8f} "
        f"relL2={current['gemm']['relative_l2']:.8f} "
        f"nMAE={current['gemm']['normalized_mae']:.8f} "
        f"bias={current['gemm']['bias_ratio']:.8f} "
        f"WCE={current['product']['WCE']}",
        flush=True,
    )

    for round_index in range(args.bit_rounds):
        trials = []
        for bit in bits:
            trial_inits = _flip(current["inits"], [bit])
            trial = evaluate(trial_inits)
            if valid(trial):
                trials.append((float(trial["gemm"]["score"]), bit, trial))
        trials.sort(key=lambda item: (item[0], int(item[2]["product"]["WCE"])))
        if not trials or trials[0][0] >= float(current["gemm"]["score"]) - 1e-12:
            print(f"[single] round={round_index + 1}: no improvement", flush=True)
            break
        new_score, bit, new = trials[0]
        old_score = float(current["gemm"]["score"])
        accepted.append(
            {
                "round": round_index + 1,
                "stage": "single",
                "flips": [_label_bit(bit)],
                "old_score": old_score,
                "new_score": new_score,
                "product": new["product"],
                "gemm": new["gemm"],
            }
        )
        current = new
        print(
            f"[single] round={round_index + 1} ACCEPT {_label_bit(bit)} "
            f"{old_score:.8f}->{new_score:.8f} "
            f"relL2={new['gemm']['relative_l2']:.8f}",
            flush=True,
        )

    for round_index in range(args.pair_rounds):
        ranked = []
        for bit in bits:
            trial = evaluate(_flip(current["inits"], [bit]))
            if valid(trial):
                ranked.append((float(trial["gemm"]["score"]), bit))
        ranked.sort(key=lambda item: item[0])
        pool = [bit for _, bit in ranked[: args.pair_candidate_bits]]
        best: tuple[float, list[tuple[str, int]], dict[str, Any]] | None = None
        for i, first in enumerate(pool):
            for second in pool[i + 1 :]:
                trial = evaluate(_flip(current["inits"], [first, second]))
                if not valid(trial):
                    continue
                score = float(trial["gemm"]["score"])
                if best is None or score < best[0]:
                    best = (score, [first, second], trial)
        if best is None or best[0] >= float(current["gemm"]["score"]) - 1e-12:
            print(f"[pair] round={round_index + 1}: no improvement", flush=True)
            break
        new_score, flips, new = best
        old_score = float(current["gemm"]["score"])
        accepted.append(
            {
                "round": round_index + 1,
                "stage": "pair",
                "flips": [_label_bit(bit) for bit in flips],
                "old_score": old_score,
                "new_score": new_score,
                "product": new["product"],
                "gemm": new["gemm"],
            }
        )
        current = new
        print(
            f"[pair] round={round_index + 1} ACCEPT "
            f"{', '.join(_label_bit(bit) for bit in flips)} "
            f"{old_score:.8f}->{new_score:.8f}",
            flush=True,
        )

    artifact = design.artifact(
        current["inits"],
        metrics=current["product"],
        extra={
            "stage": "gemm_aware_refined",
            "base_inits_json": str(Path(args.base_inits_json).resolve())
            if args.base_inits_json
            else "design.spec.base_inits",
            "sample_path": str(sample_path),
            "gemm_metrics": current["gemm"],
            "search_args": vars(args),
            "accepted_steps": accepted,
        },
    )
    json_path = out / "best_signed88_gemm_aware.json"
    write_json(json_path, artifact)
    rtl_path = design.export_rtl(
        Path(args.rtl_template_root),
        out / "best_rtl",
        current["inits"],
        metadata={"product": current["product"], "gemm": current["gemm"]},
    )
    lut_dir = PROJECT_ROOT / "outputs" / "fpga_luts"
    lut_dir.mkdir(parents=True, exist_ok=True)
    lut_name = f"s88gemm_{out.name}_signed_int8_lut.npy"
    lut_path = lut_dir / lut_name
    signed_lut = _signed_lut_from_low(design, current["inits"])
    _verify_signed_lut(design, current["inits"], signed_lut)
    np.save(lut_path, signed_lut)
    data = {
        "config": {
            "design": design.spec.name,
            "model": args.model,
            "dataset": args.dataset,
            "sampled_layers": len(samples),
            "max_rows_per_layer": args.max_rows_per_layer,
            "max_cols_per_layer": args.max_cols_per_layer,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "max_wce": args.max_wce,
            "sample_path": str(sample_path),
            "rtl_path": str(rtl_path),
            "lut_path": str(lut_path),
        },
        "summary": {
            "baseline": {
                "product": evaluate(base_inits)["product"],
                "gemm": {
                    **_sample_metrics(samples, design.hard_low_numpy(base_inits)),
                    "score": _score(
                        _sample_metrics(samples, design.hard_low_numpy(base_inits)), args
                    ),
                },
            },
            "final": current,
            "accepted_steps": accepted,
            "cache_size": len(cache),
        },
    }
    report_json = out / "signed88_gemm_aware_refinement.json"
    report_md = out / "signed88_gemm_aware_refinement.md"
    report_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_report(report_md, data)
    print(f"[artifact] {json_path}", flush=True)
    print(f"[rtl] {rtl_path}", flush=True)
    print(f"[lut] {lut_path}", flush=True)
    print(f"[report] {report_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
