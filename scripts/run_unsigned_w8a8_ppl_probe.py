"""Run a small end-to-end PPL probe for uint8 W8A8 zero-point Linear layers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.tcasi24 import mul8_unsigned


DESIGN_LABELS = {
    "fp32": "FP32/BF16 original",
    "exact_uint8_w8a8": "Exact uint8 W8A8",
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=256, help="0 means use all available tokens")
    parser.add_argument(
        "--eval-style",
        choices=["current", "axcore"],
        default="current",
        help="axcore uses full WikiText-style concatenation, non-overlapping seq_len blocks, and drops the remainder",
    )
    parser.add_argument("--activation-scale", choices=["per_tensor", "per_token"], default="per_token")
    parser.add_argument("--weight-scale", choices=["per_tensor", "per_channel"], default="per_channel")
    parser.add_argument(
        "--designs",
        nargs="+",
        default=["fp32", "exact_uint8_w8a8", "fpga_dist1550_cand2", "fpga_cand20", "fpga_cand17"],
    )
    parser.add_argument("--max-linear-layers", type=int, default=0, help="0 means replace all eligible Linear layers")
    parser.add_argument("--include-lm-head", action="store_true", help="also quantize/replace lm_head; slow for LUT designs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="unsigned_w8a8_ppl_probe")
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ["torch", "transformers", "datasets"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _build_tcasi_unsigned_lut(mode: str) -> np.ndarray:
    lut = np.empty((256, 256), dtype=np.uint32)
    for a in range(256):
        for b in range(256):
            lut[a, b] = mul8_unsigned(a, b, mode)
    return lut


def _load_luts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    fpga_dir = Path(args.fpga_lut_dir)
    luts: dict[str, np.ndarray] = {
        "tcasi24_lsam1": _build_tcasi_unsigned_lut("lsam1"),
        "tcasi24_csam2": _build_tcasi_unsigned_lut("csam2"),
        "fpga_cand17": np.load(fpga_dir / "fpga_cand17_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand20": np.load(fpga_dir / "fpga_cand20_unsigned8_lut.npy").astype(np.uint32),
        "fpga_cand10": np.load(fpga_dir / "fpga_cand10_unsigned8_lut.npy").astype(np.uint32),
    }
    for path in sorted(fpga_dir.glob("fpga_dist*_cand*_unsigned8_lut.npy")):
        design = path.name.removesuffix("_unsigned8_lut.npy")
        luts[design] = np.load(path).astype(np.uint32)
        label = design.replace("fpga_dist", "Dist").replace("_cand", " cand")
        DESIGN_LABELS.setdefault(design, label)
    for name, lut in luts.items():
        if lut.shape != (256, 256):
            raise ValueError(f"{name} LUT shape must be (256, 256), got {lut.shape}")
    return luts


def _quantize_uint8_tensor(x: Any, mode: str) -> tuple[Any, Any, Any]:
    import torch

    arr = x.float()
    qmin = 0.0
    qmax = 255.0
    if mode == "per_tensor":
        x_min = torch.min(arr)
        x_max = torch.max(arr)
        scale = (x_max - x_min) / (qmax - qmin)
        if not torch.isfinite(scale) or float(scale) == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=arr.device)
            q = torch.zeros_like(arr, dtype=torch.uint8)
            zp = torch.tensor(0, dtype=torch.int32, device=arr.device)
            return q, scale.reshape(1), zp.reshape(1)
        zp = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax).to(torch.int32)
        q = torch.clamp(torch.round(arr / scale + zp.float()), qmin, qmax).to(torch.uint8)
        return q, scale.reshape(1), zp.reshape(1)

    if mode == "per_token":
        x_min = torch.min(arr, dim=1, keepdim=True).values
        x_max = torch.max(arr, dim=1, keepdim=True).values
        scale = (x_max - x_min) / (qmax - qmin)
        invalid = (scale == 0.0) | ~torch.isfinite(scale)
        scale = torch.where(invalid, torch.ones_like(scale), scale)
        zp = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax).to(torch.int32)
        q = torch.clamp(torch.round(arr / scale + zp.float()), qmin, qmax).to(torch.uint8)
        q = torch.where(invalid, torch.zeros_like(q), q)
        return q, scale, zp

    raise ValueError(f"unknown uint8 quantization mode: {mode}")


def _quantize_weight_uint8(weight: Any, mode: str) -> tuple[Any, Any, Any]:
    import torch

    w = weight.detach().float()
    if mode == "per_tensor":
        return _quantize_uint8_tensor(w.reshape(1, -1), "per_tensor")
    if mode == "per_channel":
        return _quantize_uint8_tensor(w, "per_token")
    raise ValueError(f"unknown weight scale mode: {mode}")


def _uint8_product_sum_lut(a_q: np.ndarray, b_q_out_in: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if a_q.dtype != np.uint8:
        a_q = a_q.astype(np.uint8)
    if b_q_out_in.dtype != np.uint8:
        b_q_out_in = b_q_out_in.astype(np.uint8)
    if a_q.shape[1] != b_q_out_in.shape[1]:
        raise ValueError(f"incompatible GEMM shapes: {a_q.shape}, {b_q_out_in.shape}")
    acc = np.zeros((a_q.shape[0], b_q_out_in.shape[0]), dtype=np.int64)
    b_t = b_q_out_in.T
    for k in range(a_q.shape[1]):
        acc += lut[np.ix_(a_q[:, k], b_t[k, :])].astype(np.int64)
    return acc


class ExactUint8W8A8Linear:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch

        class _ExactUint8W8A8Linear(torch.nn.Module):
            def __init__(self, source: torch.nn.Linear, activation_scale: str, weight_scale: str):
                super().__init__()
                q_weight, w_scale, w_zp = _quantize_weight_uint8(source.weight, weight_scale)
                if weight_scale == "per_tensor":
                    q_weight = q_weight.reshape_as(source.weight).contiguous()
                self.register_buffer("q_weight", q_weight.cpu())
                self.register_buffer("w_scale", w_scale.float().cpu().reshape(-1))
                self.register_buffer("w_zp", w_zp.to(torch.int32).cpu().reshape(-1))
                self.activation_scale = activation_scale
                self.out_features = source.out_features
                self.in_features = source.in_features
                if source.bias is not None:
                    self.register_buffer("bias", source.bias.detach().float().cpu())
                else:
                    self.bias = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape[:-1]
                x_flat = x.reshape(-1, self.in_features)
                q_a, a_scale, a_zp = _quantize_uint8_tensor(x_flat, self.activation_scale)
                a_center = q_a.to(torch.int32) - a_zp.to(torch.int32)
                b_center = self.q_weight.to(torch.int32) - self.w_zp.reshape(-1, 1).to(torch.int32)
                out_int = a_center @ b_center.T
                out = out_int.float()
                out = out * a_scale.float()
                out = out * self.w_scale.reshape(1, -1).to(out.device)
                if self.bias is not None:
                    out = out + self.bias.to(out.device).reshape(1, -1)
                return out.reshape(*shape, self.out_features).to(dtype=x.dtype)

        return _ExactUint8W8A8Linear(*args, **kwargs)


class LutUint8W8A8Linear:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch

        class _LutUint8W8A8Linear(torch.nn.Module):
            def __init__(self, source: torch.nn.Linear, activation_scale: str, weight_scale: str, lut: np.ndarray):
                super().__init__()
                q_weight, w_scale, w_zp = _quantize_weight_uint8(source.weight, weight_scale)
                if weight_scale == "per_tensor":
                    q_weight = q_weight.reshape_as(source.weight).contiguous()
                self.register_buffer("q_weight", q_weight.cpu())
                self.register_buffer("w_scale", w_scale.float().cpu().reshape(-1))
                self.register_buffer("w_zp", w_zp.to(torch.int32).cpu().reshape(-1))
                self.activation_scale = activation_scale
                self.out_features = source.out_features
                self.in_features = source.in_features
                self.lut = np.asarray(lut, dtype=np.uint32)
                self.q_weight_np = self.q_weight.numpy().astype(np.uint8, copy=False)
                self.sum_b_np = np.sum(self.q_weight_np.astype(np.int64), axis=1, keepdims=False)
                self.w_zp_np = self.w_zp.numpy().astype(np.int64, copy=False)
                if source.bias is not None:
                    self.register_buffer("bias", source.bias.detach().float().cpu())
                else:
                    self.bias = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape[:-1]
                x_flat = x.reshape(-1, self.in_features)
                q_a, a_scale, a_zp = _quantize_uint8_tensor(x_flat, self.activation_scale)
                a_np = q_a.detach().cpu().numpy().astype(np.uint8, copy=False)
                za_np = a_zp.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1, 1)
                approx_products = _uint8_product_sum_lut(a_np, self.q_weight_np, self.lut)
                sum_a = np.sum(a_np.astype(np.int64), axis=1, keepdims=True)
                k_dim = a_np.shape[1]
                approx_int = (
                    approx_products
                    - sum_a * self.w_zp_np.reshape(1, -1)
                    - za_np * self.sum_b_np.reshape(1, -1)
                    + k_dim * za_np * self.w_zp_np.reshape(1, -1)
                )
                out = torch.from_numpy(approx_int.astype(np.float32, copy=False)).to(x.device)
                out = out * a_scale.float()
                out = out * self.w_scale.reshape(1, -1).to(out.device)
                if self.bias is not None:
                    out = out + self.bias.to(out.device).reshape(1, -1)
                return out.reshape(*shape, self.out_features).to(dtype=x.dtype)

        return _LutUint8W8A8Linear(*args, **kwargs)


def _iter_named_linears(model: Any, torch_module: Any) -> list[tuple[str, Any, str, Any]]:
    items: list[tuple[str, Any, str, Any]] = []

    def visit(parent: Any, prefix: str) -> None:
        for child_name, child in parent.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch_module.nn.Linear):
                items.append((full_name, parent, child_name, child))
            else:
                visit(child, full_name)

    visit(model, "")
    return items


def _replace_linears(model: Any, args: argparse.Namespace, design: str, lut: np.ndarray | None) -> list[str]:
    import torch

    replaced: list[str] = []
    for full_name, parent, child_name, module in _iter_named_linears(model, torch):
        if not args.include_lm_head and full_name == "lm_head":
            continue
        if args.max_linear_layers and len(replaced) >= args.max_linear_layers:
            break
        if design == "exact_uint8_w8a8":
            new_module = ExactUint8W8A8Linear(module, args.activation_scale, args.weight_scale)
        else:
            if lut is None:
                raise ValueError(f"LUT required for design {design}")
            new_module = LutUint8W8A8Linear(module, args.activation_scale, args.weight_scale, lut)
        setattr(parent, child_name, new_module)
        replaced.append(full_name)
    return replaced


def _load_eval_token_ids(args: argparse.Namespace, tokenizer: Any) -> Any:
    import torch
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.eval_style == "axcore":
        joined = "\n\n".join(str(row.get("text", "")) for row in dataset)
        ids = tokenizer(joined, return_tensors="pt").input_ids
        if args.max_eval_tokens > 0:
            ids = ids[:, : args.max_eval_tokens + 1]
        if ids.shape[1] < 2:
            raise RuntimeError("not enough tokens for PPL evaluation")
        return ids.to(torch.long)

    texts: list[str] = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
        if args.max_eval_tokens > 0 and len(" ".join(texts)) > args.max_eval_tokens * 8:
            break
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    if args.max_eval_tokens > 0:
        ids = ids[:, : args.max_eval_tokens + 1]
    if ids.shape[1] < 2:
        raise RuntimeError("not enough tokens for PPL evaluation")
    return ids.to(torch.long)


def _evaluate_ppl(model: Any, input_ids: Any, *, seq_len: int, device: str, eval_style: str) -> dict[str, float]:
    import torch

    model.eval().to(device)
    nll_sum = 0.0
    token_count = 0
    chunk_count = 0
    with torch.no_grad():
        if eval_style == "axcore":
            nsamples = input_ids.numel() // seq_len
            loss_fct = torch.nn.CrossEntropyLoss()
            for i in range(nsamples):
                chunk = input_ids[:, i * seq_len : (i + 1) * seq_len].to(device)
                lm_logits = model(chunk).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = chunk[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                nll_sum += float(loss.detach().cpu()) * seq_len
                token_count += seq_len
                chunk_count += 1
            mean_nll = nll_sum / max(token_count, 1)
            return {
                "nll": mean_nll,
                "ppl": float(math.exp(mean_nll)) if mean_nll < 100 else float("inf"),
                "eval_tokens": float(token_count),
                "chunks": float(chunk_count),
            }

        for start in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(start + seq_len, input_ids.shape[1] - 1)
            if end <= start:
                continue
            chunk = input_ids[:, start : end + 1].to(device)
            outputs = model(chunk, labels=chunk)
            effective_tokens = chunk.shape[1] - 1
            nll_sum += float(outputs.loss.detach().cpu()) * effective_tokens
            token_count += effective_tokens
            chunk_count += 1
    mean_nll = nll_sum / max(token_count, 1)
    return {
        "nll": mean_nll,
        "ppl": float(math.exp(mean_nll)) if mean_nll < 100 else float("inf"),
        "eval_tokens": float(token_count),
        "chunks": float(chunk_count),
    }


def _run_design(args: argparse.Namespace, design: str, luts: dict[str, np.ndarray]) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    input_ids = _load_eval_token_ids(args, tokenizer)
    replaced: list[str] = []
    if design != "fp32":
        lut = None if design == "exact_uint8_w8a8" else luts[design]
        replaced = _replace_linears(model, args, design, lut)
    metrics = _evaluate_ppl(model, input_ids, seq_len=args.seq_len, device=args.device, eval_style=args.eval_style)
    elapsed = time.perf_counter() - start
    return {
        "design": design,
        "label": DESIGN_LABELS.get(design, design),
        "metrics": metrics,
        "replaced_linear_count": len(replaced),
        "replaced_linears": replaced,
        "elapsed_sec": elapsed,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    exact_ppl = None
    for result in data["results"]:
        if result["design"] == "exact_uint8_w8a8":
            exact_ppl = float(result["metrics"]["ppl"])
            break
    rows = []
    for result in data["results"]:
        ppl = float(result["metrics"]["ppl"])
        delta_exact = "" if exact_ppl is None else f"{ppl - exact_ppl:.4f}"
        rel_exact = "" if exact_ppl is None else f"{(ppl / exact_ppl - 1.0) * 100.0:.2f}%"
        rows.append(
            {
                "设计": result["label"],
                "PPL": f"{ppl:.4f}",
                "NLL": f"{float(result['metrics']['nll']):.6f}",
                "vs exact": delta_exact,
                "vs exact %": rel_exact,
                "替换Linear数": result["replaced_linear_count"],
                "耗时(s)": f"{float(result['elapsed_sec']):.1f}",
            }
        )

    lines = [
        "# Unsigned uint8 W8A8 PPL Smoke Test",
        "",
        "## 实验说明",
        "",
        "本实验把模型中的 `nn.Linear` 替换为 asymmetric uint8 activation / asymmetric uint8 weight 行为模型，并用 LUT 近似原始乘法项。",
        "",
        "$$",
        r"\sum_k (q_{a,k}-z_a)(q_{b,k}-z_b)=\sum_k q_{a,k}q_{b,k}-z_b\sum_k q_{a,k}-z_a\sum_k q_{b,k}+Kz_az_b",
        "$$",
        "",
        "其中 approximate multiplier 只替换 \\(q_aq_b\\)，zero-point correction 暂按精确整数加减处理。",
        "",
        "## 配置",
        "",
        f"- model: `{data['config']['model']}`",
        f"- dataset: `{data['config']['dataset']}` / `{data['config']['dataset_config']}` / `{data['config']['split']}`",
        f"- seq_len: `{data['config']['seq_len']}`",
        f"- max_eval_tokens: `{data['config']['max_eval_tokens']}`",
        f"- activation quantization: `uint8 asymmetric {data['config']['activation_scale']}`",
        f"- weight quantization: `uint8 asymmetric {data['config']['weight_scale']}`",
        f"- include_lm_head: `{data['config']['include_lm_head']}`",
        f"- max_linear_layers: `{data['config']['max_linear_layers']}`",
        "",
        "## 结果",
        "",
        _markdown_table(rows, ["设计", "PPL", "NLL", "vs exact", "vs exact %", "替换Linear数", "耗时(s)"]),
        "",
    ]
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
    luts = _load_luts(args)
    unknown_designs = [design for design in args.designs if design not in {"fp32", "exact_uint8_w8a8"} and design not in luts]
    if unknown_designs:
        raise ValueError(f"unknown designs: {', '.join(unknown_designs)}")

    results = []
    for design in args.designs:
        print(f"running {design} ...", flush=True)
        results.append(_run_design(args, design, luts))
        print(
            f"finished {design}: ppl={results[-1]['metrics']['ppl']:.4f}, "
            f"replaced={results[-1]['replaced_linear_count']}",
            flush=True,
        )

    data = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "seq_len": args.seq_len,
            "max_eval_tokens": args.max_eval_tokens,
            "eval_style": args.eval_style,
            "activation_scale": args.activation_scale,
            "weight_scale": args.weight_scale,
            "designs": args.designs,
            "max_linear_layers": args.max_linear_layers,
            "include_lm_head": args.include_lm_head,
            "device": args.device,
            "local_files_only": args.local_files_only,
        },
        "results": results,
    }
    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(md_path, data)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
