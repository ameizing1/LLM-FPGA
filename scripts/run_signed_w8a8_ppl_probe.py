"""Run a small end-to-end PPL probe for signed W8A8 LUT-backed Linear layers."""

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

from am_lut_tcasi24.gemm import lut_gemm


DESIGN_LABELS = {
    "fp32": "FP32/BF16 original",
    "exact_w8a8": "Exact signed W8A8",
    "tcasi24_lsam1": "TCASI24 LSAM1",
    "tcasi24_csam2": "TCASI24 CSAM2",
    "fpga_cand17": "FPGA cand17",
    "fpga_cand20": "FPGA cand20",
    "fpga_cand10": "FPGA cand10",
}


def _s88_label(variant: str, family: str) -> str:
    if variant.startswith("balanced_"):
        return f"S88-Balanced: approx low/mid 6x2, high exact ({variant})"
    if variant.startswith("fast_"):
        return f"S88-Fast: all 6x2 use no-CARRY4 local carry prediction ({variant})"
    if variant.startswith("area_"):
        return f"S88-Area: AL quantized to multiples of 16, truncated LL low carry ({variant})"
    if variant.startswith("aggressive_"):
        return f"S88-Aggressive: LL LUT-only, CARRY4 from exact fused MACs ({variant})"
    return f"{family} {variant}"


def _manualu88_label(variant: str) -> str:
    labels = {
        "approx2": "Manual-Exact-ish: accurate66 + accurate22, signed-wrapper exact",
        "approx5_1": "Manual-Comp66-Accurate: approx66 + accurate62 cross terms",
        "approx5_2": "Manual-Comp66-Accurate variant: approx66 + accurate62 cross terms",
        "approx5_7": "Manual-LowOR-HighCarry comp66",
        "approx5_8": "Manual-LowOR-HighCarry comp66 optimized",
        "approx5_9": "Manual-LowOR-HighCarry comp66 optimized+",
        "approx5_10": "Manual-AllOR comp66",
        "approx5_11": "Manual-AllOR comp66 optimized INIT",
        "approx5_12": "Manual-LUTBudget6 comp66 remap",
        "approx5_13": "Manual-LUTBudget6 comp66 remap variant",
        "approx5_14": "Manual-LUTBudget6 comp66 remap variant",
    }
    return labels.get(variant, f"Manual unsigned8x8 {variant} signed-wrapper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=256, help="0 means use all available tokens")
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="skip this many concatenated dataset tokens before evaluation",
    )
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
        default=["fp32", "exact_w8a8", "tcasi24_lsam1", "tcasi24_csam2", "fpga_cand17"],
    )
    parser.add_argument("--max-linear-layers", type=int, default=0, help="0 means replace all eligible Linear layers")
    parser.add_argument("--include-lm-head", action="store_true", help="also quantize/replace lm_head; slow for LUT designs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tcasi-lut-dir", default="outputs/luts")
    parser.add_argument("--tcasi8x8-lut-dir", default="outputs/tcasi24_8x8_luts")
    parser.add_argument("--fpga-lut-dir", default="outputs/fpga_luts")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--report-name", default="signed_w8a8_ppl_probe")
    return parser.parse_args()


def _require_runtime() -> None:
    missing = [name for name in ["torch", "transformers", "datasets"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def _load_luts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tcasi_dir = Path(args.tcasi_lut_dir)
    tcasi8x8_dir = Path(args.tcasi8x8_lut_dir)
    fpga_dir = Path(args.fpga_lut_dir)
    luts = {
        "tcasi24_lsam1": np.load(tcasi_dir / "lsam1_int8_lut.npy").astype(np.int32),
        "tcasi24_csam2": np.load(tcasi_dir / "csam2_int8_lut.npy").astype(np.int32),
        "fpga_cand17": np.load(fpga_dir / "fpga_cand17_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand20": np.load(fpga_dir / "fpga_cand20_signed_wrapper_int8_lut.npy").astype(np.int32),
        "fpga_cand10": np.load(fpga_dir / "fpga_cand10_signed_wrapper_int8_lut.npy").astype(np.int32),
    }
    for path in sorted(fpga_dir.glob("fpga_dist2055_cand*_signed_wrapper_int8_lut.npy")):
        design = path.name.removesuffix("_signed_wrapper_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        cand = design.removeprefix("fpga_dist2055_cand")
        DESIGN_LABELS.setdefault(design, f"Dist2055 cand{cand}")
    for path in sorted(fpga_dir.glob("s8862_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s8862_")
        DESIGN_LABELS.setdefault(design, f"signed8x8_6x2 {variant}")
    for path in sorted(fpga_dir.glob("s8888_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s8888_")
        DESIGN_LABELS.setdefault(design, _s88_label(variant, "signed8x8_202688_1000"))
    for path in sorted(fpga_dir.glob("s8889_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s8889_")
        DESIGN_LABELS.setdefault(design, _s88_label(variant, "signed8x8_202689_1800"))
    for path in sorted(fpga_dir.glob("s88ref_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s88ref_")
        DESIGN_LABELS.setdefault(design, f"Refined signed8x8 {variant}")
    for path in sorted(fpga_dir.glob("s88bias_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s88bias_")
        DESIGN_LABELS.setdefault(design, f"Bias-constrained signed8x8 {variant}")
    for path in sorted(fpga_dir.glob("s88gemm_*_signed_int8_lut.npy")):
        design = path.name.removesuffix("_signed_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("s88gemm_")
        DESIGN_LABELS.setdefault(design, f"GEMM-aware signed8x8 {variant}")
    for path in sorted(fpga_dir.glob("manualu88_*_signed_wrapper_int8_lut.npy")):
        design = path.name.removesuffix("_signed_wrapper_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("manualu88_")
        DESIGN_LABELS.setdefault(design, _manualu88_label(variant))
    for path in sorted(tcasi8x8_dir.glob("tcasi8x8_*_signed_wrapper_int8_lut.npy")):
        design = path.name.removesuffix("_signed_wrapper_int8_lut.npy")
        luts[design] = np.load(path).astype(np.int32)
        variant = design.removeprefix("tcasi8x8_").upper()
        DESIGN_LABELS.setdefault(design, f"TCASI24 {variant} RTL signed-wrapper")
    return luts


def _quantize_weight(weight: Any, mode: str) -> tuple[Any, Any]:
    import torch

    w = weight.detach().float()
    qmax = 127.0
    if mode == "per_tensor":
        scale = torch.max(torch.abs(w)) / qmax
        if not torch.isfinite(scale) or float(scale) == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=w.device)
            q = torch.zeros_like(w, dtype=torch.int8)
        else:
            q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        return q.cpu(), scale.detach().cpu().reshape(1)

    if mode == "per_channel":
        scale = torch.max(torch.abs(w), dim=1, keepdim=True).values / qmax
        scale = torch.where((scale == 0.0) | ~torch.isfinite(scale), torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        return q.cpu(), scale.detach().cpu().reshape(-1)

    raise ValueError(f"unknown weight scale mode: {mode}")


def _quantize_activation(x_flat: Any, mode: str) -> tuple[Any, Any]:
    import torch

    x = x_flat.float()
    qmax = 127.0
    if mode == "per_tensor":
        scale = torch.max(torch.abs(x)) / qmax
        if not torch.isfinite(scale) or float(scale) == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
            return torch.zeros_like(x, dtype=torch.int8), scale.reshape(1)
        return torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8), scale.reshape(1)

    if mode == "per_token":
        scale = torch.max(torch.abs(x), dim=1, keepdim=True).values / qmax
        scale = torch.where((scale == 0.0) | ~torch.isfinite(scale), torch.ones_like(scale), scale)
        return torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8), scale

    raise ValueError(f"unknown activation scale mode: {mode}")


class ExactW8A8Linear:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch

        class _ExactW8A8Linear(torch.nn.Module):
            def __init__(self, source: torch.nn.Linear, activation_scale: str, weight_scale: str):
                super().__init__()
                q_weight, w_scale = _quantize_weight(source.weight, weight_scale)
                self.register_buffer("q_weight", q_weight)
                self.register_buffer("w_scale", w_scale.float())
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
                q_a, a_scale = _quantize_activation(x_flat, self.activation_scale)
                out_int = q_a.to(torch.int32) @ self.q_weight.to(torch.int32).T
                out = out_int.float()
                if a_scale.numel() == 1:
                    out = out * a_scale.reshape(1, 1).float()
                else:
                    out = out * a_scale.float()
                out = out * self.w_scale.reshape(1, -1).to(out.device)
                if self.bias is not None:
                    out = out + self.bias.to(out.device).reshape(1, -1)
                return out.reshape(*shape, self.out_features).to(dtype=x.dtype)

        return _ExactW8A8Linear(*args, **kwargs)


class LutW8A8Linear:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch

        class _LutW8A8Linear(torch.nn.Module):
            def __init__(self, source: torch.nn.Linear, activation_scale: str, weight_scale: str, lut: np.ndarray):
                super().__init__()
                q_weight, w_scale = _quantize_weight(source.weight, weight_scale)
                self.register_buffer("q_weight", q_weight)
                self.register_buffer("w_scale", w_scale.float())
                self.activation_scale = activation_scale
                self.out_features = source.out_features
                self.in_features = source.in_features
                self.lut = np.asarray(lut, dtype=np.int32)
                if source.bias is not None:
                    self.register_buffer("bias", source.bias.detach().float().cpu())
                else:
                    self.bias = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = x.shape[:-1]
                x_flat = x.reshape(-1, self.in_features)
                q_a, a_scale = _quantize_activation(x_flat, self.activation_scale)
                a_np = q_a.detach().cpu().numpy().astype(np.int8, copy=False)
                b_np = self.q_weight.detach().cpu().numpy().astype(np.int8, copy=False).T.copy()
                out_int_np = lut_gemm(a_np, b_np, self.lut)
                out = torch.from_numpy(out_int_np.astype(np.float32, copy=False)).to(x.device)
                if a_scale.numel() == 1:
                    out = out * a_scale.reshape(1, 1).float()
                else:
                    out = out * a_scale.float()
                out = out * self.w_scale.reshape(1, -1).to(out.device)
                if self.bias is not None:
                    out = out + self.bias.to(out.device).reshape(1, -1)
                return out.reshape(*shape, self.out_features).to(dtype=x.dtype)

        return _LutW8A8Linear(*args, **kwargs)


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
    candidates = _iter_named_linears(model, torch)
    for full_name, parent, child_name, module in candidates:
        if not args.include_lm_head and full_name == "lm_head":
            continue
        if args.max_linear_layers and len(replaced) >= args.max_linear_layers:
            break
        if design == "exact_w8a8":
            new_module = ExactW8A8Linear(module, args.activation_scale, args.weight_scale)
        else:
            if lut is None:
                raise ValueError(f"LUT required for design {design}")
            new_module = LutW8A8Linear(module, args.activation_scale, args.weight_scale, lut)
        setattr(parent, child_name, new_module)
        replaced.append(full_name)
    return replaced


def _load_eval_token_ids(args: argparse.Namespace, tokenizer: Any) -> Any:
    import torch
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.eval_style == "axcore":
        joined = "\n\n".join(str(row.get("text", "")) for row in dataset)
        encoded = tokenizer(joined, return_tensors="pt")
        ids = encoded.input_ids
        offset = max(int(args.token_offset), 0)
        if args.max_eval_tokens > 0:
            ids = ids[:, offset : offset + args.max_eval_tokens + 1]
        elif offset:
            ids = ids[:, offset:]
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
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt")
    ids = encoded.input_ids
    offset = max(int(args.token_offset), 0)
    if args.max_eval_tokens > 0:
        ids = ids[:, offset : offset + args.max_eval_tokens + 1]
    elif offset:
        ids = ids[:, offset:]
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
        replaced = _replace_linears(model, args, design, None if design == "exact_w8a8" else luts[design])
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
    rows = []
    exact_ppl = None
    for result in data["results"]:
        if result["design"] == "exact_w8a8":
            exact_ppl = float(result["metrics"]["ppl"])
            break
    for result in data["results"]:
        ppl = float(result["metrics"]["ppl"])
        delta_exact = "" if exact_ppl is None else f"{ppl - exact_ppl:.4f}"
        rows.append(
            {
                "设计": result["label"],
                "PPL": f"{ppl:.4f}",
                "NLL": f"{float(result['metrics']['nll']):.6f}",
                "vs exact W8A8": delta_exact,
                "替换Linear数": result["replaced_linear_count"],
                "耗时(s)": f"{float(result['elapsed_sec']):.1f}",
            }
        )

    lines = [
        "# Signed W8A8 端到端 PPL Smoke Test",
        "",
        "## 实验说明",
        "",
        "本实验把 OPT 模型中的 `nn.Linear` 临时替换为 signed W8A8 行为模型，并在 WikiText2 上跑小规模 perplexity smoke test。默认不替换 `lm_head`，因为词表投影很大，第一版先聚焦 Transformer block 内部 Linear。",
        "",
        "近似版本使用 signed int8 product LUT 替换每个标量乘法，activation 使用运行时量化，weight 使用加载模型后静态量化。",
        "",
        "## 配置",
        "",
        f"- model: `{data['config']['model']}`",
        f"- dataset: `{data['config']['dataset']}` / `{data['config']['dataset_config']}` / `{data['config']['split']}`",
        f"- seq_len: `{data['config']['seq_len']}`",
        f"- max_eval_tokens: `{data['config']['max_eval_tokens']}`",
        f"- token_offset: `{data['config']['token_offset']}`",
        f"- activation quantization: `{data['config']['activation_scale']}`",
        f"- weight quantization: `{data['config']['weight_scale']}`",
        f"- include_lm_head: `{data['config']['include_lm_head']}`",
        f"- max_linear_layers: `{data['config']['max_linear_layers']}`",
        "",
        "## 结果",
        "",
        _markdown_table(rows, ["设计", "PPL", "NLL", "vs exact W8A8", "替换Linear数", "耗时(s)"]),
        "",
        "## 解读口径",
        "",
        "- `FP32/BF16 original` 是未替换 Linear 的原模型结果。",
        "- `Exact signed W8A8` 是主 baseline，近似乘法器的 PPL 退化应优先相对它比较。",
        "- 这是 smoke test，token 数较少，适合判断是否跑通和是否明显爆炸；正式结论需要扩大 token 数。",
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
    unknown_designs = [
        design
        for design in args.designs
        if design not in {"fp32", "exact_w8a8"} and design not in luts
    ]
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
            "token_offset": args.token_offset,
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
