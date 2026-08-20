"""Evaluate a GEMM-aware signed88 artifact on an independent sample archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "multiplier_models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from signed88.common import ObjectiveWeights, read_json
from signed88.data import load_calibration_csv
from signed88.hardware import get_design
from signed88.metrics import evaluate_design

from refine_signed88_gemm_aware import (
    _prepare_samples,
    _sample_metrics,
)
import run_signed_w8a8_layerwise_bias_report as layerwise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument(
        "--calibration-csv",
        default=str(PROJECT_ROOT / "tests" / "data" / "w8a8_calibration_hist_smoke_pcalib_nonzero.csv"),
    )
    p.add_argument("--calibration-weight-column", default="auto")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "reports"))
    p.add_argument("--report-name", default="signed88_gemm_aware_holdout")
    p.add_argument("--gemm-rel-l2-weight", type=float, default=1.0)
    p.add_argument("--gemm-nmae-weight", type=float, default=0.25)
    p.add_argument("--gemm-bias-weight", type=float, default=0.25)
    p.add_argument("--gemm-directionality-weight", type=float, default=0.05)
    return p.parse_args()


def _score(metrics: dict, args: argparse.Namespace) -> float:
    return (
        args.gemm_rel_l2_weight * metrics["relative_l2"]
        + args.gemm_nmae_weight * metrics["normalized_mae"]
        + args.gemm_bias_weight * metrics["bias_ratio"]
        + args.gemm_directionality_weight * metrics["directionality"]
    )


def _load_samples(path: Path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"].item()))
    raw = []
    for item in metadata["layers"]:
        prefix = item["prefix"]
        raw.append(
            layerwise.LayerSample(
                name=item["name"],
                activation=archive[f"{prefix}_activation"],
                activation_scale=archive[f"{prefix}_activation_scale"],
                weight=archive[f"{prefix}_weight"],
                weight_scale=archive[f"{prefix}_weight_scale"],
            )
        )
    return _prepare_samples(raw), metadata.get("config", {})


def _write_md(path: Path, data: dict) -> None:
    rows = [data["baseline"], data["candidate"]]
    lines = [
        "# Signed88 GEMM-aware Holdout Evaluation",
        "",
        "本报告只读取独立 holdout 样本，不参与 INIT 搜索。候选与 baseline 使用完全相同的量化矩阵和 scale。",
        "",
        "| 设计 | GEMM score | rel L2 | nMAE | bias ratio | directionality | product WCE | product workload MRED |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    f"{row['gemm']['score']:.8f}",
                    f"{row['gemm']['relative_l2']:.8f}",
                    f"{row['gemm']['normalized_mae']:.8f}",
                    f"{row['gemm']['bias_ratio']:.8f}",
                    f"{row['gemm']['directionality']:.8f}",
                    str(row["product"]["WCE"]),
                    f"{row['product']['workload_MRED']:.8f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- holdout layers: `{data['sample_config'].get('layers', data['sample_count'])}`",
            f"- sample archive: `{data['samples']}`",
            f"- artifact: `{data['artifact']}`",
            "",
            "该结果用于判断小规模搜索得到的 INIT 是否能在未参与搜索的层上保持改善；最终仍需端到端 PPL 验证。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)
    artifact = read_json(artifact_path)
    design = get_design(artifact.get("design", "balanced"))
    inits = design.normalize_inits(artifact["inits"])
    baseline = design.normalize_inits(design.spec.base_inits)
    samples, sample_config = _load_samples(Path(args.samples))
    profile = load_calibration_csv(Path(args.calibration_csv), args.calibration_weight_column)
    objective = ObjectiveWeights()

    def evaluate(label: str, candidate: dict[str, str]) -> dict:
        gemm = _sample_metrics(samples, design.hard_low_numpy(candidate))
        gemm["score"] = _score(gemm, args)
        return {
            "label": label,
            "product": evaluate_design(design, candidate, profile, objective).to_dict(),
            "gemm": gemm,
        }

    data = {
        "artifact": str(artifact_path.resolve()),
        "samples": str(Path(args.samples).resolve()),
        "sample_config": sample_config,
        "sample_count": len(samples),
        "baseline": evaluate("Balanced baseline", baseline),
        "candidate": evaluate("GEMM-aware candidate", inits),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.report_name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_md(out / f"{args.report_name}.md", data)
    print(json.dumps({
        "baseline": data["baseline"]["gemm"],
        "candidate": data["candidate"]["gemm"],
        "report": str(out / f"{args.report_name}.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
