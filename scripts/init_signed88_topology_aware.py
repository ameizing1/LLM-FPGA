#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "multiplier_models"))

from signed88.common import ObjectiveWeights, bits_int, hex_to_int, int_to_hex, write_json
from signed88.data import load_calibration_csv
from signed88.hardware import get_design
from signed88.metrics import evaluate_design

DEFAULT_CALIBRATION_CSV = (
    PROJECT_ROOT / "tests" / "data" / "w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
)
RTL_TEMPLATE_ROOT = PROJECT_ROOT / "FPGA_multiplier" / "signed8x8_6x2"


@dataclass(frozen=True)
class Vote:
    zero: float = 0.0
    one: float = 0.0

    def add(self, target: int, weight: float) -> "Vote":
        if target:
            return Vote(self.zero, self.one + weight)
        return Vote(self.zero + weight, self.one)


BIT_TO_TABLE_AND_ADDR = {
    0: ("cp_lut01", "o5"),
    1: ("cp_lut01", "o6"),
    2: ("cp_lut23", "o5"),
    3: ("cp_lut23", "o6"),
    4: ("cp_lut45", "o5"),
    5: ("cp_lut45", "o6"),
    6: ("cp_lut67", "o5"),
    7: ("cp_lut67", "o6"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a topology-aware INIT for the signed88 CP-hybrid designs. "
            "The script keeps the fixed RTL topology and chooses mutable LUT "
            "entries by calibration-weighted bit votes."
        )
    )
    parser.add_argument("--design", default="balanced", choices=("quality", "balanced"))
    parser.add_argument("--calibration-csv", default=str(DEFAULT_CALIBRATION_CSV))
    parser.add_argument(
        "--calibration-weight-column",
        default="auto",
        choices=("auto", "count", "p_calib", "weight", "probability"),
    )
    parser.add_argument(
        "--vote-weighting",
        default="shifted_bit_value",
        choices=("probability", "local_bit_value", "shifted_bit_value"),
        help="how much a local product bit vote should count",
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "tmp" / "signed88_topology_init"))
    parser.add_argument("--rtl-template-root", default=str(RTL_TEMPLATE_ROOT))
    parser.add_argument("--score-mred-weight", type=float, default=1.0)
    parser.add_argument("--score-er-weight", type=float, default=0.25)
    parser.add_argument("--score-ned-weight", type=float, default=0.10)
    parser.add_argument("--score-bias-weight", type=float, default=0.05)
    parser.add_argument("--score-uniform-mred-weight", type=float, default=0.05)
    return parser.parse_args()


def cp62_addr(al: int, bit: int) -> tuple[str, int]:
    table, out = BIT_TO_TABLE_AND_ADDR[bit]
    if table == "cp_lut01":
        addr = 3 + (((al >> 0) & 1) << 2) + (((al >> 1) & 1) << 3)
        addr += 16 if out == "o5" else 48
        return table, addr
    if table == "cp_lut23":
        addr = 3 + (((al >> 1) & 1) << 2) + (((al >> 2) & 1) << 3) + (((al >> 3) & 1) << 4)
        if out == "o6":
            addr += 32
        return table, addr
    if table in ("cp_lut45", "cp_lut67"):
        addr = 3 + (((al >> 3) & 1) << 2) + (((al >> 4) & 1) << 3) + (((al >> 5) & 1) << 4)
        if out == "o6":
            addr += 32
        return table, addr
    raise ValueError(table)


def vote_weight(probability: float, local_bit: int, shift: int, mode: str) -> float:
    if mode == "probability":
        return probability
    if mode == "local_bit_value":
        return probability * float(1 << local_bit)
    if mode == "shifted_bit_value":
        return probability * float(1 << (local_bit + shift))
    raise ValueError(mode)


def build_votes(design, profile, mode: str) -> dict[str, dict[int, Vote]]:
    votes: dict[str, dict[int, Vote]] = {name: {} for name in design.spec.train_names}
    approx_positions = [pos for pos in range(3) if (design.approx_mask >> pos) & 1]
    for a, b, probability in zip(profile.a, profile.b, profile.probability):
        al = int(a) & 63
        bl = int(b) & 63
        exact_local = al * 3
        for pos in approx_positions:
            shift = 2 * pos
            digit = (bl >> shift) & 3
            if digit != 3:
                continue
            for bit in range(8):
                table, addr = cp62_addr(al, bit)
                if addr not in design.spec.mutable_bits[table]:
                    continue
                target = (exact_local >> bit) & 1
                weight = vote_weight(float(probability), bit, shift, mode)
                votes[table][addr] = votes[table].get(addr, Vote()).add(target, weight)
    return votes


def apply_votes(base_inits: Mapping[str, str], votes: Mapping[str, Mapping[int, Vote]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, init_hex in base_inits.items():
        bits = [(hex_to_int(init_hex) >> i) & 1 for i in range(64)]
        for addr, vote in votes.get(name, {}).items():
            if vote.one > vote.zero:
                bits[int(addr)] = 1
            elif vote.zero > vote.one:
                bits[int(addr)] = 0
        out[name] = int_to_hex(bits_int(bits))
    return out


def vote_summary(votes: Mapping[str, Mapping[int, Vote]], base_inits: Mapping[str, str], new_inits: Mapping[str, str]) -> dict:
    rows = {}
    for name, table_votes in votes.items():
        base = hex_to_int(base_inits[name])
        new = hex_to_int(new_inits[name])
        changed = []
        undecided = []
        for addr, vote in sorted(table_votes.items()):
            old_bit = (base >> addr) & 1
            new_bit = (new >> addr) & 1
            row = {
                "addr": int(addr),
                "old": int(old_bit),
                "new": int(new_bit),
                "vote_zero": vote.zero,
                "vote_one": vote.one,
                "margin": abs(vote.one - vote.zero),
            }
            if old_bit != new_bit:
                changed.append(row)
            if abs(vote.one - vote.zero) <= 1e-15:
                undecided.append(row)
        rows[name] = {
            "voted_entries": len(table_votes),
            "changed_entries": changed,
            "undecided_entries": undecided,
        }
    return rows


def main() -> int:
    args = parse_args()
    design = get_design(args.design)
    profile = load_calibration_csv(Path(args.calibration_csv), args.calibration_weight_column)
    objective = ObjectiveWeights(
        args.score_mred_weight,
        args.score_er_weight,
        args.score_ned_weight,
        args.score_bias_weight,
        args.score_uniform_mred_weight,
    )
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_inits = design.normalize_inits(design.spec.base_inits)
    votes = build_votes(design, profile, args.vote_weighting)
    new_inits = design.normalize_inits(apply_votes(base_inits, votes))
    base_metrics = evaluate_design(design, base_inits, profile, objective)
    new_metrics = evaluate_design(design, new_inits, profile, objective)

    artifact = design.artifact(
        new_inits,
        metrics=new_metrics.to_dict(),
        extra={
            "stage": "topology_aware_bit_vote_init",
            "calibration": profile.metadata(),
            "objective_weights": asdict(objective),
            "vote_weighting": args.vote_weighting,
            "base_metrics": base_metrics.to_dict(),
            "vote_summary": vote_summary(votes, base_inits, new_inits),
        },
    )
    write_json(out / "topology_aware_signed88_inits.json", artifact)
    design.export_rtl(
        Path(args.rtl_template_root),
        out / "topology_aware_rtl",
        new_inits,
        metadata={
            "metrics": new_metrics.to_dict(),
            "calibration": profile.metadata(),
            "objective_weights": asdict(objective),
            "vote_weighting": args.vote_weighting,
        },
    )
    summary = {
        "design": design.spec.name,
        "resources": design.spec.resource_summary,
        "vote_weighting": args.vote_weighting,
        "base_metrics": base_metrics.to_dict(),
        "new_metrics": new_metrics.to_dict(),
        "vote_summary": artifact["vote_summary"],
        "artifact": str(out / "topology_aware_signed88_inits.json"),
        "rtl": str(out / "topology_aware_rtl"),
    }
    write_json(out / "summary.json", summary)
    print(f"[design] {design.spec.name} resources={design.spec.resource_summary}")
    print(f"[vote-weighting] {args.vote_weighting}")
    print(f"[base] {base_metrics.short()}")
    print(f"[topology-aware] {new_metrics.short()}")
    print(f"[artifact] {out / 'topology_aware_signed88_inits.json'}")
    print(f"[rtl] {out / 'topology_aware_rtl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
