"""Generate signed-wrapper LUTs from FPGA_multiplier unsigned Verilog candidates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_unsigned8_lut


PRIMITIVES_VERILOG = r"""
module LUT6 #(parameter [63:0] INIT = 64'h0)
(
    input I0, input I1, input I2, input I3, input I4, input I5,
    output O
);
    assign O = INIT[{I5, I4, I3, I2, I1, I0}];
endmodule

module LUT6_2 #(parameter [63:0] INIT = 64'h0)
(
    input I0, input I1, input I2, input I3, input I4, input I5,
    output O5, output O6
);
    assign O5 = INIT[{1'b0, I4, I3, I2, I1, I0}];
    assign O6 = INIT[{I5, I4, I3, I2, I1, I0}];
endmodule

module CARRY4
(
    output [3:0] CO,
    output [3:0] O,
    input CI,
    input CYINIT,
    input [3:0] DI,
    input [3:0] S
);
    wire c0, c1, c2, c3;
    assign c0 = CI | CYINIT;
    assign O[0] = S[0] ^ c0;
    assign CO[0] = S[0] ? c0 : DI[0];
    assign c1 = CO[0];
    assign O[1] = S[1] ^ c1;
    assign CO[1] = S[1] ? c1 : DI[1];
    assign c2 = CO[1];
    assign O[2] = S[2] ^ c2;
    assign CO[2] = S[2] ? c2 : DI[2];
    assign c3 = CO[2];
    assign O[3] = S[3] ^ c3;
    assign CO[3] = S[3] ? c3 : DI[3];
endmodule
"""


TESTBENCH_VERILOG = r"""
module tb;
    reg [7:0] a;
    reg [7:0] b;
    wire [15:0] prod;
    integer i;
    integer j;

    approx88_cascade dut(.a(a), .b(b), .prod(prod));

    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                a = i[7:0];
                b = j[7:0];
                #1;
                $display("%0d,%0d,%0d", i, j, prod);
            end
        end
        $finish;
    end
endmodule
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpga-root", default="FPGA_multiplier", help="directory containing approx_unsigned8x8")
    parser.add_argument("--out-dir", default="outputs/fpga_luts", help="directory for generated .npy LUTs")
    parser.add_argument("--candidates", nargs="+", default=["17", "20", "10"], help="candidate ids to generate")
    parser.add_argument("--iverilog", default="iverilog", help="iverilog executable")
    parser.add_argument("--vvp", default="vvp", help="vvp executable")
    return parser.parse_args()


def _candidate_dir(fpga_root: Path, candidate: str) -> Path:
    path = fpga_root / "approx_unsigned8x8" / str(candidate)
    if not path.is_dir():
        raise FileNotFoundError(f"candidate directory not found: {path}")
    return path


def _simulate_unsigned_lut(verilog_path: Path, *, iverilog: str, vvp: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="fpga_lut_") as tmp:
        tmp_dir = Path(tmp)
        primitives = tmp_dir / "xilinx_primitives_sim.v"
        testbench = tmp_dir / "tb_dump_unsigned_lut.v"
        sim_out = tmp_dir / "sim.vvp"
        primitives.write_text(PRIMITIVES_VERILOG, encoding="utf-8")
        testbench.write_text(TESTBENCH_VERILOG, encoding="utf-8")

        compile_cmd = [iverilog, "-g2012", "-o", str(sim_out), str(primitives), str(verilog_path), str(testbench)]
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        run = subprocess.run([vvp, str(sim_out)], check=True, capture_output=True, text=True)

    lut = np.empty((256, 256), dtype=np.uint32)
    line_count = 0
    for line in run.stdout.splitlines():
        if not re.fullmatch(r"\d+,\d+,\d+", line.strip()):
            continue
        a_s, b_s, p_s = line.strip().split(",")
        lut[int(a_s), int(b_s)] = int(p_s)
        line_count += 1
    if line_count != 256 * 256:
        raise AssertionError(f"expected 65536 simulation rows, got {line_count}")
    return lut


def _metrics_unsigned(lut: np.ndarray) -> dict[str, float]:
    exact = exact_unsigned8_lut().astype(np.int64)
    approx = lut.astype(np.int64)
    err = approx - exact
    abs_err = np.abs(err)
    nonzero = exact != 0
    red = abs_err[nonzero] / exact[nonzero]
    return {
        "total_cases": float(exact.size),
        "error_cases": float(np.count_nonzero(err)),
        "ER": float(np.mean(err != 0)),
        "MED": float(np.mean(abs_err)),
        "NED": float(np.mean(abs_err) / ((2**8 - 1) ** 2)),
        "MRED": float(np.mean(red)),
        "WCE": float(np.max(abs_err)),
    }


def _load_expected_metrics(candidate_dir: Path) -> dict[str, Any] | None:
    json_path = candidate_dir / "final_best_approx88_cascade_inits.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8")).get("metrics")


def _check_against_expected(candidate: str, observed: dict[str, float], expected: dict[str, Any] | None) -> None:
    if expected is None:
        return
    # MRED definitions vary slightly across scripts when exact product is zero
    # or tiny. The hard checks below are enough to catch wrong primitive
    # semantics while avoiding a false failure from a reporting convention.
    for key in ("error_cases", "ER", "MED", "WCE"):
        if key not in expected:
            continue
        obs = observed[key]
        exp = float(expected[key])
        if not np.isclose(obs, exp, rtol=1e-6, atol=1e-6):
            raise AssertionError(f"candidate {candidate} {key} mismatch: simulated={obs}, expected={exp}")


def _write_metadata(out_dir: Path, metadata: dict[str, Any]) -> None:
    path = out_dir / "fpga_signed_wrapper_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    fpga_root = Path(args.fpga_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "indexing": "signed_lut[a + 128, b + 128] for signed int8 operands a and b",
        "signed_wrapper": "abs(a), abs(b) -> unsigned approx88_cascade -> restore sign",
        "unsigned_lut_indexing": "unsigned_lut[a, b] for raw uint8 magnitudes",
        "candidates": {},
    }

    for candidate in args.candidates:
        candidate_dir = _candidate_dir(fpga_root, candidate)
        verilog_path = candidate_dir / "final_best_approx88_cascade.v"
        unsigned_lut = _simulate_unsigned_lut(verilog_path, iverilog=args.iverilog, vvp=args.vvp)
        observed = _metrics_unsigned(unsigned_lut)
        _check_against_expected(candidate, observed, _load_expected_metrics(candidate_dir))

        signed_lut = build_signed_wrapper_lut(unsigned_lut)
        unsigned_path = out_dir / f"fpga_cand{candidate}_unsigned8_lut.npy"
        signed_path = out_dir / f"fpga_cand{candidate}_signed_wrapper_int8_lut.npy"
        np.save(unsigned_path, unsigned_lut)
        np.save(signed_path, signed_lut.astype(np.int32))

        metadata["candidates"][candidate] = {
            "source_verilog": str(verilog_path),
            "unsigned_lut": str(unsigned_path),
            "signed_lut": str(signed_path),
            "unsigned_metrics": observed,
        }
        print(
            f"candidate {candidate}: wrote {signed_path} "
            f"ER={observed['ER']:.6f} MED={observed['MED']:.3f} WCE={observed['WCE']:.0f}"
        )

    _write_metadata(out_dir, metadata)


if __name__ == "__main__":
    main()
