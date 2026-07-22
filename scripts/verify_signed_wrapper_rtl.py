"""Verify signed_approx88_wrapper RTL against generated signed-wrapper LUTs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_fpga_signed_wrapper_luts import PRIMITIVES_VERILOG


TESTBENCH_VERILOG = r"""
module tb;
    reg signed [7:0] a;
    reg signed [7:0] b;
    wire signed [16:0] prod;
    integer i;
    integer j;

    signed_approx88_wrapper dut(.a(a), .b(b), .prod(prod));

    initial begin
        for (i = -128; i < 128; i = i + 1) begin
            for (j = -128; j < 128; j = j + 1) begin
                a = i[7:0];
                b = j[7:0];
                #1;
                $display("%0d,%0d,%0d", i, j, $signed(prod));
            end
        end
        $finish;
    end
endmodule
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpga-root", default="FPGA_multiplier", help="directory containing multiplier RTL")
    parser.add_argument("--lut-dir", default="outputs/fpga_luts", help="directory containing generated signed LUTs")
    parser.add_argument("--candidates", nargs="+", default=["17", "20", "10"], help="candidate ids to verify")
    parser.add_argument("--iverilog", default="iverilog", help="iverilog executable")
    parser.add_argument("--vvp", default="vvp", help="vvp executable")
    return parser.parse_args()


def _simulate_wrapper(candidate_verilog: Path, wrapper_verilog: Path, *, iverilog: str, vvp: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="signed_wrapper_rtl_") as tmp:
        tmp_dir = Path(tmp)
        primitives = tmp_dir / "xilinx_primitives_sim.v"
        testbench = tmp_dir / "tb_signed_wrapper.v"
        sim_out = tmp_dir / "sim.vvp"
        primitives.write_text(PRIMITIVES_VERILOG, encoding="utf-8")
        testbench.write_text(TESTBENCH_VERILOG, encoding="utf-8")

        compile_cmd = [
            iverilog,
            "-g2012",
            "-o",
            str(sim_out),
            str(primitives),
            str(candidate_verilog),
            str(wrapper_verilog),
            str(testbench),
        ]
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        run = subprocess.run([vvp, str(sim_out)], check=True, capture_output=True, text=True)

    lut = np.empty((256, 256), dtype=np.int32)
    line_count = 0
    for line in run.stdout.splitlines():
        if not re.fullmatch(r"-?\d+,-?\d+,-?\d+", line.strip()):
            continue
        a_s, b_s, p_s = line.strip().split(",")
        lut[int(a_s) + 128, int(b_s) + 128] = int(p_s)
        line_count += 1
    if line_count != 256 * 256:
        raise AssertionError(f"expected 65536 simulation rows, got {line_count}")
    return lut


def main() -> None:
    args = parse_args()
    fpga_root = Path(args.fpga_root)
    lut_dir = Path(args.lut_dir)
    wrapper_verilog = fpga_root / "signed_wrapper" / "signed_approx88_wrapper.v"
    if not wrapper_verilog.exists():
        raise FileNotFoundError(wrapper_verilog)

    for candidate in args.candidates:
        candidate_verilog = fpga_root / "approx_unsigned8x8" / str(candidate) / "final_best_approx88_cascade.v"
        expected_lut_path = lut_dir / f"fpga_cand{candidate}_signed_wrapper_int8_lut.npy"
        expected = np.load(expected_lut_path).astype(np.int32)
        observed = _simulate_wrapper(candidate_verilog, wrapper_verilog, iverilog=args.iverilog, vvp=args.vvp)
        if not np.array_equal(observed, expected):
            diff = observed - expected
            idx = np.argwhere(diff != 0)[0]
            a = int(idx[0]) - 128
            b = int(idx[1]) - 128
            raise AssertionError(
                f"candidate {candidate} mismatch at a={a}, b={b}: "
                f"rtl={observed[idx[0], idx[1]]}, lut={expected[idx[0], idx[1]]}"
            )
        print(f"candidate {candidate}: signed wrapper RTL matches LUT for 65536 signed int8 cases")


if __name__ == "__main__":
    main()

