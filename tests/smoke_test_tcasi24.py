"""Smoke tests that run without pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.gemm import exact_gemm, lut_gemm
from am_lut_tcasi24.tcasi24 import build_int8_lut, mul4_csam2, mul4_lsam1, mul8_unsigned


def main() -> None:
    for a in range(16):
        for b in range(16):
            assert 0 <= mul4_lsam1(a, b) <= 255
            assert 0 <= mul4_csam2(a, b) <= 255

    for a in range(256):
        for b in range(256):
            assert mul8_unsigned(a, b, "exact") == a * b

    exact = build_int8_lut("exact")
    assert exact.shape == (256, 256)
    assert exact.dtype == np.int16
    assert int(exact[0, 0]) == 16384
    assert int(exact[255, 255]) == 16129
    assert int(exact[127, 127]) == 1

    a = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int8)
    b = np.array([[7, -8], [9, -10], [11, -12]], dtype=np.int8)
    assert np.array_equal(exact_gemm(a, b), lut_gemm(a, b, exact))
    print("smoke tests passed")


if __name__ == "__main__":
    main()
