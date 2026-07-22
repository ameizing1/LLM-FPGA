"""Smoke tests for signed-wrapper LUT construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiplier_models.signed_wrapper import build_signed_wrapper_lut, exact_int8_lut, exact_unsigned8_lut


def main() -> None:
    unsigned_exact = exact_unsigned8_lut()
    signed_exact = build_signed_wrapper_lut(unsigned_exact)
    assert signed_exact.shape == (256, 256)
    assert signed_exact.dtype == np.int32
    assert np.array_equal(signed_exact, exact_int8_lut())

    # Boundary cases that often expose wrong sign-magnitude handling.
    assert int(signed_exact[0, 0]) == 16384  # -128 * -128
    assert int(signed_exact[0, 255]) == -16256  # -128 * 127
    assert int(signed_exact[127, 127]) == 1  # -1 * -1
    assert int(signed_exact[128, 127]) == 0  # 0 * -1
    assert int(signed_exact[255, 255]) == 16129  # 127 * 127
    print("signed-wrapper smoke tests passed")


if __name__ == "__main__":
    main()
