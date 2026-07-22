"""Signed int8 wrappers around unsigned 8x8 product LUTs."""

from __future__ import annotations

import numpy as np

INT8_VALUES = np.arange(-128, 128, dtype=np.int16)
UINT8_VALUES = np.arange(256, dtype=np.uint16)


def exact_unsigned8_lut() -> np.ndarray:
    """Return the exact unsigned 8x8 product LUT indexed by raw uint8 values."""

    return (UINT8_VALUES[:, None].astype(np.uint32) * UINT8_VALUES[None, :].astype(np.uint32)).astype(np.uint32)


def exact_int8_lut() -> np.ndarray:
    """Return the exact signed int8 product LUT indexed by operand + 128."""

    return (INT8_VALUES[:, None].astype(np.int32) * INT8_VALUES[None, :].astype(np.int32)).astype(np.int32)


def build_signed_wrapper_lut(unsigned_lut: np.ndarray) -> np.ndarray:
    """Wrap an unsigned 8x8 product LUT for signed int8 operands.

    The model is:

    - take two's-complement int8 operands a and b;
    - multiply abs(a) and abs(b) with the unsigned core;
    - restore the sign with sign(a) xor sign(b).

    This is a behavior-model baseline, not a claim about the final signed RTL
    structure or area.
    """

    lut = np.asarray(unsigned_lut)
    if lut.shape != (256, 256):
        raise ValueError(f"unsigned_lut must have shape (256, 256), got {lut.shape}")

    mags = np.abs(INT8_VALUES.astype(np.int32))
    signs = np.where(INT8_VALUES < 0, -1, 1).astype(np.int32)
    mag_products = lut[np.ix_(mags, mags)].astype(np.int32)
    return mag_products * signs[:, None] * signs[None, :]


def signed_wrapper_product(a: int, b: int, unsigned_lut: np.ndarray) -> int:
    """Return one signed-wrapper product for int8 operands a and b."""

    a_i = int(np.asarray(a, dtype=np.int8))
    b_i = int(np.asarray(b, dtype=np.int8))
    mag = int(np.asarray(unsigned_lut)[abs(a_i), abs(b_i)])
    return -mag if (a_i < 0) ^ (b_i < 0) else mag

