"""GEMM helpers for exact and LUT-backed int8 product accumulation."""

from __future__ import annotations

import numpy as np


def _as_int8_matrix(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    return arr


def exact_gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference int8 GEMM with int32 accumulation."""

    a8 = _as_int8_matrix("a", a)
    b8 = _as_int8_matrix("b", b)
    if a8.shape[1] != b8.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {a8.shape} and {b8.shape}")
    return a8.astype(np.int32) @ b8.astype(np.int32)


def lut_gemm(a: np.ndarray, b: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """GEMM where every scalar product is read from a 256x256 int8 LUT."""

    a8 = _as_int8_matrix("a", a)
    b8 = _as_int8_matrix("b", b)
    product_lut = np.asarray(lut)
    if product_lut.shape != (256, 256):
        raise ValueError(f"lut must have shape (256, 256), got {product_lut.shape}")
    if a8.shape[1] != b8.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {a8.shape} and {b8.shape}")

    acc = np.zeros((a8.shape[0], b8.shape[1]), dtype=np.int32)
    for k in range(a8.shape[1]):
        a_idx = a8[:, k].astype(np.int16) + 128
        b_idx = b8[k, :].astype(np.int16) + 128
        acc += product_lut[np.ix_(a_idx, b_idx)].astype(np.int32)
    return acc
