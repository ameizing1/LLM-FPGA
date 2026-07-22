"""Behavior models for TCASI24 LUT-based approximate multipliers.

The 4x4 LSAM1 and CSAM2 models mirror the authors' open Verilog modules:
https://github.com/YnuGuoLab/DATE_FPGA_Approx_Mul

The int8 wrapper in this file is intentionally a first-week behavior model:
it applies the unsigned 8x8 approximate multiplier to absolute values and
then restores the signed result. It is not a replacement for signed RTL.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

INT8_VALUES = np.arange(-128, 128, dtype=np.int16)


def _bits(value: int, width: int) -> list[int]:
    return [(int(value) >> i) & 1 for i in range(width)]


def _pack(bits: list[int]) -> int:
    out = 0
    for i, bit in enumerate(bits):
        out |= (int(bit) & 1) << i
    return out


def _lut(init: int, inputs: list[int], *, o5: bool = False) -> int:
    """Evaluate a Xilinx LUT INIT value with I0 as the address LSB."""

    width = 5 if o5 else 6
    if len(inputs) != width:
        raise ValueError(f"expected {width} LUT inputs, got {len(inputs)}")
    addr = sum((int(bit) & 1) << i for i, bit in enumerate(inputs))
    return (int(init) >> addr) & 1


def _lut6(init: int, i0: int, i1: int, i2: int, i3: int, i4: int, i5: int) -> int:
    return _lut(init, [i0, i1, i2, i3, i4, i5])


def _lut5_from_lut6_2(init: int, i0: int, i1: int, i2: int, i3: int, i4: int) -> int:
    return _lut(init, [i0, i1, i2, i3, i4], o5=True)


def _carry4(di: list[int], s: list[int], *, ci: int = 0, cyinit: int = 0) -> tuple[list[int], list[int]]:
    """Evaluate the CARRY4 primitive for the subset used in the Verilog."""

    if len(di) != 4 or len(s) != 4:
        raise ValueError("CARRY4 expects four DI bits and four S bits")

    carry = (int(ci) | int(cyinit)) & 1
    co: list[int] = []
    out: list[int] = []
    for i in range(4):
        out.append((int(s[i]) ^ carry) & 1)
        carry = carry if int(s[i]) else int(di[i])
        co.append(carry & 1)
    return co, out


def mul4_lsam1(a: int, b: int) -> int:
    """Unsigned 4x4 LSAM1 approximate multiplication."""

    if not (0 <= int(a) < 16 and 0 <= int(b) < 16):
        raise ValueError("mul4_lsam1 expects unsigned 4-bit operands")

    a_bits = _bits(a, 4)
    b_bits = _bits(b, 4)
    prod = [0] * 8
    gen = [0] * 4
    prop = [0] * 4
    l1 = [0] * 6
    l2 = [0] * 6

    init = 0xDAF02A0078887888
    l1[2] = _lut6(init, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], 1)
    prod[1] = _lut5_from_lut6_2(init, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2])

    l1[3] = _lut6(0xFA5A70F00AAA8000, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], a_bits[3])
    l1[4] = _lut6(0x0A0A2AAAA0000000, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], a_bits[3])
    l1[5] = _lut6(0xA0A0800000000000, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], a_bits[3])

    l2[2] = _lut6(init, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], 1)
    l2[1] = _lut5_from_lut6_2(init, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2])

    l2[3] = _lut6(0xFA5A70F00AAA8000, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], a_bits[3])
    prop[3] = _lut6(0x0A0A2AAAA0000000, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], a_bits[3])
    gen[3] = _lut6(0xA0A0800000000000, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], a_bits[3])

    init9 = 0x5FA05FA088888888
    prod[2] = _lut6(init9, a_bits[0], b_bits[0], b_bits[2], l1[2], 1, 1)
    prod[0] = _lut5_from_lut6_2(init9, a_bits[0], b_bits[0], b_bits[2], l1[2], 1)

    init10 = 0x007F7F80FF808000
    prop[0] = _lut6(init10, l1[2], a_bits[0], b_bits[2], l1[3], l2[1], 1)
    gen[0] = _lut5_from_lut6_2(init10, l1[2], a_bits[0], b_bits[2], l1[3], l2[1])

    init_xor_and = 0x6666666688888880
    prop[1] = _lut6(init_xor_and, l1[4], l2[2], 1, 1, 1, 1)
    gen[1] = _lut5_from_lut6_2(init_xor_and, l1[4], l2[2], 1, 1, 1)
    prop[2] = _lut6(init_xor_and, l1[5], l2[3], 1, 1, 1, 1)
    gen[2] = _lut5_from_lut6_2(init_xor_and, l1[5], l2[3], 1, 1, 1)

    cout, summation = _carry4(gen, prop)
    prod[3:7] = summation
    prod[7] = cout[3]
    return _pack(prod)


def mul4_csam2(a: int, b: int) -> int:
    """Unsigned 4x4 CSAM2 approximate multiplication."""

    if not (0 <= int(a) < 16 and 0 <= int(b) < 16):
        raise ValueError("mul4_csam2 expects unsigned 4-bit operands")

    a_bits = _bits(a, 4)
    b_bits = _bits(b, 4)
    prod = [0] * 8
    l1 = [0] * 6
    l2 = [0] * 6

    init = 0xDAF02A0078887888
    l1[2] = _lut6(init, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], 1)
    prod[1] = _lut5_from_lut6_2(init, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2])

    l1[3] = _lut6(0xFA5A70F00AAA8000, b_bits[1], a_bits[0], b_bits[0], a_bits[1], a_bits[2], a_bits[3])

    init4 = 0xA00000000AAA8000
    l1[5] = _lut6(init4, b_bits[1], a_bits[1], b_bits[0], a_bits[2], a_bits[3], 1)
    l1[4] = _lut5_from_lut6_2(init4, b_bits[1], a_bits[1], b_bits[0], a_bits[2], a_bits[3])

    l2[2] = _lut6(init, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], 1)
    l2[1] = _lut5_from_lut6_2(init, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2])

    l2[3] = _lut6(0xFA5A70F00AAA8000, b_bits[3], a_bits[0], b_bits[2], a_bits[1], a_bits[2], a_bits[3])

    l2[4] = _lut5_from_lut6_2(init4, b_bits[3], a_bits[1], b_bits[2], a_bits[2], a_bits[3])
    prod[7] = _lut6(init4, b_bits[3], a_bits[1], b_bits[2], a_bits[2], a_bits[3], 1)

    init9 = 0xFFFFFF8078787878
    prod[3] = _lut6(init9, a_bits[0], b_bits[2], l1[2], l1[3], l2[1], 1)
    prod[2] = _lut5_from_lut6_2(init9, a_bits[0], b_bits[2], l1[2], l1[3], l2[1])

    init10 = 0x0FF00FF088888888
    prod[4] = _lut6(init10, a_bits[0], b_bits[0], l1[4], l2[2], 1, 1)
    prod[0] = _lut5_from_lut6_2(init10, a_bits[0], b_bits[0], l1[4], l2[2], 1)

    init11 = 0xFFFFEC80936C936C
    prod[6] = _lut6(init11, l1[4], l1[5], l2[2], l2[3], l2[4], 1)
    prod[5] = _lut5_from_lut6_2(init11, l1[4], l1[5], l2[2], l2[3], l2[4])

    return _pack(prod)


_MUL4: dict[str, Callable[[int, int], int]] = {
    "exact": lambda a, b: int(a) * int(b),
    "lsam1": mul4_lsam1,
    "csam2": mul4_csam2,
}


def mul8_unsigned(a: int, b: int, mode: str) -> int:
    """Unsigned 8x8 multiplication built from four 4x4 blocks."""

    if mode not in _MUL4:
        raise ValueError(f"unknown multiplier mode: {mode}")
    if not (0 <= int(a) < 256 and 0 <= int(b) < 256):
        raise ValueError("mul8_unsigned expects unsigned 8-bit operands")
    if mode == "exact":
        return int(a) * int(b)

    mul4 = _MUL4[mode]
    al, ah = int(a) & 0xF, (int(a) >> 4) & 0xF
    bl, bh = int(b) & 0xF, (int(b) >> 4) & 0xF
    ll = mul4(al, bl)
    lh = mul4(al, bh)
    hl = mul4(ah, bl)
    hh = mul4(ah, bh)
    return (ll + ((lh + hl) << 4) + (hh << 8)) & 0xFFFF


def mul_int8_signed(a: int, b: int, mode: str) -> int:
    """Signed int8 product using an unsigned approximate magnitude core."""

    a_i = int(np.int8(a))
    b_i = int(np.int8(b))
    if mode == "exact":
        return a_i * b_i

    sign = -1 if (a_i < 0) ^ (b_i < 0) else 1
    mag = mul8_unsigned(abs(a_i), abs(b_i), mode)
    return sign * mag


def build_int8_lut(mode: str) -> np.ndarray:
    """Build a 256x256 LUT indexed by operand + 128."""

    if mode not in _MUL4:
        raise ValueError(f"unknown multiplier mode: {mode}")

    lut = np.empty((256, 256), dtype=np.int16)
    for i, a in enumerate(INT8_VALUES):
        for j, b in enumerate(INT8_VALUES):
            lut[i, j] = mul_int8_signed(int(a), int(b), mode)
    return lut
