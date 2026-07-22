"""Generate exact, LSAM1, and CSAM2 signed int8 product LUTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_lut_tcasi24.tcasi24 import build_int8_lut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/luts", help="directory for .npy LUT files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for mode, filename in {
        "exact": "exact_int8_lut.npy",
        "lsam1": "lsam1_int8_lut.npy",
        "csam2": "csam2_int8_lut.npy",
    }.items():
        lut = build_int8_lut(mode)
        path = out_dir / filename
        np.save(path, lut)
        written[mode] = str(path)
        print(f"{mode:6s} {path} shape={lut.shape} dtype={lut.dtype}")

    metadata = {
        "indexing": "lut[a + 128, b + 128] for signed int8 operands a and b",
        "signed_wrapper": "sign-magnitude wrapper around unsigned TCASI24 8x8 behavior model",
        "source": "https://github.com/YnuGuoLab/DATE_FPGA_Approx_Mul",
        "files": written,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
