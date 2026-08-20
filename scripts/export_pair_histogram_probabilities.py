"""Export a pair histogram as concrete P_calib values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histogram-npy", required=True, help="input 256x256 pair-count histogram")
    parser.add_argument("--probability-npy", required=True, help="output 256x256 probability matrix")
    parser.add_argument("--nonzero-csv", required=True, help="output CSV with nonzero (a,b,count,p_calib) rows")
    parser.add_argument(
        "--index-mode",
        choices=["signed_int8", "raw_uint8"],
        default="signed_int8",
        help="how to display row/column indices in the CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hist_path = Path(args.histogram_npy)
    prob_path = Path(args.probability_npy)
    csv_path = Path(args.nonzero_csv)

    hist = np.load(hist_path).astype(np.int64, copy=False)
    if hist.shape != (256, 256):
        raise ValueError(f"histogram must have shape (256, 256), got {hist.shape}")

    total = int(hist.sum())
    if total <= 0:
        raise ValueError("histogram is empty")

    prob = hist.astype(np.float64) / float(total)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, prob)

    nonzero = np.argwhere(hist > 0)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "count", "p_calib"])
        for i, j in nonzero:
            if args.index_mode == "signed_int8":
                a = int(i - 128)
                b = int(j - 128)
            else:
                a = int(i)
                b = int(j)
            writer.writerow([a, b, int(hist[i, j]), f"{prob[i, j]:.18e}"])

    print(f"wrote {prob_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
