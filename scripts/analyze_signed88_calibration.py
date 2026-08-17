#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'multiplier_models'))

from signed88.data import load_calibration_csv

DEFAULT_CALIBRATION_CSV = PROJECT_ROOT / 'tests' / 'data' / 'w8a8_calibration_hist_smoke_pcalib_nonzero.csv'

def main():
 p=argparse.ArgumentParser();p.add_argument('csv',nargs='?',default=str(DEFAULT_CALIBRATION_CSV));p.add_argument('--weight-column',default='auto');a=p.parse_args();prof=load_calibration_csv(Path(a.csv),a.weight_column)
 print(f'source={prof.source}');print(f'rows={prof.row_count} weight_column={prof.weight_column} raw_weight_sum={prof.raw_weight_sum:g}');print(f'low_state_coverage={np.count_nonzero(prof.state_probability)}/4096');print(f'nonzero_probability={prof.nonzero_probability:.12g}');print(f'zero_probability={prof.zero_probability:.12g}');
 order=np.argsort(prof.state_probability)[::-1][:20];print('top low6 states:')
 for idx in order: print(f'  AL={idx//64:2d} BL={idx%64:2d} p={prof.state_probability[idx]:.12g}')
 return 0
if __name__=='__main__':raise SystemExit(main())
