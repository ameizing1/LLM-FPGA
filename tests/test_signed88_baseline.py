import sys
import unittest
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'multiplier_models'))

from signed88.common import ObjectiveWeights
from signed88.data import load_calibration_csv
from signed88.hardware import get_design
from signed88.metrics import evaluate_design

CALIBRATION_CSV=PROJECT_ROOT/'tests/data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv'
EXPECTED={
 'balanced':(5.625,80,0.0),
 'quality':(1.125,16,0.0),
}
class BaselineMetricsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.profile=load_calibration_csv(CALIBRATION_CSV)
 def test_readme_baselines(self):
  for name,(mae,wce,bias) in EXPECTED.items():
   with self.subTest(name=name):
    d=get_design(name);m=evaluate_design(d,d.spec.base_inits,self.profile,ObjectiveWeights())
    self.assertAlmostEqual(m.MED,mae,places=10);self.assertEqual(m.WCE,wce);self.assertAlmostEqual(m.bias,bias,places=10)
if __name__=='__main__':unittest.main()
