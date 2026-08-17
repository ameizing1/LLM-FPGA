import sys,tempfile,unittest
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'multiplier_models'))

from signed88.hardware import get_design
from signed88.common import read_json
RTL_TEMPLATE_ROOT=PROJECT_ROOT/'FPGA_multiplier/signed8x8_6x2'
class RtlExportTest(unittest.TestCase):
 def test_export_all(self):
  for name in ['balanced','quality']:
   with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
    d=get_design(name);out=d.export_rtl(RTL_TEMPLATE_ROOT,Path(td)/name,d.spec.base_inits);obj=read_json(out/'trained_artifact.json');self.assertEqual(obj['design'],name);self.assertEqual(d.normalize_inits(obj['inits']),d.normalize_inits(d.spec.base_inits))
if __name__=='__main__':unittest.main()
