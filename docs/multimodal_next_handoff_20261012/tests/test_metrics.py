import math,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metrics_core import phase,number,summarize,joined_by_sensor_time

class MetricTests(unittest.TestCase):
    def test_phase_domain(self):
        for t in [0,.01,.1,float('nan'),float('inf')]:self.assertTrue(math.isnan(phase(t)))
        for t in [-2,-.05,.2,3,60]:self.assertTrue(math.isfinite(phase(t)))
    def test_exact_zero(self):self.assertEqual(summarize([(2.,2.),(-2.,-2.)])['strict_mean_MiD'],0)
    def test_missing_is_not_zero(self):self.assertTrue(math.isnan(number('')))
    def test_missing_retained(self):
        x=summarize([(2.,math.nan),(3.,3.)]);self.assertEqual(x['labelled'],2);self.assertIsNone(x['strict_mean_MiD']);self.assertEqual(x['invalid_MiD'],1)
    def test_unlabelled(self):self.assertEqual(summarize([(math.nan,2.)])['labelled'],0)
    def test_signed(self):
        x=summarize([(2.,-2.)]);self.assertEqual(x['wrong_sign_count'],1);self.assertIsNotNone(x['strict_mean_MiD'])
    def test_band_aggregation_not_created(self):self.assertNotIn('overall_MiD',summarize([(2.,2.)]))
    def test_reindex(self):
        a=[{'sequence_id':'s','anchor_us':'5','query_id':'s:0'}];b=[{'sequence_id':'s','anchor_us':'5','query_id':'s:15'}]
        self.assertEqual(len(joined_by_sensor_time(a,b)),1)
    def test_duplicate_time_rejected(self):
        a=[{'sequence_id':'s','anchor_us':'5'}]*2
        with self.assertRaises(ValueError):joined_by_sensor_time(a,a)
    def test_cap_count(self):self.assertEqual(summarize([(2.,-60.)])['capped_count'],1)
if __name__=='__main__':unittest.main()
