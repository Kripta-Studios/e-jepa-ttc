import copy
import unittest
import numpy as np
from reference.rgb_port_contracts import *

class Contracts(unittest.TestCase):
    def timeline(self):
        return [RGBFrame(str(i),10_000_000+i*100_000,10_000_000+i*100_000) for i in range(12)]
    def test_uint8(self):
        a=rgb_unit_interval(np.full((2,3,4,4),255,np.uint8));self.assertEqual(a.dtype,np.float32);self.assertTrue((a==1).all())
    def test_imagenet_is_rejected(self):
        with self.assertRaises(ValueError):rgb_unit_interval(np.full((3,4,4),-1,np.float32))
    def test_wrong_order_shape(self):
        with self.assertRaises(ValueError):rgb_unit_interval(np.zeros((4,4,3),np.uint8))
    def test_nonfinite(self):
        with self.assertRaises(ValueError):rgb_unit_interval(np.full((3,4,4),np.nan,np.float32))
    def test_uint16_not_guessed(self):
        with self.assertRaises(TypeError):rgb_unit_interval(np.ones((3,4,4),np.uint16))
    def test_stats_shape(self):
        a=rgb_statistics(np.zeros((2,8,3,4,4),np.uint8));self.assertEqual(a.shape,(2,8,2))
    def test_flat_gray(self):
        a=rgb_statistics(np.full((3,4,4),.5,np.float32));self.assertAlmostEqual(float(a[0]),.5);self.assertEqual(a[1],0)
    def test_gradient(self):
        a=np.zeros((3,4,4),np.float32);a[:,:,:2]=1;self.assertGreater(rgb_statistics(a)[1],0)
    def test_exposure_black(self):self.assertEqual(legacy_rgb_support(np.zeros((3,4,4),np.uint8)),0)
    def test_exposure_white(self):self.assertEqual(legacy_rgb_support(np.full((3,4,4),255,np.uint8)),0)
    def test_flat_colour_is_not_certified_texture(self):
        a=np.zeros((3,4,4),np.float32);a[0]=1;self.assertGreater(legacy_rgb_support(a),0);self.assertEqual(rgb_statistics(a)[1],0)
    def test_selection_no_future(self):
        a=select_frames(self.timeline(),cutoff_us=10_750_000);self.assertTrue(all(f.timestamp_us<=10_750_000 for f in a))
    def test_availability(self):
        a=[RGBFrame('x',100,150),RGBFrame('y',80,90)];self.assertEqual([f.frame_id for f in select_frames(a,cutoff_us=120,lookback_us=100)],['y'])
    def test_no_duplicate_ids(self):
        with self.assertRaises(ValueError):select_frames([RGBFrame('x',1,1),RGBFrame('x',2,2)],cutoff_us=3)
    def test_duplicate_timestamp(self):
        with self.assertRaises(ValueError):select_frames([RGBFrame('x',1,1),RGBFrame('y',1,1)],cutoff_us=3)
    def test_10hz_no_fake_20hz(self):
        c=triplet_context(self.timeline(),cutoff_us=11_000_000);self.assertEqual(len(c),5)
    def test_triplets_are_real(self):
        c=triplet_context(self.timeline(),cutoff_us=11_000_000);self.assertTrue(all(len({f.frame_id for f in v})==3 for v in c))
    def test_empty_context_not_faked(self):self.assertEqual(triplet_context([RGBFrame('x',1,1)],cutoff_us=2),[])
    def test_int_precision(self):
        a=deltas_seconds([10**16,10**16+100_003,10**16+200_006]);np.testing.assert_allclose(a,[.100003,.100003],rtol=1e-6)
    def test_repeated_time(self):
        with self.assertRaises(ValueError):deltas_seconds([1,1])
    def test_reversal_time(self):
        with self.assertRaises(ValueError):deltas_seconds([3,2])
    def test_positive_config(self):
        c={'modality':'event','in_channels':12,'transport_mode':'adaptive_pyramid','hidden_dim':64};old=copy.deepcopy(c);new=rgb_config(c);self.assertEqual(c,old);self.assertEqual(new['in_channels'],3);self.assertEqual(new['transport_mode'],'adaptive_pyramid')
    def test_bin_gate_not_silently_reused(self):
        with self.assertRaises(ValueError):rgb_config({'temporal_channel_gate_enabled':True})
    def test_feature_schema(self):
        a=phase17(np.zeros((2,2)),np.zeros((2,3)),np.zeros((2,3)),np.array([[1,2,4],[1,3,2]]));self.assertEqual(a.shape,(2,17));np.testing.assert_array_equal(a[0,-6:],[3,2,1,3,2,1])
    def test_shape_is_not_semantics(self):self.assertTrue(all(not name.startswith('event_') for name in RGB_PHASE17_NAMES));self.assertEqual(len(RGB_SCHEMA_HASH),64)
    def test_feature_nonfinite(self):
        with self.assertRaises(ValueError):phase17(np.zeros(2),np.zeros(3),np.zeros(3),np.array([1,np.nan,2]))
    def test_roles_valid(self):validate_group_roles({'a'},{'b'},{'c'})
    def test_roles_overlap(self):
        with self.assertRaises(ValueError):validate_group_roles({'a'},{'b'},{'a'})
    def test_fulltrain_weights_rejected(self):
        with self.assertRaises(ValueError):assert_checkpoint_excludes({'a','b','c'},{'b'})
    def test_height_anchor(self):
        self.assertAlmostEqual(current_ttc_from_heights(50,55,.1),1.0);self.assertAlmostEqual(previous_ttc_from_heights(50,55,.1),1.1)
    def test_common_resize_invariance(self):
        self.assertAlmostEqual(current_ttc_from_heights(50,55,.1),current_ttc_from_heights(100,110,.1))
    def test_signed_phase(self):
        x=phase_from_ttc(np.array([-2.,2.]));self.assertLess(x[0],0);self.assertGreater(x[1],0)
    def test_bad_phase_domain(self):
        with self.assertRaises(ValueError):phase_from_ttc(np.array([.05]))

if __name__=='__main__':unittest.main()
