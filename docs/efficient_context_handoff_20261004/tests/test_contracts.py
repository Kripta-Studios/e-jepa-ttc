import unittest,tempfile,math
from pathlib import Path
from reference.context_plan import *
from reference.journal import atomic_json,completed_fragment

class Contracts(unittest.TestCase):
    def key(self,**kw):
        d=dict(source_digest='raw-sha',sequence='seq',start_us=100,end_us=200,roi_xyxy=(0.,0.,20.,20.),roi_size=128,event_pixel_diff=5.,preprocessing_digest='prep-sha')
        d.update(kw);return WindowKey(**d)
    def test_unique(self):
        a,b=self.key(),self.key(start_us=101)
        u,i=deduplicate([a,b,a]);self.assertEqual((u,i),([a,b],[0,1,0]))
    def test_roi_is_identity(self):self.assertNotEqual(self.key(),self.key(roi_xyxy=(1.,0.,21.,20.)))
    def test_time_exact(self):self.assertNotEqual(self.key(),self.key(end_us=201))
    def test_source_identity(self):self.assertNotEqual(self.key(),self.key(source_digest='other'))
    def test_prep_identity(self):self.assertNotEqual(self.key(),self.key(preprocessing_digest='other'))
    def test_offset_identity(self):self.assertNotEqual(self.key(),self.key(event_pixel_diff=0.))
    def test_nonfinite(self):
        with self.assertRaises(ValueError):self.key(event_pixel_diff=float('nan'))
    def test_invalid_interval(self):
        with self.assertRaises(ValueError):self.key(start_us=200)
    def test_encoder_calls(self):
        calls=[]
        def enc(k):calls.append(k);return (k.start_us,k.end_us)
        a,b=self.key(),self.key(start_us=101)
        self.assertEqual(evaluate_unique([a,b,a],enc),[(100,200),(101,200),(100,200)])
        self.assertEqual(len(calls),2)
    def context(self,valid=None,**kw):
        a=[1_000_000+i*50_000 for i in range(16)]
        d=dict(anchors_us=a,available_us=[1_800_000]*16,valid=valid or [True]*16,current_anchor_us=a[-1],current_available_us=1_800_000)
        d.update(kw);return selected_context(**d)
    def test_span(self):
        s,m,t=self.context();self.assertEqual(len(s),8);self.assertEqual(t[0][0],.75);self.assertEqual(t[-1][0],0.)
    def test_gap_recomputed(self):
        _,_,t=self.context();self.assertEqual(t[1][2],.1);self.assertEqual(t[4][2],.15)
    def test_cold_start(self):
        _,m,t=self.context([False]*15+[True]);self.assertEqual(sum(m),1);self.assertEqual(t[-1][2],0.)
    def test_partial(self):
        _,m,t=self.context([False]*5+[True]*11);self.assertEqual(m[:3],(False,False,False));self.assertEqual(t[3][2],0.)
    def test_future(self):
        with self.assertRaises(ValueError):self.context(current_available_us=1_000_000)
    def test_holes(self):
        with self.assertRaises(ValueError):self.context([True]+[False]*14+[True])
    def test_current_mandatory(self):
        with self.assertRaises(ValueError):self.context([True]*15+[False])
    def test_timestamp_int_precision(self):
        off=2**54
        a=[off+i*50_000 for i in range(16)]
        _,_,t=self.context(anchors_us=a,available_us=[off+800_000]*16,current_anchor_us=a[-1],current_available_us=off+800_000)
        self.assertEqual(t[0][0],.75)
    def test_phase_anchor_identity(self):
        ratio=1.1;dt=.1;current=current_ttc_from_ratio(ratio,dt);previous=current+dt
        self.assertAlmostEqual(benchmark_phase(previous,dt),math.log(ratio),places=13)
        self.assertNotAlmostEqual(benchmark_phase(current,dt),math.log(ratio),places=8)
    def test_contact_task_differs(self):
        self.assertGreater(abs(benchmark_phase(2.)-benchmark_phase(1.8))*10000,58.)
    def test_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'one.json';atomic_json(p,dict(binding='v1',status='COMPLETE',draws=[1,2,3]));self.assertEqual(completed_fragment(Path(d),'one','v1')['draws'],[1,2,3])
    def test_bad_binding(self):
        with tempfile.TemporaryDirectory() as d:
            atomic_json(Path(d)/'one.json',dict(binding='v1',status='COMPLETE'))
            with self.assertRaises(ValueError):completed_fragment(Path(d),'one','v2')
    def test_bad_id(self):
        with self.assertRaises(ValueError):completed_fragment(Path('.'),'../a','v1')
    def test_pending(self):
        with tempfile.TemporaryDirectory() as d:self.assertIsNone(completed_fragment(Path(d),'one','v1'))

if __name__=='__main__':unittest.main()
