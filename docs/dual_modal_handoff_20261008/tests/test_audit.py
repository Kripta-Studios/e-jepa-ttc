from pathlib import Path
import sys, csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_saved_predictions import metric,audit,MODELS

def test_nonfinite_prediction_is_not_removed():
    r=metric(np.array([1.,np.nan]),np.array([1.,2.]))
    assert r['coverage']==.5 and r['mae'] is None and r['n']==2

def test_common_cap_is_secondary_and_no_target_dropping(tmp_path):
    path=tmp_path/'scores.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['query_id','sequence_id','truth_ttc_seconds',*MODELS]);w.writeheader()
        for i,(truth,pred) in enumerate([('2','200'),('','3'),('3','2')]):
            w.writerow({'query_id':str(i),'sequence_id':'s','truth_ttc_seconds':truth,**{m:pred for m in MODELS}})
    r=audit(path,tmp_path/'new')
    assert r['queries']==3 and r['labelled']==2
    for m in MODELS:
        assert r['models'][m]['native']['n']==2 and r['models'][m]['common_signed_cap60']['n']==2
        assert r['models'][m]['native']['mae']>r['models'][m]['common_signed_cap60']['mae']

def test_invalid_phase_never_silently_clamped_for_metric():
    r=metric(np.array([.05,2]),np.array([1.,2.]))
    assert r['invalid_phase_predictions']==1 and r['unweighted_phase_absolute_error_x1e4'] is None
