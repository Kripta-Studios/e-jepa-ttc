import numpy as np
from e_jepa_ttc.object_event_v4_13 import ObjectEventV413Config, conservative_dual_head_prediction, selective_fusion_gates

def test_low_confidence_is_soft_residual_without_forced_flip():
    base=np.array([0.1,-0.1]); p=np.array([0.2,0.8])
    out,soft,blend,override=conservative_dual_head_prediction(base,p)
    assert np.allclose(soft,[0.06,-0.06])
    assert np.allclose(blend,0.2)
    assert not override.any()
    assert out[0]>0 and out[1]<0

def test_extreme_negative_probability_can_flip_positive_baseline():
    out,_,blend,override=conservative_dual_head_prediction(np.array([0.1]),np.array([0.99]))
    assert override.item()
    assert blend.item()>0.5
    assert out.item()<0

def test_negative_baseline_is_never_override_candidate():
    out,_,_,override=conservative_dual_head_prediction(np.array([-0.1]),np.array([0.999]))
    assert not override.item()
    assert out.item()<0

def test_invalid_probability_is_rejected():
    try:
        conservative_dual_head_prediction(np.array([0.1]),np.array([1.1]))
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')

def test_gates_preserve_scientific_failures():
    base={'pearson':0.67,'expansion_mae':0.015,'positive_accuracy':0.89}
    routed={'pearson':0.68,'expansion_mae':0.0154,'balanced_sign_accuracy':0.77,'negative_accuracy':0.69,'minimum_sequence_negative_accuracy':0.4,'positive_accuracy':0.85}
    diag={'override_rate':0.05,'zero_event_pearson_drop':0.68,'shuffled_event_pearson_drop':0.70}
    th={'pearson_floor':0.66,'pearson_max_drop':0.005,'mae_tolerance':0.00075,'balanced_sign_gate':0.765,'negative_accuracy_gate':0.68,'minimum_sequence_negative_accuracy_gate':0.35,'positive_accuracy_max_drop':0.06,'maximum_override_rate':0.08,'zero_event_pearson_drop_gate':0.55,'shuffled_event_pearson_drop_gate':0.55}
    assert all(selective_fusion_gates(routed=routed,baseline=base,diagnostics=diag,thresholds=th).values())
    routed['minimum_sequence_negative_accuracy']=0.2
    assert not selective_fusion_gates(routed=routed,baseline=base,diagnostics=diag,thresholds=th)['minimum_sequence_negative_accuracy']
