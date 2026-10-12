"""Synthetic operator checks: no optimizer updates, sensors or TTC claims."""
from dataclasses import asdict
import math
import pytest
import torch
from reference.operators import (
    ResidualSpec, current_log_ratio, equal_slope_residual,
    importance_estimate, ordered_residual, require_same_operator,
)


def test_exact_importance_expectation():
    w=torch.tensor([.1,.3,.6],dtype=torch.float64)
    p=torch.tensor([.2,.5,.3],dtype=torch.float64)
    l=torch.tensor([1.,4.,9.],dtype=torch.float64,requires_grad=True)
    estimates=torch.stack([importance_estimate(l[i:i+1],w[i:i+1],p[i:i+1]) for i in range(3)])
    actual=(p*estimates).sum()
    torch.testing.assert_close(actual,(w*l).sum(),rtol=1e-14,atol=1e-14)
    actual.backward();torch.testing.assert_close(l.grad,w)


def test_uniform_is_native_Nw():
    losses=torch.tensor([2.,5.,3.],dtype=torch.float64)
    w=torch.tensor([.01,.1,.02],dtype=torch.float64)
    p=torch.full_like(w,.01)
    torch.testing.assert_close(importance_estimate(losses,w,p),(100*w*losses).mean())


def test_p_equals_w_has_no_second_weight():
    l=torch.tensor([2.,5.,3.]);w=torch.tensor([.01,.1,.02])
    torch.testing.assert_close(importance_estimate(l,w,w),l.mean(),rtol=0,atol=0)


def test_double_weight_would_change_objective():
    w=torch.tensor([.1,.9],dtype=torch.float64);l=torch.tensor([8.,1.],dtype=torch.float64)
    correct=(w*l).sum();wrong=(w*2*w*l).sum()
    assert not torch.isclose(correct,wrong)


def test_sampled_masses_need_not_sum_to_one():
    w=torch.tensor([.8,.8,.8]); l=torch.tensor([2.,2.,2.])
    assert importance_estimate(l,w,w).item()==2.


@pytest.mark.parametrize("case",["empty","shape","nonfinite","negative","zero_p","over_one","learnable","integer"])
def test_bad_estimator_inputs(case):
    l=torch.tensor([1.,2.]);w=torch.tensor([.2,.8]);p=w.clone()
    if case=='empty':l=w=p=torch.tensor([])
    elif case=='shape':w=w[:1]
    elif case=='nonfinite':l[0]=float('nan')
    elif case=='negative':w[0]=-.1
    elif case=='zero_p':p[0]=0
    elif case=='over_one':p[0]=2
    elif case=='learnable':w.requires_grad_(True)
    elif case=='integer':l=l.to(torch.int64)
    with pytest.raises((ValueError,TypeError)):
        importance_estimate(l,w,p)


def test_sampling_resume_same_call_boundaries():
    p=torch.tensor([.01,.09,.2,.7],dtype=torch.float64)
    gen=torch.Generator().manual_seed(7)
    torch.multinomial(p,128,True,generator=gen)
    state=gen.get_state().clone()
    expected=torch.multinomial(p,128,True,generator=gen)
    restored=torch.Generator();restored.set_state(state)
    actual=torch.multinomial(p,128,True,generator=restored)
    assert torch.equal(actual,expected)


def test_C05_equals_native():
    s=torch.linspace(-8,8,97,dtype=torch.float64)
    assert torch.equal(equal_slope_residual(s,ResidualSpec(.05)),.05*torch.tanh(s))


@pytest.mark.parametrize('bound',[.05,.20])
def test_odd_and_bounded(bound):
    s=torch.linspace(-100,100,101,dtype=torch.float64)
    r=equal_slope_residual(s,ResidualSpec(bound))
    assert torch.equal(equal_slope_residual(-s,ResidualSpec(bound)),-r)
    assert float(r.abs().max())<=bound


@pytest.mark.parametrize('bound',[.05,.20])
def test_same_initial_slope(bound):
    s=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    r=equal_slope_residual(s,ResidualSpec(bound));r.backward()
    assert r.item()==0.
    assert s.grad.item()==.05


def test_endpoint_swap_antisymmetry():
    f=torch.tensor([1.,2.,3.]);r=torch.tensor([3.,-1.,2.]);spec=ResidualSpec(.20)
    assert torch.equal(ordered_residual(f,r,spec),-ordered_residual(r,f,spec))


def test_no_new_parameters_in_operator():
    # Operator definition has only immutable scalars; parameters remain the scorer's.
    assert asdict(ResidualSpec(.20))=={'bound':.20,'slope':.05,'operator':'equal_slope_tanh_v1'}


def test_operator_identity_rejected():
    with pytest.raises(ValueError):require_same_operator(ResidualSpec(.05),ResidualSpec(.20))
    require_same_operator(ResidualSpec(.05),ResidualSpec(.05))


@pytest.mark.parametrize('kwargs',[{'bound':.1},{'bound':.2,'slope':.2},{'bound':.2,'operator':'native'}])
def test_unregistered_spec(kwargs):
    with pytest.raises(ValueError):ResidualSpec(**kwargs)


def test_wide_has_more_capacity_not_same_native_function():
    s=torch.tensor(5.,dtype=torch.float64)
    narrow=equal_slope_residual(s,ResidualSpec(.05))
    wide=equal_slope_residual(s,ResidualSpec(.20))
    assert wide>narrow
    assert not torch.isclose(wide,.2*torch.tanh(s))


def test_fixed_current_pair_example():
    # Illustrative constant closing velocity, not a dataset or trained result.
    required=current_log_ratio(.8,.1)-.006
    assert .05<required<.2
    score=(.2/.05)*math.atanh(required/.2)
    actual=equal_slope_residual(torch.tensor(score,dtype=torch.float64),ResidualSpec(.2))
    assert abs(actual.item()-required)<1e-14


def test_benchmark_phase_is_not_current_ratio():
    assert not math.isclose(current_log_ratio(.8,.1),-math.log1p(-.1/.8))
