from __future__ import annotations
import copy
import io
import numpy as np
import pytest
import torch
from torch import nn
from dual_ttc.phase import phase_to_ttc, ttc_to_phase, emitted_phase, validate_initial_prior, weighted_loss, per_row_loss
from dual_ttc.contracts import CheckpointOrigin, require_heldout, split_groups, seconds_since, nearest_rgb, validate_modality_input
from dual_ttc.correlation import local_correlation
from dual_ttc.model import EventModel, RGBExtension, HeadOnlyRGBBridge, FusionHead, PhaseHead

class TinyEncoder(nn.Module):
    def __init__(self,c:int,dim:int=16):
        super().__init__();self.conv=nn.Conv2d(c,dim,3,stride=2,padding=1);self.calls=0
    def forward(self,x):
        self.calls+=1
        z=torch.tanh(self.conv(x));return z,z.mean((-2,-1))

def event_model():
    return EventModel(.03,encoder=TinyEncoder(20),dim=16,hidden=32,radius=1)

def inputs(b=2,t=2):
    x=torch.randn(b,t,20,8,8)
    times=torch.zeros(b,t-1,4);times[:,:,0:3]=.1
    return x,times

@pytest.mark.parametrize('value',[-60.,-10.,-1.,-.2,.2,1.,3.,60.])
def test_phase_roundtrip(value):
    x=torch.tensor([value],dtype=torch.float64)
    torch.testing.assert_close(phase_to_ttc(ttc_to_phase(x)),x,rtol=1e-12,atol=1e-12)

def test_zero_is_positive60(): assert phase_to_ttc(torch.tensor([0.])).item()==pytest.approx(60.)
@pytest.mark.parametrize('value',[0.,.01,.1,float('nan'),float('inf')])
def test_bad_ttc(value):
    with pytest.raises(ValueError):ttc_to_phase(torch.tensor([value]))
@pytest.mark.parametrize('prior',[0.,.0001,-.0001,9.,float('nan')])
def test_bad_prior(prior):
    with pytest.raises(ValueError):validate_initial_prior(prior)

def test_loss_uses_emitted_phase():
    h=PhaseHead(8,.03);out=h(torch.ones(2,8));target=torch.tensor([.06,.05]);mass=torch.tensor([.25,.25])
    assert torch.isfinite(weighted_loss(out,target,mass,4))
    torch.testing.assert_close(out['point_phase'],ttc_to_phase(out['ttc']))

def test_empty_batch_mass():
    h=PhaseHead(8,.03);out=h(torch.ones(2,8))
    with pytest.raises(ValueError):weighted_loss(out,torch.ones(2),torch.ones(3),4)

def test_corr_identity_and_borders():
    a=torch.eye(25).T.reshape(1,25,5,5)
    p,m=local_correlation(a,a,radius=1)
    assert p.shape==(1,9,5,5)
    assert bool((p[~m]==0).all());assert bool((p[:,4]>.999).all())
    torch.testing.assert_close(p.sum(1),torch.ones(1,5,5))

def test_corr_known_right_shift():
    a=torch.eye(25).T.reshape(1,25,5,5);b=torch.zeros_like(a);b[:,:,:,1:]=a[:,:,:,:-1]
    p,_=local_correlation(a,b,radius=1)
    assert bool((p.argmax(1)[:,:,:-1]==5).all())

def test_corr_gradients():
    a=torch.randn(1,8,5,5,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    p,_=local_correlation(a,b,1);p[:,4].square().sum().backward()
    assert a.grad is not None and a.grad.abs().sum()>0
    assert b.grad is not None and b.grad.abs().sum()>0

def test_bad_corr():
    with pytest.raises(ValueError):local_correlation(torch.zeros(1,2,2,2),torch.zeros(1,2,3,3))

def test_context4_and_coverage():
    model=event_model();x,t=inputs(t=4);out=model(x,t)
    assert out['ttc'].shape==(2,) and torch.isfinite(out['ttc']).all()
    assert out['has_pair'].all()

def test_padding_nan_not_used():
    model=event_model().eval();x,t=inputs(t=4);valid=torch.tensor([[False,False,True,True]]*2)
    original=model(x,t,valid)
    x[:,:2]=float('nan');same=model(x,t,valid)
    for k in ['ttc','point_phase','hidden']:torch.testing.assert_close(original[k],same[k],atol=0,rtol=0)

def test_current_only_cold_start_is_retained():
    model=event_model();x,t=inputs();valid=torch.tensor([[False,True]]*2)
    out=model(x,t,valid);assert not out['has_pair'].any();assert torch.isfinite(out['ttc']).all()

def test_bad_time():
    model=event_model();x,t=inputs();t[:,:,0]=0
    with pytest.raises(ValueError):model(x,t)

def test_no_rgb_argument_on_event_interface():
    model=event_model();x,t=inputs()
    with pytest.raises(TypeError):model(x,t,rgb=torch.zeros(1))

def test_encoder_receives_training_gradient():
    torch.manual_seed(7);model=event_model();optim=torch.optim.AdamW(model.parameters(),lr=.003)
    before=model.encoder.conv.weight.detach().clone();x,t=inputs()
    for _ in range(3):
        optim.zero_grad();out=model(x,t);per_row_loss(out,torch.tensor([.05,.07])).mean().backward();optim.step()
    assert not torch.equal(before,model.encoder.conv.weight)

def test_missing_rgb_skips_rgb_encoder_exact_fallback():
    model=RGBExtension(event_model(),.03,rgb_encoder=TinyEncoder(3),dim=16,hidden=32,radius=1).eval()
    x,t=inputs();e=model.event(x,t);out=model(x,t,None,torch.zeros(2,4),torch.zeros(2,dtype=torch.bool),torch.zeros(2,4))
    assert model.rgb_encoder.calls==0
    for k in ['ttc','raw_location','point_phase','q10','q90']:torch.testing.assert_close(e[k],out[k],atol=0,rtol=0)

def test_mixed_rgb_nan_invalid_ignored():
    model=RGBExtension(event_model(),.03,rgb_encoder=TinyEncoder(3),dim=16,hidden=32,radius=1).eval()
    x,t=inputs();rgb=torch.randn(2,2,3,8,8);rgb[1]=float('nan');mask=torch.tensor([True,False])
    out=model(x,t,rgb,torch.ones(2,4)*.1,mask,torch.zeros(2,4))
    assert torch.isfinite(out['ttc']).all();assert model.rgb_encoder.calls==1

def test_rgb_learns_and_event_remains_frozen():
    torch.manual_seed(8)
    model=RGBExtension(event_model(),.03,rgb_encoder=TinyEncoder(3),dim=16,hidden=32,radius=1).train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003)
    ev_before=copy.deepcopy(model.event.state_dict());r_before=model.rgb_encoder.conv.weight.detach().clone()
    x,t=inputs();rgb=torch.randn(2,2,3,8,8);rt=torch.ones(2,4)*.1;mask=torch.ones(2,dtype=torch.bool)
    for _ in range(3):
        opt.zero_grad();o=model(x,t,rgb,rt,mask,rt)
        loss=per_row_loss(o,torch.tensor([.05,.08])).mean()+.25*per_row_loss(o['rgb_aux'],torch.tensor([.05,.08])).mean()
        loss.backward();opt.step()
    assert not torch.equal(r_before,model.rgb_encoder.conv.weight)
    assert all(torch.equal(ev_before[k],v) for k,v in model.event.state_dict().items())
    assert all(p.grad is None for p in model.event.parameters())

def test_fusion_can_leave_expert_value():
    h=PhaseHead(32,.03);e=h(torch.randn(2,32));f=FusionHead(32)
    with torch.no_grad():f.residual.bias.fill_(2.)
    o=f(e,torch.ones(2,32),torch.zeros(2,4),torch.ones(2,dtype=torch.bool))
    assert bool((o['point_phase']>e['point_phase']).all())

def test_zero_initial_fusion_does_not_double_zero_gradient():
    e=PhaseHead(32,.03)(torch.randn(2,32));f=FusionHead(32)
    o=f(e,torch.ones(2,32),torch.zeros(2,4),torch.ones(2,dtype=torch.bool))
    (o['point_phase']-.07).square().sum().backward()
    assert f.residual.bias.grad.abs().sum()>0

def test_bridge_zero_missing_rgb():
    bridge=HeadOnlyRGBBridge(8,32);e=PhaseHead(32,.03)(torch.ones(2,32))
    o=bridge(e,torch.randn(2,2,8),torch.zeros(2,4),torch.zeros(2,dtype=torch.bool))
    torch.testing.assert_close(o['ttc'],e['ttc'],atol=0,rtol=0)

def test_ancestry_forbids_train40_in_new_split():
    origin=CheckpointOrigin('TRAIN40',frozenset(['a','b']),'audited_task_training')
    with pytest.raises(ValueError):require_heldout(origin,{'b'})

def test_unknown_genealogy_not_clean():
    with pytest.raises(ValueError):require_heldout(CheckpointOrigin('public',frozenset(),'unknown'),{'dev'})

def test_generic_allowed(): require_heldout(CheckpointOrigin('imagenet',frozenset(),'generic_pretraining'),{'dev'})

def test_group_split_deterministic():
    g=[f'g{i}' for i in range(40)];a,b=split_groups(g);c,d=split_groups(list(reversed(g)))
    assert a==c and b==d and len(b)==8 and not set(a)&set(b)

def test_timestamp_subtraction_int_before_float():
    assert seconds_since(1_700_000_000_000_001,1_700_000_000_000_000)==1e-6

def test_nearest_tie_earlier(): assert nearest_rgb(np.array([900,1100],dtype=np.int64),1000,100,1200)==0

def test_nearest_no_hidden_future(): assert nearest_rgb(np.array([980,1001],dtype=np.int64),1000,100,1000) is None

def test_nearest_explicit_wait_allowed(): assert nearest_rgb(np.array([980,1001],dtype=np.int64),1000,100,1001)==1

def test_event_mode_does_not_accept_rgb():
    with pytest.raises(ValueError):validate_modality_input('event',object())

def test_resume_reference_training():
    def run(m,opt,n):
        for _ in range(n):
            x,t=inputs();y=torch.rand(2)*.04+.03
            opt.zero_grad();per_row_loss(m(x,t),y).mean().backward();opt.step()
    torch.manual_seed(123);full=event_model();opt=torch.optim.AdamW(full.parameters(),lr=.001)
    initial=copy.deepcopy(full.state_dict());rng=torch.get_rng_state();run(full,opt,4)
    partial=event_model();partial.load_state_dict(initial);op=torch.optim.AdamW(partial.parameters(),lr=.001)
    torch.set_rng_state(rng);run(partial,op,2)
    buf=io.BytesIO();torch.save({'model':partial.state_dict(),'optim':op.state_dict(),'rng':torch.get_rng_state()},buf);buf.seek(0)
    saved=torch.load(buf,weights_only=True);restored=event_model();restored.load_state_dict(saved['model']);op2=torch.optim.AdamW(restored.parameters(),lr=.001);op2.load_state_dict(saved['optim']);torch.set_rng_state(saved['rng']);run(restored,op2,2)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in full.state_dict().items())

def test_resnet50_full_resolution_forward_bn_policy():
    from dual_ttc.encoder import DenseResNet50
    torch.manual_seed(1);m=DenseResNet50(20).train()
    assert not m.pretraining_loaded
    assert all(not b.training for b in m.modules() if isinstance(b,nn.BatchNorm2d))
    with torch.no_grad():f,g=m(torch.randn(1,20,128,128))
    assert f.shape==(1,128,32,32) and g.shape==(1,128)
    assert torch.isfinite(f).all() and torch.isfinite(g).all()
