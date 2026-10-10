"""Executable reference model. Production data/training integration is deliberately separate."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .correlation import local_correlation
from .encoder import DenseResNet50
from .phase import MIN_PHASE, MAX_PHASE, phase_to_ttc, emitted_phase, validate_initial_prior

class Residual(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.f=nn.Sequential(nn.GroupNorm(8,dim),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1),nn.GroupNorm(8,dim),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1))
    def forward(self,x:Tensor)->Tensor: return x+self.f(x)

class PairEvidence(nn.Module):
    def __init__(self,dim:int=128,hidden:int=160,radius:int=4):
        super().__init__()
        self.radius=radius;k=(2*radius+1)**2
        self.spatial=nn.Sequential(nn.Conv2d(3*dim+4*k,hidden,1),Residual(hidden),Residual(hidden))
        self.project=nn.Sequential(nn.Linear(2*hidden+3*dim+4,hidden),nn.LayerNorm(hidden),nn.SiLU())
    def forward(self,a:Tensor,b:Tensor,ga:Tensor,gb:Tensor,timing:Tensor,correlation:bool=True)->Tensor:
        if timing.shape!=(a.shape[0],4): raise ValueError('Four per-pair times required')
        fw,fv=local_correlation(a,b,self.radius)
        rv,rm=local_correlation(b,a,self.radius)
        if not correlation: fw,rv=torch.zeros_like(fw),torch.zeros_like(rv)
        x=torch.cat((a.float(),b.float(),b.float()-a.float(),fw,fv.float(),rv,rm.float()),1)
        z=self.spatial(x)
        summary=torch.cat((z.mean((-2,-1)),z.amax((-2,-1)),ga.float(),gb.float(),gb.float()-ga.float(),timing.float()),1)
        return self.project(summary)

class PhaseHead(nn.Module):
    def __init__(self,hidden:int,prior:float):
        super().__init__();validate_initial_prior(prior)
        self.loc=nn.Linear(hidden,1);self.width=nn.Linear(hidden,2)
        nn.init.zeros_(self.loc.weight);nn.init.zeros_(self.loc.bias)
        nn.init.zeros_(self.width.weight);nn.init.zeros_(self.width.bias)
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32))
    def forward(self,z:Tensor)->dict[str,Tensor]:
        location=self.prior+.03*self.loc(z).squeeze(-1)
        widths=.03*F.softplus(self.width(z))
        return {'raw_location':location,'point_phase':emitted_phase(location),'ttc':phase_to_ttc(location),
                'q10':(location-widths[:,0]).clamp(MIN_PHASE,MAX_PHASE),'q90':(location+widths[:,1]).clamp(MIN_PHASE,MAX_PHASE),'hidden':z}

class EventModel(nn.Module):
    def __init__(self,prior:float,correlation:bool=True,encoder:nn.Module|None=None,dim:int=128,hidden:int=160,radius:int=4):
        super().__init__()
        self.encoder=encoder if encoder is not None else DenseResNet50(20)
        self.evidence=PairEvidence(dim,hidden,radius)
        self.cells=nn.ModuleList([nn.GRUCell(hidden,hidden),nn.GRUCell(hidden,hidden)])
        self.head=PhaseHead(hidden,prior);self.use_correlation=correlation;self.hidden=hidden
    def forward(self,events:Tensor,timing:Tensor,valid:Tensor|None=None)->dict[str,Tensor]:
        if events.ndim!=5 or events.shape[1] not in {2,4} or events.shape[2]!=20:
            raise ValueError('Events must be [B,T=2|4,20,H,W]')
        b,t,c,h,w=events.shape
        if b==0 or timing.shape!=(b,t-1,4) or not bool(torch.isfinite(timing).all()):
            raise ValueError('Invalid batch/pair timing')
        if bool((timing[...,0]<=0).any()): raise ValueError('Pair interval must be positive')
        if valid is None: valid=torch.ones((b,t),dtype=torch.bool,device=events.device)
        if valid.shape!=(b,t) or valid.dtype!=torch.bool or not bool(valid[:,-1].all()):
            raise ValueError('Invalid temporal mask; current observation required')
        if bool((valid[:,:-1]&~valid[:,1:]).any()): raise ValueError('Expected valid suffix')
        clean=torch.where(valid[:,:,None,None,None],events,torch.zeros_like(events))
        if not bool(torch.isfinite(clean).all()): raise ValueError('Nonfinite valid events')
        maps,glob=self.encoder(clean.reshape(b*t,c,h,w))
        maps=maps.reshape(b,t,*maps.shape[1:]);glob=glob.reshape(b,t,-1)
        states=[maps.new_zeros((b,self.hidden),dtype=torch.float32) for _ in range(2)]
        for j in range(t-1):
            token=self.evidence(maps[:,j],maps[:,j+1],glob[:,j],glob[:,j+1],timing[:,j],self.use_correlation)
            keep=(valid[:,j]&valid[:,j+1])[:,None]
            states[0]=torch.where(keep,self.cells[0](token,states[0]),states[0])
            states[1]=torch.where(keep,self.cells[1](states[0],states[1]),states[1])
        out=self.head(states[-1]);out['has_pair']=(valid[:,:-1]&valid[:,1:]).any(1)
        return out

class FusionHead(nn.Module):
    def __init__(self,hidden:int=160):
        super().__init__()
        self.readout=nn.Sequential(nn.Linear(hidden*2+4,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.gate=nn.Linear(hidden,1);self.residual=nn.Linear(hidden,1);self.width=nn.Linear(hidden,2)
        for m in [self.gate,self.residual,self.width]: nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,event:dict[str,Tensor],rgb_hidden:Tensor,quality:Tensor,available:Tensor)->dict[str,Tensor]:
        b=event['hidden'].shape[0]
        if available.dtype!=torch.bool or available.shape!=(b,) or quality.shape!=(b,4) or rgb_hidden.shape!=event['hidden'].shape:
            raise ValueError('RGB fusion mask/shape mismatch')
        clean=torch.where(available[:,None],rgb_hidden,torch.zeros_like(rgb_hidden))
        q=torch.where(available[:,None],quality,torch.zeros_like(quality))
        if not bool(torch.isfinite(clean).all() and torch.isfinite(q).all()): raise ValueError('Nonfinite valid RGB features')
        z=self.readout(torch.cat((event['hidden'],clean,q),1))
        gate=torch.sigmoid(self.gate(z).squeeze(-1))
        delta=.03*gate*self.residual(z).squeeze(-1)
        location=event['raw_location']+delta
        widths=.03*F.softplus(self.width(z))
        proposal={'raw_location':location,'point_phase':emitted_phase(location),'ttc':phase_to_ttc(location),
                  'q10':(location-widths[:,0]).clamp(MIN_PHASE,MAX_PHASE),'q90':(location+widths[:,1]).clamp(MIN_PHASE,MAX_PHASE)}
        # Select the exact existing E values for missing RGB, including its interval.
        out=dict(event)
        for k,value in proposal.items(): out[k]=torch.where(available,value,event[k])
        out['rgb_gate']=torch.where(available,gate,torch.zeros_like(gate))
        out['rgb_available']=available
        return out

class RGBExtension(nn.Module):
    def __init__(self,event_model:EventModel,prior:float,rgb_encoder:nn.Module|None=None,dim:int=128,hidden:int=160,radius:int=4):
        super().__init__();self.event=event_model
        for p in self.event.parameters(): p.requires_grad_(False)
        self.rgb_encoder=rgb_encoder if rgb_encoder is not None else DenseResNet50(3)
        self.rgb_pair=PairEvidence(dim,hidden,radius);self.rgb_head=PhaseHead(hidden,prior);self.fusion=FusionHead(hidden)
        self.hidden=hidden
    def train(self,mode:bool=True):
        super().train(mode);self.event.eval();return self
    def forward(self,events:Tensor,event_times:Tensor,rgb:Tensor|None,rgb_times:Tensor,available:Tensor,quality:Tensor,event_valid:Tensor|None=None)->dict[str,Tensor]:
        with torch.no_grad(): e=self.event(events,event_times,event_valid)
        b=events.shape[0]
        if available.shape!=(b,) or available.dtype!=torch.bool: raise ValueError('RGB availability mask required')
        indices=available.nonzero(as_tuple=False).flatten()
        h=e['hidden'].new_zeros((b,self.hidden))
        if len(indices):
            if rgb is None or rgb.ndim!=5 or rgb.shape[:3]!=(b,2,3) or rgb_times.shape!=(b,4):
                raise ValueError('Available RGB must be [B,2,3,H,W] with pair times')
            r=rgb.index_select(0,indices)
            if not bool(torch.isfinite(r).all()): raise ValueError('Nonfinite available RGB pixels')
            n,t,c,hh,ww=r.shape
            f,g=self.rgb_encoder(r.reshape(n*t,c,hh,ww))
            f=f.reshape(n,2,*f.shape[1:]);g=g.reshape(n,2,-1)
            token=self.rgb_pair(f[:,0],f[:,1],g[:,0],g[:,1],rgb_times.index_select(0,indices))
            h=h.index_copy(0,indices,token)
        out=self.fusion(e,h,quality,available)
        out['rgb_aux']=self.rgb_head(h)
        return out

class HeadOnlyRGBBridge(nn.Module):
    """For cached RGB descriptors and exact H8 hidden states; production wrapper is separate."""
    def __init__(self,rgb_dim:int=2048,hidden:int=160):
        super().__init__()
        self.rgb_project=nn.Sequential(nn.Linear(3*rgb_dim+4,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.fusion=FusionHead(hidden)
    def forward(self,event:dict[str,Tensor],rgb_pair:Tensor,timing:Tensor,available:Tensor,zero:bool=False,quality:Tensor|None=None)->dict[str,Tensor]:
        if rgb_pair.ndim!=3 or rgb_pair.shape[1]!=2: raise ValueError('Expected [B,2,D]')
        clean=torch.where(available[:,None,None],rgb_pair,torch.zeros_like(rgb_pair))
        safe_timing=torch.where(available[:,None],timing,torch.zeros_like(timing))
        if not bool(torch.isfinite(clean).all() and torch.isfinite(safe_timing).all()): raise ValueError('Nonfinite available RGB features/times')
        if zero: clean=torch.zeros_like(clean)
        a,b=clean.unbind(1)
        h=self.rgb_project(torch.cat((a,b,b-a,safe_timing),1))
        return self.fusion(event,h,torch.zeros_like(timing) if quality is None else quality,available)
