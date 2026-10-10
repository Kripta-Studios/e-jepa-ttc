"""Local distributions retained, not converted to expected displacement before reading."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F

def local_correlation(a: Tensor, b: Tensor, radius: int=4, temperature: float=.07) -> tuple[Tensor, Tensor]:
    if a.ndim != 4 or a.shape != b.shape or not 0 <= radius <= 4 or temperature <= 0:
        raise ValueError('Expected matching [B,C,H,W], radius0..4 and positive temperature')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite dense features')
    n,c,h,w = a.shape
    k=2*radius+1
    # Explicit precision boundary: similarities, normalization and masks in FP32.
    with torch.autocast(device_type=a.device.type, enabled=False):
        aa=F.normalize(a.float(),dim=1,eps=1e-6)
        bb=F.normalize(b.float(),dim=1,eps=1e-6)
        candidates=F.unfold(bb,kernel_size=k,padding=radius).reshape(n,c,k*k,h,w)
        logits=(aa[:,:,None]*candidates).sum(1)/temperature
        valid=F.unfold(aa.new_ones((1,1,h,w)),kernel_size=k,padding=radius).reshape(1,k*k,h,w)>0
        valid=valid.expand(n,-1,-1,-1)
        probability=torch.softmax(logits.masked_fill(~valid, -1e9),dim=1)
        probability=torch.where(valid,probability,torch.zeros_like(probability))
    return probability,valid
