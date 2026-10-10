"""Trainable ResNet50/FPN. No network downloads occur in this module."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class DenseResNet50(nn.Module):
    def __init__(self, channels: int, generic_state: str | Path | None=None, dense_dim: int=128):
        super().__init__()
        if channels not in {3,20}: raise ValueError('Expected RGB3 or native-event20')
        from torchvision.models import resnet50
        base=resnet50(weights=None)
        if generic_state is not None:
            state=torch.load(str(generic_state),map_location='cpu',weights_only=True)
            base.load_state_dict(state,strict=True)
        self.pretraining_loaded = generic_state is not None
        if channels!=3:
            conv=nn.Conv2d(channels,64,7,stride=2,padding=3,bias=False)
            with torch.no_grad():
                conv.weight.copy_(base.conv1.weight.mean(1,keepdim=True).repeat(1,channels,1,1)*(3.0/channels))
            base.conv1=conv
        base.fc=nn.Identity()
        self.net=base
        self.p2=nn.Conv2d(256,dense_dim,1)
        self.p3=nn.Conv2d(512,dense_dim,1)
        self.p4=nn.Conv2d(1024,dense_dim,1)
        self.refine=nn.Sequential(nn.GroupNorm(8,dense_dim),nn.GELU(),nn.Conv2d(dense_dim,dense_dim,3,padding=1))
        self.global_project=nn.Sequential(nn.Linear(2048,dense_dim),nn.LayerNorm(dense_dim))
        self.channels=channels
        self.train(True)
    def train(self, mode: bool=True):
        super().train(mode)
        # Freeze running statistics, NOT convolution gradients or BN affine weights.
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        return self
    def forward(self,x: Tensor)->tuple[Tensor,Tensor]:
        if x.ndim!=4 or x.shape[1]!=self.channels:
            raise ValueError('Encoder modality shape mismatch')
        m=self.net
        z=m.maxpool(m.relu(m.bn1(m.conv1(x))))
        c2=m.layer1(z); c3=m.layer2(c2); c4=m.layer3(c3); c5=m.layer4(c4)
        dense=self.refine(self.p2(c2)+F.interpolate(self.p3(c3),size=c2.shape[-2:],mode='bilinear',align_corners=False)+F.interpolate(self.p4(c4),size=c2.shape[-2:],mode='bilinear',align_corners=False))
        glob=self.global_project(c5.mean((-2,-1)))
        return dense,glob
