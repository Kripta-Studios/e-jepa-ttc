from pathlib import Path
import json, sys, platform
import torch
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True,help='V17 source checkout, not a checkpoint')
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
SOURCE=a.source.resolve()
if a.output.exists(): raise SystemExit('Output exists; do not overwrite a prior receipt')
sys.path[:0]=[str(ROOT), str(SOURCE/'src'), str(SOURCE)]
from reference.relative_measurement import PairMeasurement,pair_objective
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC,CausalScaleTTCConfig
from e_jepa_ttc.simplex_t.phase import emitted_phase

torch.set_num_threads(1)
torch.manual_seed(7)
config=CausalScaleTTCConfig(in_channels=12,hidden_dim=64,geometry_dim=128,residual_depth=2,dropout=0.05,foreground_decoder='resize_conv',modality='event')
native=CausalScaleTTC(config).eval()
original={k:v.detach().clone() for k,v in native.state_dict().items()}
records=[]
for name,correlation,frozen in [('DIRECT',False,False),('CORR',True,False),('CORR_FROZEN',True,True)]:
 torch.manual_seed(19)
 model=PairMeasurement.from_native(native,use_correlation=correlation,freeze_encoder=frozen).train()
 x=torch.randn(2,2,12,128,128)
 dt=torch.tensor([0.1,0.08])
 out=model(x,dt)
 loss=pair_objective(out['raw_phase'],torch.tensor([0.08,-0.015]),torch.tensor([0.5,0.5]),2,emitted_phase)
 loss.backward()
 encnorm=sum(float(p.grad.square().sum()) for p in model.encoder.parameters() if p.grad is not None)**0.5
 assert torch.isfinite(loss) and torch.isfinite(out['raw_phase']).all()
 assert out['descriptor'].shape==(2,194)
 assert (encnorm>0) if not frozen else (encnorm==0)
 assert all(torch.equal(v,original[k]) for k,v in native.state_dict().items())
 assert all(p.grad is None for p in native.parameters())
 assert not any(p.data_ptr()==q.data_ptr() for p in model.encoder.parameters() for q in native.encoder.features.parameters())
 records.append({'name':name,'input_shape':list(x.shape),'output_shape':list(out['raw_phase'].shape),'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'encoder_gradient_norm':encnorm,'native_parent_unchanged':True,'private_encoder_copy':True,'synthetic_loss':float(loss.detach()),'optimizer_updates':0})
 print(records[-1],flush=True)
result={'status':'PASS_SYNTHETIC_NATIVE_INTERFACE','records':records,'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},'scope':'Native CNN instantiated from source with random weights; synthetic 128x128 inputs. No dataset, trained A5 parent, GPU, optimizer update or TTC accuracy claim.'}
a.output.write_text(json.dumps(result,indent=2)+'\n')
