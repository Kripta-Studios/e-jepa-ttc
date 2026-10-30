"""Real CPU/GPU input parity and fixed-state training checks for resident inputs."""
import copy
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/workspace/phase_release')
PROJECT=ROOT/'workspace/phase-ttc-s0-evaluation'
OLD=PROJECT/'output/detection_campaign/rgb_screen_v5_seed7_v1/producer'
NEW=PROJECT/'output/cloud_input_optimized_v1/producer'
OUT=ROOT/'optimization_v1'
TAG=sys.argv[1]
os.chdir(PROJECT)
sys.path[:0]=[str(OLD/'src'),str(OLD/'scripts')]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

old=load('original_producer',OLD/'scripts/train_detection_campaign.py')
import e_jepa_ttc.detection
e_jepa_ttc.detection.__path__.append(str(NEW/'src/e_jepa_ttc/detection'))
new=load('resident_producer',NEW/'scripts/train_detection_campaign.py')
config=old.load_campaign_config(ROOT/'cloud_configs/r0_seed7.yaml')
manifest=old.validate_campaign_manifest(config)
tokens=manifest['roles']['train']['sample_tokens']
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
sampler=old.ResumablePermutationSampler(len(tokens),7)
indices=sum([sampler.next_batch(4) for _ in range(16)],[])
datasets=[old.EAPRGBDataset(config.data.root,tokens,max_open_shards=128,max_label_sequences=24)
          for _ in range(2)]
original=old.PreparedCache(datasets[0],config,{})
resident=new.PreparedCache(datasets[1],config,{})
original_rows=[];resident_rows=[]

def compare(left,right,exact=True):
    if isinstance(left,torch.Tensor):
        assert left.shape==right.shape and left.dtype==right.dtype
        if exact:assert torch.equal(left.detach().cpu(),right.detach().cpu())
        else:torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:compare(left[key],right[key],exact)
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for x,y in zip(left,right):compare(x,y,exact)
    else:assert left==right

for index in indices:
    a=original.get(index);b=resident.get(index)
    assert a[1].member_sha256==b[1].member_sha256
    assert np.array_equal(a[1].image,b[1].image)
    for key in ('image','calibration','image_size','target','native_ground_truth','report'):
        compare(getattr(a[0],key),getattr(b[0],key))
    original_rows.append(a);resident_rows.append(b)
print(json.dumps({'input_parity':'EXACT','samples':len(indices)}),flush=True)
deadline=time.monotonic()+600
while not (OUT/'original_checkpoint_pause.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Original producer did not checkpoint and pause')
    time.sleep(2)
batch_cpu=old.batch_to_cuda(original_rows[:4],config,training=True)
batch_gpu=new.batch_to_cuda(resident_rows[:4],config,training=True)
compare(batch_cpu,batch_gpu)
free_cache=new.PreparedCache(datasets[1],config,{},label_free_model_input=True)
for index in indices[:8]:
    prepared,raw=free_cache.get(index)
    reference=old.prepare_model_input(raw,config)
    compare(prepared.image,reference.image)
    compare(prepared.target,reference.target)
    assert prepared.target['labels'].numel()==0 and not prepared.native_ground_truth

portable=PROJECT/('output/cloud_input_optimized_v1/portable_bench_'+TAG)
old.prepare_source(Path(config.model.source_root),portable,OUT/('bench_'+TAG+'_upstream.patch'),
                   predict_velocity=True)
model,criterion,_=old.build_detector(portable,config.model.group_num,predict_velocity=True,
                                     velocity_loss_coef=config.model.velocity_loss_coef)
checkpoint=torch.load(ROOT/'cloud_runs/hardware_smoke/checkpoints/last.pt',map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model'],strict=True)
initial={k:v.detach().clone() for k,v in model.state_dict().items()}

def step(batch,reset=True):
    if reset:
        model.load_state_dict(initial,strict=True)
        random.seed(23);np.random.seed(23);torch.manual_seed(23);torch.cuda.manual_seed_all(23)
    model.train();criterion.train();model.zero_grad(set_to_none=True)
    images,calibrations,targets,sizes=batch
    outputs=model(images,calibrations,targets,sizes)
    old.require_finite(outputs)
    losses=criterion(outputs,targets);old.require_finite(losses,'training_losses')
    terms=[v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict]
    loss=terms[0]
    for term in terms[1:]:loss=loss+term
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),config.optimization.gradient_clip_norm,
                                   error_if_nonfinite=True)
    return outputs,losses,{k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}

initial.clear();del checkpoint
padding=torch.empty((12*2**30//4,),device='cuda')
optimizer=torch.optim.AdamW(model.parameters(),lr=config.optimization.learning_rate,
                            weight_decay=config.optimization.weight_decay)
timings=[]
for iteration in range(132):
    if iteration==4:steady_start=time.time()
    selected=[indices[(iteration*4+i)%len(indices)] for i in range(4)]
    torch.cuda.synchronize();start=time.perf_counter()
    rows=[resident.get(i) for i in selected]
    batch=new.batch_to_cuda(rows,config,training=True)
    prepared=time.perf_counter()
    model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
    outputs=model(*batch[:2],batch[2],batch[3]);old.require_finite(outputs)
    losses=criterion(outputs,batch[2]);old.require_finite(losses,'training_losses')
    terms=[v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict]
    loss=terms[0]
    for term in terms[1:]:loss=loss+term
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),config.optimization.gradient_clip_norm,
                                   error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize()
    if iteration>=4:timings.append({'input':prepared-start,'total':time.perf_counter()-start})
    del outputs,losses,loss

import resource
report={'status':'PASS_BENCHMARK','tag':TAG,'updates':len(timings),
        'steady_start':steady_start,'steady_end':time.time(),
        'steady_cached_update_seconds_median':statistics.median(x['total'] for x in timings),
        'steady_seconds':sum(x['total'] for x in timings),
        'cpu_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'padding_bytes':padding.numel()*padding.element_size(),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
(OUT/('concurrency_'+TAG+'.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
