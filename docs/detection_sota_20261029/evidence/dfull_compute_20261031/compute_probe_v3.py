"""Bounded disposable fixed-checkpoint compute experiment; never writes campaign weights."""
import os,sys,json,time,signal,subprocess,importlib.util,statistics,copy,hashlib
from pathlib import Path
from contextlib import nullcontext
import torch

B=Path('/workspace/phase_release'); P=B/'workspace/phase-ttc-s0-evaluation'
C=B/'dfull_campaign_v1'; RUN=C/'r2_dfull_seed7_v2'; OUT=C/'compute_probe_v3'
OUT.mkdir(exist_ok=False); os.chdir(P)
pid=7818
assert 'train_detection_campaign.py' in Path(f'/proc/{pid}/cmdline').read_text()
assert str(RUN) in Path(f'/proc/{pid}/cmdline').read_text()
os.link(RUN/'checkpoints/last.pt', OUT/'source_checkpoint.pt')
sys.path[:0]=[str(RUN/'code/src'),str(RUN/'code/scripts')]
spec=importlib.util.spec_from_file_location('producer',RUN/'code/scripts/train_detection_campaign.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
config=runner.load_campaign_config(C/'config_v2.yaml');torch.set_num_threads(4)
payload=torch.load(OUT/'source_checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
tokens=json.loads(Path(config.data.manifest).read_text())['roles']['train']['sample_tokens']
indices=runner.ResumablePermutationSampler(len(tokens),7).next_batch(4)
with runner.open_campaign_dataset(config.data.root,[tokens[i] for i in indices],config) as dataset:
 rows=[]
 for i in range(4):
  raw=dataset[i]; prepared=runner.prepare_sample(raw,width=640,height=384,include_velocity=True)
  rows.append((prepared,raw))
model,criterion,_=runner.build_detector(RUN/'code/portable_source',11,predict_velocity=True,velocity_loss_coef=.1)
groups,_=runner.optimizer_parameter_groups(model,config.optimization)
optimizer=torch.optim.AdamW(groups)
batch=runner.batch_to_cuda(rows,config,training=True)
original_finite=runner.require_finite
def leaves(x):
 if isinstance(x,torch.Tensor):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from leaves(v)
 elif isinstance(x,(tuple,list)):
  for v in x:yield from leaves(v)
def fused(x,name='output'):
 tensors=list(leaves(x))
 if tensors and not torch.stack([torch.isfinite(t).all() for t in tensors]).all().item():
  original_finite(x,name)
def reset():
 model.load_state_dict(payload['model'],strict=True)
 optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
 from e_jepa_ttc.detection.campaign_training import restore_rng_state
 restore_rng_state(payload['rng'])
 model.train();criterion.train()
def step(finite,amp=False,sections=False):
 optimizer.zero_grad(set_to_none=True)
 times={};t=time.perf_counter()
 def mark(name):
  nonlocal t
  if sections:torch.cuda.synchronize()
  now=time.perf_counter();times[name]=now-t;t=now
 with torch.profiler.record_function('MODEL_FORWARD'), (torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()):
  outputs=model(*batch)
 mark('forward')
 with torch.profiler.record_function('OUTPUT_FINITE'):finite(outputs)
 mark('output_checks')
 with torch.profiler.record_function('CRITERION_MATCHING'):
  losses=criterion(outputs,batch[2])
 mark('criterion')
 with torch.profiler.record_function('LOSS_FINITE'):finite(losses,'losses')
 mark('loss_checks')
 loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
 with torch.profiler.record_function('BACKWARD'):loss.backward()
 mark('backward')
 with torch.profiler.record_function('CLIP_OPTIMIZER'):
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  optimizer.step()
 torch.cuda.synchronize();mark('optimizer')
 return times,outputs,float(loss.detach()),float(norm)

result={'checkpoint_update':payload['update'],'source_checkpoint_sha256':runner.sha256_file(OUT/'source_checkpoint.pt'),'samples':[tokens[i] for i in indices],'torch':torch.__version__,'variants':{},'scope':'disposable short fixed-batch training timings, not AP or convergence evidence'}
print(json.dumps({'stage':'ready','checkpoint_update':payload['update']}),flush=True)
paused=False;watch=None;pause_at=None
try:
 # Independent emergency continuation, even if the diagnostic process fails.
 watch=subprocess.Popen([sys.executable,'-c',f'import time,os,signal;time.sleep(120);os.kill({pid},signal.SIGCONT)'],start_new_session=True)
 os.kill(pid,signal.SIGSTOP);paused=True;pause_at=time.perf_counter();time.sleep(.5)
 variants=[('baseline',4,'highest',False,False),('threads1',1,'highest',False,False),('fused_checks',4,'highest',True,False),('fused_threads1',1,'highest',True,False),('tf32_fused',1,'high',True,False),('bf16_fused',1,'highest',True,True)]
 reference=None
 for name,threads,precision,fuse,amp in variants:
  if time.perf_counter()-pause_at>90:break
  try:
   torch.set_num_threads(threads);torch.set_float32_matmul_precision(precision);reset()
   finite=fused if fuse else original_finite
   elapsed=[];losses=[]
   for i in range(8):
    torch.cuda.synchronize();started=time.perf_counter()
    _,outputs,loss,norm=step(finite,amp)
    elapsed.append(time.perf_counter()-started);losses.append(loss)
    if i==0:
     vals=[x.detach().float().cpu().clone() for x in leaves(outputs)]
     if reference is None:reference=vals
     error=max(float((a-b).abs().max()) for a,b in zip(vals,reference))
     exact=all(torch.equal(a,b) for a,b in zip(vals,reference))
    del outputs
   result['variants'][name]={'median_last6_seconds':statistics.median(elapsed[2:]),'seconds':elapsed,'loss':losses,'first_outputs_exact':exact,'first_outputs_max_abs_error':error,'last_gradient_norm':norm}
  except Exception as e:
   result['variants'][name]={'error':str(e),'type':type(e).__name__}
  (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({name:result['variants'][name]}),flush=True)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');reset()
 stages=[]
 for _ in range(3):
  times,outputs,_,_=step(original_finite,sections=True);stages.append(times);del outputs
 result['synchronized_stage_median_seconds']={k:statistics.median(t[k] for t in stages) for k in stages[0]}
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False) as prof:
  _,outputs,_,_=step(original_finite);del outputs
 (OUT/'profiler_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
 (OUT/'profiler_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=35))
 prof.export_chrome_trace(str(OUT/'trace.json'))
 result['status']='COMPLETE'
finally:
 if paused:os.kill(pid,signal.SIGCONT)
 if watch is not None:watch.terminate()
 result['pause_seconds']=None if pause_at is None else time.perf_counter()-pause_at
 result['training_resumed']=paused
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'stage':'finished','pause_seconds':result['pause_seconds'],'training_resumed':paused}),flush=True)
