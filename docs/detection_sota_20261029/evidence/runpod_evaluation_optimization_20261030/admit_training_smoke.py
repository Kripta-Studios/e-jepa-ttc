from pathlib import Path
import hashlib,json,subprocess,time,yaml
root=Path('/workspace/phase_release'); out=root/'evaluation_optimization_v1'
while Path('/proc/9843/cmdline').exists() and 'benchmark_thread_count.py' in Path('/proc/9843/cmdline').read_text():
 time.sleep(2)
producer=root/'workspace/phase-ttc-s0-evaluation/output/cloud_evaluation_prefetch_v1/producer'
config=yaml.safe_load((root/'cloud_configs_input_v1/r0_seed7.yaml').read_text())
config['experiment_id']='evaluation-prefetch-gpu-admission'
config['model']['portable_root']='output/cloud_evaluation_prefetch_v1/portable_smoke'
path=out/'smoke_config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
subprocess.run([str(root/'.venv-detector/bin/python'),str(producer/'scripts/train_detection_campaign.py'),'--config',str(path),'--output',str(out/'hardware_smoke'),'--stop-after-update','16'],cwd=root/'workspace/phase-ttc-s0-evaluation',check=True)
summary=json.loads((out/'hardware_smoke/summary.json').read_text())
assert summary['status']=='BUDGET_STOPPED' and summary['final_update']==16
profile=json.loads((out/'admission.json').read_text());assert profile['status']=='PASS_EXACT_PAYLOAD' and profile['unprofiled_speedup']>1.1
receipt=json.loads((producer/'EVALUATION_PREFETCH_RECEIPT.json').read_text())
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert receipt['runner_sha256']==digest(producer/'scripts/train_detection_campaign.py')
evidence=['evaluation_optimization_v1/admission.json','evaluation_optimization_v1/hardware_smoke/summary.json']
result={'status':'PASS_EXACT_PAYLOAD_AND_TRAINING_SMOKE','runner_sha256':receipt['runner_sha256'],'modules':receipt['modules'],'evidence_sha256':{p:digest(root/p) for p in evidence},'speedup_128_frame_panel':profile['unprofiled_speedup'],'cpu_threads':4,'batch_size':1}
(out/'deployment_admission.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
