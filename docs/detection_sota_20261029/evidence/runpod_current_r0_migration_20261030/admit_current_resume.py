import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/workspace/phase_release');out=root/'evaluation_optimization_v1/r0_migration'
pids={9167:'train_detection_campaign.py',9162:'runpod_input_optimized_queue.py',10080:'runpod_evaluation_prefetch_queue.py'}
for pid,name in pids.items():assert name in Path(f'/proc/{pid}/cmdline').read_text()
watchdog=out/'resume_watchdog.py'
watchdog.write_text("import time,os,signal\nfrom pathlib import Path\ntime.sleep(900)\nif not Path('/workspace/phase_release/evaluation_optimization_v1/r0_migration/SWITCH_COMMITTED').exists():\n for pid in [9167,9162,10080]:\n  try:os.kill(pid,signal.SIGCONT)\n  except ProcessLookupError:pass\n")
p=subprocess.Popen([sys.executable,str(watchdog)],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(out/'pause.json').write_text(json.dumps({'pids':pids,'watchdog_pid':p.pid,'paused_epoch':time.time()}))
for pid in pids:os.kill(pid,signal.SIGSTOP)
try:
 producer=root/'workspace/phase-ttc-s0-evaluation/output/cloud_evaluation_prefetch_v1/producer/scripts/train_detection_campaign.py'
 subprocess.run([sys.executable,str(producer),'--config',str(root/'cloud_configs_input_v1/r0_seed7.yaml'),'--output',str(out/'resume_smoke'),'--resume',str(out/'checkpoints/last.pt'),'--stop-after-update','4097'],cwd=root/'workspace/phase-ttc-s0-evaluation',check=True,timeout=300)
 summary=json.loads((out/'resume_smoke/summary.json').read_text());migration=json.loads((out/'MIGRATION.json').read_text())
 assert summary['status']=='BUDGET_STOPPED' and summary['start_update']==4096 and summary['final_update']==4097
 assert summary['resumed_from_sha256']==migration['checkpoint_sha256']
 (out/'RESUME_ADMISSION.json').write_text(json.dumps({'status':'PASS_STRICT_RESUME_4096_TO_4097','checkpoint_sha256':migration['checkpoint_sha256'],'original_processes_paused':list(pids)},indent=2))
 print('PASS_STRICT_RESUME_4096_TO_4097',flush=True)
except BaseException:
 for pid in pids:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 raise
