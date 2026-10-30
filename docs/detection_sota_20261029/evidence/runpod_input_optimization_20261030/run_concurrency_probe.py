"""Compare one and two producers with reserved full-D4096 image VRAM budgets."""
import json
import subprocess
from pathlib import Path

root=Path('/workspace/phase_release')
out=root/'optimization_v1'
python=root/'.venv-detector/bin/python'
for tags in [('single',),('pair0','pair1')]:
    processes=[]
    for tag in tags:
        log=(out/f'concurrency_{tag}.log').open('w')
        process=subprocess.Popen([str(python),str(out/'benchmark_resident_concurrency.py'),tag],
                                  stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        processes.append((process,log))
    for process,log in processes:
        status=process.wait(timeout=1200);log.close()
        if status:raise RuntimeError(f'Concurrency worker failed with status {status}')
single=json.loads((out/'concurrency_single.json').read_text())
pair=[json.loads((out/f'concurrency_pair{i}.json').read_text()) for i in range(2)]
single_rate=single['updates']/(single['steady_end']-single['steady_start'])
pair_rate=sum(r['updates'] for r in pair)/(max(r['steady_end'] for r in pair)-min(r['steady_start'] for r in pair))
result={'status':'PASS','single_updates_per_second':single_rate,
        'pair_aggregate_updates_per_second':pair_rate,'aggregate_speedup':pair_rate/single_rate,
        'recommended_jobs':2 if pair_rate>single_rate*1.10 else 1,
        'scope':'128 updates per worker over 64 images, with 12 GiB extra VRAM reserved per worker'}
(out/'concurrency_admission.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
