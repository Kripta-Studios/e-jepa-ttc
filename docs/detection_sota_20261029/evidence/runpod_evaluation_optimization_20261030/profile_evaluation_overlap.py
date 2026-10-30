"""Paired B1 evaluation scheduling profile on fixed real selection images."""
import cProfile
import gc
import hashlib
import importlib.util
import json
import os
import pstats
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

root = Path('/workspace/phase_release')
project = root/'workspace/phase-ttc-s0-evaluation'
parent = project/'output/cloud_input_optimized_v1/producer'
candidate = project/'output/cloud_evaluation_prefetch_v1/producer'
out = root/'evaluation_optimization_v1'
os.chdir(project)
sys.path[:0] = [str(parent/'src'), str(parent/'scripts')]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

old = load('evaluation_baseline', parent/'scripts/train_detection_campaign.py')
import e_jepa_ttc.detection
e_jepa_ttc.detection.__path__.append(str(candidate/'src/e_jepa_ttc/detection'))
new = load('evaluation_candidate', candidate/'scripts/train_detection_campaign.py')
config = old.load_campaign_config(root/'cloud_configs_input_v1/r0_seed7.yaml')
manifest = old.validate_campaign_manifest(config)
all_tokens = manifest['roles'][config.data.selection_role]['sample_tokens']
tokens = all_tokens[::8][:128]
torch.set_num_threads(config.resources.num_cpu_threads)
torch.set_float32_matmul_precision('highest')
model, _, _ = old.build_detector(project/'output/cloud_input_optimized_v1/portable_r0_seed7',
                                 config.model.group_num, predict_velocity=True,
                                 velocity_loss_coef=config.model.velocity_loss_coef)
checkpoint_path = root/'cloud_runs_input_v1/r0_seed7/checkpoints/selected.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
model.load_state_dict(checkpoint['model'], strict=True)
with checkpoint_path.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
del checkpoint
datasets = [old.open_campaign_dataset(config.data.root, tokens, config) for _ in range(2)]
caches = [module.PreparedCache(dataset, config, {}, label_free_model_input=True)
          for module, dataset in zip([old, new], datasets)]
for cache in caches:
    for i in range(len(tokens)):
        cache.get(i)
protocol = old.load_protocol(config.evaluation.protocol)
original_pid = 9167
assert 'train_detection_campaign.py' in Path(f'/proc/{original_pid}/cmdline').read_text()
watchdog = subprocess.Popen([sys.executable, '-c',
    f'import os,signal,time;time.sleep(240);os.kill({original_pid},signal.SIGCONT)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True)
os.kill(original_pid, signal.SIGSTOP)
rows = []
reference = None
try:
    for iteration, mode in enumerate([0, 1, 1, 0]):
        module = [old, new][mode]
        path = out/f'panel_{iteration}_{mode}.json'
        profiler = cProfile.Profile()
        start = time.perf_counter()
        if iteration == 0:
            profiler.enable()
        result = module.evaluate(model, caches[mode], protocol, config, path)
        if iteration == 0:
            profiler.disable()
            profiler.dump_stats(str(out/'baseline_cprofile.prof'))
            with (out/'baseline_cprofile.txt').open('w') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(35)
        elapsed = time.perf_counter()-start
        payload = json.loads(path.read_text())
        for key in ['inference_seconds_median', 'inference_seconds_total']:
            payload['metrics_v2'].pop(key)
        if reference is None:
            reference = payload
        assert payload == reference, f'Exact payload mismatch in mode {mode}'
        rows.append({'mode':mode, 'seconds':elapsed, 'profiled':iteration==0,
                     'inference_seconds':result['inference_seconds_total'],
                     'exact_payload':True})
        print(json.dumps(rows[-1]), flush=True)
        del payload, result
        gc.collect()
    baseline = rows[-1]['seconds']
    speedup = baseline/(sum(r['seconds'] for r in rows if r['mode']==1)/2)
    report = {'status':'PASS_EXACT_PAYLOAD', 'checkpoint_sha256':checkpoint_sha,
              'sample_tokens':tokens, 'rows':rows, 'unprofiled_speedup':speedup,
              'batch_precision_model_metric_unchanged':True}
    (out/'admission.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status':report['status'],'speedup':speedup}),flush=True)
finally:
    os.kill(original_pid, signal.SIGCONT)
    watchdog.terminate()
    for dataset in datasets:
        dataset.close()
