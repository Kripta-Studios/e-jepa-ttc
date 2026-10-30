"""Measure frozen reader stages and a larger bounded TAR-handle cache on CPU."""
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/phase_release')
PROJECT = ROOT / 'workspace/phase-ttc-s0-evaluation'
FROZEN = PROJECT / 'output/detection_campaign/rgb_screen_v5_seed7_v1/producer'
os.chdir(PROJECT)
sys.path[:0] = [str(FROZEN/'src'), str(FROZEN/'scripts')]
import numpy as np
import psutil
import torch
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.campaign_training import load_campaign_config, ResumablePermutationSampler
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample

torch.set_num_threads(1)
config = load_campaign_config(ROOT/'cloud_configs/r0_seed7.yaml')
manifest = json.loads(Path(config.data.manifest).read_text())
tokens = manifest['roles']['train']['sample_tokens']
sampler = ResumablePermutationSampler(len(tokens), 7)
indices = sum([sampler.next_batch(4) for _ in range(12)], [])
baseline = {}
reports = []
for handles in (2, 128):
    dataset = EAPRGBDataset(config.data.root, tokens, max_open_shards=handles,
                            max_label_sequences=24)
    stages = {'shard_index': [], 'labels': [], 'raw_total': [], 'prepare': []}
    original_shard, original_labels = dataset._shard, dataset._frame_labels
    def shard(relative):
        start=time.perf_counter(); value=original_shard(relative)
        stages['shard_index'].append(time.perf_counter()-start); return value
    def labels(row):
        start=time.perf_counter(); value=original_labels(row)
        stages['labels'].append(time.perf_counter()-start); return value
    dataset._shard, dataset._frame_labels = shard, labels
    started=time.perf_counter()
    for position,index in enumerate(indices+indices):
        start=time.perf_counter(); raw=dataset[index]
        stages['raw_total'].append(time.perf_counter()-start)
        start=time.perf_counter()
        prepared=prepare_sample(raw, width=config.model.width, height=config.model.height,
                                include_velocity=True)
        stages['prepare'].append(time.perf_counter()-start)
        fingerprint=(raw.member_sha256, prepared.image.numpy().tobytes())
        if handles==2 and position<len(indices): baseline[index]=fingerprint
        else: assert baseline[index]==fingerprint, f'Image parity failed: {index}'
        if position==0:
            def tensor_bytes(value):
                if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
                if isinstance(value,dict):return sum(tensor_bytes(x) for x in value.values())
                return 0
            sizes={'image':tensor_bytes(prepared.image),'raw_image':raw.image.nbytes,
                   'target_tensors':tensor_bytes(prepared.target)}
        if (position+1)%24==0:
            print(json.dumps({'handles':handles,'samples':position+1,
                 'seconds':time.perf_counter()-started}),flush=True)
    report={'handles':handles,'samples':len(indices)*2,'seconds':time.perf_counter()-started,
            'sample_bytes':sizes,'rss':psutil.Process().memory_info().rss,
            'first_pass':{k:sum(v[:len(indices)]) for k,v in stages.items()},
            'repeat_pass':{k:sum(v[len(indices):]) for k,v in stages.items()},
            'image_and_png_hash_parity':'EXACT'}
    reports.append(report);dataset.close()
    print(json.dumps(report),flush=True)
(ROOT/'optimization_v1/input_profile.json').write_text(json.dumps(reports,indent=2))
