"""Restore sealed runtime and input cache on replacement Pod; no training."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

OPS = Path('/workspace/ejepa_event_operations')
BASE = OPS / 'manual_pause_20261026'
OUT = OPS / 'resume_20261027'

def status(stage, **kwargs):
    p = OUT / 'RESTORE_STATUS.json'
    t = p.with_suffix('.pending')
    t.write_text(json.dumps({'stage':stage, **kwargs}, indent=2))
    os.replace(t, p)

def main():
    OUT.mkdir(exist_ok=True)
    receipt = json.loads((BASE/'PRESERVATION_STATUS.json').read_text())
    assert receipt['status']=='PAUSED_PRESERVED'
    source = Path(receipt['runtime_archive'])
    local = Path('/opt/ejepa_runtime_restore.tar')
    size = source.stat().st_size
    assert size == receipt['runtime_bytes']
    fd = os.open(source, os.O_RDONLY)
    step = 8*1024*1024
    digest = hashlib.sha256()
    def read(offset):
        wanted=min(step,size-offset)
        data=os.pread(fd,wanted,offset)
        assert len(data)==wanted
        return data
    with cf.ThreadPoolExecutor(8) as pool, local.open('wb') as out:
        offsets=iter(range(0,size,step)); pending=[]; done=0
        for _ in range(8):
            offset=next(offsets,None)
            if offset is not None:pending.append(pool.submit(read,offset))
        while pending:
            data=pending.pop(0).result();digest.update(data);out.write(data);done+=len(data)
            status('VERIFY_RUNTIME',bytes=done,total=size)
            offset=next(offsets,None)
            if offset is not None:pending.append(pool.submit(read,offset))
    os.close(fd)
    assert digest.hexdigest()==receipt['runtime_sha256']
    status('EXTRACT_RUNTIME')
    subprocess.run(['tar','-C','/opt','-xf',str(local)],check=True)
    # Copy independently into disjoint destination files; preserve symlink semantics.
    src=Path('/workspace/ejepa_event_bundle');dst=Path('/opt/ejepa_event_bundle')
    jobs=[]
    for root,dirs,files in os.walk(src,followlinks=False):
        rp=Path(root);target=dst/rp.relative_to(src);target.mkdir(parents=True,exist_ok=True)
        for name in list(dirs):
            p=rp/name
            if p.is_symlink():
                q=target/name
                if not q.exists():q.symlink_to(os.readlink(p),target_is_directory=True)
                dirs.remove(name)
        jobs.extend((rp/name,target/name) for name in files)
    def copy(pair):
        a,b=pair
        shutil.copy2(a,b,follow_symlinks=False)
        return a.lstat().st_size
    done=0;total_bytes=0
    with cf.ThreadPoolExecutor(8) as pool:
        for n in pool.map(copy,jobs):
            done+=1;total_bytes+=n
            if done%100==0:status('COPY_BUNDLE',files=done,total_files=len(jobs),bytes=total_bytes)
    subprocess.run(['/opt/ejepa_env/bin/python','-c',
                    'import torch,sys; assert sys.version_info[:2]==(3,11); assert torch.__version__.startswith("2.11.0"); assert torch.cuda.is_available(); print(sys.version,torch.__version__,torch.cuda.get_device_name())'],check=True)
    status('RESTORED',files=done,bytes=total_bytes,training_started=False)

if __name__=='__main__':main()
