"""Wait for the exact retained RGB cache archive and verify it on container SSD."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

ARCHIVE=Path('/tmp/ejepa_final_RGB_PILOT.tar')
ROOT=Path('/opt/ejepa_final_rgb')
DEADLINE=1793145600


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    while not ARCHIVE.exists() or ARCHIVE.stat().st_size!=10569820160:
        if time.time()>=DEADLINE:raise TimeoutError('Authorized deadline reached')
        time.sleep(5)
    expected='a3dadc7e37e35735c9dad2d70bae5d3e28634a2dd6b802fe10d57ae3f2821cf0'
    if digest(ARCHIVE)!=expected:raise ValueError('RGB archive checksum mismatch')
    ROOT.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE) as tar:
        manifest=json.load(tar.extractfile('BUNDLE_MANIFEST.json'))
        members={m.name:m for m in tar.getmembers()}
        files={row['relative']:row for row in manifest['files']}
        if set(members)!=set(files)|{'BUNDLE_MANIFEST.json'}:raise ValueError('RGB archive coverage mismatch')
        for name,member in members.items():
            if time.time()>=DEADLINE:raise TimeoutError('Authorized deadline reached')
            target=ROOT/name
            if not member.isfile() or not target.resolve().is_relative_to(ROOT):raise ValueError('Unsafe RGB member')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                tmp=target.with_suffix(target.suffix+'.pending')
                with tar.extractfile(member) as incoming,tmp.open('wb') as outgoing:
                    shutil.copyfileobj(incoming,outgoing)
                    outgoing.flush();os.fsync(outgoing.fileno())
                os.replace(tmp,target)
            if name in files and (target.stat().st_size!=files[name]['bytes'] or digest(target)!=files[name]['sha256']):
                raise ValueError('RGB cache member checksum mismatch: '+name)
            target.chmod(0o444)
    receipt={'status':'VERIFIED','archive_sha256':expected,'files':len(files),'timestamp':time.time(),
             'input_copy_is_ephemeral':True,'durable_outputs':'/workspace/ejepa_final_runs',
             'local_archive_preserved':True}
    tmp=ROOT/'RGB_READY.pending';tmp.write_text(json.dumps(receipt));os.replace(tmp,ROOT/'RGB_READY.json')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
