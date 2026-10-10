"""Small durable JSON journal example; no automatic retries or model execution."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path


def atomic_json(path: Path, payload: dict)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as out:
            json.dump(payload,out,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def sha256(path: Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def completed_fragment(root: Path, ident: str, binding: str)->dict|None:
    if not ident or Path(ident).name!=ident or '/' in ident or '\\' in ident:
        raise ValueError('Not a safe fragment identifier')
    path=Path(root)/(ident+'.json')
    if not path.exists():return None
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('binding')!=binding:raise ValueError('Fragment input/protocol binding changed')
    if data.get('status')!='COMPLETE':return None
    return data
