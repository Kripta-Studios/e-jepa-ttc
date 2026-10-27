"""Chunked authenticated upload with end-to-end SHA verification and durable progress."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/runpod_event_migration_20261026/resume_20261027'))
from rpc import call


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination');p.add_argument('--status',type=Path,required=True)
    a=p.parse_args()
    if not a.destination.startswith(('/workspace/ejepa_final_','/tmp/ejepa_final_')) or '..' in a.destination:
        raise ValueError('Unregistered destination')
    with a.source.open('rb') as f:expected=hashlib.file_digest(f,'sha256').hexdigest()
    servers=call("import pathlib,json;print('PODSTATUS:'+json.dumps([json.loads(p.read_text()) for p in pathlib.Path('/root/.local/share/jupyter/runtime').glob('jpserver-*.json')]))")
    session=requests.Session();session.headers['Authorization']='token '+servers[0]['token']
    url='https://lreiivpf18vxu7-8888.proxy.runpod.net/api/contents/'+a.destination.lstrip('/')
    size=a.source.stat().st_size;sent=0;chunk=0;started=time.time()
    a.status.parent.mkdir(parents=True,exist_ok=True)
    with a.source.open('rb') as f:
        while data:=f.read(8*1024*1024):
            chunk+=1;sent+=len(data)
            response=session.put(url,json={'type':'file','format':'base64','content':base64.b64encode(data).decode(),'chunk':-1 if sent==size else chunk},timeout=120)
            response.raise_for_status()
            a.status.write_text(json.dumps({'status':'UPLOADING','sent':sent,'total':size,'elapsed':time.time()-started,'timestamp':time.time()}))
    code=f"import pathlib,json,hashlib;p=pathlib.Path({a.destination!r});h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert h=={expected!r};print('PODSTATUS:'+json.dumps({{'status':'VERIFIED','bytes':p.stat().st_size,'sha256':h,'path':str(p)}}))"
    result=call(code,timeout=120);a.status.write_text(json.dumps(result));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
