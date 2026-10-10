"""Bounded, read-only archive verification; downloads only on explicit request.

No extraction, pickle loading, model training or automatic execution of archive code.
"""
from __future__ import annotations
import argparse,hashlib,json,os,stat,urllib.request,urllib.parse,zipfile
from pathlib import Path,PurePosixPath


def file_hash(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def safe_member(name: str)->None:
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe ZIP member: {name!r}')


def deep_check(path: Path,max_uncompressed: int=16*1024**3)->dict:
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(set(names))!=len(names):raise ValueError('Duplicate ZIP names')
        if sum(i.file_size for i in infos)>max_uncompressed:raise ValueError('Archive exceeds verification byte cap')
        for i in infos:
            safe_member(i.filename)
            if stat.S_ISLNK(i.external_attr>>16):raise ValueError('Symlink in ZIP')
        hashes={};sizes={}
        for i in infos:
            if i.is_dir():continue
            h=hashlib.sha256();count=0
            with z.open(i) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);count+=len(b)
            if count!=i.file_size:raise ValueError('Member size mismatch')
            hashes[i.filename]=h.hexdigest();sizes[i.filename]=count
        checked=0;manifest_kind=None
        if 'CONTENT_MANIFEST.json' in names:
            data=json.loads(z.read('CONTENT_MANIFEST.json'))
            members=data.get('members')
            if isinstance(members,dict):
                manifest_kind='CONTENT_MANIFEST.json'
                for name,v in members.items():
                    safe_member(name)
                    if not isinstance(v,dict) or hashes.get(name)!=v.get('sha256'):
                        raise ValueError(f'Member hash mismatch: {name}')
                    if 'bytes' in v and sizes[name]!=v['bytes']:raise ValueError(f'Member byte mismatch: {name}')
                    checked+=1
                if set(hashes)-{'CONTENT_MANIFEST.json'} != set(members):raise ValueError('Incomplete root content manifest')
        elif 'SHA256SUMS.txt' in names:
            manifest_kind='SHA256SUMS.txt';seen=set()
            for line in z.read(manifest_kind).decode('utf-8-sig').splitlines():
                if not line.strip():continue
                expected,name=line.split('  ',1);name=name.lstrip('*');safe_member(name)
                if name in seen or hashes.get(name)!=expected.lower():raise ValueError(f'Member hash mismatch: {name}')
                seen.add(name);checked+=1
            if set(hashes)-{manifest_kind}!=seen:raise ValueError('Incomplete root sums manifest')
        return dict(members=len(infos),crc_verified=True,uncompressed_bytes=sum(sizes.values()),root_manifest=manifest_kind,member_hashes_verified=checked,scope='CRC of every file; hashes only when a recognized complete root manifest exists')


def acquire(row:dict,directory:Path,download:bool)->Path:
    name=row['filename'];safe_member(name)
    if '/' in name:raise ValueError('Expected basename')
    path=directory/name
    if not path.exists():
        if not download:raise FileNotFoundError(str(path))
        u=urllib.parse.urlparse(row['url'])
        if u.scheme!='https' or u.hostname!='github.com' or not u.path.startswith('/Kripta-Studios/e-jepa-ttc/releases/download/'):
            raise ValueError('Download URL outside fixed release scope')
        directory.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
        if part.exists():raise FileExistsError(f'Preserve prior partial download; inspect or rename: {part}')
        count=0
        req=urllib.request.Request(row['url'],headers={'User-Agent':'E-JEPA-TTC-audit'})
        with urllib.request.urlopen(req,timeout=60) as src,part.open('xb') as out:
            while b:=src.read(1024*1024):
                count+=len(b)
                if count>row['bytes']:raise ValueError('Remote asset exceeds pinned length')
                out.write(b)
            out.flush();os.fsync(out.fileno())
        if count!=row['bytes'] or file_hash(part)!=row['sha256']:raise ValueError('Downloaded asset does not match pin')
        part.rename(path)
    if path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']:
        raise ValueError(f'Existing asset differs; not overwritten: {path}')
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'RELEASE_ASSETS.json')
    p.add_argument('--download',action='store_true');p.add_argument('--deep',action='store_true');p.add_argument('--only',action='append');p.add_argument('--output',type=Path)
    args=p.parse_args();data=json.loads(args.manifest.read_text(encoding='utf-8'))
    rows=[r for r in data['archives'] if not args.only or r['filename'] in args.only]
    if not rows:raise ValueError('No selected assets in manifest')
    result=[]
    for row in rows:
        try:
            path=acquire(row,args.directory,args.download)
            receipt=dict(filename=row['filename'],status='VERIFIED_BYTES',sha256=row['sha256'])
            if args.deep:receipt.update(deep_check(path))
        except Exception as e:receipt=dict(filename=row['filename'],status='NOT_VERIFIED',error=f'{type(e).__name__}: {e}')
        result.append(receipt);print(json.dumps(receipt),flush=True)
    report=dict(assets=result,all_verified=all(r['status']=='VERIFIED_BYTES' for r in result))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if report['all_verified'] else 2
if __name__=='__main__':raise SystemExit(main())
