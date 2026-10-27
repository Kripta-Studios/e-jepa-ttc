"""Admit byte-identical relocated files while preserving historical science bindings."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

OPS=Path('/workspace/ejepa_event_operations')
OUT=OPS/'resume_20261027'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    receipt=OUT/'PHYSICAL_RELOCATION.json'
    assert digest(receipt)==os.environ['EJEPA_PHYSICAL_RELOCATION_SHA256']
    relocation=json.loads(receipt.read_text())
    assert relocation['scientific_files_modified'] is False
    base=Path('/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement')
    runtime=OPS/'event_downstream_runtime_v25'
    producer=OPS/'runtime_v12/operational/runpod_migration'
    sys.path[:0]=[str(base/'src'),str(base),str(runtime)]
    import operational.runpod_migration as migration
    migration.__path__.insert(0,str(producer))
    import scripts
    scripts.__path__=[str(runtime/'scripts'),str(producer.parents[1]/'scripts'),*scripts.__path__]
    from operational.runpod_migration import event_runner
    original_verify=event_runner.verify_seal

    def bound(root):
        root=Path(root).resolve(strict=True);row=relocation['roots'][str(root)]
        for key,sha_key in [('original_seal','original_seal_sha256'),('manifest','manifest_sha256'),('new_seal','new_seal_sha256')]:
            assert digest(Path(row[key]))==row[sha_key],key
        return row

    def verify_bundle(root,seal):
        row=bound(root)
        assert str(seal)==row['original_seal']
        return original_verify(root,Path(row['new_seal']))

    def verify_supplement(root_value,admission_value,*,root_expected,admission_schema,seal_schema,files):
        root=Path(root_value).resolve(strict=True);row=bound(root)
        admission=json.loads(Path(admission_value).read_text());seal=json.loads(Path(row['new_seal']).read_text())
        assert root==root_expected
        assert admission['schema']==admission_schema and admission['status']=='VERIFIED'
        assert admission['root']==str(root) and admission['immutable_seal_sha256']==row['original_seal_sha256']
        assert admission['manifest_sha256']==row['manifest_sha256']
        assert seal['schema']==seal_schema and seal['status']=='SEALED' and seal['root']==str(root)
        assert seal['manifest_sha256']==row['manifest_sha256'] and len(seal['files'])==files
        for item in seal['files']:
            p=root/item['relative'];s=p.stat()
            assert not p.is_symlink() and p.is_file() and not s.st_mode & 0o222
            assert (s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino,s.st_mode)==tuple(item[k] for k in ['bytes','mtime_ns','device','inode','mode']),str(p)
        return root

    event_runner.verify_seal=verify_bundle
    event_runner._verified_supplement=verify_supplement
    if '--preflight' in sys.argv:
        manifest=verify_bundle(Path('/opt/ejepa_event_bundle'),Path('/opt/ejepa_event_bundle.SEAL.json'))
        event_runner._mapping(manifest,Path('/opt/ejepa_event_bundle'))
        print(json.dumps({'status':'PASS','scientific_updates':0,'relocated_roots':len(relocation['roots'])}))
        return
    path=runtime/'scripts/runpod_event_queue_bootstrap_v7.py'
    assert digest(path)==os.environ['EJEPA_ORIGINAL_V25_BOOTSTRAP_SHA256']
    spec=importlib.util.spec_from_file_location('ejepa_original_v25_relocated',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert digest(Path(__file__).with_name('runpod_event_context_overlap_v1.py')) == os.environ['EJEPA_CONTEXT_OVERLAP_SHA256']
    from runpod_event_context_overlap_v1 import install, HeadFinished
    original_load = module.load
    def load_with_overlap(name, path):
        loaded = original_load(name, path)
        if name == 'ejepa_event_queue_v1_v7':
            install(loaded)
        return loaded
    module.load = load_with_overlap
    try:
        module.main()
    except HeadFinished:
        return

if __name__=='__main__':main()
