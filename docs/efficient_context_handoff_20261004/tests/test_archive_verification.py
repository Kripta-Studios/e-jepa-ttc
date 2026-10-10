import unittest,tempfile,zipfile,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_release_assets import deep_check,safe_member

class ArchiveVerification(unittest.TestCase):
    def build(self,d,items):
        p=Path(d)/'a.zip'
        with zipfile.ZipFile(p,'w') as z:
            for name,b in items:z.writestr(name,b)
        return p
    def test_good(self):
        b=b'test';manifest={'members':{'a.txt':{'bytes':4,'sha256':hashlib.sha256(b).hexdigest()}}}
        with tempfile.TemporaryDirectory() as d:
            p=self.build(d,[('a.txt',b),('CONTENT_MANIFEST.json',json.dumps(manifest))]);self.assertEqual(deep_check(p)['member_hashes_verified'],1)
    def test_bad_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.build(d,[('a',b'x'),('CONTENT_MANIFEST.json',json.dumps({'members':{'a':{'sha256':'0'*64}}}))])
            with self.assertRaises(ValueError):deep_check(p)
    def test_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.build(d,[('a',b'x'),('CONTENT_MANIFEST.json','{"members":{}}')])
            with self.assertRaises(ValueError):deep_check(p)
    def test_traversal(self):
        with self.assertRaises(ValueError):safe_member('../x')
    def test_drive(self):
        with self.assertRaises(ValueError):safe_member('C:/x')
    def test_backslash(self):
        with self.assertRaises(ValueError):safe_member('a\\b')
    def test_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.build(d,[('a',b'xxxx')])
            with self.assertRaises(ValueError):deep_check(p,max_uncompressed=3)
    def test_crc_scope_without_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.build(d,[('a',b'x')]);r=deep_check(p);self.assertTrue(r['crc_verified']);self.assertEqual(r['member_hashes_verified'],0)
if __name__=='__main__':unittest.main()
