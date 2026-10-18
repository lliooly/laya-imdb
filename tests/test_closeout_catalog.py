"""Verify catalog boundaries, static analysis, and link checking on fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_closeout_catalog as catalog


class CloseoutCatalogTests(unittest.TestCase):
    def test_source_selection_excludes_weights_cache_and_virtualenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in catalog.ROOT_FILES:
                (root/name).write_text('metadata')
            for folder in catalog.SOURCE_DIRS:
                (root/folder).mkdir(parents=True,exist_ok=True)
            for name in ['scripts/entry.py','environments/x/uv.lock','environments/x/.venv/secret.py',
                         'rust-tokenizer/target/binary','scripts/__pycache__/cache.py',
                         'artifacts/model.safetensors','data/test.jsonl']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            with patch.object(catalog,'ROOT',root),patch.object(catalog,'DOC',root/'docs/closeout'):
                selected={str(p.relative_to(root)) for p in catalog.source_files()}
            self.assertIn('scripts/entry.py',selected)
            self.assertIn('environments/x/uv.lock',selected)
            self.assertFalse(any('secret' in p or 'target' in p or '__pycache__' in p or p.startswith(('data/','artifacts/')) for p in selected))

    def test_static_imports_do_not_execute_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir()
            (root/'scripts/helper.py').write_text('value=1')
            p=root/'scripts/entry.py'
            p.write_text('"""Fixture entry."""\nimport helper\nraise RuntimeError("must not run")\ndef main(): pass\n')
            with patch.object(catalog,'ROOT',root):
                info=catalog.module_info(p)
            self.assertEqual(info['local_imports'],['helper'])
            self.assertTrue(info['has_main'])
            self.assertEqual(info['description'],'Fixture entry.')

    def test_link_check_ignores_external_anchors_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'index.md'
            (root/'a b.md').write_text('target')
            p.write_text('[ok](a%20b.md#part) [web](https://example.com)\n```sh\n[x](not-real)\n```\n[missing](absent.md)')
            with patch.object(catalog,'ROOT',root):
                missing=catalog.check_links([p])
            self.assertEqual(missing,[dict(file='index.md',target='absent.md')])

    def test_digest_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(b'first');a=catalog.digest(p)
            p.write_bytes(b'second');self.assertNotEqual(a,catalog.digest(p))


if __name__=='__main__':
    unittest.main()
