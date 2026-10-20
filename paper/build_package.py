"""Package the report, not the project source or videos; verify every ZIP entry."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
files=[HERE/name for name in ['main.tex','main.pdf','IMDB项目报告.docx','README.md',
    'REVISION_NOTES.md','submission.json','manuscript.py','build_report.py','build_figures.py','build_architecture.py','build_package.py','validate_report.py']]
files+=sorted((HERE/'figures').glob('*.pdf'))+sorted((HERE/'figures').glob('*.png'))
files+=sorted((HERE/'figures').glob('*.svg'))
files += [HERE/name for name in ['review/REVIEW.md', 'review/audit.py', 'review/audit.json',
    'review/references.json', 'review/original/main.tex']]
manifest_path=HERE/'analysis/package_manifest.json'
files+=sorted(p for p in (HERE/'analysis').iterdir() if p.is_file() and p!=manifest_path)
assert all(p.is_file() for p in files)
records=[{'path':p.relative_to(HERE).as_posix(),'bytes':p.stat().st_size,
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
records.append({'path':'analysis/package_manifest.json','sha256':None,
                'integrity':'Self-entry is listed but not hashed to avoid self-reference.'})
manifest={'archive_root':'IMDB报告','archive_entry_count':len(records),
          'hashed_payload_count':len(records)-1,'files':records}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
files.append(manifest_path)
archive=HERE/'IMDB报告与TeX源码.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,'IMDB报告/'+p.relative_to(HERE).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==manifest['archive_entry_count']
    assert set(z.namelist())=={'IMDB报告/'+r['path'] for r in records}
    for r in records:
        if r['sha256']:
            assert hashlib.sha256(z.read('IMDB报告/'+r['path'])).hexdigest()==r['sha256']
print(json.dumps({'archive':str(archive),'entries':len(records),'sha256_verified_payloads':len(records)-1},ensure_ascii=False))
