"""Build reproducible source/docs/report catalogs without model or dataset access."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/closeout'
RUN = 'project-closeout-20261018-v1'
OUT = ROOT/'reports'/RUN
ART = ROOT/'artifacts'/RUN
SKIP = {'.git', '.venv', '__pycache__', 'target', '.DS_Store'}
SOURCE_DIRS = ('scripts','tests','configs','docs','environments','archive','rust-tokenizer')
ROOT_FILES = ('README.md','AGENTS.md','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml','uv.lock','.python-version','.gitattributes','.gitignore')
GENERATED = {'catalog.json','metrics.json','metrics.md','files.md','reports.md','code.md','designs.md','local-assets.json','research-ledger.md'}
SCIENCE_TESTS = {'test_bow_nb_branch.py','test_laya_deberta_calibration_weight.py'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def files(folder):
    for parent, dirs, names in os.walk(ROOT/folder):
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv'))
        for name in sorted(names):
            p=Path(parent)/name
            if name not in SKIP and not p.is_symlink():
                yield p


def source_files():
    result=[ROOT/p for p in ROOT_FILES]
    for folder in SOURCE_DIRS:
        result.extend(p for p in files(folder) if not (p.parent==DOC and p.name in GENERATED))
    return sorted(result)


def link(path, label=None):
    p=Path(path)
    relative=os.path.relpath(ROOT/p,DOC)
    return '['+(label or str(p))+']('+relative.replace(' ','%20')+')'


def category(name):
    name=name.lower()
    groups=[('收尾与汇总',('closeout','build_all_experiment','consolidat')),
      ('异构模型',('deberta',)),('全序列查询头',('token_query',)),
      ('头部与表示',('head_control','head_study','frozen_probe')),
      ('训练数据与语义',('semantic','resampl','review','train_candidate','hardcase')),
      ('损失与调度',('robust','rdrop','regularization','optimization','warmup','combined')),
      ('长文与决策策略',('context','length_router','long_','window','posthoc','decision','checkpoint_average')),
      ('融合、评估与拒答',('fusion','risk_coverage','frozen_test','frozen_official','probability','bow_nb','final_cpu','calibration_weight')),
      ('后端与数值',('mlx','parity','trace','determin','numerics','short_drift')),
      ('数据与分词',('data','tokeniz','cache_tokens')),
      ('训练基础设施',('modal','delivery','receive','cuda','checkpoint','training','train_laya')),
      ('早期模型与课程',('day','baseline','modernbert','model-comparison'))]
    return next((label for label,keywords in groups if any(k in name for k in keywords)), '共享实现与历史辅助')


def module_info(path):
    tree=ast.parse(path.read_text(),filename=str(path))
    modules={p.stem for p in (ROOT/'scripts').glob('*.py')}
    imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    return dict(description=(ast.get_docstring(tree) or '无模块说明；按文件名与调用关系定位。').split('\n')[0],
                local_imports=sorted(imports & modules),
                has_main=any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='main' for n in tree.body))


def metric_rows():
    old=read('reports/all-experiments-metrics-20261016-v1/summary.json')
    rows=[]
    for r in old['rows']:
        rows.append(dict(r,id='H-'+r['id'],catalog_origin='historical_20261016'))
    sources=read('docs/closeout/studies.json')['metric_sources']
    for source in sources:
        d=read(source['file'])
        for leaf in source['entries']:
            keys=leaf['keys']
            metric=d
            for key in keys:
                metric=metric[key]
            assert 'accuracy' in metric and 'count' in metric
            assert metric['count']==(25000 if source['split']=='official_test' else 2500)
            if 'expected' in leaf:
                assert abs(metric['accuracy']-leaf['expected'])<1e-12
            rows.append(dict(id=f"N-{len(rows)-len(old['rows'])+1:03d}",group=source['group'],
                method=leaf['label'],seed=leaf.get('seed',source.get('seed','见协议')),
                protocol=source['protocol'],split=source['split'],coverage=1.,
                metrics=metric,source=dict(file=source['file'],keys=keys),note=source['note'],catalog_origin='added_20261018'))
    return rows


def assets():
    result={}
    for folder in ('artifacts','checkpoints','delivery'):
        entries=[]
        for p in sorted((ROOT/folder).iterdir()):
            if not p.is_dir():
                continue
            owned=[]
            for parent, dirs, names in os.walk(p):
                dirs[:]=[d for d in dirs if d not in SKIP and d not in ('data','raw')]
                for name in names:
                    q=Path(parent)/name
                    if q.is_symlink():
                        continue
                    if q.suffix in ('.safetensors','.pt','.pth','.ckpt'):
                        owned.append(dict(path=str(q.relative_to(ROOT)),bytes=q.stat().st_size))
            entries.append(dict(directory=str(p.relative_to(ROOT)),weights=owned,
                status='metadata_only_not_loaded_or_rehashed'))
        result[folder]=entries
    return dict(asset_directories=result,data_and_cache_contents_opened=False,
        interpretation='存在不等于已通过加载；交付准入只引用各实验 local-delivery / verification 收据。L2缺失权重不由预测恢复。')


def build():
    DOC.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    studies=read('docs/closeout/studies.json')
    for study in studies['studies']:
        for key in ('evidence','code','design'):
            for p in study.get(key,[]):
                assert (ROOT/p).is_file(), p
    ledger=['# 研究台账：设计、实现、结果与验证','','更新：2026-10-18。24类覆盖研究脉络，类数不等于训练次数。历史43条思路与123条指标快照保持原文；补充记录见指标索引。','',
            '| 类别 | 状态 | 核心边界 |','|---|---|---|']
    for s in studies['studies']:
        ledger.append(f"| {s['id']} {s['title']} | {s['status']} | {s['note']} |")
    for s in studies['studies']:
        ledger+=['','## '+s['id']+' '+s['title'],'','状态：'+s['status']+'。'+s['note'],'']
        for key,label in [('design','设计/计划'),('code','实现'),('evidence','最终结果/核验')]:
            ledger += [label+'：'+('；'.join(link(p) for p in s[key]) if s[key] else '以对应报告原协议为准；不补造独立规格。'),'']
    ledger+=['## 尚未完成','','端到端新头、领域预训练、无标签自训练、第三模型和ModernBERT-base/Laya-RLCD公平迁移未完成；L2缺失权重、历史数据使用及后端差异保留披露。旧提案存在不表示已执行。','',
             '[设计与配置](designs.md) · [代码与测试](code.md) · [报告与验证](reports.md) · [指标](metrics.md)']
    (DOC/'research-ledger.md').write_text('\n'.join(ledger)+'\n')
    inventory=[]
    for p in source_files():
        r=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p),category=category(p.stem))
        if p.suffix=='.py':
            r.update(module_info(p))
        inventory.append(r)
    report_inventory=[]
    for p in files('reports'):
        if OUT in p.parents:
            continue
        report_inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
            role='历史快照，按原路径保留',category=category(str(p.relative_to(ROOT)))))
    rows=metric_rows()
    write(DOC/'metrics.json',dict(as_of='2026-10-18',rows=rows,
        historical_rows_preserved=123,new_rows=len(rows)-123,
        record_count_is_not_training_count=True,no_new_scoring_or_prediction=True,
        source_sha256={p:digest(ROOT/p) for p in sorted({r['source']['file'] for r in rows})}))
    write(DOC/'catalog.json',dict(schema=1,as_of='2026-10-18',source_files=inventory,
        historical_report_files=report_inventory,
        counts=dict(source_files=len(inventory),historical_report_files=len(report_inventory),metric_records=len(rows)),
        report_inventory_method='paths and file sizes only; individual predictions not opened',
        generated_files_excluded_from_own_digest=sorted(GENERATED)))
    write(DOC/'local-assets.json',assets())
    text=['# 实验指标完整索引','','保留 2026-10-16 的全部 123 条记录，并补入后续已完成实验。记录包括同权重多视图、后端复核、分组和选择性覆盖；条数不是独立训练次数。',
          '', '开发集、正式测试和选择性覆盖必须分别比较。所有新增行来自冻结摘要，未重新计算预测。JSON 保留原指标、来源文件及键路径。', '',
          '| ID | 分区 / 覆盖 | 方法 | Accuracy | 数量 | 协议 | 证据 |','|---|---|---|---:|---:|---|---|']
    for r in rows:
        text.append(f"| {r['id']} | {r['split']} / {r['coverage']:.3f} | {r['method']} | {100*r['metrics']['accuracy']:.3f}% | {r['metrics'].get('count','—')} | {r['protocol']} | {link(r['source']['file'],'摘要')} |")
    (DOC/'metrics.md').write_text('\n'.join(text).rstrip()+'\n')
    text=['# 源码与测试索引','','路径保持不变；模块说明由 AST 静态读取，不导入或执行训练入口。CLI 列只表示定义了 main，执行前仍需阅读对应协议。局部导入包括函数内部的静态导入，不保证运行时全部经过。','']
    for group in sorted({r['category'] for r in inventory if r['path'].startswith(('scripts/','tests/'))}):
        text+=['## '+group,'','| 文件 | 说明 | CLI | 本地依赖 |','|---|---|---|---|']
        for r in inventory:
            if r['category']==group and r['path'].startswith(('scripts/','tests/')):
                desc=r.get('description','').replace('|','／')
                text.append(f"| {link(r['path'])} | {desc} | {'有' if r.get('has_main') else '—'} | {', '.join(r.get('local_imports',[]))} |")
        text.append('')
    (DOC/'code.md').write_text('\n'.join(text).rstrip()+'\n')
    text=['# 设计、计划与配置索引','','日期文件记录当时方案，存在文件不表示已执行。执行结论以收尾总览和最终评估为准。','']
    for prefix,title in [('docs/superpowers/specs/','设计规格'),('docs/superpowers/plans/','实现计划'),('configs/','实验配置与模型锁'),('environments/','依赖环境'),('docs/archive/','历史归档')]:
        text+=['## '+title,'']
        text += ['- '+link(r['path']) for r in inventory if r['path'].startswith(prefix)]
        text.append('')
    (DOC/'designs.md').write_text('\n'.join(text).rstrip()+'\n')
    text=['# 报告与验证证据索引','','顶层 REPORT 可能是启动快照；若存在 evaluation/REPORT.md，优先阅读最终评估。完整文件元数据见 catalog.json；不改写历史 manifest、哈希或 reproduction。','']
    runs=sorted({Path(r['path']).parts[1] for r in report_inventory if len(Path(r['path']).parts)>2})
    for run in runs:
        prefix='reports/'+run+'/'
        selected=[r for r in report_inventory if r['path'].startswith(prefix)]
        evidence=[r['path'] for r in selected if Path(r['path']).name in ('REPORT.md','README.md','summary.json') or 'verification' in Path(r['path']).name or Path(r['path']).name in ('local-delivery.json','completion.json') or Path(r['path']).suffix in ('.png','.svg','.pdf')]
        evidence.sort(key=lambda p:('evaluation/REPORT.md' not in p, p))
        text += ['## '+run,'',f"{category(run)}；本地文件 {len(selected)} 份。",'']
        text += ['- '+link(p) for p in evidence]
        if not evidence:
            text.append('- 原始材料见 '+link('reports/'+run,'报告目录')+'；不由文件名推断实验成功。')
        text.append('')
    (DOC/'reports.md').write_text('\n'.join(text).rstrip()+'\n')
    text=['# 文件清单','','源码、文档、配置、测试和环境锁逐文件记录 SHA256；历史报告只盘点路径和大小。权重只做存在性盘点，不读取数据目录、缓存或测试逐样本预测。','',
        '| 类别 | 文件数 |','|---|---:|']
    text += [f'| {k} | {v} |' for k,v in sorted(Counter(Path(r['path']).parts[0] for r in inventory).items())]
    text += ['', '机器清单：[catalog.json](catalog.json)；本地权重元数据：[local-assets.json](local-assets.json)。', '',
        '路径盘点不证明权重可加载；最终准入仍以各实验保存的验证收据为准。']
    (DOC/'files.md').write_text('\n'.join(text).rstrip()+'\n')
    catalog=read('docs/closeout/catalog.json')
    catalog['generated_sha256']={name:digest(DOC/name) for name in sorted(GENERATED-{'catalog.json'})}
    write(DOC/'catalog.json',catalog)
    print(json.dumps(dict(state='catalog_built',source_files=len(inventory),report_files=len(report_inventory),metric_records=len(rows))))


def check_links(paths):
    missing=[]
    for path in paths:
        text=re.sub(r'```.*?```','',path.read_text(),flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            target=target.split('#')[0].replace('%20',' ')
            if not target or re.match(r'[a-z]+:',target):
                continue
            if not (path.parent/target).resolve().exists():
                missing.append(dict(file=str(path.relative_to(ROOT)),target=target))
    return missing


def verify():
    catalog=read('docs/closeout/catalog.json')
    for r in catalog['source_files']:
        assert digest(ROOT/r['path'])==r['sha256'],r['path']
    for name,sha in catalog['generated_sha256'].items():
        assert digest(DOC/name)==sha,name
    for r in catalog['historical_report_files']:
        assert (ROOT/r['path']).stat().st_size==r['bytes'],r['path']
    metrics=read('docs/closeout/metrics.json')
    for path,sha in metrics['source_sha256'].items():
        assert digest(ROOT/path)==sha,path
    assert len({r['id'] for r in metrics['rows']})==len(metrics['rows'])
    assert metrics['historical_rows_preserved']==123
    targets=list(DOC.glob('*.md'))+[ROOT/'README.md',ROOT/'docs/README.md',ROOT/'docs/experiments/current-status.md',ROOT/'docs/experiments/improvement-roadmap.md',ROOT/'docs/guides/inference.md']
    missing=check_links(targets)
    assert not missing, missing
    checks=dict(source_hashes_verified=True,report_sizes_verified=True,metric_provenance_verified=True,
        latest_document_links_verified=True,python_ast_parse_passed=True,
        official_test_inference_or_scoring=False,models_loaded=False)
    write(OUT/'verification.json',checks)
    print(json.dumps(checks))


def run_tests():
    OUT.mkdir(parents=True,exist_ok=True)
    def one(path):
        scientific=path.name in SCIENCE_TESTS
        cmd=['uv','run']+(['--project','environments/bow-cpu'] if scientific else [])
        cmd+=['--locked','--offline','--no-sync','python','-m','unittest','discover','-s','tests','-p',path.name,'-v']
        env=os.environ.copy()
        env['UV_CACHE_DIR']='/private/tmp/end-imdb-closeout-uv-cache'
        env['PYTHONPATH']=str(ROOT/'scripts')
        env['TOKENIZERS_PARALLELISM']='false'
        env['OMP_NUM_THREADS']='2'
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        output=result.stdout+'\n'+result.stderr
        (OUT/(path.stem+'.log')).write_text(output)
        count=re.search(r'Ran (\d+) tests?',output)
        item=dict(module=str(path.relative_to(ROOT)),environment='bow-cpu' if scientific else 'main',
            passed=result.returncode==0,returncode=result.returncode,tests=int(count[1]) if count else 0,
            command=cmd)
        print(json.dumps(item),flush=True)
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(one,sorted((ROOT/'tests').glob('test_*.py'))))
    receipt=dict(passed=all(r['passed'] for r in results),modules=len(results),tests=sum(r['tests'] for r in results),
        results=results,isolated_process_per_module=True,reason='Several scientific tests install permanent Python audit hooks.')
    write(OUT/'tests.json',receipt)
    if not receipt['passed']:
        raise RuntimeError('Some checks failed; inspect per-module logs')


def package():
    verify()
    ART.mkdir(parents=True,exist_ok=True)
    archive=ART/'end-imdb-source-and-research-20261018-v1.zip'
    if archive.exists():
        raise FileExistsError('Preserve existing archive')
    paths=set(source_files())
    paths.update(p for p in DOC.iterdir() if p.is_file())
    for p in files('reports'):
        if (p.name in ('REPORT.md','README.md','summary.json','local-delivery.json','completion.json')
                or 'verification' in p.name or p.suffix in ('.png','.svg','.pdf')
                or ('reproduction' in p.parts and p.suffix in ('.py','.toml','.lock','.json','.md'))):
            paths.add(p)
    paths.update(ROOT/p for p in read('docs/closeout/metrics.json')['source_sha256'])
    for study in read('docs/closeout/studies.json')['studies']:
        paths.update(ROOT/p for p in study['evidence'])
    for entry in read('docs/closeout/studies.json').get('additional_materials',[]):
        paths.update(ROOT/p for p in entry['files'])
    # Preserve the fixed data split identity, without copying review text files.
    paths.add(ROOT/'reports/split_manifest.json')
    for p in OUT.glob('*.json'):
        paths.add(p)
    manifest=[]
    for p in sorted(paths):
        rel=p.relative_to(ROOT)
        assert not any(part in SKIP for part in rel.parts)
        assert rel.parts[0] not in ('data','artifacts','checkpoints','delivery','.cache')
        assert p.suffix not in ('.npy','.npz','.safetensors','.pt','.pth','.ckpt','.bin')
        manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=digest(p)))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for entry in manifest:
            info=zipfile.ZipInfo(entry['path'],date_time=(2026,10,18,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            z.writestr(info,(ROOT/entry['path']).read_bytes())
        z.writestr('PACKAGE-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(manifest)+1
        for entry in manifest:
            assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    receipt=dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),
        files=len(manifest),manifest_embedded=True,every_entry_rehashed=True,
        excluded=['data directories','weights','virtual environments','caches','individual prediction exports','full recovery bundles'],
        note='Historical reports can contain identifiers and limited excerpts; source licenses remain in force. This is not a standalone model bundle.')
    write(OUT/'package-receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('build','verify','test','package'))
    args=parser.parse_args()
    dict(build=build,verify=verify,test=run_tests,package=package)[args.mode]()
