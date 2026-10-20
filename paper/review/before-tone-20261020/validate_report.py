"""Verify preserved scientific content and update metadata after the final render."""
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile
from lxml import etree
from PIL import Image
from pypdf import PdfReader
from manuscript import BLOCKS, REFERENCES, SOURCES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(rel):return json.loads((HERE/rel).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def source_json(key):return json.loads((ROOT/SOURCES[key]).read_text())
word_render=HERE/'qa/word-supplement-20261020'
tex_render=HERE/'qa/tex-supplement-20261020'
baseline=read('analysis/revision-baseline.json')
assert [b['tex'] for b in BLOCKS if b['kind']=='equation']==baseline['equations']
reference_keys=[r[0] for r in REFERENCES]
assert reference_keys[:len(baseline['reference_keys'])]==baseline['reference_keys']
assert len(reference_keys)==len(set(reference_keys))
plots=read('analysis/plot_data.json')
for k,v in baseline['plot_data_hashes'].items():assert digest(plots[k])==v,k
for rel,expected in baseline['source_sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
recomputed=read('analysis/verification.json')
for model,metrics in baseline['metrics'].items():
    for metric,expected in metrics.items():assert recomputed['metrics'][model][metric]==expected,(model,metric)

# Independently check the new probability claims using only Python's standard library.
frozen=json.loads((ROOT/'reports/deberta-laya-final-cpu-evaluation-20261018-v1/test-fused-predictions.json').read_text())
labels=frozen['true_labels']
probability_checks={}
for model,probabilities in frozen['positive_probabilities'].items():
    wrong=[max(p,1-p) for y,p in zip(labels,probabilities) if int(p>=.5)!=y]
    high=sum(c>=.95 for c in wrong)
    brier=math.fsum((p-y)**2 for y,p in zip(labels,probabilities))/len(labels)
    measured=recomputed['metrics'][model]
    assert high==measured['high_confidence_errors']
    assert abs(brier-measured['brier_positive'])<1e-12
    probability_checks[model]={'errors':len(wrong),'high_confidence_errors':high,
        'fraction':high/len(wrong),'error_mean_confidence':math.fsum(wrong)/len(wrong),'brier_positive':brier}
weight=json.loads((ROOT/'reports/laya-deberta-calibration-weight-cpu-20261018-v1/development-predictions.json').read_text())
a=[int(p>=.5)==y for y,p in zip(weight['true_labels'],weight['p_frozen_50_50'])]
b=[int(p>=.5)==y for y,p in zip(weight['true_labels'],weight['p_calibration_nll_weight'])]
fixed=sum(not x and y for x,y in zip(a,b));broken=sum(x and not y for x,y in zip(a,b))
p_value=min(1.,2*sum(math.comb(fixed+broken,k) for k in range(min(fixed,broken)+1))/2**(fixed+broken))
assert (fixed,broken)==(8,11) and abs(p_value-.6476058959960938)<1e-12

# Bind the newly added table and prose to the completed supplementary reports.
supp=source_json('supplement');diagnostic=source_json('suppweights')
robustness=source_json('supprobust');benchmark=source_json('suppcost');cpu=source_json('cpufull')
assert supp['state']==diagnostic['state']==robustness['state']=='completed'
assert not supp['official_test_accessed'] and not robustness['official_test_accessed']
assert not supp['best_seed_selection'] and not robustness['best_policy_selected']
expected_rows=[]
for key,name in [('deberta','DeBERTa'),('equal','等权融合'),('fixed_deberta_080','DeBERTa 80%'),('calibration_nll','校准拟合权重')]:
    values=[r['policies'][key]['metrics']['accuracy'] for r in diagnostic['results']]
    expected_rows.append([name]+[f'{100*x:.2f}%' for x in values]+[f'{100*math.fsum(values)/3:.3f}%'])
supp_table=next(b for b in BLOCKS if b['kind']=='table' and '三组完整重训' in b['caption'])
assert supp_table['rows']==expected_rows
for i,run in enumerate(supp['results']):
    for name,policy in [('deberta','deberta'),('fusion','equal')]:
        assert run['metrics'][name]==diagnostic['results'][i]['policies'][policy]['metrics']
    assert run['metrics']['fusion']['ece_15_equal_width_top_label']<run['metrics']['deberta']['ece_15_equal_width_top_label']
means={name:{metric:math.fsum(r['metrics'][name][metric] for r in supp['results'])/3
             for metric in ['accuracy','nll','ece_15_equal_width_top_label']}
       for name in ['deberta','fusion']}
assert f"{means['deberta']['nll']:.6f}"=='0.091995'
assert f"{means['fusion']['nll']:.6f}"=='0.089649'
assert f"{100*means['deberta']['ece_15_equal_width_top_label']:.3f}"=='1.328'
assert f"{100*means['fusion']['ece_15_equal_width_top_label']:.3f}"=='0.741'
intervals=robustness['development_conditional_intervals']
for key,rounded in [('fixed_deberta_080',[-.040,.160]),('calibration_nll',[-.027,.160])]:
    assert [round(v,3) for v in intervals[key]['percentile_95_pp']]==rounded
    assert intervals[key]['bootstrap_samples']==10000 and intervals[key]['unique_text_groups']==2487
cv=robustness['calibration_cross_validation']
assert [round(r['statistics']['equal']['delta_mean_pp'],3) for r in cv]==[-.325,-.600,-.100]
assert [round(r['statistics']['fixed_deberta_080']['delta_mean_pp'],3) for r in cv]==[0.,0.,0.]
assert [round(r['statistics']['calibration_nll']['delta_mean_pp'],3) for r in cv]==[0.,0.,-.150]
assert all(r['statistics']['fixed_deberta_080']['nll_mean']<r['statistics']['deberta']['nll_mean'] for r in cv)
assert (benchmark['sample_count'],benchmark['batch_size'],benchmark['warmup_repetitions'],benchmark['measured_repetitions'])==(64,8,2,5)
assert not benchmark['single_request_latency'] and not benchmark['test_accessed']
validation={r['id']:r for r in map(json.loads,(ROOT/'data/processed/validation.jsonl').read_text().splitlines())}
assert all(validation[s]['label']==0 for s in benchmark['common_sample_ids'])
assert [f"{r['reviews_per_second']:.3f}" for r in benchmark['results']]==['78.576','28.234','20.683']
assert [f"{r['peak_allocated_bytes']/2**30:.3f}" for r in benchmark['results']]==['2.447','3.499','5.586']
assert (cpu['count'],cpu['class_disagreement_count'],cpu['fusion_disagreement_count'])==(2500,1,0)
assert f"{cpu['max_abs_probability_difference']:.6f}"=='0.011909'
prose='\n'.join(b['text'] for b in BLOCKS if b['kind']=='p')
assert not re.search(r'我们|笔者|(?<!自)我|\b(?:we|our|ours|I)\b',prose)

ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
docx=HERE/'IMDB项目报告.docx'
with zipfile.ZipFile(docx) as z:
    body=z.read('word/document.xml');tree=etree.fromstring(body)
    paragraphs=[''.join(p.itertext()) for p in tree.xpath('//w:body/w:p',namespaces=ns)]
    # itertext is independent of python-docx and includes the inline hyperlink text.
    for block in BLOCKS:
        if block['kind']=='p':assert block['text'] in paragraphs,block['text'][:50]
    tables=tree.xpath('//w:body/w:tbl',namespaces=ns)
    expected_tables=[b for b in BLOCKS if b['kind']=='table']
    assert len(tables)==len(expected_tables)==7
    for node,block in zip(tables,expected_tables):
        rows=node.xpath('./w:tr',namespaces=ns)
        actual=[[''.join(cell.xpath('.//w:t/text()',namespaces=ns)) for cell in row.xpath('./w:tc',namespaces=ns)] for row in rows]
        assert actual==[block['headers']]+block['rows']
    links=tree.xpath('//w:hyperlink',namespaces=ns)
    assert len(links)==len(REFERENCES)
    media_hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
    for p in (HERE/'figures').glob('*.png'):assert hashlib.sha256(p.read_bytes()).hexdigest() in media_hashes,p.name
    for p in (HERE/'qa/equations').glob('equation-*.png'):assert hashlib.sha256(p.read_bytes()).hexdigest() in media_hashes,p.name
    images=tree.xpath('//*[local-name()="inline"]')
    assert len(images)==19
    reference_paras=[p for p in tree.xpath('//w:body/w:p',namespaces=ns) if re.match(r'^\[[1-9]\d*\]', ''.join(p.itertext()))]
    assert len(reference_paras)==len(REFERENCES)
    for p in reference_paras:
        ind=p.find('w:pPr/w:ind',ns);assert ind is not None and int(ind.get('{'+ns['w']+'}hanging'))>0

tex=(HERE/'main.tex').read_text();log=(HERE/'main.log').read_text()
for formula in baseline['equations']:assert formula in tex
assert re.findall(r'\\bibitem\{([^}]+)\}',tex)==reference_keys
assert set(re.findall(r'\\cite\{([^}]+)\}',tex))==set(reference_keys)
assert len(re.findall(r'\\label\{',tex))==sum(b['kind'] in ['equation','figure','table'] for b in BLOCKS)
assert not re.search(r'Overfull|Missing character|undefined',log)
for rel,entry in read('analysis/source_manifest.json').items():
    if 'path' in entry:assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
figure_files=list((HERE/'figures').glob('*.png'));assert len(figure_files)==11
architecture=read('analysis/architecture_sources.json')
for rel,expected in architecture['source_sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
assert architecture['laya_head']['norm_first'] is True
assert architecture['laya_head']['ffn_activation']=='ReLU'
assert (architecture['laya_encoder']['layers'],architecture['deberta']['layers'])==(28,24)
for p in figure_files:
    with Image.open(p) as im:
        assert all(abs(d-300)<1 for d in im.info['dpi'])
pdf=PdfReader(HERE/'main.pdf')
word_pdf=PdfReader(word_render/'IMDB项目报告.pdf')
pages=list(word_render.glob('page-*.png'))
assert len(pages)==len(word_pdf.pages)
assert len(list(tex_render.glob('page-*.png')))==len(pdf.pages)
identity=read('submission.json')
assert pdf.metadata.author==(identity.get('name') or '待填写') and pdf.metadata.subject

# Patch only document properties after rendering; the rendered body is unchanged.
with zipfile.ZipFile(docx) as z:entries=[(info,z.read(info.filename)) for info in z.infolist()]
with zipfile.ZipFile(docx,'w') as z:
    for info,data in entries:
        if info.filename=='docProps/app.xml':
            app=etree.fromstring(data)
            app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages').text=str(len(word_pdf.pages))
            data=etree.tostring(app,xml_declaration=True,encoding='UTF-8',standalone=True)
        z.writestr(info,data)
with zipfile.ZipFile(docx) as z:assert z.read('word/document.xml')==body

result={'tex_pages':len(pdf.pages),'word_render_pages':len(word_pdf.pages), 'references':len(REFERENCES),
    'figures':11,'tables':len(expected_tables),'equations':8,'docx_images':19,
    'architecture_source_config_and_code_sha256':'passed',
    'formula_and_reference_key_preservation':'passed','original_nine_figure_data_preservation':'passed',
    'frozen_metric_recomputation':'passed','new_probability_claims':probability_checks,
    'calibration_weight_paired_comparison':{'fixed':fixed,'broken':broken,'net':fixed-broken,'p':p_value},
    'source_sha256_check':'passed','tex_word_body_and_tables':'passed',
    'supplementary_table_and_probability_means':means,
    'supplementary_intervals_cross_validation_benchmark_and_cpu_parity':'passed',
    'first_person_prose_check':'passed',
    'word_embedded_figures_match_current_pngs':'passed','reference_hanging_indent_and_hyperlinks':'passed',
    'tex_missing_character_count':0,'tex_overfull_box_count':0,'tex_unresolved_references':0,
    'rendered_word_body_sha256':hashlib.sha256(body).hexdigest(),
    'layout_review':'pending latest full-page visual review',
    'word_render_directory':word_render.relative_to(ROOT).as_posix(),'tex_render_directory':tex_render.relative_to(ROOT).as_posix(),
    'compiler':'local XeLaTeX / TeX Live 2026','word_renderer':'bundled LibreOffice with explicit CJK font aliases',
    'target_word_wps_validation':'not performed','identity':'all fields deliberately left 待填写 at user request',
    'scope':'report revision incorporates already completed supplementary experiments; this revision performs no training, fitting or new test forward pass and changes no historical experimental reports'}
(HERE/'analysis/document_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
