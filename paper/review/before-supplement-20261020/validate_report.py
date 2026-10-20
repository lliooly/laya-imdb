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
    assert len(tables)==len(expected_tables)==6
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
assert len(re.findall(r'\\label\{',tex))==25
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
word_pdf=PdfReader(HERE/'qa/word-review/IMDB项目报告.pdf')
pages=list((HERE/'qa/word-review').glob('page-*.png'))
assert len(pages)==len(word_pdf.pages)
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
    'figures':11,'tables':6,'equations':8,'docx_images':19,
    'architecture_source_config_and_code_sha256':'passed',
    'formula_and_reference_key_preservation':'passed','original_nine_figure_data_preservation':'passed',
    'frozen_metric_recomputation':'passed','new_probability_claims':probability_checks,
    'calibration_weight_paired_comparison':{'fixed':fixed,'broken':broken,'net':fixed-broken,'p':p_value},
    'source_sha256_check':'passed','tex_word_body_and_tables':'passed',
    'word_embedded_figures_match_current_pngs':'passed','reference_hanging_indent_and_hyperlinks':'passed',
    'tex_missing_character_count':0,'tex_overfull_box_count':0,'tex_unresolved_references':0,
    'rendered_word_body_sha256':hashlib.sha256(body).hexdigest(),
    'layout_review':'pending latest full-page visual review',
    'word_render_directory':'paper/qa/word-review','tex_render_directory':'paper/qa/tex-review',
    'compiler':'local XeLaTeX / TeX Live 2026','word_renderer':'bundled LibreOffice with explicit CJK font aliases',
    'target_word_wps_validation':'not performed','identity':'all fields deliberately left 待填写 at user request',
    'scope':'report only; no historical report edits, training, fitting, new test forward pass, videos, deployment or project source repack'}
(HERE/'analysis/document_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
