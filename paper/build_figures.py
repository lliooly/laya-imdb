"""Recompute descriptive report statistics from frozen predictions; never fit models."""
from pathlib import Path
import csv, hashlib, json, math, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/imdb-paper-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
DATA = HERE / 'analysis'
DATA.mkdir(exist_ok=True)
sources = {}
def load(rel):
    p = ROOT / rel
    sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

S = load('reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json')
P = load('reports/deberta-laya-final-cpu-evaluation-20261018-v1/test-fused-predictions.json')
OLD = load('reports/laya-frozen-official-test-20261014-v1/test-actual.json')
D = load('reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/summary.json')
AUDIT = load('reports/data_audit.json')
for rel in ['reports/laya-head-control-analysis-20261015-v1/REPORT.md',
            'reports/laya-robust-loss-analysis-20261016-v1/REPORT.md',
            'reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md',
            'reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/REPORT.md']:
    sources[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
y = np.array(P['true_labels'])
probs = {k: np.array(v) for k,v in P['positive_probabilities'].items()}
assert len(y) == len(set(P['sample_ids'])) == 25000
assert P['sample_ids'] == OLD['sample_ids']
assert np.bincount(y).tolist() == [12500,12500]
names = ['original_laya_main','deberta_cpu','fusion_50_50']
labels = ['Laya ensemble','DeBERTa-v3','Heterogeneous fusion']
palette = {'rose':'#DEA3BB', 'blue':'#8CAFC9', 'teal':'#5FA9B9',
           'light_rose':'#F6CCDF', 'light_blue':'#AFD8F0', 'lilac':'#D9CEE4'}
colors = [palette['rose'], palette['blue'], palette['teal']]
heatmap = LinearSegmentedColormap.from_list('reference_blue', ['#FFFFFF','#AFD8F0','#5FA9B9'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white','axes.titleweight':'bold'})
figures=[]
def save(fig,name,description):
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=300)
    plt.close(fig)
    figures.append({'id':name,'description':description,'width_inches':7.0,'palette':palette,
                    'source_files':list(sources)})
def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den; delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-delta,center+delta]
def exact(a,b):
    n=a+b
    if n==0:return 1.
    return min(1.,2*sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)) for k in range(min(a,b)+1)))
metrics={}
for name,p in probs.items():
    pred=(p>=.5).astype(int); ok=pred==y
    cm=np.zeros((2,2),int)
    np.add.at(cm,(y,pred),1)
    pp=np.clip(p,1e-15,1-1e-15)
    nll=float(-np.mean(y*np.log(pp)+(1-y)*np.log(1-pp)))
    ref=S['groups']['all'][name]
    assert int(ok.sum())==ref['correct']
    assert abs(nll-ref['nll'])<1e-12
    assert cm.tolist()==ref['confusion_matrix_actual_rows_predicted_columns']
    confidence=np.maximum(p,1-p)
    high_errors=int(np.sum((~ok)&(confidence>=.95)))
    brier=float(np.mean((p-y)**2))
    assert high_errors==ref['errors_with_confidence_ge_095']
    assert abs(brier-ref['brier_positive'])<1e-12
    metrics[name]={'correct':int(ok.sum()),'count':len(y),'accuracy':float(ok.mean()),
      'macro_f1':ref['macro_f1'],'brier_positive':brier,'high_confidence_errors':high_errors,
      'high_confidence_error_fraction':high_errors/int((~ok).sum()),
      'error_mean_confidence':float(confidence[~ok].mean()),
      'nll':nll,'wilson95':wilson(int(ok.sum()),len(y)),'confusion_matrix':cm.tolist()}
fused=probs['fusion_50_50']>=.5
paired={}
for name in names[:2]:
    base=probs[name]>=.5
    fixed=int(np.sum((base!=y)&(fused==y)))
    broken=int(np.sum((base==y)&(fused!=y)))
    paired[name]={'fixed':fixed,'broken':broken,'net':fixed-broken,'p':exact(fixed,broken)}
with (DATA/'metrics.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['split','method','n','correct','accuracy','macro_f1','auroc','nll','ece'])
    for split,gs in [('official_test',S['groups']['all']),('development',D['groups']['all'])]:
        for key,v in gs.items():
            if 'accuracy' in v:w.writerow([split,key,v['count'],v['correct'],v['accuracy'],v['macro_f1'],v['auroc'],v['nll'],v['ece_15_equal_width_top_label']])

# Architecture diagram: schematic, not an empirical measurement.
fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');ax.axis('off');ax.set(xlim=(0,10),ylim=(0,5))
def box(x,y,w,h,text,color='#F6CCDF'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=color,ec='#425563',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#425563'})
box(.1,2,1.3,1,'Movie\nreview',palette['lilac'])
box(2,3,2.4,1.4,'Laya: baseline-s43\n+ combined-s42\nprefix / head-tail')
box(2,.5,2.4,1.2,'DeBERTa-v3-large\nown tokenizer\nhead-tail',palette['light_blue'])
box(5.1,3.1,1.7,1.2,'Calibrated T\nFrozen Laya weight\n0.702 / 0.298')
box(5.1,.5,1.7,1.2,'Temperature\nT = 1.137',palette['light_blue'])
box(7.5,1.8,2.2,1.6,'Fixed probability\naverage: 50 : 50\nthreshold = 0.5','#C9E8ED')
for a,b in [((1.4,2.6),(2,3.7)),((1.4,2.4),(2,1.1)),((4.4,3.7),(5.1,3.7)),((4.4,1.1),(5.1,1.1)),((6.8,3.7),(7.5,2.9)),((6.8,1.1),(7.5,2.2))]:arrow(a,b)
save(fig,'01_pipeline','Schematic of frozen three-checkpoint inference and probability fusion.')

lengths=np.array([s['review_tokens'] for s in OLD['stats']])
fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
axs[0].bar(['Gradient','Calibration','Development','Official test'],[22100,400,2500,25000],color=palette['teal'])
axs[0].tick_params(axis='x',rotation=25);axs[0].set(ylabel='Reviews',ylim=(0,29000),title='(a) Fixed partitions')
for i,v in enumerate([22100,400,2500,25000]):axs[0].text(i,v+400,f'{v:,}',ha='center',fontsize=8)
edges=[0,128,256,466,768,1024,2048,4096,8192,16384]
counts,bins=np.histogram(lengths,bins=edges)
assert counts.sum()==len(lengths), 'Histogram must include every test review'
axs[1].bar(np.arange(len(counts)),counts,color=[palette['rose']]*3+[palette['blue']]*6)
axs[1].set_xticks(np.arange(len(counts)),['0–128','128–256','256–466','466–768','768–1k','1k–2k','2k–4k','4k–8k','8k–16k'],rotation=55,fontsize=7)
axs[1].set(ylabel='Test reviews',title='(b) Original Laya review tokens')
axs[1].text(.97,.94,f'>466 tokens: {sum(lengths>466):,}\n({np.mean(lengths>466):.3%})',ha='right',va='top',transform=axs[1].transAxes,fontsize=9)
save(fig,'02_data','Fixed partition sizes and official-test token-length distribution. Bins are left-closed except the last.')

fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
for ax,group,title in [(axs[0],D['groups']['all'],'(a) Development: n = 2,500'),(axs[1],S['groups']['all'],'(b) Official test: n = 25,000')]:
    kk=['frozen_main','deberta','fusion_50_50'] if ax is axs[0] else names
    vv=[group[k]['correct'] for k in kk];n=group[kk[0]]['count'];acc=np.array(vv)/n*100
    ci=np.array([wilson(v,n) for v in vv])*100
    ax.errorbar(acc,range(3),xerr=[acc-ci[:,0],ci[:,1]-acc],fmt='none',ecolor='#555555',capsize=3)
    for i,c in enumerate(colors):ax.scatter(acc[i],i,c=c,s=45,zorder=3,marker=['o','s','^'][i],edgecolors='#425563',linewidths=.5)
    for i,a in enumerate(acc):ax.text(a+.04,i+.17,f'{a:.3f}%',fontsize=8)
    ax.set_yticks(range(3),['Laya','DeBERTa','Fusion']);ax.set_ylim(2.5,-.5)
    ax.set(xlim=(95.6,98.1),xlabel='Accuracy (%) — local scale',title=title)
save(fig,'03_accuracy','Wilson 95% intervals for fixed-model accuracy; they exclude training and model-selection uncertainty.')

fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
cm=np.array(metrics['fusion_50_50']['confusion_matrix']);axs[0].imshow(cm,cmap=heatmap,vmin=0,vmax=12500)
for i in range(2):
 for j in range(2):axs[0].text(j,i,f'{cm[i,j]:,}\n{cm[i,j]/12500:.2%}',ha='center',va='center',color='#172B35')
axs[0].set(xticks=[0,1],xticklabels=['Negative','Positive'],yticks=[0,1],yticklabels=['Negative','Positive'],xlabel='Predicted',ylabel='Actual',title='(a) Fusion confusion matrix')
axs[1].bar(['Fixed','Broken','Net'],[paired[names[0]][k] for k in ['fixed','broken','net']],color=[colors[2],colors[0],colors[1]])
for i,v in enumerate([265,94,171]):axs[1].text(i,v+4,str(v),ha='center')
axs[1].set(ylim=(0,310),ylabel='Reviews',title='(b) Fusion versus Laya')
save(fig,'04_errors','Full official-test confusion matrix and paired prediction transitions relative to old Laya ensemble.')

fig,axs=plt.subplots(1,3,figsize=(7,2.5),layout='constrained')
for ax,key,title,scale in zip(axs,['nll','ece_15_equal_width_top_label','brier_positive'],['NLL (lower is better)','ECE (%) / 15 bins','Brier / positive class'],[1,100,1]):
 vals=[S['groups']['all'][k][key]*scale for k in names]
 ax.bar(['Laya','DeBERTa','Fusion'],vals,color=colors);ax.tick_params(axis='x',rotation=30)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.4f}' if scale==1 else f'{v:.3f}',ha='center',fontsize=8)
 ax.set(ylim=(0,max(vals)*1.24),title=title)
save(fig,'05_probability','Official-test NLL, top-label ECE, and positive-class Brier score, with zero-based bar axes.')

fig,ax=plt.subplots(figsize=(7,2.9),layout='constrained');ax.plot([.5,1],[.5,1],'--',color='#777777',label='Perfect calibration')
for k,c,lab,mark in zip(names,colors,labels,['o','s','^']):
 bins=[b for b in S['groups']['all'][k]['reliability_bins'] if b['count']]
 ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],marker=mark,color=c,label=lab,markersize=4)
ax.set(xlim=(.49,1.01),ylim=(0,1.01),xlabel='Mean confidence in bin',ylabel='Empirical accuracy',title='Official-test reliability (15 equal-width confidence bins)');ax.legend(fontsize=8,loc='upper left')
assert all(0 <= b['accuracy'] <= 1 for k in names for b in S['groups']['all'][k]['reliability_bins'] if b['count'])
save(fig,'06_reliability','Nonempty original 15 equal-width top-label bins. Low-confidence bins have smaller denominators; see reliability_bins.csv.')
with (DATA/'reliability_bins.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','lower','upper','count','confidence','accuracy'])
 for k in names:
  for b in S['groups']['all'][k]['reliability_bins']:w.writerow([k,b['lower'],b['upper'],b['count'],b['confidence'],b['accuracy']])

fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
for i,k in enumerate(names):
 vals=[100*S['groups'][g][k]['accuracy'] for g in ['laya_complete','laya_truncated']]
 axs[0].plot(range(2),vals,marker=['o','s','^'][i],label=labels[i],color=colors[i])
axs[0].set(xticks=[0,1],xticklabels=['≤466 (n=21,287)','>466 (n=3,713)'],ylabel='Accuracy (%) — local scale',ylim=(96,97.6),title='(a) Official-test length strata');axs[0].legend(fontsize=7)
for i,(lab,vals,c,m) in enumerate([('CE',[96.24,96.16,96.56],colors[0],'o'),('0.5 CE + 0.5 GCE',[96.40,96.16,96.64],colors[1],'s')]):
 axs[1].plot([42,43,44],vals,marker=m,color=c,label=lab)
axs[1].set(xticks=[42,43,44],xlabel='Training seed',ylabel='Development accuracy (%)',ylim=(96,96.9),title='(b) Paired robust-loss study');axs[1].legend(fontsize=7)
save(fig,'07_strata_ablation','Test-length descriptive stratification and a separate three-seed development loss ablation; panels are different experiments.')

fig,axs=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
for lab,vals,c,m in [('Original head',[96.52,96.68,96.36],colors[0],'o'),('Linear head',[96.72,96.48,96.28],colors[1],'s')]:axs[0].plot([42,43,44],vals,marker=m,color=c,label=lab)
axs[0].set(xticks=[42,43,44],xlabel='Seed',ylabel='Development accuracy (%)',ylim=(96.1,96.9),title='(a) Paired decision-head control');axs[0].legend(fontsize=8)
axs[1].bar(['Long / >466','Short / ≤466','All'],[6,-6,0],color=[colors[2],colors[0],colors[1]]);axs[1].axhline(0,color='#555555',lw=.7)
axs[1].set(ylabel='Net correct: 2048 minus 512',ylim=(-8,8),title='(b) Context length: 512 vs 2048')
for i,v in enumerate([6,-6,0]):axs[1].text(i,v+(0.4 if v>=0 else -1),f'{v:+d}',ha='center')
save(fig,'08_controls','All seeds in the matched-head experiment; long/short net changes in the separate 512/2048 control.')

fig,axs=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
for k,c,lab,m in zip(names,colors,labels,['o','s','^']):
 rows=S['selective_at_calibration_only_thresholds'][k]
 axs[0].plot([r['actual_test_coverage']*100 for r in rows],[r['accuracy']*100 for r in rows],color=c,marker=m,label=lab)
axs[0].set(xlabel='Actual retained coverage (%)',ylabel='Retained accuracy (%)',ylim=(96,100.1),title='(a) Frozen calibration thresholds');axs[0].legend(fontsize=7)
axs[1].bar(['Single Laya','Two-Laya fusion'],[17.030,7.531],color=[colors[0],colors[2]])
for i,v in enumerate([17.030,7.531]):axs[1].text(i,v+.4,f'{v:.3f}',ha='center')
axs[1].set(ylim=(0,21),ylabel='Reviews / second',title='(b) MLX precise FP32 only')
save(fig,'09_selective_cost','Left: actual test coverage using pre-frozen calibration-only thresholds. Right: historical two-Laya MLX benchmark, excluding DeBERTa; not final-system throughput.')

# Existing historical run only; no inferred steps or smoothing.
curve=load('reports/laya-rtx4090-ce-combined-s43-epochs2-fp32-v1/epoch_validation_curve.json')
epochs=[r['epoch'] for r in curve['epochs']]
loss=[r['training_loss'] for r in curve['epochs']]
accuracy=[r['raw_metrics']['accuracy']*100 for r in curve['epochs']]
fig,axs=plt.subplots(1,2,figsize=(7,2.6),layout='constrained')
for ax,values,color,marker,title,ylabel,fmt in [
    (axs[0],loss,colors[0],'o','(a) Epoch-average training CE','Training loss','{:.4f}'),
    (axs[1],accuracy,colors[2],'s','(b) Development before calibration','Accuracy (%) — local scale','{:.2f}%')]:
    ax.plot(epochs,values,color=color,marker=marker,linewidth=1.5,markeredgecolor='#425563',markeredgewidth=.5)
    ax.set(xticks=epochs,xlim=(.75,2.25),xlabel='Epoch',ylabel=ylabel,title=title)
    ax.margins(y=.35)
    for x,v in zip(epochs,values):ax.annotate(fmt.format(v),(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
save(fig,'10_training','Historical combined-s43: two saved epoch observations; not the final combined-s42 fusion member. No smoothing; development n=2500 before final temperature calibration.')

from build_architecture import create_architecture
figures.append(create_architecture(HERE,palette))

result={'count':len(y),'metrics':metrics,'paired_fusion_vs':paired,
 'relative_error_reduction_vs_laya':(914-743)/914,'verified_against_frozen_summary':True,
 'sources':sources,'figures':figures,'analysis_type':'descriptive recomputation; no new training or fitting',
 'human_verification':'pending human review; automated verification is not a human sign-off'}
(DATA/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(DATA/'figure_manifest.json').write_text(json.dumps(figures,indent=2,ensure_ascii=False)+'\n')
(DATA/'plot_data.json').write_text(json.dumps({
 'partitions':{'gradient':22100,'calibration':400,'development':2500,'test':25000},
 'test_laya_token_histogram':{'edges':edges,'counts':counts.tolist(),'boundary':'left closed; last bin right closed'},
 'paired_fusion_vs':paired,'test_summary_groups':S['groups'],
 'development_summary_groups':D['groups'],
 'robust_loss':{'seed':[42,43,44],'ce_accuracy_percent':[96.24,96.16,96.56],'mixed_accuracy_percent':[96.40,96.16,96.64]},
 'heads':{'seed':[42,43,44],'original_accuracy_percent':[96.52,96.68,96.36],'linear_accuracy_percent':[96.72,96.48,96.28]},
 'context_net_correct':{'long':6,'short':-6,'all':0},
 'selective_at_frozen_calibration_thresholds':S['selective_at_calibration_only_thresholds'],
 'mlx_precise_fp32_reviews_per_second':{'single_laya':17.030,'two_laya':7.531},
 'full_precision_parameters':{'laya_internal_lambda':0.7020326697613102,
   'deberta_temperature':1.1374971945707235,
   'mlx_max_logits_difference':0.00035738945,'mlx_max_probability_difference':6.2526942e-5},
 'historical_combined_s43_curve':curve,
 'source_sha256':sources},indent=2,ensure_ascii=False)+'\n')
(DATA/'palette.json').write_text(json.dumps({'source':'User-supplied reference image; approximate pastel palette, not an exact colorimetric extraction',
 'colors':palette,'series':dict(zip(names,colors)),'redundant_encoding':'direct labels; line/point markers o, s, ^; dark text on light heatmap'},indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'metrics':metrics,'paired':paired,'figures':len(figures)},ensure_ascii=False))
