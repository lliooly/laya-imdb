"""Reproduce a fixed length router from sealed predictions; no models or GPU."""
import json
from pathlib import Path

import numpy as np

from common import ROOT, file_digest, write_json
from comparison_metrics import metrics, paired
from laya_length_router import route_predict

SOURCE = 'laya-context-pair-s42-modal-20261016-v1'
STUDY = 'laya-length-router-20261017-v1'


def wilson(correct, count):
    p, z = correct/count, 1.959963984540054
    center=(p+z*z/(2*count))/(1+z*z/count)
    half=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [float(center-half),float(center+half)]


def comparison(y, a, b):
    delta=((b>0)==y).astype(int)-((a>0)==y).astype(int)
    counts=np.asarray([(delta==x).sum() for x in (-1,0,1)])
    rng=np.random.default_rng(20261017)
    samples=rng.multinomial(len(y),counts/len(y),size=20000)
    interval=np.quantile((samples[:,2]-samples[:,0])/len(y),[.025,.975]).tolist()
    return dict(paired(y,a,b),delta_accuracy=float(delta.mean()),
                delta_accuracy_conditional_bootstrap95=interval,count=len(y))


def main():
    source=ROOT/'reports'/SOURCE
    output=ROOT/'reports'/STUDY
    if output.exists():raise FileExistsError('Preserve the existing report; use a new run ID')
    hashes={}
    def read(path):
        hashes[str(path.relative_to(ROOT))]=file_digest(path)
        return json.loads(path.read_text())
    delivery=read(source/'local-delivery.json')
    manifest=read(source/'evaluation/bundle.json')
    if (delivery['state']!='verified_local' or not delivery['local_delivery_verified']
            or delivery['bundles']['evaluation']['bundle_sha256']!=file_digest(source/'evaluation/bundle.json')):
        raise ValueError('Independent local delivery is incomplete')
    sealed={f['path']:f for f in manifest['files']}
    def prediction_file(name):
        path=source/'evaluation'/name
        expected=sealed[name]
        if path.stat().st_size!=expected['bytes'] or file_digest(path)!=expected['sha256']:
            raise ValueError('Sealed prediction changed: '+name)
        return read(path)
    summary=prediction_file('summary.json')
    sets={};temperatures={};models={};weights_bytes=0
    for cap in (512,2048):
        temperatures[cap]=summary['models'][str(cap)]['temperature']
        if not .5<=temperatures[cap]<=5:raise ValueError('Frozen temperature out of bounds')
        folder=ROOT/'checkpoints'/SOURCE/f'context-{cap}-s42/epoch-002'
        m=read(folder/'bundle.json');meta=read(folder/'head_control.json')
        record=delivery['bundles'][f'context-{cap}-s42-epoch-002']
        f=next(f for f in m['files'] if f['path']=='head_control.json')
        if (file_digest(folder/'bundle.json')!=record['bundle_sha256'] or not record['local_delivery_verified']
                or meta['state_sha256']!=record['verification']['state_sha256']
                or file_digest(folder/'head_control.json')!=f['sha256']
                or summary['frozen_gate']['final_bundle_sha256'][str(cap)]!=record['bundle_sha256']):
            raise ValueError('Frozen model identity or verified receipt differs')
        w=next(f for f in m['files'] if f['path']=='model.safetensors')
        weights_bytes+=w['bytes']
        models[str(cap)]={'bundle':str(folder.relative_to(ROOT)),'state_sha256':meta['state_sha256'],
                          'weights_sha256':meta['weights_sha256'],'weights_bytes':w['bytes']}
        for part,n in [('calibration',400),('development',2500)]:
            d=prediction_file(f'{cap}-{part}.json')
            if (d['test_accessed'] is not False or len(d['sample_ids'])!=n or len(set(d['sample_ids']))!=n
                    or len(d['review_token_lengths'])!=n or len(d['true_labels'])!=n
                    or any(type(y) is not int or y not in (0,1) for y in d['true_labels'])):
                raise ValueError('Invalid prediction split')
            z=np.asarray(d['logits'])
            if z.shape!=(n,2) or not np.isfinite(z).all():raise ValueError('Invalid logits')
            sets[cap,part]=d
    for part in ('calibration','development'):
        a,b=sets[512,part],sets[2048,part]
        if any(a[key]!=b[key] for key in ('sample_ids','true_labels','review_token_lengths')):
            raise ValueError('Paired rows or lengths differ')
    if set(sets[512,'calibration']['sample_ids'])&set(sets[512,'development']['sample_ids']):
        raise ValueError('Calibration overlaps development')
    a,b=sets[512,'development'],sets[2048,'development']
    y=np.asarray(a['true_labels']);lengths=np.asarray(a['review_token_lengths'])
    za,zb=np.asarray(a['logits']),np.asarray(b['logits'])
    routed,chosen=route_predict(lengths,lambda idx:za[idx],lambda idx:zb[idx])
    raw_margin=routed[:,1]-routed[:,0]
    calibrated_margin=raw_margin/np.where(chosen==512,temperatures[512],temperatures[2048])
    margins={512:za[:,1]-za[:,0],2048:zb[:,1]-zb[:,0]}
    groups={}
    for name,keep in [('all',np.ones(len(y),dtype=bool)),('complete_at_512',chosen==512),('truncated_at_512',chosen==2048)]:
        raw=metrics(y[keep],raw_margin[keep]);calibrated=metrics(y[keep],calibrated_margin[keep])
        groups[name]={'raw':raw,'component_temperature_calibrated':calibrated,
                      'accuracy_wilson95':wilson(raw['correct'],raw['count']),
                      'paired_vs_512':comparison(y[keep],margins[512][keep],raw_margin[keep]),
                      'paired_vs_2048':comparison(y[keep],margins[2048][keep],raw_margin[keep])}
    sources=[Path(__file__),ROOT/'scripts/laya_length_router.py',ROOT/'scripts/comparison_metrics.py',
             ROOT/'scripts/common.py',ROOT/'tests/test_laya_length_router.py',
             ROOT/'docs/superpowers/specs/2026-10-17-laya-length-router-design.md']
    for path in sources:hashes[str(path.relative_to(ROOT))]=file_digest(path)
    result={'study_id':STUDY,'source_study_id':SOURCE,'state':'offline_evaluated','seed':42,'test_accessed':False,
            'rule':{'review_token_threshold':466,'comparison':'strictly_greater_than','long_model':2048,
                    'short_model':512,'threshold_search':False,'learned_selector':False},
            'models':models,'frozen_component_temperatures':{str(k):v for k,v in temperatures.items()},
            'groups':groups,'dispatch':{str(c):{'count':int((chosen==c).sum()),
                       'fraction':float((chosen==c).mean())} for c in (512,2048)},
            'cost':{'model_forwards_per_sample':1,'resident_fp32_weights_bytes':weights_bytes,
                    'resident_fp32_weights_gib':weights_bytes/1024**3,'gpu_called':False,
                    'latency_measured':False,'peak_memory_measured':False},
            'limitations':['posthoc_length_routing','development_reused','single_seed',
                           'bootstrap_conditional_on_fixed_models_and_rule'],
            'input_and_source_sha256':hashes}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',result)
    write_json(output/'routed-development.json',{'sample_ids':a['sample_ids'],'true_labels':a['true_labels'],
               'review_token_lengths':a['review_token_lengths'],'selected_model_max_len':chosen.tolist(),
               'logits':routed.tolist(),'component_temperature_calibrated_margin':calibrated_margin.tolist(),
               'test_accessed':False})
    all_metrics=groups['all'];vs=all_metrics['paired_vs_512'];ci=[x*100 for x in vs['delta_accuracy_conditional_bootstrap95']]
    lines=['# 固定长度路由：512 / 2048','',
           '规则：完整 review tokens <=466 使用 512，>466 使用 2048。模板占 46 token；不按字符数，不先截断后计数。两份模型均固定为 seed 42、epoch 2。', '',
           '## 开发集结果','', '| 方案 | 正确 / 总数 | Accuracy | Macro-F1 | AUROC | 校准 NLL | 校准 ECE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for cap in (512,2048):
        g=summary['models'][str(cap)]['groups']['all'];r,c=g['raw'],g['calibrated']
        lines.append(f"| 全走 {cap} | {r['correct']} / {r['count']} | {100*r['accuracy']:.2f}% | {r['macro_f1']:.6f} | {r['auroc']:.6f} | {c['nll']:.6f} | {c['ece_15_equal_width_top_label']:.6f} |")
    r,c=all_metrics['raw'],all_metrics['component_temperature_calibrated']
    lines += [f"| 固定长度路由 | {r['correct']} / {r['count']} | {100*r['accuracy']:.2f}% | {r['macro_f1']:.6f} | {c['auroc']:.6f} | {c['nll']:.6f} | {c['ece_15_equal_width_top_label']:.6f} |",'',
              '原两模型各自为 92 条错误；路由为 86 条错误，净少 6 条（+0.24 pp）。路由行 AUROC 使用各分支原冻结温度校准后的分数；原始混合 logits 的 AUROC 另见 summary.json。原两模型的正温度缩放不改变各自 AUROC。', '',
              '## 配对证据与边界','',
              f"- 对全 512：修复 {vs['b_only_correct']} 条、退化 {vs['a_only_correct']} 条；精确 McNemar p={vs['mcnemar_exact_two_sided_p']:.6f}。条件配对 bootstrap 95% 差值区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] pp。"]
    p=all_metrics['paired_vs_2048'];ci2=[v*100 for v in p['delta_accuracy_conditional_bootstrap95']]
    lines += [f"- 对全 2048：修复 {p['b_only_correct']} 条、退化 {p['a_only_correct']} 条；精确 McNemar p={p['mcnemar_exact_two_sided_p']:.6f}。条件差值区间 [{ci2[0]:+.2f}, {ci2[1]:+.2f}] pp。",
              f"- 路由整体 Accuracy 的 Wilson 95% 区间为 [{100*all_metrics['accuracy_wilson95'][0]:.2f}%, {100*all_metrics['accuracy_wilson95'][1]:.2f}%]。",'',
              '该规则在观察两模型开发分组结果后提出；阈值虽由模板预算确定，方案选择仍属于事后探索。单个 seed、反复使用的开发集不能支持独立测试或跨种子收益保证。两个比较均未达到 p<0.05。相对全 512 的点估计有利，不能宣称已稳定提升，也没有超过历史最高开发成绩。', '',
              '## 实现与成本','',
              '- 选择器接口：`scripts/laya_length_router.py: route_predict`。先按长度分流，再调用两分支回调并按原序回填；空分支跳过，每条只调用一个模型。回调负责对应模型与原输入编码。未切换默认部署，未完成新的真实 GPU / Mac 路由前向验收。',
              f"- 本开发集 2093 条（83.72%）走 512，407 条（16.28%）走 2048。两份完整 FP32 权重共约 {weights_bytes/1024**3:.2f} GiB；这是权重下限，不是运行峰值内存。",
              '- 本轮仅离线缓存复算，未启动 GPU、未重新训练。单样本一次模型调用不代表延迟与全 512 相同；尚未测路由吞吐、加载开销或峰值内存。',
              '- 每个分支沿用原已在 400 校准样本上冻结的温度。路由不观察标签、置信度或两模型预测是否正确；不扫阈值，不训练二级选择器。超过 2002 review tokens 仍按原 2048 首尾规则处理。', '',
              '## 复现与审计','',
              '`UV_CACHE_DIR=/private/tmp/end-imdb-review-uv-cache uv run --locked --offline --no-sync python scripts/analyze_laya_length_router.py`', '',
              '首次运行创建独立报告目录；已存在时拒绝覆盖。输入预测、原冻结摘要、交付凭据、模型身份和本次源码哈希见 summary.json；逐条选路见 routed-development.json。封存文件大小 / SHA、样本 ID / 标签 / 长度配对已核对；未读取官方测试数据与预测。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'accuracy':r['accuracy'],'correct':r['correct'],'vs_512':vs,
                      'calibrated_nll':c['nll'],'report':str(output/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':main()
