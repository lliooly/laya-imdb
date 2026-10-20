"""Read frozen evidence and independently recompute review findings. No model fitting."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sources = {}


def read(relative, jsonl=False):
    path = ROOT / relative
    raw = path.read_bytes()
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)


def metrics(labels, probabilities):
    n = len(labels)
    assert n == len(probabilities) and n > 0
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    cm = [[0, 0], [0, 0]]
    bins = [[] for _ in range(15)]
    for y, p in zip(labels, probabilities):
        cm[y][int(p >= .5)] += 1
        confidence = max(p, 1 - p)
        bins[min(14, int(confidence * 15))].append((int((p >= .5) == bool(y)), confidence))
    f1 = sum(2 * cm[c][c] / (2 * cm[c][c] + cm[1-c][c] + cm[c][1-c]) for c in (0, 1)) / 2
    ordered = sorted(zip(probabilities, labels))
    rank_sum = 0.
    i = 0
    while i < n:
        j = i + 1
        while j < n and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in ordered[i:j])
        i = j
    positive = sum(labels)
    auroc = (rank_sum - positive * (positive + 1) / 2) / (positive * (n - positive))
    wrong_conf = [max(p, 1-p) for y, p in zip(labels, probabilities) if int(p >= .5) != y]
    return dict(count=n, correct=cm[0][0] + cm[1][1], accuracy=(cm[0][0]+cm[1][1])/n,
                macro_f1=f1, auroc=auroc, confusion_matrix=cm,
                nll=-math.fsum(math.log(max(1e-15, min(1-1e-15, p if y else 1-p)))
                                   for y, p in zip(labels, probabilities))/n,
                brier_positive=math.fsum((p-y)**2 for y, p in zip(labels, probabilities))/n,
                ece=math.fsum(abs(sum(y for y, _ in row)-math.fsum(p for _, p in row))
                              for row in bins)/n,
                high_confidence_errors=sum(c >= .95 for c in wrong_conf),
                error_mean_confidence=math.fsum(wrong_conf)/len(wrong_conf))


prediction = read('reports/deberta-laya-final-cpu-evaluation-20261018-v1/test-fused-predictions.json')
summary = read('reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json')
ids, labels, probabilities = (prediction[k] for k in ('sample_ids', 'true_labels', 'positive_probabilities'))
assert len(ids) == len(set(ids)) == len(labels) == 25000
assert Counter(labels) == {0: 12500, 1: 12500}
computed = {name: metrics(labels, p) for name, p in probabilities.items()}
for name, measured in computed.items():
    expected = summary['groups']['all'][name]
    for key in ['accuracy', 'correct', 'macro_f1', 'auroc', 'nll', 'brier_positive']:
        assert abs(measured[key] - expected[key]) < 1e-12, (name, key)
    assert abs(measured['ece']-expected['ece_15_equal_width_top_label']) < 1e-12
    assert measured['confusion_matrix'] == expected['confusion_matrix_actual_rows_predicted_columns']
assert max(abs(p - (a+b)/2) for p, a, b in zip(probabilities['fusion_50_50'],
    probabilities['original_laya_main'], probabilities['deberta_cpu'])) < 1e-15
assert not any(p == .5 for values in probabilities.values() for p in values)
paired = {}
for name in ['original_laya_main', 'deberta_cpu']:
    a = [int(p >= .5) == y for y, p in zip(labels, probabilities[name])]
    b = [int(p >= .5) == y for y, p in zip(labels, probabilities['fusion_50_50'])]
    fixed, broken = sum(not x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))
    n = fixed + broken
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(fixed, broken)+1)) / 2**n)
    paired[name] = dict(fixed=fixed, broken=broken, net=fixed-broken, exact_two_sided_mcnemar_p=p)

train = read('data/processed/train.jsonl', True)
test = read('data/processed/test.jsonl', True)
development = read('data/processed/validation.jsonl', True)
manifest = read('reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json')
byid = {row['id']: row for row in train}
gradient_ids, calibration_ids = manifest['gradient_train_ids'], manifest['calibration_ids']
gt = {byid[sid]['text'] for sid in gradient_ids}
ct = {byid[sid]['text'] for sid in calibration_ids}
tt = {r['text'] for r in test}
assert [r['id'] for r in test] == ids and [r['label'] for r in test] == labels
overlap_ids = [r['id'] for r in test if r['text'] in gt]
duplicates = dict(official_training_test_unique_text_overlap=len({r['text'] for r in train+development} & tt),
                  gradient_test_unique_text_overlap=len(gt & tt),
                  gradient_test_affected_test_rows=len(overlap_ids),
                  gradient_test_affected_test_ids=overlap_ids,
                  test_duplicate_extra_rows=len(test)-len(tt),
                  gradient_calibration_unique_text_overlap=len(gt & ct),
                  gradient_development_unique_text_overlap=len(gt & {r['text'] for r in development}),
                  calibration_label_counts=dict(Counter(byid[sid]['label'] for sid in calibration_ids)))
assert duplicates['official_training_test_unique_text_overlap'] == 123
assert duplicates['gradient_test_affected_test_rows'] == 113
assert duplicates['gradient_calibration_unique_text_overlap'] == 0
assert duplicates['gradient_development_unique_text_overlap'] == 0
admission = read('reports/deberta-laya-final-cpu-evaluation-20261018-v1/admission.json')
parameters = {name: read(path)['native_temperature'] for name, path in [
    ('baseline_s43_temperature', 'reports/laya-posthoc-baseline-s43-fp32-v1/metadata.json'),
    ('combined_s42_temperature', 'reports/laya-posthoc-combined-s42-fp32-v1/metadata.json')]}
thresholds = read('reports/deberta-laya-final-cpu-evaluation-20261018-v1/calibration-thresholds.json')
selective = []
for target, threshold in thresholds['fusion_50_50'].items():
    kept = [(y, p) for y, p in zip(labels, probabilities['fusion_50_50']) if max(p, 1-p) >= threshold]
    selective.append(dict(target=target, threshold=threshold, retained=len(kept),
                          correct=sum(int(p >= .5) == y for y, p in kept)))
snapshot_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'original').iterdir()}
lowest_bin = next(b for b in summary['groups']['all']['deberta_cpu']['reliability_bins'] if b['count'])
assert lowest_bin['count'] == 37 and abs(lowest_bin['accuracy'] - 12/37) < 1e-12
display_check = dict(method='deberta_cpu', count=37, correct=12, accuracy=lowest_bin['accuracy'],
                     confidence=lowest_bin['confidence'], previous_ymin=.45,
                     previous_point_clipped=lowest_bin['accuracy'] < .45, corrected_ylim=[0, 1.01])
result = dict(review_date='2026-10-19', scope='local evidence audit; no training, fitting or new model forward pass',
              metrics=computed, paired=paired, duplicate_audit=duplicates, admission=admission,
              temperature_parameters=parameters, selective=selective, reliability_display_check=display_check,
              source_sha256=sources, original_snapshot_sha256=snapshot_hashes,
              human_signoff='not recorded')
(HERE/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(metrics=computed, paired=paired,
                     duplicates={k: v for k, v in duplicates.items() if not k.endswith('_ids')},
                     selective=selective), ensure_ascii=False, indent=2))
