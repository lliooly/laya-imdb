#!/usr/bin/env python3
"""Accuracy-headroom quantitative analysis on frozen development artifacts.

Scope (team task-1):
  * inventory frozen development (2500-row) prediction views, align by sample_id,
    verify split / row count / label agreement, dedupe by prediction vector;
  * per-view accuracy, oracle union, 68-error marginal fixability, pairwise flips;
  * reproduce the 97.40 % (2435/2500) CPU candidate from frozen probabilities;
  * fit constrained ensembles on the fixed 400-row calibration split only and
    evaluate them once on the frozen 2500-row development split;
  * export an evidence dump for the 68 fusion errors used by the qualitative audit.

Hard constraints:
  * no training, no model inference, no official-test per-sample access. Files whose
    path contains "test" are skipped and counted; only aggregate official-test counts
    already printed in published reports are copied as constants.
  * writes only under reports/accuracy-improvement-research-20261018-v1/,
    artifacts/accuracy-improvement-research-20261018-v1/ and this script.

Run:
  UV_CACHE_DIR=/private/tmp/end-imdb-headroom-uv-cache \
  uv run --locked --offline --no-sync python scripts/analyze_accuracy_headroom.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_PATH = os.path.join(ROOT, "data/processed/validation.jsonl")
TRAIN_PATH = os.path.join(ROOT, "data/processed/train.jsonl")
CAL_MANIFEST = os.path.join(ROOT, "reports/laya-deberta-calibration-weight-cpu-20261018-v1/frozen.json")
DEBERTA_EVAL_DIR = os.path.join(ROOT, "reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation")
FUSION_ANALYSIS_DIR = os.path.join(ROOT, "reports/deberta-laya-final-analysis-20261018-v1")
V2_DIR = os.path.join(ROOT, "reports/final-cpu-alternatives-20261018-v2")
BOW_DIR = os.path.join(ROOT, "reports/bow-nb-complementarity-cpu-20261018-v1")
TOKENQ_DIR = os.path.join(ROOT, "reports/laya-token-query-mps-fp32-20261018-v1")
CTX_DIR = os.path.join(ROOT, "reports/laya-context-pair-s42-modal-20261016-v1/evaluation")
REVISED_DIR = os.path.join(ROOT, "reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation")
LONGFUSION_DIR = os.path.join(ROOT, "reports/laya-long-fusion-bc-20261017-v1")
ROUTER_DIR = os.path.join(ROOT, "reports/laya-length-router-20261017-v1")
LDW_DIR = os.path.join(ROOT, "reports/laya-deberta-calibration-weight-cpu-20261018-v1")

# Published aggregate official-test counts, copied from existing reports. No per-sample
# official-test prediction file is opened anywhere in this script.
PUBLISHED_OFFICIAL_TEST = {
    "source": "reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md",
    "original_laya_main": {"correct": 24086, "n": 25000, "accuracy": 0.96344},
    "deberta_cpu": {"correct": 24233, "n": 25000, "accuracy": 0.96932},
    "fusion_50_50": {"correct": 24257, "n": 25000, "accuracy": 0.97028},
}

V2_FORMULA = (0.5319458737546259, 0.5727065867827512, -0.1161031700961201)

# Report-published development accuracies used only as an output cross-check
# (values copied from the reports listed in the analysis report; not used for selection).
REPORTED_DEV_CORRECT = {
    "baseline-s42/head_tail": (2402, "all-experiments-metrics M022 (96.08%)"),
    "combined-s42/head_tail": (2417, "all-experiments-metrics M028 (96.68%)"),
    "baseline-s43/head_tail": (2421, "all-experiments-metrics M025 (96.84%)"),
    "combined-s43/head_tail": (2416, "all-experiments-metrics M031 (96.64%)"),
    "prefix-trained/head_tail": (2416, "all-experiments-metrics M047 (96.64%)"),
    "random-window-trained/head_tail": (2411, "all-experiments-metrics M052 (96.44%)"),
    "checkpoint-average-s42/head_tail": (2416, "all-experiments-metrics M056 (96.64%)"),
    "laya-main-fusion": (2424, "deberta-laya-final-analysis REPORT (96.96%)"),
    "fusion-50_50(laya+deberta)": (2432, "deberta-heterogeneous REPORT (97.28%)"),
    "deberta-s42": (2426, "deberta-heterogeneous REPORT (97.04%)"),
    "laya-new-512(orig-data)": (2408, "laya-context-pair REPORT (96.32%)"),
    "laya-new-2048(orig-data)": (2408, "laya-context-pair REPORT (96.32%)"),
    "laya-revised-labels-512": (2410, "laya-reviewed-data REPORT (96.40%)"),
    "cpu-alt/regularized_logit": (2435, "final-cpu-alternatives-v2 REPORT (97.40%)"),
    "cpu-alt/long_context": (2432, "final-cpu-alternatives-v2 REPORT (97.28%)"),
    "cpu-alt/mlp_dropout1": (2431, "final-cpu-alternatives-v2 REPORT (97.24%)"),
    "cpu-alt/mlp_dropout4": (2431, "final-cpu-alternatives-v2 REPORT (97.24%)"),
    "bow-nb/tfidf_lr/fusion": (2433, "bow-nb REPORT (97.32%)"),
    "bow-nb/nb_ngram_lr/fusion": (2432, "bow-nb REPORT (97.28%)"),
    "bow-nb/tfidf_lr/calibrated": (2278, "bow-nb REPORT (91.12%)"),
    "bow-nb/nb_ngram_lr/calibrated": (2306, "bow-nb REPORT (92.24%)"),
    "token-query/mean/branch": (2421, "laya-token-query REPORT (96.84%)"),
    "token-query/dense/branch": (2421, "laya-token-query REPORT (96.84%)"),
    "token-query/top32/branch": (2420, "laya-token-query REPORT (96.80%)"),
    "mac-one-epoch-s42": (2394, "all-experiments-metrics M002 (95.76%)"),
    "mac-two-epoch-s42": (2424, "all-experiments-metrics M005 (96.96%)"),
    "head-control-original-s42": (2413, "laya-head-control M060 (96.52%)"),
    "head-control-linear-s42": (2418, "laya-head-control M061 (96.72%)"),
    "frozen-probe-s43/full_model": (2421, "laya-frozen-probe (96.84%)"),
    "frozen-probe-s43/encoder": (2417, "laya-frozen-probe (96.68%)"),
    "frozen-probe-s43/post_head": (2420, "laya-frozen-probe (96.80%)"),
}
# ---------------------------------------------------------------------------
# AI qualitative classification of the 68 errors of the frozen 50:50 fusion.
#
# THIS IS AI QUALITATIVE CLASSIFICATION, NOT INDEPENDENT HUMAN ADJUDICATION.
# It was produced by reading the full development review texts (data/processed/
# validation.jsonl, also dumped to fusion-68-error-evidence.json), the common-47
# materials (reports/deberta-laya-final-analysis-20261018-v1/common-47.md and
# CASE-STUDY.md) and the frozen margins/truncation flags. Classes:
#   a_label_conflict   strong in-text evidence (own score / explicit verdict) against label
#   b_ambiguity        evaluative ambiguity, irony, or quality-vs-enjoyment axis
#   c_truncation       both tokenizers truncated; decisive first-person verdict missing
#   d_model_error      text clearly supports the label; model(s) wrong
#   e_uncertain        cannot be adjudicated from the text
# Rationale strings document the evidence line used; classes were assigned once and
# were NOT optimised against any accuracy number.
FUSION_68_CLASSIFICATION: dict[str, tuple[str, str]] = {
    # --- a_label_conflict (11)
    "train/neg/3155_1.txt": ("a_label_conflict", "H006: 全文'Great movie...'肯定，无否定"),
    "train/neg/7397_4.txt": ("a_label_conflict", "H013: 'i liked this movie... Parabens'"),
    "train/neg/7647_1.txt": ("a_label_conflict", "H014: 'on the whole it's pretty good... Everybody acted well'"),
    "train/neg/7927_1.txt": ("a_label_conflict", "H016: 'AWESOME... RULEZZZ... genius'"),
    "train/neg/9409_4.txt": ("a_label_conflict", "H020: 'I liked this show!... Best Spidey show ever'"),
    "train/pos/1230_10.txt": ("a_label_conflict", "H028: 'almost Adam's worst film... bottom 100'"),
    "train/pos/1454_7.txt": ("a_label_conflict", "H029: 'I wish I could like this more, but no.'"),
    "train/pos/3160_7.txt": ("a_label_conflict", "H034: '4 out of 10... isn't good at all'"),
    "train/pos/7770_7.txt": ("a_label_conflict", "H044: 'For me this is not enough... see Aventurera'"),
    "train/pos/8647_7.txt": ("a_label_conflict", "H047: 'decidedly average... wouldn't bother to see it again'"),
    "train/pos/1710_7.txt": ("a_label_conflict", "自称'pathetic copy/flop/No credit'，标签为pos"),
    # --- c_truncation (1)
    "train/pos/12208_8.txt": ("c_truncation", "H026: Laya/DeBERTa 双截断；可见文本是Welles对配乐批评，缺第一人称总评"),
    # --- d_model_error (7)
    "train/neg/10566_3.txt": ("d_model_error", "H001: 当前作品'slow/confusing/poor'清晰，系列好评干扰"),
    "train/neg/810_1.txt": ("d_model_error", "H018: 'crap/lesser Belushi/lacks of talent'清晰贬损"),
    "train/neg/6081_4.txt": ("d_model_error", "'poor storyline/poor dialogue'为作者总评"),
    "train/neg/4632_4.txt": ("d_model_error", "'fails to achieve... still too long'为负向总评"),
    "train/pos/5506_10.txt": ("d_model_error", "为纪录片辩护、反驳0分评论"),
    "train/pos/10091_7.txt": ("d_model_error", "'I enjoyed it despite its predictability'"),
    "train/pos/9_7.txt": ("d_model_error", "'still pleasant... *** from ****'评分与标签同侧"),
    # --- e_uncertain (3)
    "train/neg/12240_1.txt": ("e_uncertain", "H005: 纯事件叙事，无影片评价"),
    "train/pos/8527_8.txt": ("e_uncertain", "H046: 隐晦玩笑/离题，无法裁定"),
    "train/pos/12122_7.txt": ("e_uncertain", "作者自述'feel unable to give an honest opinion'"),
    # --- b_ambiguity (46)
    "train/neg/11233_2.txt": ("b_ambiguity", "H002: 98美分价格反讽"),
    "train/neg/11636_3.txt": ("b_ambiguity", "H003: 为黑色幽默辩护，褒贬混杂"),
    "train/neg/12042_2.txt": ("b_ambiguity", "H004: 长篇Brad Pitt明星魅力，离题"),
    "train/neg/3182_1.txt": ("b_ambiguity", "H007: 情节复述+戏谑口吻"),
    "train/neg/4014_2.txt": ("b_ambiguity", "H008: 'clever but...'混杂"),
    "train/neg/641_4.txt": ("b_ambiguity", "H009: 'we enjoyed our moment'混杂"),
    "train/neg/6424_3.txt": ("b_ambiguity", "H010: 'mediocre cute'但肯定Perlman"),
    "train/neg/6608_1.txt": ("b_ambiguity", "H011: 破碎英语，褒贬混杂"),
    "train/neg/7026_4.txt": ("b_ambiguity", "H012: 'Beautiful but flawed'"),
    "train/neg/7719_3.txt": ("b_ambiguity", "H015: 'quite amusing... recommend as rental'"),
    "train/neg/792_1.txt": ("b_ambiguity", "H017: 反讽式'改变人生'"),
    "train/neg/8756_4.txt": ("b_ambiguity", "H019: 情节复述，'okay for a rainy day'"),
    "train/neg/9896_4.txt": ("b_ambiguity", "H021: 制作豪华/续集乏味混杂"),
    "train/pos/10016_8.txt": ("b_ambiguity", "H022: 'intriguing'但制作仓促"),
    "train/pos/11192_10.txt": ("b_ambiguity", "H023: 坎普式烂片欣赏"),
    "train/pos/11447_7.txt": ("b_ambiguity", "H024: 美学肯定+中途退场"),
    "train/pos/11853_7.txt": ("b_ambiguity", "H025: 业余制作但'watchable'"),
    "train/pos/12230_9.txt": ("b_ambiguity", "H027: 'entertaining piece of garbage'评价轴错位"),
    "train/pos/1665_7.txt": ("b_ambiguity", "H030: 反讽式'Highly recommended'"),
    "train/pos/1762_7.txt": ("b_ambiguity", "H031: 双截断；'painfully funny'与道德保留并存"),
    "train/pos/2204_10.txt": ("b_ambiguity", "H032: '5 out of 10'边界评分"),
    "train/pos/2229_7.txt": ("b_ambiguity", "H033: 'not a masterpiece... points for trying'"),
    "train/pos/3263_7.txt": ("b_ambiguity", "H035: '*** out of ****'与负向文字冲突"),
    "train/pos/4185_8.txt": ("b_ambiguity", "H036: 烂片娱乐价值，自评8"),
    "train/pos/5930_7.txt": ("b_ambiguity", "H037: 'roses among manure'"),
    "train/pos/5988_7.txt": ("b_ambiguity", "H038: 历史兴趣大于影片评价"),
    "train/pos/6093_8.txt": ("b_ambiguity", "H039: 'cracking up...odd'"),
    "train/pos/6372_7.txt": ("b_ambiguity", "H040: 'Beautiful to watch, but...'"),
    "train/pos/6588_9.txt": ("b_ambiguity", "H041: 'good dialog'但整体批评"),
    "train/pos/6892_8.txt": ("b_ambiguity", "H042: 怀旧比较，评价对象切换"),
    "train/pos/7584_7.txt": ("b_ambiguity", "H043: B级片戏谑欣赏"),
    "train/pos/8085_7.txt": ("b_ambiguity", "H045: 'okay film... will watch again'"),
    "train/neg/12360_4.txt": ("b_ambiguity", "'many wonderful aspects'与虐待主题混杂"),
    "train/neg/2291_4.txt": ("b_ambiguity", "传记叙述为主，末句才批评"),
    "train/neg/2572_1.txt": ("b_ambiguity", "反讽：把烂片称为'perfect Saturday night'"),
    "train/neg/5026_3.txt": ("b_ambiguity", "折磨式观影+结尾'relief'"),
    "train/neg/6939_4.txt": ("b_ambiguity", "'needed more sugar'混合评价"),
    "train/neg/7010_4.txt": ("b_ambiguity", "'muddled gem'：享受糟糕电影"),
    "train/neg/9359_4.txt": ("b_ambiguity", "对影片意识形态立场的疑虑，非质量评价"),
    "train/pos/1264_10.txt": ("b_ambiguity", "长篇死刑/司法主题议论，非影片质量总评"),
    "train/pos/1831_10.txt": ("b_ambiguity", "'awful'但'still an entertainer'"),
    "train/pos/1838_10.txt": ("b_ambiguity", "大量肯定+结尾'not appetizing'"),
    "train/pos/2435_8.txt": ("b_ambiguity", "被动画感动但称'depressing/too much'"),
    "train/pos/5624_7.txt": ("b_ambiguity", "表演肯定+结尾'fell short'"),
    "train/pos/6_10.txt": ("b_ambiguity", "演员好但技术层面'fails'"),
    "train/pos/9579_8.txt": ("b_ambiguity", "与Hamilton其他作品比较，褒贬混杂"),
}
CLASS_LABELS = {
    "a_label_conflict": "标签—正文冲突候选",
    "b_ambiguity": "评价歧义/反讽/评价轴错位",
    "c_truncation": "截断缺失关键证据",
    "d_model_error": "文本支持标签的模型误判",
    "e_uncertain": "不确定",
}

CAL_TEMPS = {
    "laya-new-512": 1.9321311778605548,
    "laya-new-2048": 1.9200876612990203,
    "laya-revised-labels": 2.0110113248206662,
}


# --------------------------------------------------------------------------- helpers
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Any:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def clip_logit(p: np.ndarray, bound: float = 12.0) -> np.ndarray:
    return np.clip(logit(p), -bound, bound)


def softmax2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_dev() -> tuple[list[str], np.ndarray, dict[str, int]]:
    ids: list[str] = []
    labels: list[int] = []
    for line in open(DEV_PATH, "r", encoding="utf-8"):
        row = json.loads(line)
        ids.append(row["id"])
        labels.append(int(row["label"]))
    return ids, np.asarray(labels), {sid: i for i, sid in enumerate(ids)}


def load_calibration_ids() -> list[str]:
    return list(load_json(CAL_MANIFEST)["calibration_ids"])


def load_cal_labels() -> dict[str, int]:
    cal = set(load_calibration_ids())
    out: dict[str, int] = {}
    for line in open(TRAIN_PATH, "r", encoding="utf-8"):
        row = json.loads(line)
        if row["id"] in cal:
            out[row["id"]] = int(row["label"])
    if len(out) != len(cal):
        raise RuntimeError("calibration ids missing from train.jsonl")
    return out


# ------------------------------------------------------------------ view extraction
@dataclass
class View:
    name: str
    path: str
    category: str
    pred: np.ndarray | None = None
    p_pos: np.ndarray | None = None
    group: str = ""
    origin: str = "curated"
    warn: str = ""

    def binary_vec(self) -> np.ndarray:
        if self.pred is not None:
            return self.pred
        assert self.p_pos is not None
        return (self.p_pos >= 0.5).astype(np.int8)


def rows_to_view(rows: list[dict], name: str, path: str, category: str, dev_index: dict[str, int]) -> View:
    n = len(dev_index)
    pred = np.full(n, -1, dtype=np.int8)
    prob = np.full(n, np.nan, dtype=np.float64)
    seen = 0
    for row in rows:
        sid = row.get("sample_id") or row.get("id")
        if sid not in dev_index:
            continue
        i = dev_index[sid]
        seen += 1
        if row.get("pred_label") is not None:
            pred[i] = int(row["pred_label"])
        probs = row.get("probabilities")
        if isinstance(probs, (list, tuple)) and len(probs) >= 2:
            prob[i] = float(probs[1])
        elif isinstance(row.get("logits"), (list, tuple)) and len(row["logits"]) >= 2:
            sm = softmax2(np.asarray(row["logits"], dtype=np.float64)[None, :])[0]
            prob[i] = float(sm[1])
    if seen != n:
        raise ValueError(f"{path}: only {seen}/{n} dev ids matched")
    if np.isnan(prob).any() and (pred < 0).any():
        raise ValueError(f"{path}: rows lack both prediction and probability")
    v = View(name=name, path=path, category=category)
    if not np.isnan(prob).any():
        v.p_pos = prob
    if (pred >= 0).all():
        v.pred = pred.astype(np.int8)
    return v


def wide_view(doc: dict, name: str, path: str, category: str, dev_index: dict[str, int],
              p_pos: np.ndarray) -> View:
    ids = doc.get("sample_ids") or doc.get("ids")
    if ids is None or len(ids) != len(dev_index):
        raise ValueError(f"{path}: sample_ids not aligned to dev")
    order = np.array([dev_index[sid] for sid in ids], dtype=np.int64)
    v = View(name=name, path=path, category=category)
    v.p_pos = np.asarray(p_pos, dtype=np.float64)[order]
    v.pred = (v.p_pos >= 0.5).astype(np.int8)
    return v


def _labels_in_doc(doc: Any) -> np.ndarray | None:
    if isinstance(doc, list):
        if doc and isinstance(doc[0], dict) and "true_label" in doc[0]:
            return np.asarray([int(r["true_label"]) for r in doc], dtype=np.int64)
        return None
    for key in ("true_labels", "labels", "true_label"):
        arr = doc.get(key)
        if isinstance(arr, list) and arr and not isinstance(arr[0], (list, dict)):
            return np.asarray(arr, dtype=np.int64)
    if isinstance(doc.get("rows"), list) and doc["rows"]:
        try:
            return np.asarray([int(r["true_label"]) for r in doc["rows"]], dtype=np.int64)
        except KeyError:
            return None
    return None


# ---------------------------------------------------------------- curated dev sources
def curated_dev_views(dev_ids: list[str], dev_labels: np.ndarray,
                      dev_index: dict[str, int]) -> tuple[list[View], set[str]]:
    n = len(dev_ids)
    views: list[View] = []
    used: set[str] = set()

    def add(v: View) -> None:
        views.append(v)

    def add_rows(path_rel: str, name: str, category: str, group: str,
                 pred_key: str = "pred_label") -> None:
        p = os.path.join(ROOT, path_rel)
        if not os.path.exists(p):
            print(f"  [warn] missing curated source {path_rel}", file=sys.stderr)
            return
        doc = load_json(p)
        rows = doc["rows"] if isinstance(doc, dict) else doc
        if isinstance(doc, dict) and doc.get("split") not in (None, "validation", "development"):
            raise ValueError(f"{path_rel}: unexpected split {doc.get('split')!r}")
        v = rows_to_view(rows, name, path_rel, category, dev_index)
        v.group = group
        add(v)
        used.add(path_rel)

    def add_logits(path_rel: str, name: str, category: str, group: str, temp: float = 1.0) -> None:
        p = os.path.join(ROOT, path_rel)
        if not os.path.exists(p):
            print(f"  [warn] missing curated source {path_rel}", file=sys.stderr)
            return
        doc = load_json(p)
        z = np.asarray(doc["logits"], dtype=np.float64)
        v = wide_view(doc, name, path_rel, category, dev_index, sigmoid((z[:, 1] - z[:, 0]) / temp))
        v.group = group
        add(v)
        used.add(path_rel)

    # A. legacy Laya-512 rows-style views (the 22-file / 18-vector pool)
    for model, run in {
        "baseline-s42": "laya-posthoc-baseline-s42-fp32-v1",
        "baseline-s43": "laya-posthoc-baseline-s43-fp32-v1",
        "combined-s42": "laya-posthoc-combined-s42-fp32-v1",
        "combined-s43": "laya-posthoc-combined-s43-fp32-v1",
    }.items():
        for rule in ("head", "head_tail"):
            add_rows(f"reports/{run}/validation-{rule}.json", f"{model}/{rule}",
                     "laya512-posthoc-window", f"laya512-{model}")
    for model, run in {
        "baseline-s43": "laya-decision-study-s43-s42-strict-20261013-v1/baseline-s43",
        "combined-s42": "laya-decision-study-s43-s42-strict-20261013-v1/combined-s42",
    }.items():
        for rule in ("head", "head_tail", "reverse"):
            add_rows(f"reports/{run}/validation-{rule}.json", f"{model}/{rule}",
                     "laya512-decision-study", f"laya512-{model}")
    for model in ("prefix-trained", "random-window-trained"):
        for rule in ("head", "head_tail", "reverse"):
            add_rows(f"reports/laya-decision-execution-20261014-v1-final-study/{model}/validation-{rule}.json",
                     f"{model}/{rule}", "laya512-window-training", f"laya512-{model}")
    add_rows("reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/validation-head_tail.json",
             "checkpoint-average-s42/head_tail", "laya512-checkpoint-average", "laya512-checkpoint-average-s42")
    # frozen-representation probe (3 probability columns per row)
    probe_rel = "reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/validation-predictions.json"
    probe_path = os.path.join(ROOT, probe_rel)
    if os.path.exists(probe_path):
        doc = load_json(probe_path)
        for feat in ("full_model", "encoder", "post_head"):
            p_pos = np.full(n, np.nan)
            for row in doc["rows"]:
                p_pos[dev_index[row["sample_id"]]] = float(row["probabilities"][feat][1])
            v = View(name=f"frozen-probe-s43/{feat}", path=probe_rel, category="laya512-frozen-probe", p_pos=p_pos)
            v.pred = (p_pos >= 0.5).astype(np.int8)
            v.group = f"laya512-frozen-probe-s43/{feat}"
            add(v)
        used.add(probe_rel)

    # B. frozen two-model Laya fusion (main fitted, equal-weight, CUDA-recheck exports)
    add_rows("reports/laya-probability-fusion-s43-s42-20261014-v1/validation-fused.json",
             "laya-main-fusion-70_30", "fusion", "laya-main-fusion")
    cuda_rel = "reports/laya-fusion-inference-cuda-fp32-20261014-v1/validation-actual.json"
    if os.path.exists(os.path.join(ROOT, cuda_rel)):
        doc = load_json(os.path.join(ROOT, cuda_rel))
        for label, key in (("equal-50_50(laya)", "equal"), ("primary(laya)", "primary")):
            p_pos = np.asarray([row[1] for row in doc[key]], dtype=np.float64)
            v = wide_view(doc, label, cuda_rel, "fusion", dev_index, p_pos)
            v.group = "laya-fusion"
            add(v)
        for model in ("baseline-s43", "combined-s42"):
            for rule in ("head", "head_tail"):
                v = rows_to_view(doc["models"][model][rule], f"{model}/{rule}(cuda-actual)",
                                 cuda_rel, "backend-recheck", dev_index)
                v.group = f"laya512-{model}"
                add(v)
        used.add(cuda_rel)

    # C. head-control runs (top-level calibrated probabilities) + single-model exports
    for kind in ("original", "linear"):
        for seed in (42, 43, 44):
            path_rel = f"reports/laya-head-control-{kind}-s{seed}-strict-fp32-20261015-v2/validation.json"
            p = os.path.join(ROOT, path_rel)
            if not os.path.exists(p):
                print(f"  [warn] missing curated source {path_rel}", file=sys.stderr)
                continue
            doc = load_json(p)
            rows = doc["rows"]
            sids = [r["sample_id"] for r in rows]
            vals = np.asarray([row[1] for row in doc["calibrated_probabilities"]], dtype=np.float64)
            v = View(name=f"head-control-{kind}-s{seed}", path=path_rel, category="laya512-head-control")
            arr = np.full(n, np.nan)
            for sid, val in zip(sids, vals):
                arr[dev_index[sid]] = val
            v.p_pos = arr
            v.pred = (arr >= 0.5).astype(np.int8)
            v.group = f"laya512-head-control-{kind}-s{seed}"
            add(v)
            used.add(path_rel)

    for path_rel, name, cat in (
        ("reports/laya-l512-s42-ce/validation-saved.jsonl", "mac-one-epoch-s42", "laya512-single-model"),
        ("reports/laya-mps-ce-epochs2-l512-s42-fp32-v1/saved-validation/raw.jsonl", "mac-two-epoch-s42", "laya512-single-model"),
        ("reports/laya-mps-ce-epochs2-lr06-l512-s42-fp32-v1/saved-validation/raw.jsonl", "mac-two-epoch-s42-lr06", "laya512-single-model"),
        ("reports/modernbert-l256-s42/validation-best.jsonl", "modernbert-best", "other-models"),
        ("reports/model-comparison/modernbert_base/raw.jsonl", "modernbert-final", "other-models"),
        ("reports/model-comparison/laya_ce_full/raw.jsonl", "laya-ce-full", "other-models"),
        ("reports/model-comparison/laya_ce_pilot/raw.jsonl", "laya-ce-pilot", "other-models"),
        ("reports/model-comparison/laya_rlcd_pilot/raw.jsonl", "laya-rlcd-pilot", "other-models"),
    ):
        p = os.path.join(ROOT, path_rel)
        if not os.path.exists(p):
            print(f"  [warn] missing curated source {path_rel}", file=sys.stderr)
            continue
        rows = []
        for r in load_json(p):
            sid = r.get("sample_id") or r.get("id")
            row: dict[str, Any] = {"sample_id": sid}
            if "logits" in r:
                row["logits"] = r["logits"]
            elif "probabilities" in r:
                row["probabilities"] = r["probabilities"]
            elif "positive_probability_calibrated" in r:
                q = float(r["positive_probability_calibrated"])
                row["probabilities"] = [1 - q, q]
            elif "probability" in r:
                q = float(r["probability"])
                row["probabilities"] = [1 - q, q]
            elif "prediction" in r:
                row["pred_label"] = r["prediction"]
            rows.append(row)
        v = rows_to_view(rows, name, path_rel, cat, dev_index)
        v.group = name
        add(v)
        used.add(path_rel)

    # D. heterogeneous DeBERTa + fusion (2026-10-17/18)
    deb_rel = "reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/predictions.json"
    doc = load_json(os.path.join(ROOT, deb_rel))
    for key, name, cat, group in (
        ("frozen_main", "laya-main-fusion", "fusion", "laya-main-fusion"),
        ("deberta", "deberta-s42", "deberta", "deberta"),
        ("fusion_50_50", "fusion-50_50(laya+deberta)", "fusion", "heterogeneous-fusion"),
    ):
        add(wide_view(doc, name, deb_rel, cat, dev_index,
                      np.asarray(doc["positive_probabilities"][key], dtype=np.float64)))
        views[-1].group = group
    used.add(deb_rel)

    # E. new Laya branches (2026-10-16/17) + length router
    add_logits("reports/laya-context-pair-s42-modal-20261016-v1/evaluation/512-development.json",
               "laya-new-512(orig-data)", "laya-new", "laya-new-512", temp=CAL_TEMPS["laya-new-512"])
    add_logits("reports/laya-context-pair-s42-modal-20261016-v1/evaluation/2048-development.json",
               "laya-new-2048(orig-data)", "laya-new", "laya-new-2048", temp=CAL_TEMPS["laya-new-2048"])
    add_logits("reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/512-development.json",
               "laya-revised-labels-512", "laya-new", "laya-revised", temp=CAL_TEMPS["laya-revised-labels"])
    router_rel = "reports/laya-length-router-20261017-v1/routed-development.json"
    if os.path.exists(os.path.join(ROOT, router_rel)):
        doc = load_json(os.path.join(ROOT, router_rel))
        z = np.asarray(doc["logits"], dtype=np.float64)
        v = wide_view(doc, "length-router(512/2048 routed)", router_rel, "laya-new", dev_index,
                      sigmoid(z[:, 1] - z[:, 0]))
        v.group = "length-router"
        add(v)
        used.add(router_rel)

    # F. long-fusion B/C offline combinations (2026-10-17, marked post-hoc)
    lf_rel = "reports/laya-long-fusion-bc-20261017-v1/predictions.json"
    doc = load_json(os.path.join(ROOT, lf_rel))
    for key in ("new512", "new2048", "A", "B", "C", "C_aux"):
        v = wide_view(doc, f"long-fusion-{key}", lf_rel, "fusion-posthoc", dev_index,
                      np.asarray(doc["positive_probabilities"][key], dtype=np.float64))
        v.group = "long-fusion"
        add(v)
    used.add(lf_rel)

    # G. final CPU alternatives (2026-10-18)
    v2_rel = "reports/final-cpu-alternatives-20261018-v2/development-predictions.json"
    doc = load_json(os.path.join(ROOT, v2_rel))
    for key in doc["predictions"]:
        v = wide_view(doc, f"cpu-alt/{key}", v2_rel, "cpu-alternatives", dev_index,
                      np.asarray(doc["predictions"][key], dtype=np.float64))
        v.group = f"cpu-alt/{key}"
        add(v)
    used.add(v2_rel)

    # H. bow / NB complementarity (2026-10-18)
    bow_rel = "reports/bow-nb-complementarity-cpu-20261018-v1/development-predictions.json"
    doc = load_json(os.path.join(ROOT, bow_rel))
    for key, val in doc["predictions"].items():
        if isinstance(val, dict):
            for sub in ("calibrated_positive_probabilities", "fusion_positive_probabilities"):
                if sub not in val:
                    continue
                v = wide_view(doc, f"bow-nb/{key}/{sub.split('_')[0]}", bow_rel, "bow-nb", dev_index,
                              np.asarray(val[sub], dtype=np.float64))
                v.group = f"bow-nb/{key}"
                add(v)
        else:
            v = wide_view(doc, f"bow-nb/{key}", bow_rel, "bow-nb", dev_index,
                          np.asarray(val, dtype=np.float64))
            v.group = f"bow-nb/{key}"
            add(v)
    used.add(bow_rel)

    # I. token-query heads (2026-10-18)
    tq_rel = "reports/laya-token-query-mps-fp32-20261018-v1/development-probabilities.json"
    doc = load_json(os.path.join(ROOT, tq_rel))
    for method in doc["predictions"]:
        for part in doc["predictions"][method]:
            add(wide_view(doc, f"token-query/{method}/{part}", tq_rel, "token-query", dev_index,
                          np.asarray(doc["predictions"][method][part], dtype=np.float64)))
            views[-1].group = f"token-query/{method}"
    used.add(tq_rel)

    # J. single-weight Laya+DeBERTa calibration fusion (2026-10-18)
    ldw_rel = "reports/laya-deberta-calibration-weight-cpu-20261018-v1/development-predictions.json"
    doc = load_json(os.path.join(ROOT, ldw_rel))
    for key in ("p_frozen_50_50", "p_calibration_nll_weight"):
        v = wide_view(doc, f"ldw/{key}", ldw_rel, "fusion", dev_index,
                      np.asarray(doc[key], dtype=np.float64))
        v.group = f"ldw/{key}"
        add(v)
    used.add(ldw_rel)

    return views, used


# ---------------------------------------------------------------- autoscan (2nd pass)
def autoscan_views(dev_ids: list[str], dev_labels: np.ndarray, dev_index: dict[str, int],
                   curated_paths: set[str]) -> tuple[list[View], dict]:
    """Catch dev prediction files not in the curated registry.

    Accept only full 2500-row, sample_id-aligned vectors with matching labels. Skip any
    path containing 'test' (no official-test per-sample access), tokenizer/assets copies
    and reproduction snapshots.
    """
    skipped_test = 0
    scanned = 0
    accepted: list[View] = []
    seen_vectors: set[str] = set()

    def candidates(path_rel: str, doc: Any) -> list[View]:
        out: list[View] = []
        rows = None
        if isinstance(doc, dict) and isinstance(doc.get("rows"), list) and len(doc["rows"]) == len(dev_ids):
            if doc["rows"] and isinstance(doc["rows"][0], dict) and (
                "sample_id" in doc["rows"][0] or "id" in doc["rows"][0]
            ):
                rows = doc["rows"]
        elif isinstance(doc, list) and len(doc) == len(dev_ids) and doc and isinstance(doc[0], dict) \
                and ("sample_id" in doc[0] or "id" in doc[0]):
            rows = doc
        if rows is not None:
            has = any(k in rows[0] for k in ("pred_label", "prediction", "probabilities", "logits"))
            if has:
                try:
                    out.append(rows_to_view(rows, path_rel, path_rel, "autoscan", dev_index))
                except Exception:
                    pass
        if isinstance(doc, dict) and len(doc.get("sample_ids") or doc.get("ids") or []) == len(dev_ids):
            ids = doc.get("sample_ids") or doc.get("ids")
            if set(ids) != set(dev_ids):
                return out
            for key in ("logits", "probabilities", "calibrated_probabilities", "raw_probabilities",
                        "positive_probabilities", "neural", "predictions", "pred_label", "pred_labels"):
                val = doc.get(key)
                if isinstance(val, list) and len(val) == len(dev_ids) and val:
                    if isinstance(val[0], list):
                        arr = np.asarray(val, dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            p = sigmoid(arr[:, 1] - arr[:, 0]) if key == "logits" else \
                                softmax2(arr)[:, 1] if key != "probabilities" else arr[:, 1]
                            try:
                                out.append(wide_view(doc, f"{path_rel}::{key}", path_rel,
                                                     "autoscan", dev_index, p))
                            except Exception:
                                pass
                    elif isinstance(val[0], (int, float)):
                        arr = np.asarray(val, dtype=np.float64)
                        if key in ("pred_label", "pred_labels"):
                            out.append(wide_view(doc, f"{path_rel}::{key}", path_rel, "autoscan",
                                                 dev_index, arr, ))
                            out[-1].pred = out[-1].p_pos.astype(np.int8)
                        else:
                            try:
                                out.append(wide_view(doc, f"{path_rel}::{key}", path_rel,
                                                     "autoscan", dev_index, arr))
                            except Exception:
                                pass
                elif isinstance(val, dict):
                    for sub, arr in val.items():
                        if isinstance(arr, list) and len(arr) == len(dev_ids) and arr:
                            a = np.asarray(arr, dtype=np.float64)
                            try:
                                if a.ndim == 2 and a.shape[1] >= 2:
                                    out.append(wide_view(doc, f"{path_rel}::{key}/{sub}", path_rel,
                                                         "autoscan", dev_index, sigmoid(a[:, 1] - a[:, 0])))
                                elif a.ndim == 1:
                                    out.append(wide_view(doc, f"{path_rel}::{key}/{sub}", path_rel,
                                                         "autoscan", dev_index, a))
                            except Exception:
                                pass
        for v in out:
            if v.p_pos is not None:
                if np.isnan(v.p_pos).any():
                    continue
                v.pred = (v.p_pos >= 0.5).astype(np.int8)
        return [v for v in out if v.pred is not None]

    for base in ("reports", "artifacts"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, base)):
            dir_rel = rel(dirpath)
            if any(part in dir_rel for part in ("tokenizer", "/assets", "reproduction", "/source/")):
                dirnames[:] = []
                continue
            for fname in filenames:
                if not fname.endswith((".json", ".jsonl")):
                    continue
                path = os.path.join(dirpath, fname)
                path_rel = rel(path)
                if "test" in path_rel.lower():
                    skipped_test += 1
                    continue
                if path_rel in curated_paths:
                    continue
                try:
                    if os.path.getsize(path) > 40_000_000:
                        continue
                except OSError:
                    continue
                scanned += 1
                try:
                    doc = load_json(path)
                except Exception:
                    continue
                labels = _labels_in_doc(doc)
                if labels is not None and len(labels) == len(dev_ids):
                    if not np.array_equal(labels, dev_labels):
                        continue
                try:
                    cand = candidates(path_rel, doc)
                except Exception:
                    continue
                for v in cand:
                    h = hashlib.sha256(v.pred.tobytes()).hexdigest()[:20]
                    if h in seen_vectors:
                        continue
                    seen_vectors.add(h)
                    v.group = "autoscan:" + (path_rel.split("/")[1] if "/" in path_rel else path_rel)
                    v.origin = "autoscan"
                    accepted.append(v)
    return accepted, {"autoscan_files_scanned": scanned, "skipped_test_paths": skipped_test}


# ------------------------------------------------------------------- oracle / errors
def oracle_stats(view_list: list[View], labels: np.ndarray) -> dict:
    preds = np.stack([v.binary_vec() for v in view_list], axis=1).astype(np.int16)
    correct = preds == labels[:, None]
    k = correct.sum(axis=1)
    n = len(labels)
    v = len(view_list)
    return {
        "view_count": v,
        "n": n,
        "oracle_correct": int((k > 0).sum()),
        "oracle_accuracy": float((k > 0).sum() / n),
        "all_wrong": int((k == 0).sum()),
        "all_wrong_ids": None,
        "counts_by_num_views_correct": {str(i): int((k == i).sum()) for i in range(v + 1)},
        "grouped": {
            "0": int((k == 0).sum()),
            "1-3": int(((k >= 1) & (k <= 3)).sum()),
            "4-9": int(((k >= 4) & (k <= 9)).sum()),
            "10-16": int(((k >= 10) & (k <= 16)).sum()),
            "17": int((k == 17).sum()) if v >= 18 else None,
            "all": int((k == v).sum()),
        },
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_delta(cand_correct: np.ndarray, base_correct: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(cand_correct)
    d = cand_correct.astype(np.float64) - base_correct.astype(np.float64)
    idx = rng.integers(0, n, size=(replicates, n))
    deltas = d[idx].mean(axis=1) * 100.0
    fixed = int(((~base_correct) & cand_correct).sum())
    broke = int((base_correct & (~cand_correct)).sum())
    return {
        "delta_pp": float(d.mean() * 100.0),
        "ci95_pp": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "fixed": fixed, "broke": broke,
        "mcnemar_exact_p": mcnemar_exact_p(fixed, broke),
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "bootstrap_method": "paired sample bootstrap over the fixed 2500-row development set",
    }


# ------------------------------------------------------------- calibration ensembles
def newton_logit_fit(X: np.ndarray, y: np.ndarray, l2: float, prior: np.ndarray,
                     iters: int = 200) -> np.ndarray:
    """Ridge logistic regression, mean-NLL objective + l2*sum((beta-prior)^2).

    Matches the objective documented for the published regularized_logit candidate
    (scripts/run_final_cpu_alternatives_v2.py: mean logaddexp + penalty*sum sq).
    Solved with Newton/IRLS; no external solver needed.
    """
    n = len(y)
    beta = prior.copy()
    for _ in range(iters):
        p = sigmoid(X @ beta)
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = X.T @ (p - y) / n + 2.0 * l2 * (beta - prior)
        H = (X * w[:, None]).T @ X / n + 2.0 * l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def calibration_probabilities(cal_ids: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    cal_index = {sid: i for i, sid in enumerate(cal_ids)}
    n = len(cal_ids)
    out: dict[str, np.ndarray] = {}
    prov: dict[str, dict] = {}

    def align(sample_ids: list[str], values: np.ndarray) -> np.ndarray:
        vec = np.full(n, np.nan)
        for sid, val in zip(sample_ids, values):
            if sid in cal_index:
                vec[cal_index[sid]] = float(val)
        if np.isnan(vec).any():
            raise ValueError("calibration alignment incomplete")
        return vec

    def add(name: str, path_rel: str, sample_ids: list[str], values: np.ndarray) -> None:
        out[name] = align(sample_ids, values)
        prov[name] = {"path": path_rel, "sha256": sha256_file(os.path.join(ROOT, path_rel))}

    v2_rel = "reports/final-cpu-alternatives-20261018-v2/calibration-predictions.json"
    doc = load_json(os.path.join(ROOT, v2_rel))
    add("laya-main-fusion", v2_rel, doc["sample_ids"], np.asarray(doc["p_laya"], dtype=np.float64))
    add("deberta-s42", v2_rel, doc["sample_ids"], np.asarray(doc["p_deberta"], dtype=np.float64))
    for fname, name in (("512-calibration.json", "laya-new-512"), ("2048-calibration.json", "laya-new-2048")):
        p_rel = f"reports/laya-context-pair-s42-modal-20261016-v1/evaluation/{fname}"
        d = load_json(os.path.join(ROOT, p_rel))
        z = np.asarray(d["logits"], dtype=np.float64)
        add(name, p_rel, d["sample_ids"], sigmoid((z[:, 1] - z[:, 0]) / CAL_TEMPS[name]))
    p_rel = "reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/512-calibration.json"
    d = load_json(os.path.join(ROOT, p_rel))
    z = np.asarray(d["logits"], dtype=np.float64)
    add("laya-revised-labels", p_rel, d["sample_ids"],
        sigmoid((z[:, 1] - z[:, 0]) / CAL_TEMPS["laya-revised-labels"]))
    for fname, name in (("tfidf_lr-calibration.json", "tfidf-lr"), ("nb_ngram_lr-calibration.json", "nb-ngram-lr")):
        p_rel = f"reports/bow-nb-complementarity-cpu-20261018-v1/{fname}"
        d = load_json(os.path.join(ROOT, p_rel))
        add(name, p_rel, d["sample_ids"], np.asarray(d["positive_probabilities"], dtype=np.float64))
    add("cpu-alt/mlp_dropout1", v2_rel, doc["sample_ids"],
        np.asarray(doc["predictions"]["mlp_dropout1"], dtype=np.float64))
    add("cpu-alt/mlp_dropout4", v2_rel, doc["sample_ids"],
        np.asarray(doc["predictions"]["mlp_dropout4"], dtype=np.float64))
    return out, prov


# ------------------------------------------------------------------------- audit dump
def write_error_evidence(out_dir: str, dev_ids: list[str], dev_labels: np.ndarray,
                         views: list[View], fusion: View) -> dict:
    texts: dict[str, str] = {}
    for line in open(DEV_PATH, "r", encoding="utf-8"):
        row = json.loads(line)
        texts[row["id"]] = row["text"]
    laya = load_json(os.path.join(ROOT, "reports/laya-posthoc-baseline-s43-fp32-v1/validation-head_tail.json"))
    laya_meta = {r["sample_id"]: r for r in laya["rows"]}
    deb = load_json(os.path.join(ROOT, DEBERTA_EVAL_DIR, "development.json"))
    deb_meta = {sid: {"tokens": int(t), "truncated": bool(tr)} for sid, t, tr in
                zip(deb["sample_ids"], deb["review_token_lengths"], deb["truncated"])}
    common47 = load_json(os.path.join(ROOT, FUSION_ANALYSIS_DIR, "common-47.json"))
    c47 = {c["sample_id"]: c["case_id"] for c in common47["cases"]}
    by_name = {v.name: v for v in views}
    p_laya, p_deb, p_fus = by_name["laya-main-fusion"].p_pos, by_name["deberta-s42"].p_pos, fusion.p_pos
    dump = []
    for i in np.where(fusion.binary_vec() != dev_labels)[0]:
        sid = dev_ids[i]
        y = int(dev_labels[i])
        dump.append({
            "sample_id": sid, "true_label": y, "label_name": "pos" if y else "neg",
            "case_id_common47": c47.get(sid),
            "p_laya_main": float(p_laya[i]), "p_deberta": float(p_deb[i]),
            "p_fusion_50_50": float(p_fus[i]),
            "signed_margin_fusion": float((2 * y - 1) * (p_fus[i] - 0.5)),
            "laya_review_tokens": int(laya_meta[sid]["review_tokens"]),
            "laya_truncated": bool(laya_meta[sid]["truncated"]),
            "deberta_review_tokens": deb_meta[sid]["tokens"],
            "deberta_truncated": deb_meta[sid]["truncated"],
            "text_len_chars": len(texts[sid]),
            "text": texts[sid],
        })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fusion-68-error-evidence.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, ensure_ascii=False, indent=1)
    return {"count": len(dump), "path": rel(path), "sha256": sha256_file(path)}


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/accuracy-improvement-research-20261018-v1")
    ap.add_argument("--report-dir", default="reports/accuracy-improvement-research-20261018-v1")
    ap.add_argument("--replicates", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261018)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, args.out_dir)
    report_dir = os.path.join(ROOT, args.report_dir)
    for d in (out_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    dev_ids, dev_labels, dev_index = load_dev()
    cal_ids = load_calibration_ids()
    cal_labels = load_cal_labels()
    cal_y = np.asarray([cal_labels[s] for s in cal_ids], dtype=np.int8)
    print(f"dev rows={len(dev_ids)} cal rows={len(cal_ids)}")

    # 1. inventory
    curated, curated_paths = curated_dev_views(dev_ids, dev_labels, dev_index)
    autoscan, scan_stats = autoscan_views(dev_ids, dev_labels, dev_index, curated_paths)
    all_views = curated + autoscan

    inventory = []
    valid: list[View] = []
    for v in all_views:
        pred = v.binary_vec()
        if len(pred) != len(dev_labels):
            continue
        acc = float((pred == dev_labels).mean())
        flipped_acc = float(((1 - pred) == dev_labels).mean())
        if acc < 0.5 and flipped_acc > 0.9:
            v.pred = (1 - pred).astype(np.int8)
            if v.p_pos is not None:
                v.p_pos = 1.0 - v.p_pos
            v.warn = "polarity_flipped_on_sanity_check"
            acc = flipped_acc
        if acc < 0.5:
            continue
        valid.append(v)
        inventory.append({"name": v.name, "path": v.path, "category": v.category,
                          "group": v.group, "origin": v.origin, "accuracy": acc,
                          "correct": int(round(acc * len(dev_labels))),
                          "has_probabilities": v.p_pos is not None, "warnings": v.warn})

    # label-agreement audit for curated row/wide sources that carry labels
    label_checks = {"checked": 0, "agree": 0, "disagree": []}
    for path_rel in sorted({v.path for v in valid}):
        p = os.path.join(ROOT, path_rel)
        try:
            doc = load_json(p) if not path_rel.endswith(".jsonl") else load_json(p)
        except Exception:
            continue
        labels = _labels_in_doc(doc)
        if labels is None or len(labels) != len(dev_labels):
            continue
        label_checks["checked"] += 1
        if np.array_equal(labels, dev_labels):
            label_checks["agree"] += 1
        else:
            label_checks["disagree"].append(path_rel)

    # dedupe by exact prediction vector
    dedup: dict[str, View] = {}
    for v in valid:
        h = hashlib.sha256(v.binary_vec().tobytes()).hexdigest()
        dedup.setdefault(h, v)
    distinct = list(dedup.values())
    # membership of each distinct prediction vector (representative + duplicate records)
    vector_groups: dict[str, list[str]] = {}
    for v in valid:
        h = hashlib.sha256(v.binary_vec().tobytes()).hexdigest()
        vector_groups.setdefault(h, []).append(v.name)
    distinct_records = []
    for v in distinct:
        h = hashlib.sha256(v.binary_vec().tobytes()).hexdigest()
        distinct_records.append({
            "representative": v.name, "path": v.path, "category": v.category,
            "correct": int((v.binary_vec() == dev_labels).sum()),
            "accuracy": float((v.binary_vec() == dev_labels).mean()),
            "n_records_same_vector": len(vector_groups[h]),
            "records": vector_groups[h],
        })

    by_name = {v.name: v for v in valid}
    fusion = by_name["fusion-50_50(laya+deberta)"]
    laya_main = by_name["laya-main-fusion"]
    deberta = by_name["deberta-s42"]
    base_correct = fusion.binary_vec() == dev_labels

    # 2. oracle pools
    # Documented classic pool (stage2 diagnosis, 2026-10-14): the rows-format
    # validation-{head,head_tail,reverse}.json exports plus the fused-probability view.
    # The frozen-probe export (2026-10-15) and the CUDA/MLX actual-recheck exports are
    # excluded, matching the documented "22 files -> 18 distinct vectors" scope.
    classic_files = [v for v in curated
                     if re.search(r"/validation-(head|head_tail|reverse)\.json$", v.path)
                     or v.path == "reports/laya-probability-fusion-s43-s42-20261014-v1/validation-fused.json"]
    classic_dedup = {}
    for v in classic_files:
        h = hashlib.sha256(v.binary_vec().tobytes()).hexdigest()
        classic_dedup.setdefault(h, v)
    classic_oracle = oracle_stats(list(classic_dedup.values()), dev_labels)
    classic_oracle["file_count"] = len(classic_files)
    classic_oracle["distinct_vector_count"] = len(classic_dedup)

    all_oracle = oracle_stats(distinct, dev_labels)
    all_oracle["all_wrong_ids"] = [
        dev_ids[i] for i in np.where(
            np.stack([v.binary_vec() for v in distinct], axis=1)[np.arange(len(dev_ids))].__eq__(dev_labels[:, None]).sum(axis=1) == 0
        )[0]
    ]

    # single-model/branch pool: exclude combination views and post-hoc fused views
    combo_re = re.compile(r"(fusion|fused|equal-|primary|long-fusion|regularized_logit|long_context|"
                          r"ldw/|bow-nb/(tfidf_lr|nb_ngram_lr)$|token-query/[^/]+/fusion|/A$|/B$|/C$)")
    single_pool = [v for v in distinct
                   if v.category != "fusion" and v.category != "fusion-posthoc"
                   and not combo_re.search(v.name) and v.category != "cpu-alternatives"]
    single_oracle = oracle_stats(single_pool, dev_labels)
    single_oracle["all_wrong_ids"] = [
        dev_ids[i] for i in np.where(
            (np.stack([v.binary_vec() for v in single_pool], axis=1) == dev_labels[:, None]).sum(axis=1) == 0
        )[0]
    ]

    # cumulative oracle as views are added in a pre-specified order (label-aware upper bound)
    curve_order = ["laya-main-fusion", "deberta-s42", "laya-new-2048(orig-data)",
                   "laya-revised-labels-512", "bow-nb/nb_ngram_lr/calibrated",
                   "bow-nb/tfidf_lr/calibrated", "token-query/mean/branch",
                   "cpu-alt/mlp_dropout1"]
    origin_curve = []
    pool_curve: list[View] = []
    for nm in curve_order:
        if nm not in by_name:
            continue
        pool_curve.append(by_name[nm])
        oc = oracle_stats(pool_curve, dev_labels)
        origin_curve.append({"views_added": nm, "n_views": oc["view_count"],
                             "oracle_correct": oc["oracle_correct"],
                             "oracle_accuracy": oc["oracle_accuracy"], "all_wrong": oc["all_wrong"]})

    # 3. 68-error marginal fixability
    err_idx = np.where(~base_correct)[0]
    err_views = []
    for v in distinct:
        correct = v.binary_vec() == dev_labels
        fixes = int(correct[err_idx].sum())
        if fixes:
            err_views.append({"view": v.name, "path": v.path, "category": v.category,
                              "fixes_of_68": fixes})
    err_views.sort(key=lambda r: -r["fixes_of_68"])
    fixed_by_any = int((np.stack([v.binary_vec()[err_idx] for v in distinct], axis=1)
                        == dev_labels[err_idx][:, None]).any(axis=1).sum())
    err_history = (
        (np.stack([v.binary_vec()[err_idx] for v in distinct], axis=1) == dev_labels[err_idx][:, None])
        .sum(axis=1))
    per_error = []
    for i in err_idx:
        correct_views = [v.name for v in distinct if v.binary_vec()[i] == dev_labels[i]]
        per_error.append({"sample_id": dev_ids[i], "n_views_correct": len(correct_views),
                          "correct_views": correct_views})
    err_stats = {
        "count": int(len(err_idx)),
        "fixed_by_at_least_one_existing_view": fixed_by_any,
        "wrong_in_every_existing_view": int(len(err_idx) - fixed_by_any),
        "distribution_views_correct_among_68": {str(k): int((err_history == k).sum())
                                                for k in range(0, len(distinct) + 1)
                                                if (err_history == k).any()},
        "top_fixing_views": err_views[:20],
        "views_with_zero_fix": len(distinct) - len(err_views),
        "per_error_correct_views": per_error,
    }

    # 3b. AI qualitative classification of the 68 errors (not human adjudication)
    missing_cls = [dev_ids[i] for i in err_idx if dev_ids[i] not in FUSION_68_CLASSIFICATION]
    if missing_cls:
        raise RuntimeError(f"unclassified fusion errors: {missing_cls}")
    cls_rows = []
    cls_counts = {k: 0 for k in CLASS_LABELS}
    all_wrong_set = set(all_oracle["all_wrong_ids"])
    common47_ids = {r["sample_id"] for r in load_json(
        os.path.join(ROOT, FUSION_ANALYSIS_DIR, "common-47.json"))["cases"]}
    p_fus = fusion.p_pos
    for i in err_idx:
        sid = dev_ids[i]
        cls, note = FUSION_68_CLASSIFICATION[sid]
        cls_counts[cls] += 1
        cls_rows.append({
            "sample_id": sid, "true_label": int(dev_labels[i]),
            "class": cls, "class_zh": CLASS_LABELS[cls], "ai_note": note,
            "p_fusion": float(p_fus[i]),
            "signed_margin": float((2 * dev_labels[i] - 1) * (p_fus[i] - 0.5)),
            "in_common47": sid in common47_ids,
            "wrong_in_all_57_views": sid in all_wrong_set,
        })
    hard_core = [r for r in cls_rows if r["wrong_in_all_57_views"]]
    classification = {
        "label": "AI qualitative classification, not independent human adjudication",
        "classes": CLASS_LABELS,
        "counts": cls_counts,
        "total": len(cls_rows),
        "hard_core_19_class_counts": {k: sum(1 for r in hard_core if r["class"] == k)
                                      for k in CLASS_LABELS},
        "common47_class_counts": {k: sum(1 for r in cls_rows if r["class"] == k and r["in_common47"])
                                  for k in CLASS_LABELS},
        "non_common21_class_counts": {k: sum(1 for r in cls_rows if r["class"] == k and not r["in_common47"])
                                      for k in CLASS_LABELS},
        "ids_by_class": {k: [r["sample_id"] for r in cls_rows if r["class"] == k] for k in CLASS_LABELS},
        "label_ceiling_anchors": {
            "strict_conflict_only_x": cls_counts["a_label_conflict"],
            "conflict_plus_truncation_x": cls_counts["a_label_conflict"] + cls_counts["c_truncation"],
            "broad_conflict_plus_ambiguity_x": cls_counts["a_label_conflict"] + cls_counts["c_truncation"]
            + cls_counts["b_ambiguity"],
        },
        "rows": cls_rows,
    }

    # 3c. fixed-rule combinations (no fitting; pre-specifiable) + calibration-fitted gate
    def by_pred(name: str) -> np.ndarray:
        return by_name[name].binary_vec()

    def rule_eval(name: str, pred: np.ndarray) -> dict:
        correct = pred == dev_labels
        return {"name": name, "dev_correct": int(correct.sum()),
                "dev_accuracy": float(correct.mean()),
                "paired_vs_fusion": paired_delta(correct, base_correct, args.replicates, args.bootstrap_seed)}

    fixed_rules = []
    fixed_rules.append(rule_eval("mean_prob[Laya,DeBERTa]",
                                 ((by_name["laya-main-fusion"].p_pos + by_name["deberta-s42"].p_pos) / 2 >= 0.5).astype(np.int8)))
    fixed_rules.append(rule_eval("majority3[Laya,DeBERTa,2048]",
                                 (by_pred("laya-main-fusion") + by_pred("deberta-s42")
                                  + by_pred("laya-new-2048(orig-data)") >= 2).astype(np.int8)))
    fixed_rules.append(rule_eval("majority3[Laya,DeBERTa,revised]",
                                 (by_pred("laya-main-fusion") + by_pred("deberta-s42")
                                  + by_pred("laya-revised-labels-512") >= 2).astype(np.int8)))
    fixed_rules.append(rule_eval("majority5[Laya,DeBERTa,2048,revised,tfidf]",
                                 (by_pred("laya-main-fusion") + by_pred("deberta-s42")
                                  + by_pred("laya-new-2048(orig-data)") + by_pred("laya-revised-labels-512")
                                  + by_pred("bow-nb/tfidf_lr/calibrated") >= 3).astype(np.int8)))
    fixed_rules.append(rule_eval("majority5[Laya,DeBERTa,2048,revised,nb]",
                                 (by_pred("laya-main-fusion") + by_pred("deberta-s42")
                                  + by_pred("laya-new-2048(orig-data)") + by_pred("laya-revised-labels-512")
                                  + by_pred("bow-nb/nb_ngram_lr/calibrated") >= 3).astype(np.int8)))

    # 4. pairwise flips between strong views
    strong_names = ["laya-main-fusion", "deberta-s42", "fusion-50_50(laya+deberta)",
                    "laya-new-512(orig-data)", "laya-new-2048(orig-data)",
                    "laya-revised-labels-512", "cpu-alt/regularized_logit",
                    "bow-nb/tfidf_lr/calibrated", "bow-nb/nb_ngram_lr/calibrated", "token-query/mean/branch"]
    pair_matrix = {}
    present = [nm for nm in strong_names if nm in by_name]
    for a_i, a in enumerate(present):
        for b in present[a_i + 1:]:
            va, vb = by_name[a].binary_vec(), by_name[b].binary_vec()
            pair_matrix[f"{a} | {b}"] = {
                "flip_rate": float((va != vb).mean()),
                "n_flips": int((va != vb).sum()),
                "both_correct": int(((va == dev_labels) & (vb == dev_labels)).sum()),
                "a_only_correct": int(((va == dev_labels) & (vb != dev_labels)).sum()),
                "b_only_correct": int(((va != dev_labels) & (vb == dev_labels)).sum()),
                "both_wrong": int(((va != dev_labels) & (vb != dev_labels)).sum()),
            }

    # 5. reproduce the 97.40 % candidate from frozen probabilities
    a1, a2, b0 = V2_FORMULA
    p_v2 = sigmoid(a1 * clip_logit(laya_main.p_pos) + a2 * clip_logit(deberta.p_pos) + b0)
    v2_pred = (p_v2 >= 0.5).astype(np.int8)
    v2_correct = v2_pred == dev_labels
    stored = by_name["cpu-alt/regularized_logit"]
    v2_check = {
        "formula": "sigmoid(0.5319458737546259*clip(logit(p_laya),-12,12)+0.5727065867827512*clip(logit(p_deberta),-12,12)-0.1161031700961201)",
        "recomputed_correct": int(v2_correct.sum()),
        "recomputed_accuracy": float(v2_correct.mean()),
        "stored_prediction_correct": int((stored.binary_vec() == dev_labels).sum()),
        "exact_binary_match_with_stored": bool(np.array_equal(v2_pred, stored.binary_vec())),
        "max_abs_probability_difference_vs_stored": float(np.max(np.abs(p_v2 - stored.p_pos))),
        "paired_vs_fusion_50_50": paired_delta(v2_correct, base_correct, args.replicates, args.bootstrap_seed),
        "fixed_error_ids": [dev_ids[i] for i in err_idx if v2_correct[i]],
        "new_error_ids": [dev_ids[i] for i in np.where(base_correct & (~v2_correct))[0]],
        "remaining_errors": int((~v2_correct).sum()),
        "remaining_error_ids": [dev_ids[i] for i in np.where(~v2_correct)[0]],
    }

    # 6. calibration-only ensembles
    cal_probs, cal_prov = calibration_probabilities(cal_ids)
    dev_probs: dict[str, np.ndarray] = {}
    alias = {"laya-main-fusion": "laya-main-fusion", "deberta-s42": "deberta-s42",
             "laya-new-2048": "laya-new-2048(orig-data)", "laya-new-512": "laya-new-512(orig-data)",
             "laya-revised-labels": "laya-revised-labels-512",
             "tfidf-lr": "bow-nb/tfidf_lr/calibrated", "nb-ngram-lr": "bow-nb/nb_ngram_lr/calibrated",
             "cpu-alt/mlp_dropout1": "cpu-alt/mlp_dropout1",
             "cpu-alt/mlp_dropout4": "cpu-alt/mlp_dropout4"}
    for cname, dname in alias.items():
        if cname in cal_probs and dname in by_name and by_name[dname].p_pos is not None:
            dev_probs[cname] = by_name[dname].p_pos
    missing = [k for k in cal_probs if k not in dev_probs]
    if missing:
        raise RuntimeError(f"missing dev probabilities for calibration views: {missing}")

    def fit_eval(name: str, names: list[str], l2: float) -> dict:
        Xc = np.concatenate([np.stack([clip_logit(cal_probs[k]) for k in names], axis=1),
                             np.ones((len(cal_y), 1))], axis=1)
        prior = np.zeros(len(names) + 1)
        prior[:-1] = 1.0 / len(names)
        beta = newton_logit_fit(Xc, cal_y.astype(np.float64), l2, prior)
        p_cal = sigmoid(Xc @ beta)
        Xd = np.concatenate([np.stack([clip_logit(dev_probs[k]) for k in names], axis=1),
                             np.ones((len(dev_labels), 1))], axis=1)
        p_dev = sigmoid(Xd @ beta)
        correct = (p_dev >= 0.5).astype(np.int8) == dev_labels
        loo = 0
        for i in range(len(cal_y)):
            mask = np.ones(len(cal_y), dtype=bool)
            mask[i] = False
            bi = newton_logit_fit(Xc[mask], cal_y[mask].astype(np.float64), l2, prior)
            loo += int((sigmoid(Xc[i] @ bi) >= 0.5) == cal_y[i])
        return {
            "name": name, "inputs": names,
            "coefficients": [float(x) for x in beta], "l2": l2,
            "prior": [float(x) for x in prior],
            "calibration_resubstitution_accuracy": float(((p_cal >= 0.5) == cal_y).mean()),
            "calibration_loo_accuracy": float(loo / len(cal_y)),
            "dev_correct": int(correct.sum()), "dev_accuracy": float(correct.mean()),
            "remaining_errors": int((~correct).sum()),
            "paired_vs_fusion": paired_delta(correct, base_correct, args.replicates, args.bootstrap_seed),
        }

    logit_results = []
    logit_results.append(fit_eval("logit2[Laya,DeBERTa] l2=0.01", ["laya-main-fusion", "deberta-s42"], 0.01))
    logit_results.append(fit_eval("logit2[Laya,DeBERTa] l2=0.0", ["laya-main-fusion", "deberta-s42"], 0.0))
    logit_results.append(fit_eval("logit3[Laya,DeBERTa,2048] l2=0.01",
                                  ["laya-main-fusion", "deberta-s42", "laya-new-2048"], 0.01))
    logit_results.append(fit_eval("logit4[+revised] l2=0.01",
                                  ["laya-main-fusion", "deberta-s42", "laya-new-2048", "laya-revised-labels"], 0.01))
    logit_results.append(fit_eval("logit5[+tfidf] l2=0.01",
                                  ["laya-main-fusion", "deberta-s42", "laya-new-2048", "laya-revised-labels",
                                   "tfidf-lr"], 0.01))
    logit_results.append(fit_eval("logit8[+nb,+mlp1,+mlp4] l2=0.02",
                                  ["laya-main-fusion", "deberta-s42", "laya-new-2048", "laya-revised-labels",
                                   "tfidf-lr", "nb-ngram-lr", "cpu-alt/mlp_dropout1", "cpu-alt/mlp_dropout4"], 0.02))

    def equal_eval(name: str, names: list[str]) -> dict:
        Pc = np.mean([cal_probs[k] for k in names], axis=0)
        Pd = np.mean([dev_probs[k] for k in names], axis=0)
        correct = (Pd >= 0.5).astype(np.int8) == dev_labels
        return {"name": name, "inputs": names,
                "calibration_accuracy": float(((Pc >= 0.5) == cal_y).mean()),
                "dev_correct": int(correct.sum()), "dev_accuracy": float(correct.mean()),
                "remaining_errors": int((~correct).sum()),
                "paired_vs_fusion": paired_delta(correct, base_correct, args.replicates, args.bootstrap_seed)}

    equal_results = [
        equal_eval("equal[Laya,DeBERTa]", ["laya-main-fusion", "deberta-s42"]),
        equal_eval("equal[Laya,DeBERTa,2048]", ["laya-main-fusion", "deberta-s42", "laya-new-2048"]),
        equal_eval("equal[Laya,DeBERTa,2048,revised]",
                   ["laya-main-fusion", "deberta-s42", "laya-new-2048", "laya-revised-labels"]),
        equal_eval("equal[Laya,DeBERTa,2048,revised,tfidf]",
                   ["laya-main-fusion", "deberta-s42", "laya-new-2048", "laya-revised-labels", "tfidf-lr"]),
    ]

    # low-confidence gating: if |p_fusion-0.5|<=t replace by a diverse weak branch; t fitted on
    # calibration accuracy only, then applied once to development (exploratory).
    gate_results = []
    grid = [round(x, 3) for x in np.arange(0.0, 0.2001, 0.005)]
    p_fus_cal = cal_probs["laya-main-fusion"]
    for branch in ("nb-ngram-lr", "tfidf-lr", "laya-new-2048"):
        best = None
        for t in grid:
            pred_cal = (p_fus_cal >= 0.5).astype(np.int8)
            low_cal = np.abs(p_fus_cal - 0.5) <= t
            pred_cal[low_cal] = (cal_probs[branch][low_cal] >= 0.5).astype(np.int8)
            acc = float((pred_cal == cal_y).mean())
            if best is None or acc > best[1] + 1e-12:
                best = (t, acc)
        t, cal_acc = best
        pred = (p_fus >= 0.5).astype(np.int8)
        low = np.abs(p_fus - 0.5) <= t
        pred[low] = (dev_probs[branch][low] >= 0.5).astype(np.int8)
        rec = rule_eval(f"gate(|p-0.5|<={t}) -> {branch}", pred)
        rec["threshold_fitted_on"] = "fixed_400_calibration_accuracy"
        rec["calibration_accuracy_at_threshold"] = cal_acc
        rec["calibration_low_confidence_count"] = int(low_cal.sum())
        rec["dev_low_confidence_count"] = int(low.sum())
        gate_results.append(rec)

    # 7. 68-error evidence dump
    evidence = write_error_evidence(out_dir, dev_ids, dev_labels, valid, fusion)

    # 8. headroom arithmetic + label-ceiling sensitivity
    dev_n, test_n = 2500, 25000
    dev_target, test_target = 2450, 24500
    official = PUBLISHED_OFFICIAL_TEST
    headroom = {
        "dev": {
            "current_best_correct": int(base_correct.sum()),
            "current_best_accuracy": float(base_correct.mean()),
            "current_best_errors": int((~base_correct).sum()),
            "target_98pct_correct": dev_target,
            "net_correct_needed": dev_target - int(base_correct.sum()),
            "implied_fix_rate_of_remaining_errors": (dev_target - int(base_correct.sum())) / int((~base_correct).sum()),
        },
        "official_test_aggregates_only": {
            "source": official["source"],
            "fusion_50_50_correct": official["fusion_50_50"]["correct"],
            "fusion_50_50_errors": test_n - official["fusion_50_50"]["correct"],
            "target_98pct_correct": test_target,
            "net_correct_needed": test_target - official["fusion_50_50"]["correct"],
            "implied_fix_rate_of_remaining_errors":
                (test_target - official["fusion_50_50"]["correct"]) / (test_n - official["fusion_50_50"]["correct"]),
            "dev_minus_test_pp": {
                "fusion_50_50": (float(base_correct.mean()) - official["fusion_50_50"]["accuracy"]) * 100,
                "original_laya_main": (float((laya_main.binary_vec() == dev_labels).mean())
                                       - official["original_laya_main"]["accuracy"]) * 100,
                "deberta": (float((deberta.binary_vec() == dev_labels).mean())
                            - official["deberta_cpu"]["accuracy"]) * 100,
            },
        },
        "label_ceiling_sensitivity_dev": [],
    }
    for x in (20, 30, 40, 47, 50, 55):
        rec = {
            "x_unrecoverable_of_68": x,
            "max_dev_correct": dev_n - x,
            "max_dev_accuracy": (dev_n - x) / dev_n,
            "dev_98pct_still_reachable": (dev_n - x) >= dev_target,
            "required_fix_rate_of_recoverable_errors": (dev_target - int(base_correct.sum())) / (68 - x)
            if 68 - x > 0 else None,
            "implied_max_official_test_accuracy_if_same_error_mix":
                (test_n - (test_n - official["fusion_50_50"]["correct"]) * x / 68) / test_n,
        }
        headroom["label_ceiling_sensitivity_dev"].append(rec)

    summary = {
        "run_id": "accuracy-improvement-research-20261018-v1/headroom",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "constraints": {
            "no_new_training": True, "no_new_inference": True,
            "official_test_per_sample_access": False,
            "official_test_aggregates_source": official["source"],
            "skipped_paths_containing_test": scan_stats["skipped_test_paths"],
        },
        "inputs": {
            "dev_path": "data/processed/validation.jsonl", "dev_sha256": sha256_file(DEV_PATH),
            "cal_manifest": "reports/laya-deberta-calibration-weight-cpu-20261018-v1/frozen.json",
            "cal_manifest_sha256": sha256_file(CAL_MANIFEST),
            "train_path": "data/processed/train.jsonl", "train_sha256": sha256_file(TRAIN_PATH),
        },
        "inventory": {
            "view_records_total": len(inventory),
            "curated_views": len(curated), "autoscan_views": len(autoscan),
            "valid_views": len(valid), "distinct_prediction_vectors": len(distinct),
            "autoscan_stats": scan_stats, "label_agreement_check": label_checks,
            "views": inventory, "distinct_records": distinct_records,
        },
        "references": {
            "fusion_50_50": {"correct": int(base_correct.sum()), "accuracy": float(base_correct.mean()),
                             "errors": int((~base_correct).sum())},
            "laya_main_fusion": {"correct": int((laya_main.binary_vec() == dev_labels).sum())},
            "deberta": {"correct": int((deberta.binary_vec() == dev_labels).sum())},
        },
        "oracle": {"classic_22file_rule": classic_oracle, "all_distinct_views": all_oracle,
                   "single_model_pool": single_oracle, "cumulative_curve": origin_curve},
        "fusion_errors_68": err_stats,
        "error_classification_ai": classification,
        "fixed_rule_combinations": fixed_rules,
        "calibration_fitted_gates": gate_results,
        "pairwise_flips_strong_views": {
            k: [round(v["flip_rate"], 4), v["n_flips"], v["both_correct"],
                v["a_only_correct"], v["b_only_correct"], v["both_wrong"]]
            for k, v in pair_matrix.items()
            if all(nm in k for nm in ()) and k.split(" | ")[0] in {
                "laya-main-fusion", "deberta-s42", "fusion-50_50(laya+deberta)",
                "laya-new-2048(orig-data)", "laya-revised-labels-512",
                "cpu-alt/regularized_logit"}},
        "v2_candidate_reproduction": v2_check,
        "calibration_ensembles": {
            "calibration_views": list(cal_probs), "calibration_provenance": cal_prov,
            "logit_stacking": logit_results, "equal_weight": equal_results,
            "integrity_note": "all fits use only the fixed 400-row calibration split; development is scored once",
        },
        "headroom_arithmetic": headroom,
        "error_evidence_dump": evidence,
    }

    def jsonable(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        return o

    # 9. independent cross-checks of re-derived numbers (for verification.json)
    computed = {nm: int((v.binary_vec() == dev_labels).sum()) for nm, v in by_name.items()}
    cross = []
    for nm, (rep, src) in REPORTED_DEV_CORRECT.items():
        if nm in computed:
            cross.append({"view": nm, "reported_correct": rep, "recomputed_correct": computed[nm],
                          "match": bool(rep == computed[nm]), "source": src})
        else:
            cross.append({"view": nm, "reported_correct": rep, "recomputed_correct": None,
                          "match": None, "source": src, "note": "view not found in extracted pool"})
    official = PUBLISHED_OFFICIAL_TEST
    checks = {
        "fusion_50_50_dev_2432": int(base_correct.sum()) == 2432,
        "fusion_50_50_dev_errors_68": int((~base_correct).sum()) == 68,
        "classic_oracle_2460_98_40pct": classic_oracle["oracle_correct"] == 2460
        and abs(classic_oracle["oracle_accuracy"] - 0.984) < 1e-12,
        "classic_22_files_18_vectors": classic_oracle["file_count"] == 22
        and classic_oracle["distinct_vector_count"] == 18,
        "classic_all_wrong_40": classic_oracle["all_wrong"] == 40,
        "classic_grouped_40_15_24_84_12_2325": classic_oracle["grouped"] == {
            "0": 40, "1-3": 15, "4-9": 24, "10-16": 84, "17": 12, "all": 2325},
        "v2_candidate_recomputed_2435": v2_check["recomputed_correct"] == 2435,
        "v2_candidate_exact_binary_match": v2_check["exact_binary_match_with_stored"],
        "v2_candidate_fixed3_broke0": v2_check["paired_vs_fusion_50_50"]["fixed"] == 3
        and v2_check["paired_vs_fusion_50_50"]["broke"] == 0,
        "official_test_aggregate_arithmetic": (
            official["fusion_50_50"]["correct"] == 24257
            and 25000 - 24257 == headroom["official_test_aggregates_only"]["fusion_50_50_errors"]
            and 24500 - 24257 == headroom["official_test_aggregates_only"]["net_correct_needed"]
        ),
        "reported_dev_accuracies_all_match": all(c["match"] for c in cross if c["match"] is not None),
        "calibration_views_all_have_dev_twin": set(cal_probs) == set(dev_probs),
        "dev_label_agreement_all_sources": (label_checks["disagree"] == [] and label_checks["checked"] > 0),
    }
    verification = {
        "purpose": "independent re-derivation of the report's headline numbers from frozen local files",
        "run_id": summary["run_id"],
        "generated_utc": summary["generated_utc"],
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "reported_vs_recomputed_development_correct": cross,
        "v2_candidate": {
            "formula_inputs": "positive_probabilities.frozen_main (Laya internal fusion) and "
                              "positive_probabilities.deberta from "
                              "reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/predictions.json",
            "recomputed_correct": v2_check["recomputed_correct"],
            "stored_correct": v2_check["stored_prediction_correct"],
            "max_abs_probability_difference": v2_check["max_abs_probability_difference_vs_stored"],
            "fixed_error_ids": v2_check["fixed_error_ids"],
            "new_error_ids": v2_check["new_error_ids"],
            "paired": v2_check["paired_vs_fusion_50_50"],
        },
        "input_sha256": summary["inputs"],
        "method_notes": [
            "Views are aligned to data/processed/validation.jsonl order by sample_id; every source "
            "carrying labels was checked against the canonical labels.",
            "Official-test per-sample files were never opened. Only aggregate counts copied from "
            "reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md are cited.",
            "McNemar p-values are exact two-sided binomial tests on discordant pairs; bootstrap "
            "intervals are paired sample bootstraps over the fixed 2500-row development set.",
        ],
    }
    summary["verification"] = {"checks": checks, "all_checks_pass": all(checks.values()),
                               "reported_vs_recomputed": cross}

    summary = jsonable(summary)
    data_path = os.path.join(out_dir, "headroom-analysis-data.json")
    with open(data_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    ver_path = os.path.join(report_dir, "verification.json")
    with open(ver_path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(verification), fh, ensure_ascii=False, indent=1)

    # 10. compact machine-readable deliverable (< 40 KB)
    compact = {
        "run_id": summary["run_id"],
        "generated_utc": summary["generated_utc"],
        "scope_note": "frozen development/calibration artifacts only; no training, no inference, "
                      "no official-test per-sample access",
        "inputs": summary["inputs"],
        "headline": {
            "current_best_fusion_50_50": summary["references"]["fusion_50_50"],
            "laya_main_fusion_correct": summary["references"]["laya_main_fusion"]["correct"],
            "deberta_correct": summary["references"]["deberta"]["correct"],
            "oracle_classic_22file_18view": {k: classic_oracle[k] for k in
                                             ("file_count", "distinct_vector_count", "oracle_correct",
                                              "oracle_accuracy", "all_wrong", "grouped")},
            "oracle_all_distinct_views": {k: all_oracle[k] for k in
                                          ("view_count", "oracle_correct", "oracle_accuracy", "all_wrong")},
            "oracle_single_model_pool": {k: single_oracle[k] for k in
                                         ("view_count", "oracle_correct", "oracle_accuracy", "all_wrong")},
            "oracle_cumulative_curve": origin_curve,
            "fusion_errors_68": {k: v for k, v in err_stats.items()
                                 if k not in ("top_fixing_views", "per_error_correct_views")},
            "top_fixing_views": [[r["view"], r["fixes_of_68"]] for r in err_stats["top_fixing_views"][:10]],
            "error_classification_ai_counts": classification["counts"],
            "error_classification_ai_ids": classification["ids_by_class"],
            "error_classification_hard_core_counts": classification["hard_core_19_class_counts"],
            "label_ceiling_anchors_x_of_68": classification["label_ceiling_anchors"],
        },
        "inventory": {
            "view_records_total": len(inventory), "curated_views": len(curated),
            "autoscan_views": len(autoscan), "distinct_prediction_vectors": len(distinct),
            "autoscan_stats": scan_stats, "label_agreement_check": label_checks,
            "views_compact": [[v["name"], v["category"][:18], v["correct"],
                               round(v["accuracy"], 4)] for v in inventory],
            "full_per_view_paths": rel(data_path),
        },
        "v2_candidate_reproduction": {k: v2_check[k] for k in
                                      ("recomputed_correct", "recomputed_accuracy",
                                       "exact_binary_match_with_stored",
                                       "max_abs_probability_difference_vs_stored",
                                       "fixed_error_ids", "new_error_ids", "remaining_errors")},
        "v2_candidate_paired": v2_check["paired_vs_fusion_50_50"],
        "calibration_ensembles": {
            "calibration_views": list(cal_probs),
            "calibration_provenance_paths": {k: v["path"] for k, v in cal_prov.items()},
            "logit_stacking": [{k: e[k] for k in ("name", "inputs", "coefficients", "l2",
                                                  "calibration_loo_accuracy", "dev_correct",
                                                  "dev_accuracy", "remaining_errors")}
                               | {"paired_vs_fusion": {kk: e["paired_vs_fusion"][kk] for kk in
                                                        ("delta_pp", "ci95_pp", "fixed", "broke",
                                                         "mcnemar_exact_p")}}
                               for e in logit_results],
            "equal_weight": [{k: e[k] for k in ("name", "inputs", "calibration_accuracy", "dev_correct",
                                                "dev_accuracy", "remaining_errors")}
                             | {"paired_vs_fusion": {kk: e["paired_vs_fusion"][kk] for kk in
                                                      ("delta_pp", "ci95_pp", "fixed", "broke",
                                                       "mcnemar_exact_p")}}
                             for e in equal_results],
            "fixed_rules": [{k: e[k] for k in ("name", "dev_correct", "dev_accuracy")}
                            | {"paired_vs_fusion": {kk: e["paired_vs_fusion"][kk] for kk in
                                                     ("delta_pp", "ci95_pp", "fixed", "broke",
                                                      "mcnemar_exact_p")}}
                            for e in fixed_rules],
            "calibration_fitted_gates": gate_results,
        },
        "pairwise_flips_strong_views": {
            k: [round(v["flip_rate"], 4), v["n_flips"], v["both_correct"],
                v["a_only_correct"], v["b_only_correct"], v["both_wrong"]]
            for k, v in pair_matrix.items()
            if all(nm in k for nm in ()) and k.split(" | ")[0] in {
                "laya-main-fusion", "deberta-s42", "fusion-50_50(laya+deberta)",
                "laya-new-2048(orig-data)", "laya-revised-labels-512",
                "cpu-alt/regularized_logit"}},
        "headroom_arithmetic": summary["headroom_arithmetic"],
        "verification": {"path": rel(ver_path), "sha256": sha256_file(ver_path),
                         "all_checks_pass": all(checks.values()),
                         "reported_dev_accuracies_all_match": checks["reported_dev_accuracies_all_match"],
                         "n_reported_vs_recomputed_checked": sum(1 for c in cross if c["match"] is not None),
                         "mismatches": [c for c in cross if c["match"] is False]},
        "artifact_data_json": {"path": rel(data_path), "sha256": sha256_file(data_path)},
        "commands": [
            "UV_CACHE_DIR=/private/tmp/end-imdb-headroom-uv-cache uv run --locked --offline --no-sync "
            "python scripts/analyze_accuracy_headroom.py",
        ],
        "caveats": [
            "oracle numbers are label-aware post-hoc upper bounds, not deployable accuracy; the pool "
            "contains near-duplicate and post-hoc-derived views",
            "the 68-error classification is AI qualitative review of review text, not human adjudication",
            "all new candidate deltas are exploratory; development set was reused across many studies",
            "official-test counts are aggregate values copied from an existing report",
        ],
    }
    summary_path = os.path.join(report_dir, "headroom-summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(compact), fh, ensure_ascii=False, indent=1)
    print(json.dumps({
        "valid_views": len(valid),
        "distinct_vectors": len(distinct),
        "fusion_errors": summary["references"]["fusion_50_50"]["errors"],
        "classic_oracle": {k: classic_oracle[k] for k in
                           ("file_count", "distinct_vector_count", "oracle_correct", "oracle_accuracy", "all_wrong", "grouped")},
        "all_views_oracle": {k: all_oracle[k] for k in ("view_count", "oracle_correct", "oracle_accuracy", "all_wrong")},
        "single_pool_oracle": {k: single_oracle[k] for k in ("view_count", "oracle_correct", "oracle_accuracy", "all_wrong")},
        "fixed_by_any": err_stats["fixed_by_at_least_one_existing_view"],
        "wrong_in_every_view": err_stats["wrong_in_every_existing_view"],
        "v2": {k: v2_check[k] for k in ("recomputed_correct", "exact_binary_match_with_stored",
                                        "max_abs_probability_difference_vs_stored")},
        "v2_pair": v2_check["paired_vs_fusion_50_50"],
        "calibration_views": list(cal_probs),
        "logit_dev_correct": [(e["name"], e["dev_correct"]) for e in logit_results],
        "logit_coefficients": [(e["name"], [round(x, 4) for x in e["coefficients"]]) for e in logit_results],
        "logit_paired": [(e["name"], e["paired_vs_fusion"]["delta_pp"],
                          e["paired_vs_fusion"]["ci95_pp"], e["paired_vs_fusion"]["mcnemar_exact_p"])
                         for e in logit_results],
        "equal_dev_correct": [(e["name"], e["dev_correct"]) for e in equal_results],
        "label_checks": {"checked": label_checks["checked"], "agree": label_checks["agree"],
                         "disagree": label_checks["disagree"]},
        "oracle_curve": [(c["views_added"], c["oracle_correct"], round(c["oracle_accuracy"], 4))
                         for c in origin_curve],
        "classification_counts": classification["counts"],
        "hard_core_class_counts": classification["hard_core_19_class_counts"],
        "label_ceiling_anchors": classification["label_ceiling_anchors"],
        "fixed_rules": [(e["name"], e["dev_correct"], e["paired_vs_fusion"]["delta_pp"])
                        for e in fixed_rules],
        "gates": [(e["name"], e["dev_correct"], e["paired_vs_fusion"]["delta_pp"],
                   e["threshold_fitted_on"]) for e in gate_results],
        "verification_all_pass": all(checks.values()),
        "wrote": {"data_json": rel(data_path), "verification": rel(ver_path),
                  "headroom_summary": rel(summary_path)},
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
