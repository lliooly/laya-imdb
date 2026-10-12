"""Exposure budget, source contract and numerical control-path regressions."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT
from laya_resampling import make_plan, resample_training, validate_sampling, ACTIVE_CANDIDATE_SHA256 as CANDIDATE_SHA256
from laya_optimization import optimize_training
from cuda_numerics import guard_training

CANDIDATES = 'reports/laya-resampling-45-preparation-v2/frozen-candidates.json'


def source_data():
    with (ROOT / 'data/processed/train.jsonl').open() as source:
        rows = [json.loads(line) for line in source]
    m = json.loads((ROOT / 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json').read_text())
    hard = json.loads((ROOT / CANDIDATES).read_text())['rows']
    gradient = set(m['gradient_train_ids'])
    return m['gradient_train_ids'], {r['id']: r['label'] for r in rows if r['id'] in gradient}, {r['sample_id'] for r in hard}


def options(mode):
    return {'mode': mode, 'candidate_file': CANDIDATES, 'candidate_sha256': CANDIDATE_SHA256}


class ResamplingTests(unittest.TestCase):
    def test_full_frozen_pool_budget_classes_coverage_and_rng(self):
        ids, labels, hard = source_data()
        before = random.getstate()
        plans, removed, counts = make_plan(ids, labels, hard, 43, 'resampled')
        self.assertEqual(random.getstate(), before)
        self.assertEqual((plans, removed, counts), make_plan(ids, labels, hard, 43, 'resampled'))
        self.assertEqual(len(removed), 45)
        self.assertEqual(len(counts), 22100)
        self.assertEqual(sum(counts.values()), 44200)
        self.assertTrue(all(counts[i] == 3 for i in hard))
        self.assertTrue(all(counts[i] == 1 for i in removed))
        for i in ('train/neg/10383_2.txt','train/pos/7504_7.txt','train/pos/9854_7.txt'):
            self.assertNotIn(i,hard)
            self.assertEqual(counts[i],2)
        contract=json.loads((ROOT/'reports/laya-resampling-45-preparation-v2/execution-contract.json').read_text())
        from laya_resampling import id_hash
        for epoch,plan in enumerate(plans):
            order=list(plan);random.Random(43+epoch).shuffle(order)
            self.assertEqual(id_hash([ids[i] for i in order]),contract['epoch_ordered_id_sha256']['resampled'][epoch])
        from collections import Counter
        for plan in plans:
            self.assertEqual(Counter(labels[ids[i]] for i in plan), {0: 11065, 1: 11035})
            self.assertEqual((len(plan) + 7) // 8, 2763)

    def test_bad_modes_memberships_and_options_rejected(self):
        for value in ({'mode': 'weighted'}, options('control') | {'candidate_sha256':'bad'}, options('control') | {'smoke_items':True}):
            with self.assertRaises(ValueError): validate_sampling(value)
        with self.assertRaises(ValueError):make_plan(['a','a'], {'a':0}, set(), 43, 'control')
        with self.assertRaises(ValueError):make_plan(['a'], {'a':0}, {'b'}, 43, 'resampled')
        with self.assertRaises(ValueError):make_plan(['a'], {'a':0}, {'a'}, 43, 'resampled')

    def test_control_numerically_identical_and_treatment_uses_budget(self):
        import torch
        import laya.train as upstream
        torch.set_num_threads(1)
        ids, labels, hard = source_data()
        selected = set(hard)
        for label in (0,1):
            selected.update([i for i in ids if labels[i]==label and i not in hard][:64-sum(labels[i]==label for i in hard)])
        ids = [i for i in ids if i in selected]
        labels = {i:labels[i] for i in ids}
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Embedding(8,4);self.head=torch.nn.Linear(4,1);self.dropout=torch.nn.Dropout(.2)
            def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype,detach_encoder=False):
                return self.head(self.dropout(self.encoder(input_ids))).squeeze(-1).gather(1,marker_pos),None
        config=upstream.TrainConfig(epochs=2,micro_batch=8,grad_accum=1,encoder_lr=.02,head_lr=.04,min_lr=.001,loss='soft-ce',amp=False,gradient_checkpointing=False,log_every=0,label_smoothing=.05,seed=43)
        items=[{'sample_id':i,'ids':[1,2+labels[i]],'markers':[0,1],'qtype':0,'q':{'t':'choice'},'k':2,'target':[1.-labels[i],float(labels[i])]} for i in ids]
        original_items=copy.deepcopy(items);original_train=upstream.train_model
        results=[]
        for mode in (None,'control','resampled','failure'):
            torch.manual_seed(17);model=Toy();observed=[]
            fail = mode == 'failure'
            if fail:
                model.head.weight.data.fill_(float('nan'))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)
                def encode(tok,item,*args,**kwargs):observed.append((item['sample_id'],list(item['target'])));return item
                with patch.object(upstream,'encode_item',encode),ExitStack() as stack:
                    stack.enter_context(guard_training(torch,upstream))
                    stack.enter_context(resample_training(upstream,options('resampled' if fail else mode) if mode else None,ids,labels,path))
                    stack.enter_context(optimize_training(torch,upstream,{'warmup_ratio':.05},path))
                    if fail:
                        with self.assertRaises(FloatingPointError):
                            upstream.train_model(model,SimpleNamespace(pad_token_id=0),items,config,torch.device('cpu'),512,192)
                    else:
                        history=upstream.train_model(model,SimpleNamespace(pad_token_id=0),items,config,torch.device('cpu'),512,192)
                audit=json.loads((path/'sampling_audit.json').read_text()) if mode else None
                update=json.loads((path/'optimization_diagnostics.json').read_text())
                if fail:
                    self.assertEqual(update['completed_optimizer_updates'],0)
                    self.assertEqual(audit['state'],'failed')
                    self.assertIs(upstream.train_model,original_train)
                    continue
                self.assertEqual(update['completed_optimizer_updates'],32)
                if mode:self.assertEqual(audit['state'],'completed');self.assertEqual(audit['epochs_observed'],[1,2])
                results.append((model.state_dict(),torch.get_rng_state(),history,observed))
            self.assertIs(upstream.train_model,original_train)
            self.assertEqual(items,original_items)
        self.assertEqual(results[0][2:],results[1][2:])
        self.assertTrue(torch.equal(results[0][1],results[1][1]))
        for key in results[0][0]:self.assertTrue(torch.equal(results[0][0][key],results[1][0][key]))
        self.assertNotEqual(results[1][3],results[2][3])


if __name__=='__main__':unittest.main()
