import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch

from deberta_inputs import encode,training_rows
from deberta_training import (RECIPE,seed_all,optimizer_for,groups_for,update,snapshot,restore)
from deberta_study import seal,check_bundle


class FakeTokenizer:
    def encode(self,text,**kwargs):return list(range(int(text)))
    def num_special_tokens_to_add(self,pair=False):return 2
    def build_inputs_with_special_tokens(self,ids):return [9998]+ids+[9999]


class Tiny(torch.nn.Module):
    def __init__(self,dropout=0.):
        super().__init__();self.emb=torch.nn.Embedding(30,6)
        self.dropout=torch.nn.Dropout(dropout);self.classifier=torch.nn.Linear(6,2)
    def forward(self,input_ids,attention_mask):
        x=self.emb(input_ids);mask=attention_mask.unsqueeze(-1)
        x=(x*mask).sum(1)/mask.sum(1)
        return SimpleNamespace(logits=self.classifier(self.dropout(x)))


class DebertaTests(unittest.TestCase):
    def test_full_and_head_tail_specials(self):
        t=FakeTokenizer()
        for n in [1,509,510,511,900]:
            x=encode(t,str(n));self.assertEqual(x['review_tokens'],n)
            want=list(range(n)) if n<=510 else list(range(255))+list(range(n-255,n))
            self.assertEqual(x['input_ids'],[9998]+want+[9999]);self.assertLessEqual(len(x['input_ids']),512)

    def test_partial_accumulation_matches_full_batch(self):
        seed_all();a=Tiny();b=copy.deepcopy(a);items=[dict(input_ids=[i+1,2],label=i%2) for i in range(5)]
        oa,sa=optimizer_for(a);ob,sb=optimizer_for(b)
        x=update(a,items,0,oa,sa,torch.device('cpu'),micro=2)
        y=update(b,items,0,ob,sb,torch.device('cpu'),micro=8)
        self.assertAlmostEqual(x['loss'],y['loss'],places=6)
        for u,v in zip(a.parameters(),b.parameters()):torch.testing.assert_close(u,v,rtol=1e-6,atol=1e-7)

    def test_exact_optimizer_scheduler_rng_resume(self):
        seed_all();model=Tiny(.25);opt,sched=optimizer_for(model)
        items=[dict(input_ids=[i+1,2],label=i%2) for i in range(7)];device=torch.device('cpu')
        update(model,items,0,opt,sched,device,micro=3)
        state=snapshot(model,opt,sched,device,{'id':'tiny'},{'updates':1});weights=copy.deepcopy(model.state_dict())
        update(model,items,0,opt,sched,device,micro=3);expected=copy.deepcopy(model.state_dict())
        other=Tiny(.25);other.load_state_dict(weights);oo,ss=optimizer_for(other)
        restore(state,other,oo,ss,device,{'id':'tiny'});update(other,items,0,oo,ss,device,micro=3)
        for name,v in expected.items():self.assertTrue(torch.equal(v,other.state_dict()[name]),name)
        self.assertEqual(sched.state_dict(),ss.state_dict())

    def test_frozen_members_and_two_epoch_counts(self):
        rows,split=training_rows();self.assertEqual(len(rows),22500)
        for epoch in [0,1]:
            groups=groups_for(22100,epoch);self.assertEqual(len(groups),691)
            self.assertEqual(len(groups[-1]),20);self.assertEqual(sorted(sum(groups,[])),list(range(22100)))
        self.assertEqual(RECIPE['total_steps'],1382)

    def test_immutable_bundle_rejects_tamper(self):
        import deberta_study as study
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'bundle';folder.mkdir();(folder/'weights').write_bytes(b'a')
            from unittest.mock import patch
            with patch.object(study,'ROOT',root),patch.object(study,'identity',lambda:{'study_id':study.STUDY}):
                seal(folder,'test');check_bundle(folder);(folder/'weights').write_bytes(b'b')
                with self.assertRaises(ValueError):check_bundle(folder)


if __name__=='__main__':unittest.main()
