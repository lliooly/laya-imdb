import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
import deberta_training as original
import deberta_bf16_training as bf16
from deberta_bf16_study import install,RECIPE


class Tiny(torch.nn.Module):
    def __init__(self,dropout=0.):
        super().__init__();self.embedding=torch.nn.Embedding(20,5)
        self.drop=torch.nn.Dropout(dropout);self.classifier=torch.nn.Linear(5,2)
    def forward(self,input_ids,attention_mask):
        mask=attention_mask.unsqueeze(-1);x=(self.embedding(input_ids)*mask).sum(1)/mask.sum(1)
        return SimpleNamespace(logits=self.classifier(self.drop(x)))


class Bf16Tests(unittest.TestCase):
    def test_only_recipe_precision_changes(self):
        self.assertEqual(RECIPE,dict(original.RECIPE,precision='bfloat16'))

    def test_cpu_control_exact_including_tail_group(self):
        original.seed_all();a=Tiny();b=copy.deepcopy(a)
        oa,sa=original.optimizer_for(a);ob,sb=bf16.optimizer_for(b)
        items=[dict(input_ids=[i+1,2],label=i%2) for i in range(5)]
        x=original.update(a,items,0,oa,sa,torch.device('cpu'),micro=2)
        y=bf16.update(b,items,0,ob,sb,torch.device('cpu'),micro=2)
        self.assertEqual(x['loss'],y['loss'])
        for u,v in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(u,v))

    def test_precision_bound_resume_restores_rng_exactly(self):
        original.seed_all();a=Tiny(.2);opt,sched=bf16.optimizer_for(a)
        items=[dict(input_ids=[i+1,3],label=i%2) for i in range(7)];device=torch.device('cpu')
        bf16.update(a,items,0,opt,sched,device,micro=3)
        state=bf16.snapshot(a,opt,sched,device,{'id':'bf16'},{'updates':1});weights=copy.deepcopy(a.state_dict())
        bf16.update(a,items,0,opt,sched,device,micro=3);expected=copy.deepcopy(a.state_dict())
        b=Tiny(.2);b.load_state_dict(weights);oo,ss=bf16.optimizer_for(b)
        bf16.restore(state,b,oo,ss,device,{'id':'bf16'});bf16.update(b,items,0,oo,ss,device,micro=3)
        for n,v in expected.items():self.assertTrue(torch.equal(v,b.state_dict()[n]))
        bad=dict(state,compute_precision='float16')
        with self.assertRaises(ValueError):bf16.restore(bad,b,oo,ss,device,{'id':'bf16'})

    def test_private_adapter_leaves_historical_objects_unchanged(self):
        before=original.update;recipe=copy.deepcopy(original.RECIPE)
        cp=install();self.assertIs(original.update,before);self.assertEqual(original.RECIPE,recipe)
        self.assertEqual(cp.STUDY,'deberta-heterogeneous-bf16-s42-modal-20261017-v1')

    def test_cpu_and_storage_references_do_not_enable_cuda_autocast(self):
        from unittest.mock import patch
        with patch.object(torch,'autocast',side_effect=AssertionError('Autocast must stay off')):
            with bf16.autocast_context(torch.device('cpu')):pass
            with bf16.autocast_context(torch.device('cuda'),reference=True):pass


if __name__=='__main__':unittest.main()
