"""Real upstream loop and tokenizer, synthetic reviews; optional CUDA mechanical check."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, file_digest
from semantic_pair_pilot import digest, make_schedule
from laya_semantic_training import load_semantic, semantic_training, validate_semantic, schedule_digest
from laya_optimization import optimize_training
from cuda_numerics import guard_training
from laya_runtime import prepare_runtime, apply_runtime
from train_laya import QUESTION


def fixture():
    ids = [str(i) for i in range(128)]
    source = {i:{'id':i,'text':'A pleasant movie.' if int(i)%2 else 'A dull movie.',
                 'label':int(i)%2} for i in ids}
    variants = {0:('A dull movie.','A boring movie.','I dislike this movie.'),
                1:('A pleasant movie.','An enjoyable movie.','I like this movie.')}
    rows = [{'parent_id':i,'label':source[i]['label'],
             'variants':{k:{'text':text,'sha256':digest(text)}
                         for k,text in zip(('original','ordinary','semantic'),variants[int(i)%2])}}
            for i in ('0','1')]
    return {'options':{'arm':'A'},'pack':{'question':QUESTION,'rows':rows},'source':source,
            'gradient_ids':ids,'calibration_ids':['synthetic-calibration']}


class SemanticTrainingTests(unittest.TestCase):
    def test_option_hashes_and_paths_rejected(self):
        self.assertIsNone(validate_semantic(None))
        options={'arm':'A',**{k+s:('reports/'+k+'.json' if s=='_file' else '0'*64)
                             for k in ('pack','review','manifest') for s in ('_file','_sha256')}}
        validate_semantic(options)
        for key,value in [('arm','D'),('pack_file','../data.json'),('review_sha256','bad')]:
            with self.assertRaises(ValueError):validate_semantic(dict(options,**{key:value}))

    def test_real_pending_review_and_stale_files_are_rejected(self):
        bundle=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'reports/fixture';folder.mkdir(parents=True)
            manifest=folder/'manifest.json';manifest.write_text('{}')
            pack=dict(bundle['pack'],source_sha256='source',manifest_sha256=file_digest(manifest),tokenizer_sha256='tokenizer')
            (folder/'pack.json').write_text(json.dumps(pack));(folder/'review.json').write_text(json.dumps({'reviewer_kind':None}))
            options={'arm':'A',**{k+s:(f'reports/fixture/{k}.json' if s=='_file' else file_digest(folder/(k+'.json')))
                                 for k in ('pack','review','manifest') for s in ('_file','_sha256')}}
            with patch('laya_semantic_training.load_source',return_value=(bundle['source'],bundle['gradient_ids'],
                       {'question':QUESTION,'calibration_ids':bundle['calibration_ids']},'source')):
                with self.assertRaisesRegex(ValueError,'human review'):load_semantic(options,root)
                (folder/'review.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):load_semantic(options,root)

    def run_arm(self,arm=None,fail=False,reverse=False,smoke=None):
        # No real IMDB review or human acceptance is fabricated for this test.
        device_name=os.environ.get('LAYA_SEMANTIC_TEST_DEVICE','cpu')
        if device_name=='cuda':prepare_runtime({'runtime_profile':'strict','device':'cuda','training':{'amp':False}})
        import torch
        import laya.train as t
        from transformers import AutoTokenizer
        torch.set_num_threads(1)
        if device_name=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA mechanical test requested but unavailable')
            apply_runtime(torch,'strict')
        tok=AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'),local_files_only=True)
        bundle=fixture();bundle['options']['arm']=arm or 'A'
        if smoke:bundle['options']['smoke_items']=smoke
        typed=[{'state':bundle['source'][i]['text'],'questions':{'sentiment':QUESTION},
                'expected':{'sentiment':('negative','positive')[bundle['source'][i]['label']]}}
               for i in bundle['gradient_ids']]
        items,skipped=t.items_from_rows(tok,typed,512,192,label_smoothing=0.0)
        self.assertFalse(skipped);before=copy.deepcopy(items)
        if reverse:items=list(reversed(items))
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Embedding(len(tok),4)
                self.head=torch.nn.Linear(4,1);self.dropout=torch.nn.Dropout(.2)
            def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype,detach_encoder=False):
                return self.head(self.dropout(self.encoder(input_ids))).squeeze(-1).gather(1,marker_pos),None
        torch.manual_seed(17)
        if device_name=='cuda':torch.cuda.manual_seed_all(17)
        model=Toy()
        if fail:model.head.weight.data.fill_(float('nan'))
        config=t.TrainConfig(epochs=2,micro_batch=8,grad_accum=1,encoder_lr=.02,head_lr=.04,
                             min_lr=.001,loss='soft-ce',amp=False,gradient_checkpointing=False,
                             log_every=0,label_smoothing=.05,seed=42)
        original_train=t.train_model;original_encode=t.encode_item;epochs=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with ExitStack() as stack:
                stack.enter_context(guard_training(torch,t))
                stack.enter_context(semantic_training(t,bundle if arm else None,path))
                stack.enter_context(optimize_training(torch,t,{'warmup_ratio':.05},path))
                if fail or reverse:
                    with self.assertRaises(FloatingPointError if fail else ValueError):
                        t.train_model(model,tok,items,config,torch.device(device_name),512,192)
                    history=[]
                else:
                    history=t.train_model(model,tok,items,config,torch.device(device_name),512,192,
                                          on_epoch_end=lambda epoch,loss:epochs.append(epoch+1))
            self.assertIs(t.train_model,original_train);self.assertIs(t.encode_item,original_encode)
            if not reverse:self.assertEqual(items,before)
            audit=json.loads((path/'semantic_audit.json').read_text()) if arm and not reverse else None
            journal=[json.loads(line) for line in (path/'semantic_consumption.jsonl').read_text().splitlines()] if arm and not reverse else []
            optimization=json.loads((path/'optimization_diagnostics.json').read_text())
            if fail:self.assertEqual(optimization['completed_optimizer_updates'],0);self.assertEqual(audit['state'],'failed')
            elif not reverse:self.assertEqual(optimization['completed_optimizer_updates'],2*(((smoke or 128)+7)//8));self.assertEqual(epochs,[1,2])
        return {'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state().cpu() if device_name=='cuda' else None,
                'history':history,'audit':audit,'journal':journal}

    def test_control_exactness_and_three_actual_schedules(self):
        import torch
        baseline=self.run_arm()
        arms={a:self.run_arm(a) for a in ('A','B','C')}
        self.assertEqual(baseline['history'],arms['A']['history'])
        self.assertTrue(torch.equal(baseline['rng'],arms['A']['rng']))
        if baseline['cuda_rng'] is not None:self.assertTrue(torch.equal(baseline['cuda_rng'],arms['A']['cuda_rng']))
        for key in baseline['weights']:self.assertTrue(torch.equal(baseline['weights'][key],arms['A']['weights'][key]))
        bundle=fixture();expected=make_schedule(bundle['gradient_ids'],{i:r['label'] for i,r in bundle['source'].items()},['0','1'])
        for arm,result in arms.items():
            self.assertEqual(result['audit']['state'],'completed')
            self.assertEqual(result['audit']['consumed_counts'],[128,128])
            self.assertEqual(result['audit']['actual_schedule_sha256'],[schedule_digest(e) for e in expected])
            actual=[[{k:r[k] for k in ('original_id','parent_id','augmented','label')} for r in result['journal'] if r['epoch']==epoch] for epoch in (1,2)]
            self.assertEqual(actual,expected)
        for records in zip(*(arms[a]['journal'] for a in ('A','B','C'))):
            if not records[0]['augmented']:self.assertEqual(records[0],records[1]);self.assertEqual(records[1],records[2])
            else:self.assertEqual(len({r['input_sha256'] for r in records}),3)

    def test_smoke_subset_preserves_original_partition_and_actual_budget(self):
        result=self.run_arm('C',smoke=64)
        self.assertEqual(result['audit']['original_gradient_count'],128)
        self.assertEqual(result['audit']['actual_train_count'],64)
        self.assertEqual(result['audit']['consumed_counts'],[64,64])
        self.assertEqual(len(result['journal']),128)

    def test_nonfinite_stops_without_updates_and_restores_hooks(self):
        self.run_arm('C',fail=True)

    def test_reordered_original_partition_is_rejected(self):
        self.run_arm('B',reverse=True)


if __name__=='__main__':unittest.main()
