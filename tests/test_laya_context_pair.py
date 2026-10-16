"""Scientific invariance, length retention and genuine epoch recovery tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from common import ROOT
from laya_context_pair import items_for, training_config, seal, check_bundle, wilson
from laya_context_training import setup, run_epoch, snapshot, restore, Journal
from laya_head_control import HeadControl, state_digest, save_fp32, load_fp32


def tiny():
    torch.manual_seed(79)
    encoder = BertModel(BertConfig(vocab_size=50368, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=32, max_position_embeddings=2048,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    return HeadControl(encoder, 'original', 42)


class ContextPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)
        cls.rows = [{'id':str(i), 'text':('Wonderful movie. ' if i%2 else 'Awful film. ')*(180 if i==0 else 2), 'label':i%2} for i in range(4)]
        cls.ids = [r['id'] for r in cls.rows]

    def cfg(self, cap):
        c = training_config(cap, 2); c.log_every=0
        return c

    def test_512_matches_original_ce_parameters_losses_and_rng(self):
        import laya.train as upstream
        items,_ = items_for(self.tok,self.rows,512)
        original=tiny(); newer=copy.deepcopy(original)
        expected=upstream.train_model(original,self.tok,items,self.cfg(512),torch.device('cpu'),512,192,parallel=False)
        expected_rng=torch.get_rng_state().clone()
        opt,sched,rng=setup(newer,self.cfg(512),len(items),torch.device('cpu'))
        actual=[run_epoch(newer,self.tok,items,self.ids,self.cfg(512),torch.device('cpu'),512,e,opt,sched,rng)[0] for e in (0,1)]
        self.assertEqual(expected,actual)
        self.assertEqual(state_digest(original.state_dict()),state_digest(newer.state_dict()))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))

    def test_resume_both_lengths_preserves_parameters_rng_and_audit(self):
        for cap in (512,2048):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp); items,sizes=items_for(self.tok,self.rows,cap)
                self.assertGreater(sizes[0],466)
                self.assertEqual(len(items[0]['state_ids']),min(sizes[0],cap-46))
                initial=tiny(); model=copy.deepcopy(initial); cfg=self.cfg(cap); ident={'cap':cap}
                opt,sched,rng=setup(model,cfg,len(items),torch.device('cpu'))
                j=Journal(folder/'e0')
                loss,audit=run_epoch(model,self.tok,items,self.ids,cfg,torch.device('cpu'),cap,0,opt,sched,rng,j)
                j.close(); save_fp32(model,folder/'saved')
                torch.save(snapshot(model,opt,sched,rng,torch.device('cpu'),ident,0,len(items),[{'loss':loss}],2),folder/'state.pt')
                j=Journal(folder/'continuous')
                expected=run_epoch(model,self.tok,items,self.ids,cfg,torch.device('cpu'),cap,1,opt,sched,rng,j)[0]
                j.close(); digest=state_digest(model.state_dict()); final_rng=torch.get_rng_state().clone()
                reloaded,_=load_fp32(folder/'saved')
                opt,sched,rng=setup(reloaded,cfg,len(items),torch.device('cpu'))
                state=torch.load(folder/'state.pt',weights_only=True)
                with self.assertRaises(ValueError):restore(state,reloaded,opt,sched,rng,torch.device('cpu'),{},len(items),2,1)
                restore(state,reloaded,opt,sched,rng,torch.device('cpu'),ident,len(items),2,1)
                j=Journal(folder/'resumed')
                actual=run_epoch(reloaded,self.tok,items,self.ids,cfg,torch.device('cpu'),cap,1,opt,sched,rng,j)[0]
                j.close()
                self.assertEqual(expected,actual);self.assertEqual(digest,state_digest(reloaded.state_dict()))
                self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
                self.assertEqual((folder/'continuous/updates.jsonl').read_bytes(),(folder/'resumed/updates.jsonl').read_bytes())
                import gzip
                self.assertEqual(gzip.open(folder/'continuous/inputs.jsonl.gz').read(),gzip.open(folder/'resumed/inputs.jsonl.gz').read())

    def test_same_sample_option_schedule_at_different_lengths(self):
        streams=[]
        with tempfile.TemporaryDirectory() as tmp:
            for cap in (512,2048):
                model=tiny();cfg=self.cfg(cap);items,_=items_for(self.tok,self.rows,cap)
                opt,sched,rng=setup(model,cfg,len(items),torch.device('cpu'));j=Journal(Path(tmp)/str(cap))
                _,a=run_epoch(model,self.tok,items,self.ids,cfg,torch.device('cpu'),cap,0,opt,sched,rng,j);j.close()
                streams.append(a)
        self.assertEqual(streams[0]['schedule_sha256'],streams[1]['schedule_sha256'])
        self.assertNotEqual(streams[0]['input_sha256'],streams[1]['input_sha256'])

    def test_deadline_stops_before_update_and_wilson_is_valid(self):
        model=tiny();cfg=self.cfg(512);items,_=items_for(self.tok,self.rows,512)
        opt,sched,rng=setup(model,cfg,len(items),torch.device('cpu'));before=state_digest(model.state_dict())
        with self.assertRaises(TimeoutError):run_epoch(model,self.tok,items,self.ids,cfg,torch.device('cpu'),512,0,opt,sched,rng,deadline=0)
        self.assertEqual(before,state_digest(model.state_dict()))
        lo,hi=wilson(2400,2500);self.assertLess(lo,.96);self.assertGreater(hi,.96)

    def test_sealed_bundle_detects_modified_bytes(self):
        (ROOT/'artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            from laya_context_pair import STUDY
            folder=Path(tmp);(folder/'weights.txt').write_text('original')
            seal(folder,{'study_id':STUDY},'test')
            check_bundle(folder)
            (folder/'weights.txt').write_text('changed')
            with self.assertRaises(ValueError):check_bundle(folder)


if __name__=='__main__':unittest.main()
