import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from laya_head_control import HeadControl, state_digest, save_fp32, load_fp32
from laya_robust_training import loss_values, train_model, read_resume, capture_rng
from laya_robust_audit import TrainingAudit
from run_laya_head_control import TRAINING


def tiny_model():
    torch.manual_seed(79)
    enc = BertModel(BertConfig(vocab_size=50368, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=32, max_position_embeddings=512,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    return HeadControl(enc, 'original', 42)


class RobustTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)
        from export_laya_posthoc import prepare_rules
        rows = [{'id':str(i),'text':'A wonderful film.' if i%2 else 'An awful film.','label':i%2} for i in range(8)]
        _,cls.items,_,_,_ = prepare_rules(cls.tok,rows)
        cls.ids = [r['id'] for r in rows]

    def cfg(self):
        import laya.train as t
        return t.TrainConfig(**dict(TRAINING,seed=42,micro_batch=2,log_every=0))

    def test_gradient_formula_extremes_and_slot_permutation(self):
        logits = torch.tensor([[0.,0.],[3.,-2.],[-12.,12.],[10000.,-10000.]],dtype=torch.float64,requires_grad=True)
        target = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],dtype=torch.float64)
        mask = torch.ones_like(target,dtype=torch.bool)
        ce,_ = loss_values(logits,target,mask,'ce')
        mixed,common = loss_values(logits,target,mask,'mixed')
        g_ce = torch.autograd.grad(ce,logits,retain_graph=True)[0]
        g_mix = torch.autograd.grad(mixed,logits,retain_graph=True)[0]
        py = (torch.softmax(logits.detach(),-1)*target).sum(-1)
        factor = .5+.5*py.pow(.7)
        torch.testing.assert_close(g_mix,g_ce*factor[:,None],rtol=1e-12,atol=1e-14)
        self.assertEqual(ce.item(),common.item())
        flipped,_ = loss_values(logits.flip(-1),target.flip(-1),mask,'mixed')
        self.assertEqual(mixed.item(),flipped.item())
        self.assertTrue(torch.isfinite(g_mix).all())
        import laya.train as t
        torch.testing.assert_close(ce,t.soft_ce_loss(logits,target,mask),rtol=0,atol=0)

    def test_reject_soft_missing_or_nonfinite_labels_and_logits(self):
        z=torch.zeros(2,2);y=torch.eye(2);m=torch.ones(2,2,dtype=torch.bool)
        for bad in (torch.full((2,2),.5),torch.zeros(2,2),torch.full((2,2),float('nan'))):
            with self.assertRaises((ValueError,FloatingPointError)):loss_values(z,bad,m,'mixed')
        with self.assertRaises(ValueError):loss_values(z,y,~m,'mixed')
        with self.assertRaises(FloatingPointError):loss_values(z+float('inf'),y,m,'ce')
        with self.assertRaises(ValueError):loss_values(z,y,m,'other')

    def test_ce_loop_matches_pinned_upstream_parameters_rng_and_losses(self):
        import laya.train as t
        initial=tiny_model();a=copy.deepcopy(initial);b=copy.deepcopy(initial)
        expected=t.train_model(a,self.tok,self.items,self.cfg(),torch.device('cpu'),512,192,parallel=False)
        rng=torch.get_rng_state().clone()
        actual=train_model(b,self.tok,self.items,self.cfg(),torch.device('cpu'),'ce',{'run_id':'tiny'},lambda *args:None)
        self.assertEqual(expected,actual)
        self.assertEqual(state_digest(a.state_dict()),state_digest(b.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_epoch_resume_matches_continuous_inputs_updates_and_weights(self):
        import laya.train as t
        initial=tiny_model();identity={'run_id':'resume-test','seed':42,'arm':'mixed'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=copy.deepcopy(initial)
            audit_a=TrainingAudit(t,a,self.tok,self.items,self.ids,self.cfg(),root/'continuous',True)
            expected=train_model(a,self.tok,self.items,self.cfg(),torch.device('cpu'),'mixed',identity,lambda *args:None,audit_a)
            audit_a.finish();audit_a.close();expected_rng=torch.get_rng_state().clone()
            b=copy.deepcopy(initial)
            audit_b=TrainingAudit(t,b,self.tok,self.items,self.ids,self.cfg(),root/'interrupted',True)
            class StopAfterSave(Exception):pass
            def stop(epoch,loss,ctx):
                self.assertEqual(epoch,0)
                audit_b.flush();save_fp32(b,root/'saved')
                torch.save(ctx.snapshot(),root/'saved/resume.pt')
                for leaf in ('actual_inputs.jsonl','updates.jsonl'):
                    (root/'saved'/leaf).write_bytes((root/'interrupted'/leaf).read_bytes())
                raise StopAfterSave()
            with self.assertRaises(StopAfterSave):
                train_model(b,self.tok,self.items,self.cfg(),torch.device('cpu'),'mixed',identity,stop,audit_b)
            audit_b.close()
            c,meta=load_fp32(root/'saved')
            recovery=read_resume(root/'saved/resume.pt',identity,meta['state_sha256'])
            with self.assertRaises(ValueError):read_resume(root/'saved/resume.pt',{'run_id':'other'},meta['state_sha256'])
            audit_c=TrainingAudit(t,c,self.tok,self.items,self.ids,self.cfg(),root/'resumed',True,root/'saved')
            actual=train_model(c,self.tok,self.items,self.cfg(),torch.device('cpu'),'mixed',identity,lambda *args:None,audit_c,recovery)
            audit_c.finish();audit_c.close()
            self.assertEqual(expected,actual)
            self.assertEqual(state_digest(a.state_dict()),state_digest(c.state_dict()))
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
            for leaf in ('actual_inputs.jsonl','updates.jsonl','training_audit.json'):
                self.assertEqual((root/'continuous'/leaf).read_bytes(),(root/'resumed'/leaf).read_bytes(),leaf)

    def test_callback_failure_prevents_any_second_epoch_update(self):
        model=tiny_model();count=[]
        def fail(epoch,loss,ctx):
            count.append(ctx.updates)
            raise RuntimeError('No verified local receipt')
        with self.assertRaisesRegex(RuntimeError,'No verified local receipt'):
            train_model(model,self.tok,self.items,self.cfg(),torch.device('cpu'),'mixed',{},fail)
        self.assertEqual(count,[4])

    def test_loss_arms_share_inputs_but_change_learned_parameters(self):
        import laya.train as t
        initial=tiny_model();streams=[];states=[]
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('ce','mixed'):
                model=copy.deepcopy(initial)
                audit=TrainingAudit(t,model,self.tok,self.items,self.ids,self.cfg(),Path(tmp)/arm)
                train_model(model,self.tok,self.items,self.cfg(),torch.device('cpu'),arm,{},lambda *args:None,audit)
                info=audit.finish();audit.close()
                streams.append(info['input_stream_sha256']);states.append(state_digest(model.state_dict()))
        self.assertEqual(streams[0],streams[1])
        self.assertNotEqual(states[0],states[1])


if __name__ == '__main__':unittest.main()
