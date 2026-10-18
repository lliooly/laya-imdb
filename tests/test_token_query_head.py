import sys
from pathlib import Path
import unittest
import subprocess

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from token_query_head import KINDS, TokenQueryHead
from run_token_query_study import fit_mixture


class QueryHeadTests(unittest.TestCase):
    def test_padding_cannot_change_prediction_or_receive_attention(self):
        torch.manual_seed(42)
        h = torch.randn(2, 43, 16)
        mask = torch.ones(2,43,dtype=torch.bool)
        mask[0,19:] = False
        markers=torch.tensor([[1,2],[3,4]])
        for kind in KINDS:
            model=TokenQueryHead(kind,16,8,0.).eval()
            z,w=model(h,mask,markers,True)
            changed=h.clone()
            changed[0,19:]=torch.randn_like(changed[0,19:])*100
            torch.testing.assert_close(z,model(changed,mask,markers),atol=0,rtol=0)
            self.assertEqual(w[0,:,19:].count_nonzero(),0)
            torch.testing.assert_close(w.sum(-1),torch.ones(2,2))
            if kind=='top32':
                self.assertEqual(w[0].count_nonzero(),38)
                self.assertEqual(w[1].count_nonzero(),64)

    def test_same_initialization_all_finite_gradients_and_update(self):
        initial=None
        for kind in KINDS:
            torch.manual_seed(43)
            model=TokenQueryHead(kind,16,8,0.)
            current={k:v.detach().clone() for k,v in model.state_dict().items()}
            if initial is None:
                initial=current
            else:
                self.assertTrue(all(torch.equal(v,initial[k]) for k,v in current.items()))
            h=torch.randn(3,40,16)
            mask=torch.ones(3,40,dtype=torch.bool)
            z=model(h,mask,torch.tensor([[1,2]]*3))
            torch.nn.functional.cross_entropy(z,torch.tensor([0,1,0])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(model.tokens[0].weight.grad.abs().sum(),0)
            self.assertGreater(model.queries.grad.abs().sum(),0)

    def test_short_sequences_and_invalid_markers(self):
        model=TokenQueryHead('top32',16,8,0.)
        h=torch.randn(1,3,16)
        mask=torch.tensor([[True,True,False]])
        z,w=model(h,mask,torch.tensor([[0,1]]),True)
        self.assertTrue(torch.isfinite(z).all())
        self.assertEqual(w[:,:,2].count_nonzero(),0)
        with self.assertRaises(ValueError):
            model(h,mask,torch.tensor([[0,2]]))

    def test_calibration_weight_boundaries(self):
        import numpy as np
        a=np.array([.4,.6,.4,.6])
        y=np.array([0,1,0,1])
        useful=np.array([.1,.9,.1,.9])
        harmful=1-useful
        self.assertEqual(fit_mixture(a,useful,y)['sparse_weight'],.25)
        self.assertEqual(fit_mixture(a,harmful,y)['sparse_weight'],0.)
        self.assertEqual(fit_mixture(a,a,y)['sparse_weight'],0.)

    def test_fit_guard_blocks_development_and_official_test(self):
        scriptdir=str(Path(__file__).resolve().parents[1]/'scripts')
        code='''
import sys
sys.path.insert(0,sys.argv[1])
from run_token_query_study import guard, ROOT
guard('train')
for path in ['data/processed/test.jsonl','data/processed/validation.jsonl',
             'artifacts/validation-tokens.npy','reports/example/predictions.json']:
    try:
        open(ROOT/path)
    except PermissionError:
        continue
    raise AssertionError('guard allowed '+path)
print('sealed')
'''
        result=subprocess.run([sys.executable,'-c',code,scriptdir],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('sealed',result.stdout)


if __name__=='__main__':
    unittest.main()
