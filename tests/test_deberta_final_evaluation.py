import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import write_json
from comparison_metrics import paired
from deberta_final_analysis import selective,bootstrap
import deberta_final_evaluation as final


class FinalEvaluationTests(unittest.TestCase):
    def test_fifty_extra_correct_on_25000_does_not_imply_significance(self):
        y=np.zeros(25000,dtype=int);a=-np.ones(25000);b=-np.ones(25000)
        a[:1050]=1;b[1050:2050]=1
        result=paired(y,a,b)
        self.assertEqual(result['b_only_correct']-result['a_only_correct'],50)
        self.assertGreater(result['mcnemar_exact_two_sided_p'],.05)

    def test_selective_stable_ties_and_full_prefix_risk(self):
        m=selective(np.array([0,1,1,0]),np.array([.1,.9,.8,.2]))
        self.assertEqual(m['order'].tolist(),[0,1,2,3]);self.assertEqual(m['aurc'],0.)
        bad=selective(np.array([1,0]),np.array([.1,.9]))
        np.testing.assert_array_equal(bad['risk'],[1.,1.]);self.assertEqual(bad['aurc'],1.)

    def test_identical_methods_have_zero_paired_risk_curve_differences(self):
        y=np.array([0,1]*50);p=np.linspace(.001,.999,100)
        m=selective(y,p);b=bootstrap(y,{'main':m,'fusion':m},replicates=40,seed=11)
        np.testing.assert_array_equal(b['delta_fusion_minus_main_ci95'],np.zeros((100,2)))
        self.assertEqual(b['delta_aurc_fusion_minus_main_ci95'],[0.,0.])

    def test_chunk_identity_binds_contract_order_and_input_bytes(self):
        rows=[{'sample_id':'a','input_sha256':'hashA'},{'sample_id':'b','input_sha256':'hashB'}]
        a=final.chunk_identity(rows,'contract')
        self.assertNotEqual(a,final.chunk_identity(rows[::-1],'contract'))
        self.assertNotEqual(a,final.chunk_identity(rows,'different'))
        self.assertNotEqual(a,final.chunk_identity([dict(rows[0],input_sha256='changed'),rows[1]],'contract'))

    def test_paired_interval_reduces_to_zero_for_identical_predictions(self):
        y=np.array([0,1]*20);c=np.arange(40)%3==0
        self.assertEqual(final.paired_delta_interval(y,c,c,reps=100),[0.,0.])

    def test_failed_backend_admission_never_seals_test_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir();model=root/'model';model.mkdir()
            write_json(model/'reference.json',{'items':[],'pad_token_id':0,'logits':[[0.,0.]]})
            source=root/'reports'/final.TRAIN/'evaluation';source.mkdir(parents=True)
            rows=[]
            for name,count in [('calibration',400),('development',128)]:
                ids=[f'{name}/{i}' for i in range(count)]
                write_json(source/(name+'.json'),{'sample_ids':ids,'logits':[[4.,-4.]]*count})
                rows.append([{'sample_id':sid,'logits':[-4.,4.]} for sid in ids])
            c={'temperature':1.,'engineering_gate':{'max_probability_difference':.1,'mean_probability_difference':.005,'max_class_disagreement_fraction':.01}}
            with patch.object(final,'ROOT',root),patch.object(final,'OUT',out),patch.object(final,'MODEL',model),patch('deberta_training.predict',return_value=([[0.,0.]],0.)):
                with self.assertRaises(ValueError):final.admission(None,c,*rows)
            self.assertFalse(json.loads((out/'admission.json').read_text())['passed'])
            self.assertFalse((out/'test-gate.json').exists());self.assertFalse((out/'tokens/test.json').exists())


if __name__=='__main__':unittest.main()
