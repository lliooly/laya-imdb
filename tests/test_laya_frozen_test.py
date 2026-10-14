import unittest
from unittest.mock import Mock
import numpy as np
from common import digest
from evaluate_laya_frozen_test import ProgressInference, validate_rows, wilson

class FrozenTestTests(unittest.TestCase):
    def test_progress_blocks_preserve_global_batch_groups_and_reuse(self):
        class Base:
            def _infer(self,name,items,indices,stats,ids,reuse=None):
                for start in range(0,len(indices),8):self.groups.append(indices[start:start+8])
                out=[None]*len(items)
                for i in indices:out[i]={'sample_id':ids[i],'value':i}
                return out
        class Engine(ProgressInference,Base):pass
        e=Engine();e.groups=[];e.forward_rows_completed=0;e.progress=Mock()
        n=700;changed=list(range(1,n,2));reuse=[{'sample_id':str(i),'value':i} for i in range(n)]
        out=e._infer('a',[None]*n,changed,[None]*n,list(map(str,range(n))),reuse=reuse)
        self.assertEqual(e.groups,[changed[i:i+8] for i in range(0,len(changed),8)])
        self.assertEqual(e.forward_rows_completed,len(changed))
        self.assertTrue(out[0]['reused_head_prediction']);self.assertNotIn('reused_head_prediction',out[1])
        self.assertEqual([r['value'] for r in out],list(range(n)))
    def test_sealed_rows_and_interval(self):
        rows=[{'id':str(i),'label':i%2,'text':'review '+str(i),'text_sha256':digest('review '+str(i))} for i in range(4)]
        validate_rows(rows,list(map(str,range(4))),4)
        with self.assertRaises(ValueError):validate_rows(rows,list(reversed(list(map(str,range(4))))),4)
        rows[0]['text']='changed'
        with self.assertRaises(ValueError):validate_rows(rows,list(map(str,range(4))),4)
        ci=wilson(24500,25000);self.assertLess(ci['lower'],.98);self.assertGreater(ci['upper'],.98)
        self.assertEqual(wilson(0,10)['lower'],0.)

if __name__=='__main__':unittest.main()
