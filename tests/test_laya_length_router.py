"""Check actual dispatch, no duplicate inference, order and length semantics."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_length_router import routes, route_predict


class LengthRouterTests(unittest.TestCase):
    def test_dispatches_each_sample_once_and_restores_order(self):
        calls=[]
        def callback(cap):
            def predict(indices):
                calls.append((cap,indices))
                return [[i,cap] for i in indices]
            return predict
        logits,chosen=route_predict([467,1,466,2003,0],callback(512),callback(2048))
        self.assertEqual(calls,[(512,[1,2,4]),(2048,[0,3])])
        np.testing.assert_array_equal(logits,[[0,2048],[1,512],[2,512],[3,2048],[4,512]])
        np.testing.assert_array_equal(chosen,[2048,512,512,2048,512])

    def test_skips_empty_branch_and_handles_empty_input(self):
        def unused(indices):raise AssertionError('Empty branch called')
        a,_=route_predict([466],lambda ids:[[0,1]],unused)
        self.assertEqual(a.tolist(),[[0,1]])
        b,_=route_predict([467],unused,lambda ids:[[1,0]])
        self.assertEqual(b.tolist(),[[1,0]])
        empty,_=route_predict([],unused,unused)
        self.assertEqual(empty.shape,(0,2))

    def test_rejects_ambiguous_lengths_and_invalid_callback_output(self):
        for value in (-1,466.0,True,np.bool_(False),'467',None,float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):routes([value])
        for bad in ([[float('nan'),0]],[[1]],[[1,0],[0,1]]):
            with self.assertRaises(ValueError):route_predict([0],lambda ids:bad,lambda ids:bad)
        np.testing.assert_array_equal(routes([np.int64(466),np.int32(467)]),[512,2048])


if __name__=='__main__':unittest.main()
