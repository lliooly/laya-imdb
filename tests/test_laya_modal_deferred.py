import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_modal_deferred import validate_cloud_receipt,VOLUME
from laya_checkpoint_delivery import request_sha

class DeferredTests(unittest.TestCase):
    def test_cloud_receipt_cannot_claim_local_delivery(self):
        req={'identity':{'run_id':'example'},'stage':'epoch-001','files':[]}
        r={'schema':'laya-cloud-durability-v1','state':'verified_committed_volume',
           'modal_volume':VOLUME,'local_delivery_verified':False,'request_sha256':request_sha(req),
           'identity':req['identity'],'stage':req['stage'],'files':[],
           'verification':{'strict_load':True,'state_sha256':'a'*64}}
        validate_cloud_receipt(req,r)
        for change in ({'local_delivery_verified':True},{'state':'verified_local'},
                       {'request_sha256':'b'*64},{'verification':{'strict_load':False}}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_cloud_receipt(req,{**r,**change})

    def test_real_cloud_verifier_reads_actual_weights_without_local_receipt(self):
        from test_laya_checkpoint_delivery import DeliveryTests
        import laya_checkpoint_delivery as delivery
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,_,_=DeliveryTests().bundle(root)
            with patch.object(delivery,'validate_receipt',validate_cloud_receipt):
                v=delivery.verify_checkpoint(root,req)
            self.assertTrue(v['strict_load']);self.assertEqual(v['device'],'cpu')

if __name__=='__main__':unittest.main()
