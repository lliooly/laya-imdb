from contextlib import contextmanager
import io
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_head_control import HeadControl,save_fp32,state_digest
from laya_checkpoint_delivery import (LocalTransport,seal_bundle,receive,wait_for_receipt,
    DeliveryBlocked,request_sha,atomic_json,digest,copy_verified,verify_files,validate_request)
from laya_robust_training import EpochContext


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def bundle(self,root,stage='epoch-001'):
        root=Path(root);run='tiny-delivery';folder=root/'checkpoints'/run/stage
        identity={'run_id':run,'seed':42,'arm':'mixed','contract_sha256':hashlib.sha256(b'{}').hexdigest()}
        enc=BertModel(BertConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=1,
            intermediate_size=32,max_position_embeddings=32,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
        model=HeadControl(enc,'original',42)
        save_fp32(model,folder)
        atomic_json(folder/'manifest.json',{'identity':identity,'calibration_ids':[f'cal-{i}' for i in range(400)]})
        atomic_json(folder/'epoch.json',{'epoch':int(stage[-1]),'updates':1,'exposures':2})
        for leaf in ('actual_inputs.jsonl','updates.jsonl'):(folder/leaf).write_text('{}\n')
        for leaf in ('source-lock.json','study-contract.json','laya_model_lock.json','uv.lock','cuda-uv.lock','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','source/scripts/source.py'):
            p=folder/'assets'/leaf;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        if stage=='epoch-001':
            import random
            optimizer=torch.optim.AdamW([{'params':list(model.encoder.parameters())},
                {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')]}],lr=.001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2)
            # A real optimizer state, not an empty fake recovery fixture.
            sum(p.square().sum() for p in model.parameters()).backward();optimizer.step();scheduler.step();optimizer.zero_grad()
            # Re-save after update so the recovery state is tied to actual saved weights.
            shutil.rmtree(folder/'encoder');(folder/'model.safetensors').unlink();(folder/'head_control.json').unlink()
            temp=folder.with_name('updated');save_fp32(model,temp)
            for p in temp.iterdir():shutil.move(str(p),str(folder/p.name))
            temp.rmdir()
            ctx=EpochContext(model,optimizer,scheduler,random.Random(42),torch.device('cpu'),identity,0,1,2,[.5])
            torch.save(ctx.snapshot(),folder/'resume.pt')
        req=seal_bundle(root,str(folder.relative_to(root)),identity,stage)
        return req,folder,model

    def test_actual_delivery_strict_child_reload_and_idempotent_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,folder,model=self.bundle(root/'remote');local=root/'local'
            receipt=receive(LocalTransport(root/'remote'),local,req)
            self.assertTrue(receipt['verification']['independent_process'])
            self.assertEqual(receipt['verification']['state_sha256'],state_digest(model.state_dict()))
            self.assertEqual(wait_for_receipt(root/'remote',req,timeout=0),receipt)
            verify_files(local,req)
            self.assertEqual(receive(LocalTransport(root/'remote'),local,req),receipt)
            from run_laya_robust_loss import production_receipt
            import run_laya_robust_loss
            with patch.object(run_laya_robust_loss,'ROOT',root/'remote'):
                with self.assertRaisesRegex(ValueError,'off-host SSH'):
                    production_receipt(req,0)

    def test_epoch2_freeze_and_development_results_are_all_delivered(self):
        from laya_head_control import fit_temperature,logit_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);remote=root/'remote';local=root/'local'
            req,folder,model=self.bundle(remote,'epoch-002')
            receive(LocalTransport(remote),local,req)
            model.eval()
            with torch.inference_mode():
                ids=torch.tensor([[1,2,3,4]]);markers=torch.tensor([[1,3]])
                logits=model(ids,torch.ones_like(ids),markers,torch.ones_like(markers,dtype=torch.bool),torch.zeros(1,dtype=torch.long))[0].tolist()[0]
            rows=[{'sample_id':f'cal-{i}','true_label':i%2} for i in range(400)]
            prediction={'rows':rows,'logits':[logits]*400}
            fit=fit_temperature([logits[1]-logits[0]]*400,[r['true_label'] for r in rows])
            frozen={'identity':req['identity'],'checkpoint_state_sha256':state_digest(model.state_dict()),
                'temperature_fit':fit,'independent_reload_logits_bitwise_equal':True}
            stage=folder.parent/'frozen';stage.mkdir()
            for leaf,data in [('frozen.json',frozen),('calibration-reference.json',prediction),('calibration-reloaded.json',prediction)]:
                atomic_json(stage/leaf,data)
            freeze=seal_bundle(remote,str(stage.relative_to(remote)),req['identity'],'frozen',checkpoint_bundle=req['bundle'],epoch_request_sha256=request_sha(req))
            receive(LocalTransport(remote),local,freeze)
            self.assertTrue(wait_for_receipt(remote,freeze,timeout=0)['verification']['strict_load'])
            devrows=[{'sample_id':f'dev-{i}','true_label':i%2,'truncated':i%6==0} for i in range(2500)]
            z=[logits]*2500;y=[r['true_label'] for r in devrows];trunc=[r['truncated'] for r in devrows]
            dev={'identity':req['identity'],'temperature':fit['temperature'],'rows':devrows,'logits':z,
                'raw_metrics':logit_metrics(y,z,trunc),'calibrated_metrics':logit_metrics(y,z,trunc,fit['temperature'])}
            stage=folder.parent/'evaluated';stage.mkdir();atomic_json(stage/'validation.json',dev);atomic_json(stage/'frozen.json',frozen)
            evaluated=seal_bundle(remote,str(stage.relative_to(remote)),req['identity'],'evaluated',checkpoint_bundle=req['bundle'],epoch_request_sha256=request_sha(req))
            receive(LocalTransport(remote),local,evaluated)
            self.assertEqual(json.loads((local/evaluated['bundle']/'validation.json').read_text()),dev)

    def test_real_partial_stream_interruption_then_prefix_verified_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();data=bytes(range(256))*4096
            (source/'weights').write_bytes(data);target=root/'target'
            expected={'bytes':len(data),'sha256':digest(source/'weights')}
            class Interrupted(LocalTransport):
                @contextmanager
                def open_read(self,relative,offset=0):
                    yield io.BytesIO(data[offset:offset+12345])
            with self.assertRaises(OSError):copy_verified(Interrupted(source),'weights',target,expected)
            self.assertEqual(target.with_name('target.partial').stat().st_size,12345)
            self.assertFalse(target.exists())
            copy_verified(LocalTransport(source),'weights',target,expected)
            self.assertEqual(target.read_bytes(),data)
            target.unlink();target.with_name('target.partial').write_bytes(b'bad prefix')
            copy_verified(LocalTransport(source),'weights',target,expected)
            self.assertEqual(target.read_bytes(),data)

    def test_ssh_remote_program_handles_binary_prefix_and_not_yet_created_request(self):
        from laya_checkpoint_delivery import SSHTransport
        with tempfile.TemporaryDirectory(prefix='delivery remote ') as tmp:
            root=Path(tmp);(root/'weights').write_bytes(b'\x00\x01\xffbinary')
            transport=SSHTransport('example.invalid',22,str(root))
            def remote_program(relative,action,count=0,data=None):
                # Execute the exact quoted remote agent locally; no SSH/network is used.
                command=shlex.split(transport.command(relative,action,count)[-1])
                command[0]=sys.executable
                return subprocess.run(command,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(remote_program('weights','read',2).stdout,b'\xffbinary')
            self.assertEqual(remote_program('weights','prefix',3).stdout.decode().strip(),digest(root/'weights',3))
            self.assertEqual(remote_program('delivery/pending.json','read').returncode,44)
            self.assertEqual(remote_program('delivery/acks/x.json','write',data=b'{"ok":true}').returncode,0)
            self.assertTrue(json.loads((root/'delivery/acks/x.json').read_text())['ok'])
            with self.assertRaises(ValueError):transport.command('../secret','read')

    def test_checksum_missing_file_and_stale_receipt_block_progress(self):
        for fault in ('corrupt','missing','stale'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);req,folder,_=self.bundle(root/'remote')
                if fault=='corrupt':(folder/'model.safetensors').write_bytes(b'corrupt')
                elif fault=='missing':(folder/'model.safetensors').unlink()
                else:
                    ack={'schema':'laya-local-delivery-v1','state':'verified_local','request_sha256':'0'*64}
                    atomic_json(root/'remote/delivery/acks'/f'{request_sha(req)}.json',ack)
                    with self.assertRaises(DeliveryBlocked):wait_for_receipt(root/'remote',req,timeout=0)
                    continue
                with self.assertRaises(DeliveryBlocked):receive(LocalTransport(root/'remote'),root/'local',req)
                with self.assertRaises(DeliveryBlocked):wait_for_receipt(root/'remote',req,timeout=0)
                self.assertFalse((root/'remote/delivery/acks'/f'{request_sha(req)}.json').exists())

    def test_lost_ack_keeps_durable_receipt_and_can_be_resent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,_,_=self.bundle(root/'remote')
            class LostAck(LocalTransport):
                def write_json(self,path,value):
                    if '/acks/' in path:raise OSError('connection lost before ACK')
                    return super().write_json(path,value)
            with self.assertRaises(DeliveryBlocked):receive(LostAck(root/'remote'),root/'local',req,attempts=1)
            self.assertTrue((root/'local/delivery/receipts'/f'{request_sha(req)}.json').exists())
            with self.assertRaises(DeliveryBlocked):wait_for_receipt(root/'remote',req,timeout=0)
            receipt=receive(LocalTransport(root/'remote'),root/'local',req)
            self.assertEqual(wait_for_receipt(root/'remote',req,timeout=0),receipt)

    def test_disk_failure_or_no_receiver_never_acknowledges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,_,_=self.bundle(root/'remote')
            with patch('laya_checkpoint_delivery.copy_verified',side_effect=OSError('No space left on device')):
                with self.assertRaises(DeliveryBlocked):receive(LocalTransport(root/'remote'),root/'local',req,attempts=1)
            self.assertFalse((root/'local/delivery/receipts'/f'{request_sha(req)}.json').exists())
            with self.assertRaises(DeliveryBlocked):wait_for_receipt(root/'remote',req,timeout=0)
        with tempfile.TemporaryDirectory() as tmp:
            req,_,_=self.bundle(Path(tmp))
            with self.assertRaises(DeliveryBlocked):wait_for_receipt(tmp,req,timeout=.01,poll=.001)

    def test_path_traversal_symlinks_and_incomplete_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,folder,_=self.bundle(root/'remote')
            req['files'][0]['path']='../secret'
            with self.assertRaises(ValueError):validate_request(req)
            (folder/'link').symlink_to(root/'secret')
            with self.assertRaises(ValueError):seal_bundle(root/'remote','checkpoints/tiny-delivery/epoch-001',req['identity'],'epoch-001')

    def test_valid_hash_but_unloadable_checkpoint_does_not_get_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req,folder,_=self.bundle(root/'remote')
            # Re-seal to make transport hashes valid; the tensor loader must still fail.
            (folder/'model.safetensors').write_bytes(b'not safetensors')
            (root/'remote/delivery/requests'/f'{req["identity"]["run_id"]}.epoch-001.json').unlink()
            req=seal_bundle(root/'remote',req['bundle'],req['identity'],'epoch-001')
            with self.assertRaises(DeliveryBlocked):receive(LocalTransport(root/'remote'),root/'local',req,attempts=1)
            self.assertFalse((root/'remote/delivery/acks'/f'{request_sha(req)}.json').exists())


if __name__=='__main__':unittest.main()
