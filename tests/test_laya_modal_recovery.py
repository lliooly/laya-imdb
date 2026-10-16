"""Fault injection for orchestration, not a substitute for CUDA admission."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_checkpoint_delivery import atomic_json, digest, request_sha, seal_bundle
from laya_modal_recovery import (BudgetExpired, RECOVERY, STUDY, VOLUME, check_state,
                                 execute_queue, initial_state, name, plan)


JOB = {'seed': 42, 'arm': 'mixed', 'evaluate': False}


class RecoveryTests(unittest.TestCase):
    def fixture(self, root, stage):
        """Synthetic bytes test dispatch/integrity; they are never model-loaded."""
        root = Path(root)
        contract = root/'configs'/f'{STUDY}.json'
        if not contract.exists():
            atomic_json(contract, {'fixture': 'dispatch only'})
        run = name(42, 'mixed')
        folder = root/'checkpoints'/run/stage
        identity = {'study_id': STUDY, 'run_id': run, 'seed': 42, 'arm': 'mixed',
                    'mode': 'formal', 'contract_sha256': digest(contract)}
        if stage.startswith('epoch'):
            leaves = ['model.safetensors', 'encoder/config.json', 'head_control.json',
                'manifest.json', 'epoch.json', 'actual_inputs.jsonl', 'updates.jsonl',
                'assets/source-lock.json', 'assets/study-contract.json',
                'assets/laya_model_lock.json', 'assets/uv.lock', 'assets/cuda-uv.lock',
                'assets/tokenizer/tokenizer.json', 'assets/tokenizer/tokenizer_config.json',
                'assets/source/example.py']
            if stage == 'epoch-001':
                leaves.append('resume.pt')
            extra = {}
        else:
            leaves = ['frozen.json', 'calibration-reference.json', 'calibration-reloaded.json']
            prior = json.loads((root/'delivery/requests'/f'{run}.epoch-002.json').read_text())
            extra = {'checkpoint_bundle': f'checkpoints/{run}/epoch-002',
                     'epoch_request_sha256': request_sha(prior)}
        for leaf in leaves:
            path = folder/leaf
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'unit-test-dispatch-fixture')
        req = seal_bundle(root, str(folder.relative_to(root)), identity, stage, **extra)
        receipt = {'schema': 'laya-cloud-durability-v1', 'state': 'verified_committed_volume',
            'modal_volume': VOLUME, 'local_delivery_verified': False, 'identity': identity,
            'stage': stage, 'request_sha256': request_sha(req), 'files': req['files'],
            'verification': {'strict_load': True, 'state_sha256': 'unit-test-only'}}
        for channel in ('acks', 'receipts'):
            atomic_json(root/'delivery'/channel/f'{request_sha(req)}.json', receipt)
        return folder

    def test_preemption_preserves_active_call_and_reattaches_without_spawn(self):
        state = initial_state({'previous_upper_usd': 12}, 0)
        calls, saved, finished = [], [], set()
        class Call:
            object_id = 'fc-unit-test'
            count = 0
            cancelled = False
            def get(self, timeout):
                self.count += 1
                if self.count == 1:
                    raise KeyboardInterrupt()
                finished.add('done')
                return {'state': 'frozen_cloud'}
            def cancel(self, **kw): self.cancelled = True
        call = Call()
        def spawn(job, deadline): calls.append((job, deadline)); return call
        def save(value): saved.append(json.loads(json.dumps(value)))
        def attach(call_id): self.assertEqual(call_id, call.object_id); return call
        kwargs = dict(save=save, spawn=spawn, attach=attach,
                      done=lambda job: bool(finished), clock=lambda: 1)
        with self.assertRaises(KeyboardInterrupt):
            execute_queue([JOB], state, **kwargs)
        self.assertFalse(call.cancelled)
        self.assertEqual(saved[-1]['active']['call_id'], call.object_id)
        reloaded = saved[-1]
        deadline = reloaded['deadline_unix']
        execute_queue([JOB], reloaded, **kwargs)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(reloaded['active'])
        self.assertEqual(reloaded['deadline_unix'], deadline)

    def test_unrecorded_spawn_fails_closed_without_duplicate_billing(self):
        state = initial_state({'previous_upper_usd': 12}, 0)
        state['active'] = {'key': '42-mixed-freeze', 'call_id': None}
        def never(*args): self.fail('Must not create or attach an unknown call')
        with self.assertRaisesRegex(RuntimeError, 'Unresolved submission intent'):
            execute_queue([JOB], state, save=lambda s: None, spawn=never, attach=never,
                          done=lambda j: False, clock=lambda: 1)

    def test_budget_deadline_cancels_pending_child_without_new_reservation(self):
        state = initial_state({'previous_upper_usd': 12}, 0)
        state['active'] = {'key': '42-mixed-freeze', 'call_id': 'fc-existing'}
        now = [state['deadline_unix']-100]
        class Call:
            cancelled = False
            def get(self, timeout): now[0] += 80; raise TimeoutError()
            def cancel(self, **kw): self.cancelled = True
        call = Call()
        with self.assertRaises(BudgetExpired):
            execute_queue([JOB], state, save=lambda s: None,
                          spawn=lambda *a: self.fail('duplicate spawn'), attach=lambda i: call,
                          done=lambda j: False, clock=lambda: now[0])
        self.assertTrue(call.cancelled)
        self.assertEqual(state['state'], 'stopped_budget')

    def test_completed_files_missing_do_not_trigger_silent_retraining(self):
        state = initial_state({'previous_upper_usd': 12}, 0)
        state['completed']['42-mixed-freeze'] = {'state': 'frozen_cloud'}
        with self.assertRaisesRegex(ValueError, 'no longer validates'):
            execute_queue([JOB], state, save=lambda s: None,
                          spawn=lambda *a: self.fail('silent retraining'), attach=lambda i: None,
                          done=lambda j: False, clock=lambda: 1)

    def test_changed_persisted_deadline_is_rejected(self):
        config = {'previous_upper_usd': 12}
        state = initial_state(config, 100)
        check_state(state, config)
        state['deadline_unix'] += 10000
        with self.assertRaises(ValueError): check_state(state, config)

    def test_epoch_resume_retains_partial_writes_and_saved_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);first = self.fixture(root, 'epoch-001')
            partial = first.parent/'epoch-002'
            partial.mkdir();(partial/'model.safetensors').write_bytes(b'interrupted')
            resume_sha = digest(first/'resume.pt')
            self.assertEqual(plan(root, JOB), [('train', True), ('reload', False)])
            self.assertEqual(digest(first/'resume.pt'), resume_sha)
            self.assertFalse(partial.exists())
            history = list((root/'artifacts'/RECOVERY).rglob('model.safetensors'))
            self.assertEqual([p.read_bytes() for p in history], [b'interrupted'])

    def test_frozen_model_is_skipped_and_corrupt_sealed_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);final = self.fixture(root, 'epoch-002');self.fixture(root, 'frozen')
            self.assertEqual(plan(root, JOB), [])
            (final/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): plan(root, JOB)
            self.assertEqual((final/'model.safetensors').read_bytes(), b'corrupt')

    def test_interrupted_final_epoch_is_reloaded_without_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);self.fixture(root, 'epoch-001');self.fixture(root, 'epoch-002')
            self.assertEqual(plan(root, JOB), [('train', True), ('reload', False)])

    def test_evaluation_requires_freeze_and_preserves_unsealed_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'unfrozen'):
                plan(root, {**JOB, 'evaluate': True})
            final = self.fixture(root, 'epoch-002');self.fixture(root, 'frozen')
            partial = final.parent/'evaluated'
            partial.mkdir();(partial/'validation.json').write_text('partial')
            report = root/'reports'/name(42, 'mixed');report.mkdir(parents=True)
            (report/'validation.json').write_text('partial')
            self.assertEqual(plan(root, {**JOB, 'evaluate': True}), [('evaluate', False)])
            self.assertFalse((report/'validation.json').exists())
            self.assertEqual(len(list((root/'artifacts'/RECOVERY).rglob('validation.json'))), 2)

    def test_restart_before_first_checkpoint_preserves_original_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);run = name(42, 'mixed')
            report = root/'reports'/run;report.mkdir(parents=True)
            (report/'manifest.json').write_text('original attempt')
            self.assertEqual(plan(root, JOB), [('train', False), ('reload', False)])
            self.assertEqual(next((root/'artifacts'/RECOVERY).rglob('manifest.json')).read_text(),
                             'original attempt')


if __name__ == '__main__':
    unittest.main()
