import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from laya_epoch_checkpoints import keep_epoch_checkpoints, validate_epoch_checkpoints


class EpochCheckpointsTests(unittest.TestCase):
    def test_actual_training_parameters_and_rng_are_unchanged(self):
        import torch
        import laya.train as upstream

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(12, 4)
                self.head = torch.nn.Linear(4, 1)
                self.dropout = torch.nn.Dropout(.1)

            def forward(self, input_ids, attention_mask, marker_pos, marker_mask,
                        qtype, detach_encoder=False):
                values = self.head(self.dropout(self.encoder(input_ids))).squeeze(-1)
                return values.gather(1, marker_pos), None

        items = [{'ids': [i+1, i+2], 'markers': [0, 1], 'qtype': 0,
                  'q': {'t': 'choice'}, 'k': 2, 'target': [0., 1.]} for i in range(4)]
        config = upstream.TrainConfig(epochs=2, micro_batch=2, grad_accum=1,
            encoder_lr=.02, head_lr=.04, min_lr=.001, loss='soft-ce', amp=False,
            gradient_checkpointing=False, log_every=0)
        results = []
        for enabled in (False, True):
            torch.manual_seed(42)
            model = Toy()
            with tempfile.TemporaryDirectory() as directory:
                latest = Path(directory) / 'checkpoint_latest'
                latest.mkdir()
                def save(epoch, loss):
                    (latest / 'model.safetensors').write_bytes(bytes([epoch]))
                with patch.object(upstream, 'encode_item', lambda tok, item, *a, **k: item):
                    with keep_epoch_checkpoints(upstream, enabled, directory, directory):
                        history = upstream.train_model(model, SimpleNamespace(pad_token_id=0),
                            items, config, torch.device('cpu'), 512, 192, on_epoch_end=save)
                results.append((model.state_dict(), torch.get_rng_state(), history))
        self.assertEqual(results[0][2], results[1][2])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))

    def test_disabled_leaves_function_untouched(self):
        upstream = SimpleNamespace(train_model=lambda: 7)
        original = upstream.train_model
        with keep_epoch_checkpoints(upstream, False, None, None):
            self.assertIs(upstream.train_model, original)
        for value in (None, 0, 1, 'true'):
            with self.assertRaises(ValueError):
                validate_epoch_checkpoints(value)

    def test_snapshots_survive_latest_overwrite_and_hooks_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            latest = output / 'checkpoint_latest'
            latest.mkdir()
            calls = []
            def save(epoch, loss):
                calls.append(epoch)
                (latest / 'model.safetensors').write_bytes(bytes([epoch]))
                (latest / 'config.json').write_text('{}')
            def train(on_epoch_end=None):
                for epoch in range(2):
                    on_epoch_end(epoch, .5 / (epoch + 1))
                return [1, 2]
            upstream = SimpleNamespace(train_model=train)
            with keep_epoch_checkpoints(upstream, True, output, output):
                self.assertEqual(upstream.train_model(save), [1, 2])
            self.assertIs(upstream.train_model, train)
            self.assertEqual(calls, [0, 1])
            self.assertEqual((output / 'epoch-001/model.safetensors').read_bytes(), b'\x00')
            self.assertEqual((output / 'epoch-002/model.safetensors').read_bytes(), b'\x01')
            self.assertEqual(len(json.loads((output / 'epoch_checkpoints.json').read_text())['epochs']), 2)
            with self.assertRaises(FileExistsError):
                with keep_epoch_checkpoints(upstream, True, output, output):
                    upstream.train_model(save)
            self.assertIs(upstream.train_model, train)
            self.assertEqual(calls, [0, 1])

    def test_failure_restores_hook_without_claiming_snapshot(self):
        def train(on_epoch_end=None):
            on_epoch_end(0, 1.)
        def fail(epoch, loss):
            raise RuntimeError('save failed')
        upstream = SimpleNamespace(train_model=train)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with keep_epoch_checkpoints(upstream, True, directory, directory):
                    upstream.train_model(on_epoch_end=fail)
            self.assertIs(upstream.train_model, train)
            self.assertFalse((Path(directory) / 'epoch_checkpoints.json').exists())


if __name__ == '__main__':
    unittest.main()
