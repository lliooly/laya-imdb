"""Opt-in snapshots of upstream's existing epoch save; no extra model forwards."""
from contextlib import contextmanager
from functools import wraps
import inspect
from pathlib import Path
import shutil

from common import file_digest, write_json


def validate_epoch_checkpoints(value):
    if type(value) is not bool:
        raise ValueError('epoch_checkpoints must be a boolean')
    return value


@contextmanager
def keep_epoch_checkpoints(upstream, enabled, output, report_dir):
    validate_epoch_checkpoints(enabled)
    if not enabled:
        yield
        return
    original = upstream.train_model
    signature = inspect.signature(original)
    records = []

    @wraps(original)
    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        callback = bound.arguments.get('on_epoch_end')
        if callback is None:
            raise ValueError('Epoch preservation requires the upstream save callback')

        def snapshot(epoch, loss):
            destination = Path(output) / ('epoch-%03d' % (epoch + 1))
            if destination.exists():
                raise FileExistsError('Epoch checkpoint already exists: ' + str(destination))
            callback(epoch, loss)
            # Copy the already-saved FP16 representation, including tokenizer/config.
            # No RNG use, evaluation, optimizer changes or additional GPU allocation.
            shutil.copytree(Path(output) / 'checkpoint_latest', destination)
            records.append({'epoch': epoch + 1, 'training_loss': float(loss),
                            'checkpoint': str(destination),
                            'weights_sha256': file_digest(destination / 'model.safetensors'),
                            'temperature_scope': 'pre_final_calibration'})
            write_json(Path(report_dir) / 'epoch_checkpoints.json', {
                'epochs': records, 'test_accessed': False,
                'validation_scope': 'evaluate_saved_weights_in_separate_process_after_training',
                'primary_selection': 'final_epoch; best_epoch_is_secondary_diagnostic'})

        bound.arguments['on_epoch_end'] = snapshot
        return original(*bound.args, **bound.kwargs)

    upstream.train_model = train
    try:
        yield
    finally:
        upstream.train_model = original
