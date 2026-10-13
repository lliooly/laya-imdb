"""Fresh-process 16-update admission check for the current formal strict entry."""
import argparse
import json
from pathlib import Path
import sys
from common import ROOT, file_digest, write_json
import train_laya


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if (spec.get('runtime_profile') != 'strict' or spec['sampling']['mode'] != 'control'
            or spec['train_limit'] is not None or spec['training']['grad_accum'] != 1):
        raise ValueError('Requires the full-budget strict control configuration')
    train_laya.prepare_runtime(spec)
    import torch
    import laya.train as upstream
    from laya_short_trace import DiagnosticStop, short_trace
    train_laya.apply_runtime(torch, 'strict')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.argv = [str(ROOT/'scripts/train_laya.py'), '--config', str(args.config)]
    try:
        with short_trace(torch, upstream, args.out/'trace', parameter_details=True):
            train_laya.main()
    except DiagnosticStop:
        optimization = json.loads((ROOT/'reports'/spec['run_id']/'optimization_diagnostics.json').read_text())
        if optimization['completed_optimizer_updates'] != 16 or optimization['total_optimizer_updates'] != 5526:
            raise ValueError('Diagnostic full budget mismatch')
        write_json(args.out/'result.json', {'state': 'diagnostic_completed', 'updates': 16,
            'schedule_updates': 5526, 'test_accessed': False,
            'entry_sha256': file_digest(ROOT/'scripts/train_laya.py')})


if __name__ == '__main__':
    main()
