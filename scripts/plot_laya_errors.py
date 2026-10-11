"""Plot saved validation error strata without reading review text or test data."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))

import argparse
import json

from common import ROOT, file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/laya-ce-error-analysis-v1')
    args = parser.parse_args()
    target = args.report / 'error_strata.png'
    if target.exists():
        raise FileExistsError('Plot already exists; use another analysis report directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = json.loads((args.report / 'summary.json').read_text())
    names = ['review_tokens_1_128', 'review_tokens_129_256', 'review_tokens_257_466',
             'review_tokens_467_1024', 'review_tokens_1025_plus']
    labels = ['1–128', '129–256', '257–466', '467–1024', '1025+']
    groups = [summary['groups'][n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, [100*g['accuracy'] for g in groups],
                color=['#3579af']*3 + ['#d18136']*2)
    axes[0].axhline(98, color='#a53b35', linestyle='--', label='98% target')
    axes[0].set_ylim(88, 100)
    axes[0].set_ylabel('Validation accuracy (%)')
    axes[0].set_xlabel('Full review tokens (question excluded)')
    for i, g in enumerate(groups):
        axes[0].text(i, 100*g['accuracy']+.3,
                     f"{100*g['accuracy']:.2f}%\nn={g['count']}", ha='center', fontsize=9)
    axes[0].legend(frameon=False)
    strata = [summary['groups'][n] for n in ('untruncated', 'truncated')]
    high = [g['errors_with_confidence_ge_095'] for g in strata]
    low = [g['errors']-h for g, h in zip(strata, high)]
    axes[1].bar(['Untruncated', 'Truncated'], low, color='#3579af', label='Confidence < 0.95')
    axes[1].bar(['Untruncated', 'Truncated'], high, bottom=low, color='#d18136', label='Confidence ≥ 0.95')
    axes[1].set_ylabel('Number of errors')
    for i, g in enumerate(strata):
        axes[1].text(i, g['errors']+1, f"{g['errors']} / {g['count']}", ha='center')
    axes[1].set_ylim(0, 95)
    axes[1].legend(frameon=False)
    fig.suptitle('Laya-CE: 2,500 development validation reviews, 106 errors')
    fig.savefig(target, dpi=180)
    plt.close(fig)
    write_json(args.report / 'plot_manifest.json',
               {'summary_sha256': file_digest(args.report / 'summary.json'),
                'script_sha256': file_digest(Path(__file__)), 'plot_sha256': file_digest(target),
                'confidence': 'original checkpoint temperature, not comparison-analysis temperature',
                'scope': 'descriptive development strata; not a causal length comparison'})
    print(target)


if __name__ == '__main__':
    main()
