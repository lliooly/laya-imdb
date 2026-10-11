"""Render measured CUDA FP32 step throughput and allocated memory for the report."""
import argparse
import json
from pathlib import Path

from common import ROOT, file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/cuda-preparation-v1')
    args = parser.parse_args()
    source = args.report / 'fp32-benchmark.json'
    outputs = [args.report / ('fp32-training-benchmark.' + ext) for ext in ('png', 'pdf')]
    manifest = args.report / 'benchmark_plot_manifest.json'
    if any(p.exists() for p in outputs + [manifest]):
        raise FileExistsError('Plot already exists; use another report directory')
    data = json.loads(source.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = data['results']
    labels = [str(row['micro_batch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1))
    for axis, values, color, ylabel in (
        (axes[0], [row['reviews_per_second'] for row in rows], '#3568a8', 'Reviews per second'),
        (axes[1], [row['peak_allocated_bytes'] / 1e9 for row in rows], '#478a72', 'Peak allocated memory (GB)')):
        bars = axis.bar(labels, values, color=color, width=0.58)
        axis.bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=5, fontsize=10)
        axis.set_xlabel('Micro-batch size')
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values) * 1.25)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_title('Training-step throughput')
    axes[1].set_title('PyTorch allocator peak')
    fig.suptitle('RTX 6000D: FP32, 512 tokens, gradient checkpointing', fontsize=12)
    fig.text(0.5, 0.02, '2 warmup + 5 measured steps per batch; preloaded GPU inputs; zero LR.\n'
             'Includes forward/backward/clip/AdamW; excludes tokenization, validation and checkpoint IO.',
             ha='center', fontsize=8, color='#444444')
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    for path in outputs:
        fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    write_json(manifest, {'source_sha256': file_digest(source),
        'script_sha256': file_digest(Path(__file__)),
        'plots': {p.name: file_digest(p) for p in outputs}})
    print('\n'.join(str(p) for p in outputs))


if __name__ == '__main__':
    main()
