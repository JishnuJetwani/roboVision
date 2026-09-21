"""Summarize recorded evaluations and plot training progress."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
from itertools import product
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    normal = [r for r in rows if r['intervention'] == 'normal']
    seeds = sorted({int(r['training_seed']) for r in normal})
    pairs = []
    for seed in seeds:
        selected = [r for r in normal if int(r['training_seed']) == seed]
        methods = {method: [r for r in selected if r['method'] == method]
                   for method in ('hard', 'curriculum')}
        scenes = [{int(r['scene_seed']) for r in methods[method]}
                  for method in ('hard', 'curriculum')]
        if scenes[0] != scenes[1] or not scenes[0]:
            raise ValueError('Each training seed needs matched evaluation scenarios.')
        counts = {method: sum(int(r['success']) for r in records)
                  for method, records in methods.items()}
        pairs.append({'seed': seed, 'episodes_per_method': len(scenes[0]),
                      **counts, 'difference_pp': 100 * (counts['curriculum'] - counts['hard']) / len(scenes[0])})
    differences = np.array([p['difference_pp'] for p in pairs])
    bootstrap = np.array([np.mean(sample) for sample in product(differences, repeat=len(seeds))])
    aggregates = {}
    for method in ('hard', 'curriculum'):
        records = [r for r in normal if r['method'] == method]
        successes = sum(int(r['success']) for r in records)
        aggregates[method] = {
            'successes': successes, 'episodes': len(records),
            'success_rate': successes / len(records),
            'mean_duration_seconds': float(np.mean([float(r['duration_seconds']) for r in records])),
            'mean_success_duration_seconds': float(np.mean([float(r['duration_seconds']) for r in records if int(r['success'])])),
            'failure_categories': dict(Counter(r['reason'] for r in records if not int(r['success'])))
        }
    interventions = {}
    for condition in sorted({r['intervention'] for r in rows} - {'normal'}):
        records = [r for r in rows if r['intervention'] == condition]
        keys = {(r['training_seed'], r['method'], r['scene_seed']) for r in records}
        matched = [r for r in normal if (r['training_seed'], r['method'], r['scene_seed']) in keys]
        if len(matched) != len(records):
            raise ValueError('Interventions need a normal evaluation for every model-scenario pair.')
        interventions[condition] = {
            'successes': sum(int(r['success']) for r in records),
            'normal_successes': sum(int(r['success']) for r in matched),
            'episodes': len(records)
        }
    return {'per_seed': pairs, 'aggregate': aggregates,
            'difference_pp': float(differences.mean()),
            'paired_seed_bootstrap_95_ci_pp': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            'bootstrap_resamples': len(bootstrap),
            'unique_test_scenarios': len({r['scene_seed'] for r in normal}),
            'interventions': interventions}


def write_report(directory, summary, training):
    lo, hi = summary['paired_seed_bootstrap_95_ci_pp']
    lines = ['# Evaluation results', '',
             'Both methods use the same behavior-cloned grasp controller. '
             'Each was trained with six seeds and tested on the same 500 scenes.', '',
             '| Training seed | Hard-only | Curriculum |',
             '| --- | ---: | ---: |']
    for pair in summary['per_seed']:
        n = pair['episodes_per_method']
        lines.append(f"| {pair['seed']} | {pair['hard']}/{n} | {pair['curriculum']}/{n} |")
    lines += ['', f"Curriculum gained **{summary['difference_pp']:.2f} percentage points** on average. "
              f'The 95% bootstrap interval is {lo:.2f} to {hi:.2f} points. '
              'It resamples the six seed pairs, with the grasp controller and 500 test scenes fixed.', '',
              '| Method | Successes | Mean duration | Mean duration of successes | Failures |',
              '| --- | ---: | ---: | ---: | --- |']
    for method, r in summary['aggregate'].items():
        failures = ', '.join(f"{reason.replace('_', ' ')}: {count:,}" for reason, count in r['failure_categories'].items())
        name = 'Hard-only' if method == 'hard' else 'Curriculum'
        lines.append(f"| {name} | {r['successes']:,}/{r['episodes']:,} ({100*r['success_rate']:.2f}%) | "
                     f"{r['mean_duration_seconds']:.2f} s | {r['mean_success_duration_seconds']:.2f} s | {failures} |")
    lines += ['', 'Times are simulated and include the 0.8-second approach. '
              'A timeout means no stable grasp before the time limit; the cause is not recorded.', '',
              '## Camera and grasp checks', '',
              'These checks use the first 100 test scenes for all six curriculum policies.', '',
              '| Change | Successes | Original on the same scenes |',
              '| --- | ---: | ---: |']
    for condition, r in summary['interventions'].items():
        name = 'Camera image set to black' if condition == 'black' else 'Grasp network weights set to zero'
        lines.append(f"| {name} | {r['successes']}/{r['episodes']} | {r['normal_successes']}/{r['episodes']} |")
    runs = training['approach']['runs']
    seconds = [r['elapsed_seconds'] for r in runs]
    grasp = training['grasp']
    run = grasp['training']
    collection = grasp['collection']
    development = grasp['development']
    integrated = grasp['integrated_development']
    split = grasp['split']
    lines += ['', '## Training', '',
              f"Each approach run completed 32,768 interactions in {min(seconds):.0f} to {max(seconds):.0f} seconds "
              'on an NVIDIA L4. Both methods started from identical weights within each seed pair.', '',
              'All six curriculum runs completed the stages: 60, 40, 25, 15, 10 and 6 mm.', '',
              f"The shared grasp network used {collection['successes']} successful demonstrations "
              f"({collection['transitions']:,} steps), split into {split['training_episodes']} training "
              f"and {split['validation_episodes']} validation episodes. "
              f"Collection took {collection['elapsed_seconds']:.2f} seconds and {run['updates']:,} BC updates took "
              f"{run['elapsed_seconds']:.2f} seconds.", '',
              f"On development scenes, the grasp controller scored {development['successes']}/{development['episodes']} "
              'with the hand already aligned above the cup. The complete controller scored '
              f"{integrated['hard_successes']}/{integrated['episodes_per_method']} for hard-only and "
              f"{integrated['curriculum_successes']}/{integrated['episodes_per_method']} for curriculum. "
              'These scenes were separate from the final tests.', '',
              '![Training progress](learning_curve.png)', '',
              'The curves include exploration noise during training. Shading shows the range across six seeds. '
              'Success rates from different curriculum stages use different tolerances, so they cannot '
              'be compared directly.', '',
              '## Files', '',
              '- [Episode records](episodes.csv.gz): all 7,200 evaluation episodes.',
              '- [Training progress](training_progress.csv): all 12 approach runs.',
              '- [Protocol](protocol.json): test scenes, success rules and confidence interval method.',
              '- [Training metadata](training.json): settings, costs, promotions and grasp validation.',
              '- [Collection log](demonstration_collection.csv): each attempt and its training or validation split.',
              '- [Summary](summary.json): counts, durations, failure categories and confidence interval.', '',
              'Rebuild this report with `uv run python -m robovision.report`.', '']
    (directory / 'README.md').write_text('\n'.join(lines))


def plot_progress(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (directory / 'training_progress.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout='constrained')
    colors = {'hard': '#8a8e98', 'curriculum': '#247a78'}
    for method, color in colors.items():
        records = [r for r in rows if r['method'] == method]
        seeds = sorted({r['training_seed'] for r in records})
        curves = [sorted([r for r in records if r['training_seed'] == seed], key=lambda r: int(r['steps'])) for seed in seeds]
        x = [int(r['steps']) for r in curves[0]]
        y = np.array([[float(r['mean_alignment_error_m']) * 1000 for r in curve] for curve in curves])
        axes[0].plot(x, y.mean(axis=0), label=method, color=color)
        axes[0].fill_between(x, y.min(axis=0), y.max(axis=0), color=color, alpha=.15, linewidth=0)
        if method == 'curriculum':
            for curve in curves:
                axes[1].step([int(r['steps']) for r in curve], [float(r['tolerance_m'])*1000 for r in curve], where='post', alpha=.4, color=color)
    axes[0].set(title='Target error during training', ylabel='Mean XY error (mm)')
    axes[0].legend(frameon=False)
    axes[1].axhline(6, color=colors['hard'], linestyle=':', linewidth=1)
    axes[1].set(title='Adaptive curriculum', ylabel='Reward tolerance (mm)')
    for ax in axes:
        ax.set_xlabel('Training interactions')
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(directory / 'learning_curve.png', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args(argv)
    with gzip.open(args.results / 'episodes.csv.gz', 'rt') as stream:
        summary = summarize(list(csv.DictReader(stream)))
    training = json.loads((args.results / 'training.json').read_text())
    (args.results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(args.results, summary, training)
    plot_progress(args.results)
    print(json.dumps(summary['aggregate'], indent=2))


if __name__ == '__main__':
    main()
