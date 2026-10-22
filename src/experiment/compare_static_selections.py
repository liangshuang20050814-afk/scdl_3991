"""Compare Baseline, Random-20, Similarity-20, and Diversity-20.

Random-20 runs are first averaged within each fold so that every method has
one value per patient fold. The script then writes fold-level summaries,
paired differences from Baseline, bootstrap intervals, and overview figures.
"""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOWER_IS_BETTER = ('MeanSquaredError', 'MeanAbsoluteError', 'MSE', 'MAE', 'loss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--random-runs', nargs='+', required=True)
    parser.add_argument('--similarity-run', required=True)
    parser.add_argument('--diversity-run', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2021)
    parser.add_argument('--expected-folds', type=int, default=4)
    return parser.parse_args()


def fold_number(path):
    match = re.search(r'fold(\d+)', path.name)
    return int(match.group(1))


def read_run(run_path, method, expected_folds, run_id=None):
    run_path = Path(run_path).resolve()
    records = []
    for fold_path in sorted(run_path.glob('fold*'), key=fold_number):
        metrics_path = fold_path / 'eval' / 'metrics.csv'
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = pd.read_csv(metrics_path)
        row = metrics.dropna(subset=['test_PearsonCorrCoef']).iloc[-1]
        record = {
            'method': method,
            'run_id': run_id or run_path.name,
            'fold': fold_number(fold_path),
        }
        for column in metrics.columns:
            if column.startswith('test_') and pd.notna(row[column]):
                record[column] = float(row[column])
        records.append(record)
    if len(records) != expected_folds:
        raise ValueError(
            f'Expected {expected_folds} folds in {run_path}, found {len(records)}'
        )
    return records


def direction(metric):
    return -1 if any(token in metric for token in LOWER_IS_BETTER) else 1


def friendly(metric):
    return {
        'test_PearsonCorrCoef': 'PCC',
        'test_ConcordanceCorrCoef': 'CCC',
        'test_MeanSquaredError': 'MSE',
        'test_MeanAbsoluteError': 'MAE',
        'test_ExplainedVariance': 'Explained variance',
    }.get(metric, metric.replace('test_', ''))


def bootstrap_mean_ci(values, samples, rng):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    return np.percentile(draws.mean(axis=1), [2.5, 97.5])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_run(args.baseline_run, 'Baseline', args.expected_folds)
    for index, run in enumerate(args.random_runs, start=1):
        records.extend(
            read_run(run, 'Random-20 raw', args.expected_folds, f'seed{index}')
        )
    records.extend(
        read_run(args.similarity_run, 'Similarity-20', args.expected_folds)
    )
    records.extend(
        read_run(args.diversity_run, 'Diversity-20', args.expected_folds)
    )
    raw = pd.DataFrame(records)
    metrics = [column for column in raw.columns if column.startswith('test_')]

    non_random = raw[raw['method'] != 'Random-20 raw'].copy()
    random_fold = (
        raw[raw['method'] == 'Random-20 raw']
        .groupby('fold', as_index=False)[metrics]
        .mean()
    )
    random_fold.insert(0, 'method', 'Random-20')
    fold_metrics = pd.concat(
        [non_random[['method', 'fold', *metrics]], random_fold],
        ignore_index=True,
    )

    method_order = ['Baseline', 'Random-20', 'Similarity-20', 'Diversity-20']
    fold_metrics['method'] = pd.Categorical(
        fold_metrics['method'], categories=method_order, ordered=True
    )
    fold_metrics = fold_metrics.sort_values(['method', 'fold'])

    summary_rows = []
    for method in method_order:
        data = fold_metrics[fold_metrics['method'] == method]
        for metric in metrics:
            summary_rows.append({
                'method': method,
                'metric': metric,
                'direction': 'lower' if direction(metric) < 0 else 'higher',
                'mean': data[metric].mean(),
                'std_across_folds': data[metric].std(ddof=0),
            })
    method_summary = pd.DataFrame(summary_rows)

    baseline = fold_metrics[fold_metrics['method'] == 'Baseline'].set_index('fold')
    paired_rows = []
    for method in method_order[1:]:
        data = fold_metrics[fold_metrics['method'] == method].set_index('fold')
        for fold in baseline.index:
            for metric in metrics:
                delta = data.loc[fold, metric] - baseline.loc[fold, metric]
                paired_rows.append({
                    'method': method,
                    'fold': fold,
                    'metric': metric,
                    'baseline_value': baseline.loc[fold, metric],
                    'method_value': data.loc[fold, metric],
                    'delta_method_minus_baseline': delta,
                    'improvement': direction(metric) * delta,
                })
    paired = pd.DataFrame(paired_rows)

    rng = np.random.default_rng(args.bootstrap_seed)
    paired_summary_rows = []
    for (method, metric), data in paired.groupby(['method', 'metric']):
        low, high = bootstrap_mean_ci(
            data['improvement'], args.bootstrap_samples, rng
        )
        paired_summary_rows.append({
            'method': method,
            'metric': metric,
            'mean_improvement': data['improvement'].mean(),
            'improvement_ci_95_low': low,
            'improvement_ci_95_high': high,
            'fold_win_rate': (data['improvement'] > 0).mean(),
        })
    paired_summary = pd.DataFrame(paired_summary_rows)

    raw.to_csv(output_dir / 'raw_run_fold_metrics.csv', index=False)
    fold_metrics.to_csv(output_dir / 'fold_metrics.csv', index=False)
    method_summary.to_csv(output_dir / 'method_summary.csv', index=False)
    paired.to_csv(output_dir / 'paired_differences_vs_baseline.csv', index=False)
    paired_summary.to_csv(output_dir / 'paired_summary_vs_baseline.csv', index=False)

    plot_metrics = [
        metric for metric in metrics
        if metric != 'test_ExplainedVariance'
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    colors = ['#4c78a8', '#e45756', '#f2cf5b', '#54a24b']
    for axis, metric in zip(axes, plot_metrics):
        for position, (method, color) in enumerate(zip(method_order, colors)):
            values = fold_metrics.loc[fold_metrics['method'] == method, metric]
            axis.scatter(np.full(len(values), position), values, color=color, zorder=2)
            axis.errorbar(
                position, values.mean(), yerr=values.std(ddof=0), fmt='o',
                color='black', capsize=4, zorder=3,
            )
        arrow = '↓' if direction(metric) < 0 else '↑'
        axis.set_title(f'{friendly(metric)} {arrow}')
        axis.set_xticks(range(len(method_order)), method_order, rotation=20)
        axis.grid(axis='y', alpha=0.25)
    for axis in axes[len(plot_metrics):]:
        axis.remove()
    figure.suptitle(
        f'Static neighbor selection comparison ({args.expected_folds} patient folds)'
    )
    figure.tight_layout()
    figure.savefig(output_dir / 'metric_overview.png', dpi=220, bbox_inches='tight')
    plt.close(figure)

    pcc = 'test_PearsonCorrCoef'
    figure, axis = plt.subplots(figsize=(9, 5))
    for fold in sorted(fold_metrics['fold'].unique()):
        values = [
            fold_metrics.loc[
                (fold_metrics['method'] == method)
                & (fold_metrics['fold'] == fold), pcc
            ].iloc[0]
            for method in method_order
        ]
        axis.plot(method_order, values, marker='o', alpha=0.7, label=f'fold{fold}')
    axis.set_ylabel('PCC')
    axis.set_title('Fold-paired PCC comparison')
    axis.grid(axis='y', alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / 'paired_pcc.png', dpi=220, bbox_inches='tight')
    plt.close(figure)

    pcc_summary = method_summary[method_summary['metric'] == pcc]
    report = [
        '# Static neighbor selection comparison', '',
        f'Random-20 is averaged across {len(args.random_runs)} selection seeds within each fold.',
        'Confidence intervals are descriptive fold-bootstrap intervals '
        f'(n={args.expected_folds} folds).',
        '', '## PCC', '',
    ]
    for _, row in pcc_summary.iterrows():
        report.append(
            f"- {row['method']}: {row['mean']:.6f} ± "
            f"{row['std_across_folds']:.6f}"
        )
    (output_dir / 'comparison_report.md').write_text(
        '\n'.join(report) + '\n', encoding='utf-8'
    )

    print(method_summary[method_summary['metric'] == pcc].to_string(index=False))
    print(f'Comparison written to {output_dir}')


if __name__ == '__main__':
    main()
