"""Compare original TRIPLEX runs with Random-20 runs.

The script reads fold-level evaluation logs produced by ``src/main.py`` and
writes tidy CSV summaries plus publication-ready comparison figures. Run
directories must point to the timestamp directory containing ``fold*/eval``.
"""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


LOWER_IS_BETTER = ('MeanSquaredError', 'MeanAbsoluteError', 'MSE', 'MAE', 'loss')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare baseline TRIPLEX and Random-20 evaluation runs.'
    )
    parser.add_argument(
        '--baseline-runs', nargs='+', required=True,
        help='Baseline timestamp directories containing fold*/eval/metrics.csv.'
    )
    parser.add_argument(
        '--random-runs', nargs='+', required=True,
        help='Random-20 timestamp directories containing fold*/eval/metrics.csv.'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory for comparison CSV files, report, and figures.'
    )
    parser.add_argument(
        '--primary-metric', default='test_PearsonCorrCoef',
        help='Metric used for the paired plot and headline conclusion.'
    )
    parser.add_argument(
        '--row-selection', choices=['last', 'max-primary'], default='last',
        help=('How to choose a row when eval/metrics.csv contains multiple '
              'checkpoint evaluations. "last" avoids selecting on test performance.')
    )
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2021)
    parser.add_argument('--title', default='TRIPLEX: Baseline vs Random-20')
    return parser.parse_args()


def fold_number(path):
    match = re.search(r'fold(\d+)', path.name)
    return int(match.group(1)) if match else path.name


def load_run_config(run_dir):
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        return {}, {}
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    return config.get('GENERAL', {}), config.get('DATA', {})


def choose_metric_row(metrics, primary_metric, row_selection):
    if primary_metric not in metrics.columns:
        raise ValueError(
            f"{primary_metric} not found in metrics columns: "
            f"{', '.join(metrics.columns)}"
        )

    candidates = metrics.dropna(subset=[primary_metric])
    if candidates.empty:
        raise ValueError(f'No non-null values found for {primary_metric}')

    if row_selection == 'max-primary':
        if direction(primary_metric) > 0:
            index = candidates[primary_metric].idxmax()
        else:
            index = candidates[primary_metric].idxmin()
        return candidates.loc[index]
    return candidates.iloc[-1]


def read_run(run_path, method, primary_metric, row_selection):
    run_dir = Path(run_path).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')

    general_cfg, data_cfg = load_run_config(run_dir)
    train_seed = general_cfg.get('seed', np.nan)
    selection_seed = data_cfg.get('selection_seed', np.nan)
    rows = []

    fold_dirs = sorted(run_dir.glob('fold*'), key=fold_number)
    if not fold_dirs:
        raise FileNotFoundError(f'No fold directories found under {run_dir}')

    for fold_dir in fold_dirs:
        metrics_path = fold_dir / 'eval' / 'metrics.csv'
        if not metrics_path.exists():
            print(f'Warning: skipping missing {metrics_path}')
            continue

        metrics = pd.read_csv(metrics_path)
        selected = choose_metric_row(metrics, primary_metric, row_selection)
        record = {
            'method': method,
            'run_id': run_dir.name,
            'run_path': str(run_dir),
            'train_seed': train_seed,
            'selection_seed': selection_seed,
            'fold': fold_number(fold_dir),
        }
        if 'epoch' in selected.index:
            record['checkpoint_epoch'] = selected['epoch']

        for column in metrics.columns:
            if column.startswith('test_') and pd.notna(selected[column]):
                record[column] = float(selected[column])
        rows.append(record)

    if not rows:
        raise ValueError(f'No usable evaluation metrics found under {run_dir}')
    return rows


def metric_columns(frame):
    return [column for column in frame.columns if column.startswith('test_')]


def direction(metric):
    return -1 if any(token in metric for token in LOWER_IS_BETTER) else 1


def bootstrap_ci(values, samples, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return values[0], values[0]
    sampled = rng.choice(values, size=(samples, values.size), replace=True)
    means = sampled.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def build_run_summary(raw, metrics):
    group_cols = [
        'method', 'run_id', 'run_path', 'train_seed', 'selection_seed'
    ]
    return raw.groupby(group_cols, dropna=False)[metrics].mean().reset_index()


def build_method_summary(run_summary, metrics, samples, rng):
    rows = []
    for method, method_data in run_summary.groupby('method'):
        for metric in metrics:
            values = method_data[metric].dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(values, samples, rng)
            rows.append({
                'method': method,
                'metric': metric,
                'direction': 'lower' if direction(metric) < 0 else 'higher',
                'n_runs': values.size,
                'mean': np.mean(values) if values.size else np.nan,
                'std': np.std(values, ddof=1) if values.size > 1 else 0.0,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
            })
    return pd.DataFrame(rows)


def choose_baseline_row(baseline, random_row):
    candidates = baseline[baseline['fold'] == random_row['fold']]
    if candidates.empty:
        return None

    if pd.notna(random_row['train_seed']):
        seed_matches = candidates[
            candidates['train_seed'] == random_row['train_seed']
        ]
        if len(seed_matches) == 1:
            return seed_matches.iloc[0]
        if len(seed_matches) > 1:
            candidates = seed_matches

    if len(candidates) == 1:
        return candidates.iloc[0]

    raise ValueError(
        'Multiple baseline runs match fold '
        f"{random_row['fold']}. Use one baseline per training seed or ensure "
        'baseline and Random-20 train_seed values match.'
    )


def build_paired_results(raw, metrics):
    baseline = raw[raw['method'] == 'Baseline']
    random_rows = raw[raw['method'] == 'Random-20']
    pairs = []

    for _, random_row in random_rows.iterrows():
        baseline_row = choose_baseline_row(baseline, random_row)
        if baseline_row is None:
            print(
                'Warning: no baseline match for Random-20 '
                f"run={random_row['run_id']} fold={random_row['fold']}"
            )
            continue

        for metric in metrics:
            if pd.isna(random_row[metric]) or pd.isna(baseline_row[metric]):
                continue
            delta = random_row[metric] - baseline_row[metric]
            improvement = direction(metric) * delta
            pairs.append({
                'metric': metric,
                'fold': random_row['fold'],
                'train_seed': random_row['train_seed'],
                'selection_seed': random_row['selection_seed'],
                'baseline_run': baseline_row['run_id'],
                'random_run': random_row['run_id'],
                'baseline_value': baseline_row[metric],
                'random20_value': random_row[metric],
                'delta_random_minus_baseline': delta,
                'improvement': improvement,
                'random20_better': improvement > 0,
            })
    return pd.DataFrame(pairs)


def build_paired_summary(pairs, samples, rng):
    rows = []
    for metric, metric_pairs in pairs.groupby('metric'):
        run_improvements = metric_pairs.groupby(
            ['random_run', 'train_seed', 'selection_seed'],
            dropna=False,
        )['improvement'].mean().to_numpy()
        ci_low, ci_high = bootstrap_ci(run_improvements, samples, rng)
        rows.append({
            'metric': metric,
            'direction': 'lower' if direction(metric) < 0 else 'higher',
            'n_pairs': len(metric_pairs),
            'n_random_runs': len(run_improvements),
            'baseline_mean': metric_pairs['baseline_value'].mean(),
            'random20_mean': metric_pairs['random20_value'].mean(),
            'delta_random_minus_baseline': (
                metric_pairs['delta_random_minus_baseline'].mean()
            ),
            'mean_improvement': run_improvements.mean(),
            'median_improvement': np.median(run_improvements),
            'improvement_ci_95_low': ci_low,
            'improvement_ci_95_high': ci_high,
            'random20_win_rate': metric_pairs['random20_better'].mean(),
        })
    return pd.DataFrame(rows)


def friendly_metric(metric):
    replacements = {
        'test_PearsonCorrCoef': 'PCC',
        'test_ConcordanceCorrCoef': 'CCC',
        'test_MeanSquaredError': 'MSE',
        'test_MeanAbsoluteError': 'MAE',
        'test_ExplainedVariance': 'Explained variance',
    }
    return replacements.get(metric, metric.replace('test_', ''))


def plot_metric_overview(run_summary, method_summary, metrics, output_path, title):
    cols = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = {'Baseline': '#4c78a8', 'Random-20': '#e45756'}
    rng = np.random.default_rng(7)

    for axis, metric in zip(axes, metrics):
        for x_pos, method in enumerate(['Baseline', 'Random-20']):
            values = run_summary.loc[run_summary['method'] == method, metric].dropna()
            jitter = rng.normal(0, 0.035, len(values))
            axis.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                color=colors[method],
                alpha=0.7,
                s=35,
                zorder=2,
            )
            summary = method_summary[
                (method_summary['method'] == method)
                & (method_summary['metric'] == metric)
            ]
            if not summary.empty:
                row = summary.iloc[0]
                errors = np.array([[
                    row['mean'] - row['ci_95_low']
                ], [
                    row['ci_95_high'] - row['mean']
                ]])
                axis.errorbar(
                    x_pos,
                    row['mean'],
                    yerr=errors,
                    fmt='o',
                    color='black',
                    capsize=5,
                    markersize=7,
                    zorder=3,
                )
        arrow = '↓' if direction(metric) < 0 else '↑'
        axis.set_title(f'{friendly_metric(metric)} {arrow}')
        axis.set_xticks([0, 1], ['Baseline', 'Random-20'])
        axis.grid(axis='y', alpha=0.25)

    for axis in axes[len(metrics):]:
        axis.remove()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(figure)


def plot_paired_primary(pairs, metric, output_path):
    data = pairs[pairs['metric'] == metric]
    if data.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    for _, row in data.iterrows():
        color = '#2ca02c' if row['random20_better'] else '#d62728'
        axis.plot(
            [0, 1],
            [row['baseline_value'], row['random20_value']],
            color=color,
            alpha=0.35,
            linewidth=1,
        )
        axis.scatter(
            [0, 1],
            [row['baseline_value'], row['random20_value']],
            color=color,
            s=20,
        )
    axis.set_xticks([0, 1], ['Baseline', 'Random-20'])
    axis.set_ylabel(friendly_metric(metric))
    axis.set_title(f'Paired fold/run comparison: {friendly_metric(metric)}')
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(figure)


def plot_seed_stability(raw, metric, output_path):
    random_data = raw[raw['method'] == 'Random-20'].copy()
    if random_data.empty or random_data['selection_seed'].isna().all():
        return
    summary = random_data.groupby('selection_seed')[metric].agg(['mean', 'std'])
    baseline_mean = raw.loc[raw['method'] == 'Baseline', metric].mean()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        summary.index.astype(str),
        summary['mean'],
        yerr=summary['std'].fillna(0),
        fmt='o-',
        color='#e45756',
        capsize=4,
        label='Random-20 fold mean ± SD',
    )
    axis.axhline(
        baseline_mean,
        color='#4c78a8',
        linestyle='--',
        label='Baseline fold mean',
    )
    axis.set_xlabel('Selection seed')
    axis.set_ylabel(friendly_metric(metric))
    axis.set_title('Random-20 selection-seed stability')
    axis.grid(axis='y', alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(figure)


def write_report(path, paired_summary, primary_metric):
    primary = paired_summary[paired_summary['metric'] == primary_metric]
    lines = [
        '# Baseline vs Random-20 comparison',
        '',
        'Positive improvement means Random-20 is better; metric direction is '
        'handled automatically.',
        '',
    ]
    if primary.empty:
        lines.append(f'Primary metric `{primary_metric}` was not available.')
    else:
        row = primary.iloc[0]
        low = row['improvement_ci_95_low']
        high = row['improvement_ci_95_high']
        if low > 0:
            conclusion = 'Random-20 is better in the paired bootstrap analysis.'
        elif high < 0:
            conclusion = 'The baseline is better in the paired bootstrap analysis.'
        else:
            conclusion = (
                'The 95% confidence interval crosses zero; the result is '
                'inconclusive at this level.'
            )
        lines.extend([
            f'## Primary metric: {friendly_metric(primary_metric)}',
            '',
            f"- Baseline mean: {row['baseline_mean']:.6f}",
            f"- Random-20 mean: {row['random20_mean']:.6f}",
            f"- Mean improvement: {row['mean_improvement']:.6f}",
            f"- 95% bootstrap CI: [{low:.6f}, {high:.6f}]",
            f"- Random-20 win rate: {row['random20_win_rate']:.1%}",
            f'- Conclusion: {conclusion}',
        ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run in args.baseline_runs:
        records.extend(read_run(
            run, 'Baseline', args.primary_metric, args.row_selection
        ))
    for run in args.random_runs:
        records.extend(read_run(
            run, 'Random-20', args.primary_metric, args.row_selection
        ))

    raw = pd.DataFrame(records)
    metrics = metric_columns(raw)
    if args.primary_metric not in metrics:
        raise ValueError(f'Primary metric {args.primary_metric} was not loaded')

    rng = np.random.default_rng(args.bootstrap_seed)
    run_summary = build_run_summary(raw, metrics)
    method_summary = build_method_summary(
        run_summary, metrics, args.bootstrap_samples, rng
    )
    pairs = build_paired_results(raw, metrics)
    if pairs.empty:
        raise ValueError('No fold-level baseline/Random-20 pairs could be formed')
    paired_summary = build_paired_summary(
        pairs, args.bootstrap_samples, rng
    )

    raw.to_csv(output_dir / 'fold_metrics.csv', index=False)
    run_summary.to_csv(output_dir / 'run_summary.csv', index=False)
    method_summary.to_csv(output_dir / 'method_summary.csv', index=False)
    pairs.to_csv(output_dir / 'paired_differences.csv', index=False)
    paired_summary.to_csv(output_dir / 'paired_summary.csv', index=False)

    plot_metric_overview(
        run_summary,
        method_summary,
        [metric for metric in metrics if metric != 'test_ExplainedVariance'],
        output_dir / 'metric_overview.png',
        args.title,
    )
    plot_paired_primary(
        pairs,
        args.primary_metric,
        output_dir / 'paired_primary_metric.png',
    )
    plot_seed_stability(
        raw,
        args.primary_metric,
        output_dir / 'selection_seed_stability.png',
    )
    write_report(
        output_dir / 'comparison_report.md',
        paired_summary,
        args.primary_metric,
    )

    print(f'Comparison written to {output_dir}')
    print(paired_summary.to_string(index=False))


if __name__ == '__main__':
    main()
