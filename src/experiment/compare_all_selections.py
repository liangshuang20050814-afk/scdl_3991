"""Combine static and learned neighbor-selection results into one report."""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "test_PearsonCorrCoef",
    "test_ConcordanceCorrCoef",
    "test_MeanSquaredError",
    "test_MeanAbsoluteError",
    "test_ExplainedVariance",
]
NAMES = {
    "test_PearsonCorrCoef": "PCC",
    "test_ConcordanceCorrCoef": "CCC",
    "test_MeanSquaredError": "MSE",
    "test_MeanAbsoluteError": "MAE",
    "test_ExplainedVariance": "Explained variance",
}
LOWER_IS_BETTER = {"test_MeanSquaredError", "test_MeanAbsoluteError"}
METHOD_ORDER = [
    "Baseline",
    "Random-20",
    "Similarity-20",
    "Diversity-20",
    "Soft attention",
    "ST Hard-20",
    "Two-stage Hard-20",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-fold-metrics", required=True)
    parser.add_argument("--soft-run", required=True)
    parser.add_argument("--st-run", required=True)
    parser.add_argument("--two-stage-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-folds", type=int, default=4)
    return parser.parse_args()


def fold_number(path):
    return int(re.search(r"fold(\d+)", path.name).group(1))


def read_attention_run(run_dir, method, expected_folds):
    rows = []
    for fold_dir in sorted(Path(run_dir).glob("fold*"), key=fold_number):
        metrics_path = fold_dir / "eval" / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        row = metrics.dropna(subset=["test_PearsonCorrCoef"]).iloc[-1]
        rows.append({
            "method": method,
            "fold": fold_number(fold_dir),
            **{metric: float(row[metric]) for metric in METRICS},
        })
    if len(rows) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} evaluated folds for {method}, "
            f"found {len(rows)}"
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static = pd.read_csv(args.static_fold_metrics)
    static["method"] = static["method"].astype(str)
    frames = [
        static[["method", "fold", *METRICS]],
        read_attention_run(args.soft_run, "Soft attention", args.expected_folds),
        read_attention_run(args.st_run, "ST Hard-20", args.expected_folds),
        read_attention_run(
            args.two_stage_run, "Two-stage Hard-20", args.expected_folds
        ),
    ]
    folds = pd.concat(frames, ignore_index=True)
    folds["method"] = pd.Categorical(
        folds["method"], categories=METHOD_ORDER, ordered=True
    )
    folds = folds.sort_values(["method", "fold"])

    summary_rows = []
    for method in METHOD_ORDER:
        data = folds[folds["method"] == method]
        for metric in METRICS:
            summary_rows.append({
                "method": method,
                "metric": metric,
                "mean": data[metric].mean(),
                "std_across_folds": data[metric].std(ddof=0),
            })
    summary = pd.DataFrame(summary_rows)
    wide = summary.pivot(index="method", columns="metric", values="mean")
    wide = wide.loc[METHOD_ORDER, METRICS].rename(columns=NAMES).reset_index()

    baseline = folds[folds["method"] == "Baseline"].set_index("fold")
    paired_rows = []
    for method in METHOD_ORDER[1:]:
        data = folds[folds["method"] == method].set_index("fold")
        for metric in METRICS:
            delta = data[metric] - baseline[metric]
            improvement = -delta if metric in LOWER_IS_BETTER else delta
            paired_rows.append({
                "method": method,
                "metric": metric,
                "mean_delta_method_minus_baseline": delta.mean(),
                "mean_improvement": improvement.mean(),
                "fold_win_rate": (improvement > 0).mean(),
            })
    paired = pd.DataFrame(paired_rows)

    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    wide.to_csv(output_dir / "method_summary_wide.csv", index=False)
    paired.to_csv(output_dir / "paired_summary_vs_baseline.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    for axis, metric in zip(axes, METRICS):
        means = [
            summary.loc[
                (summary["method"] == method) & (summary["metric"] == metric),
                "mean",
            ].iloc[0]
            for method in METHOD_ORDER
        ]
        stds = [
            summary.loc[
                (summary["method"] == method) & (summary["metric"] == metric),
                "std_across_folds",
            ].iloc[0]
            for method in METHOD_ORDER
        ]
        axis.errorbar(range(len(METHOD_ORDER)), means, yerr=stds, fmt="o", capsize=4)
        axis.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=35, ha="right")
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        axis.set_title(f"{NAMES[metric]} {arrow}")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].remove()
    fig.suptitle(
        f"All neighbor-selection methods ({args.expected_folds} patient folds)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "metric_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# All neighbor-selection methods", "",
        "Random-20 is averaged over five selection seeds within each fold.",
        "Soft attention uses continuous gates; all Hard-20/static methods retain at most 20 patches.",
        "The current validation and final evaluation reuse the held-out fold, so these are relative comparison results rather than unbiased external-test estimates.",
        "",
        f"## {args.expected_folds}-fold mean ± population standard deviation",
        "",
        "| Method | PCC ↑ | CCC ↑ | MSE ↓ | MAE ↓ | Explained variance ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        values = summary[summary["method"] == method].set_index("metric")
        cells = []
        for metric in METRICS:
            row = values.loc[metric]
            cells.append(f"{row['mean']:.5f} ± {row['std_across_folds']:.5f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(wide.to_string(index=False))
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
