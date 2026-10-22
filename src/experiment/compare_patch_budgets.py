"""Summarize fixed-budget neighbor selection experiments."""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "test_PearsonCorrCoef",
    "test_ConcordanceCorrCoef",
    "test_MeanSquaredError",
    "test_MeanAbsoluteError",
    "test_ExplainedVariance",
]
PLOT_METRICS = [
    "test_PearsonCorrCoef",
    "test_ConcordanceCorrCoef",
    "test_MeanSquaredError",
    "test_MeanAbsoluteError",
]
NAMES = {
    "test_PearsonCorrCoef": "PCC",
    "test_ConcordanceCorrCoef": "CCC",
    "test_MeanSquaredError": "MSE",
    "test_MeanAbsoluteError": "MAE",
    "test_ExplainedVariance": "Explained variance",
}
LOWER_IS_BETTER = {"test_MeanSquaredError", "test_MeanAbsoluteError"}
METHOD_ORDER = ["Random", "Similarity", "Diversity", "ST Hard"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("METHOD", "K", "RUN_DIR"),
        required=True,
        help="Repeat for every method-budget run.",
    )
    parser.add_argument("--expected-folds", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def fold_number(path):
    match = re.fullmatch(r"fold(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid fold directory: {path}")
    return int(match.group(1))


def read_run(run_dir, method, budget, expected_folds):
    records = []
    for fold_dir in sorted(Path(run_dir).glob("fold*"), key=fold_number):
        metrics_path = fold_dir / "eval" / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = pd.read_csv(metrics_path)
        row = metrics.dropna(subset=["test_PearsonCorrCoef"]).iloc[-1]
        records.append({
            "method": method,
            "budget": budget,
            "fold": fold_number(fold_dir),
            **{metric: float(row[metric]) for metric in METRICS},
        })
    if len(records) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} folds for {method}-{budget}, "
            f"found {len(records)} in {run_dir}"
        )
    return records


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_run(
        args.baseline_run, "Baseline", 25, args.expected_folds
    )
    seen = set()
    for method, budget_text, run_dir in args.run:
        if method not in METHOD_ORDER:
            raise ValueError(
                f"Method must be one of {METHOD_ORDER}, got {method}"
            )
        budget = int(budget_text)
        key = (method, budget)
        if key in seen:
            raise ValueError(f"Duplicate run specification: {key}")
        seen.add(key)
        records.extend(
            read_run(run_dir, method, budget, args.expected_folds)
        )

    folds = pd.DataFrame(records)
    folds["method"] = pd.Categorical(
        folds["method"],
        categories=["Baseline", *METHOD_ORDER],
        ordered=True,
    )
    folds = folds.sort_values(["method", "budget", "fold"])

    summary = (
        folds.groupby(["method", "budget"], observed=True)[METRICS]
        .agg(["mean", lambda values: values.std(ddof=0)])
        .reset_index()
    )
    summary.columns = [
        "method",
        "budget",
        *[
            f"{metric}_{stat}"
            for metric in METRICS
            for stat in ("mean", "std")
        ],
    ]

    baseline = folds[folds["method"] == "Baseline"].set_index("fold")
    paired_rows = []
    for (method, budget), data in folds[
        folds["method"] != "Baseline"
    ].groupby(["method", "budget"], observed=True):
        data = data.set_index("fold")
        for metric in METRICS:
            delta = data[metric] - baseline[metric]
            improvement = -delta if metric in LOWER_IS_BETTER else delta
            paired_rows.append({
                "method": method,
                "budget": budget,
                "metric": metric,
                "mean_delta_method_minus_baseline": delta.mean(),
                "mean_improvement": improvement.mean(),
                "std_improvement_across_folds": improvement.std(ddof=0),
                "fold_win_rate": (improvement > 0).mean(),
            })
    paired = pd.DataFrame(paired_rows)

    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "budget_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_summary_vs_baseline.csv", index=False)

    # Mean-only grouped bars use a locally scaled y-axis so that the small
    # method/budget differences remain visible. Patient variation is shown
    # separately in the fold-trajectory figure instead of as oversized SD
    # bars on this overview.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    baseline_summary = summary[summary["method"] == "Baseline"].iloc[0]
    budgets = np.array([5, 10, 15, 20])
    width = 0.9
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for axis, metric in zip(axes, PLOT_METRICS):
        plotted_values = [baseline_summary[f"{metric}_mean"]]
        for method_index, (method, color) in enumerate(zip(METHOD_ORDER, colors)):
            data = summary[summary["method"] == method].sort_values("budget")
            if data.empty:
                continue
            values = data[f"{metric}_mean"].to_numpy()
            plotted_values.extend(values)
            axis.bar(
                budgets + (method_index - 1.5) * width,
                values,
                width=width,
                color=color,
                label=method,
                alpha=0.9,
            )
        axis.axhline(
            baseline_summary[f"{metric}_mean"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Baseline-25",
        )
        axis.set_xticks(budgets)
        axis.set_xlabel("Patch budget K (includes center target)")
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        axis.set_title(f"{NAMES[metric]} {arrow}")
        value_range = max(plotted_values) - min(plotted_values)
        padding = max(value_range * 0.25, abs(np.mean(plotted_values)) * 0.003)
        axis.set_ylim(min(plotted_values) - padding, max(plotted_values) + padding)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9, ncol=2)
    fig.suptitle(
        f"BC1 patch-budget mean comparison ({args.expected_folds} patient folds)"
    )
    fig.text(
        0.5, 0.01,
        "Bars show fold means; patient-fold variability is shown in the trajectory figure.",
        ha="center", fontsize=10, color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    fig.savefig(output_dir / "budget_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Heatmap colors are normalized within each metric because PCC/CCC and
    # error metrics have different numerical scales. Cell text remains the
    # unscaled paired improvement, with positive values always meaning better.
    combinations = [
        (method, budget)
        for method in METHOD_ORDER
        for budget in (5, 10, 15, 20)
    ]
    raw_improvements = np.array([
        [
            paired.loc[
                (paired["method"] == method)
                & (paired["budget"] == budget)
                & (paired["metric"] == metric),
                "mean_improvement",
            ].iloc[0]
            for metric in PLOT_METRICS
        ]
        for method, budget in combinations
    ])
    scale = np.max(np.abs(raw_improvements), axis=0)
    scale[scale == 0] = 1.0
    normalized = raw_improvements / scale

    fig, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(normalized, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(
        range(len(PLOT_METRICS)),
        [f"{NAMES[metric]} {'↓' if metric in LOWER_IS_BETTER else '↑'}" for metric in PLOT_METRICS],
    )
    axis.set_yticks(
        range(len(combinations)),
        [f"{method} · K={budget}" for method, budget in combinations],
    )
    for row_index in range(raw_improvements.shape[0]):
        for column_index in range(raw_improvements.shape[1]):
            value = raw_improvements[row_index, column_index]
            axis.text(
                column_index, row_index, f"{value:+.4f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(normalized[row_index, column_index]) > 0.62 else "#111827",
                fontweight="bold" if value > 0 else "normal",
            )
    for boundary in (3.5, 7.5, 11.5):
        axis.axhline(boundary, color="white", linewidth=2.5)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Column-normalized improvement (green = better)")
    axis.set_title("Mean paired improvement relative to Baseline-25", pad=14)
    axis.set_xlabel("Metric (cell values are unscaled improvements)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "delta_vs_baseline.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    baseline_pcc = baseline["test_PearsonCorrCoef"].sort_index()
    for axis, method in zip(axes, METHOD_ORDER):
        method_data = folds[folds["method"] == method]
        for fold in sorted(method_data["fold"].unique()):
            fold_data = method_data[method_data["fold"] == fold].sort_values(
                "budget"
            )
            x_values = fold_data["budget"].tolist() + [25]
            y_values = fold_data["test_PearsonCorrCoef"].tolist() + [
                baseline_pcc.loc[fold]
            ]
            axis.plot(
                x_values,
                y_values,
                color="0.7",
                marker="o",
                linewidth=0.8,
                alpha=0.7,
            )
        means = (
            method_data.groupby("budget")["test_PearsonCorrCoef"]
            .mean()
            .sort_index()
        )
        axis.plot(
            means.index.tolist() + [25],
            means.tolist() + [baseline_pcc.mean()],
            color="tab:blue",
            marker="o",
            linewidth=2.5,
            label="8-fold mean",
        )
        axis.set_title(method)
        axis.set_xticks([5, 10, 15, 20, 25])
        axis.set_xlabel("Patch budget K; K=25 is Baseline")
        axis.set_ylabel("PCC")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Patient-fold PCC trajectories toward Baseline-25")
    fig.tight_layout()
    fig.savefig(
        output_dir / "pcc_fold_trajectories.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    lines = [
        "# BC1 patch-budget comparison",
        "",
        "K includes the center target patch. Training seed is fixed at 2021; "
        "Random-K uses selection seed 1. Baseline-25 is the original TRIPLEX "
        "reference.",
        "",
        f"## {args.expected_folds}-fold mean ± population standard deviation",
        "",
        "| Method | K | PCC ↑ | CCC ↑ | MSE ↓ | MAE ↓ | Explained variance ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        method = str(row["method"])
        cells = []
        for metric in METRICS:
            cells.append(
                f"{row[f'{metric}_mean']:.5f} ± "
                f"{row[f'{metric}_std']:.5f}"
            )
        lines.append(
            f"| {method} | {int(row['budget'])} | " + " | ".join(cells) + " |"
        )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"Patch-budget comparison written to {output_dir}")


if __name__ == "__main__":
    main()
