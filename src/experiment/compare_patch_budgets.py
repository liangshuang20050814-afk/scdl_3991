"""Summarize fixed-budget neighbor selection experiments."""

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

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    baseline_summary = summary[summary["method"] == "Baseline"].iloc[0]
    for axis, metric in zip(axes, METRICS):
        for method in METHOD_ORDER:
            data = summary[summary["method"] == method].sort_values("budget")
            if data.empty:
                continue
            axis.errorbar(
                data["budget"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"],
                marker="o",
                capsize=3,
                label=method,
            )
        axis.axhline(
            baseline_summary[f"{metric}_mean"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Baseline-25",
        )
        axis.set_xticks([5, 10, 15, 20])
        axis.set_xlabel("Patch budget K (includes center target)")
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        axis.set_title(f"{NAMES[metric]} {arrow}")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].remove()
    fig.suptitle(f"BC1 patch-budget comparison ({args.expected_folds} folds)")
    fig.tight_layout()
    fig.savefig(output_dir / "budget_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    for axis, metric in zip(axes, METRICS):
        metric_data = paired[paired["metric"] == metric]
        for method in METHOD_ORDER:
            data = metric_data[metric_data["method"] == method].sort_values(
                "budget"
            )
            axis.errorbar(
                data["budget"],
                data["mean_improvement"],
                yerr=data["std_improvement_across_folds"],
                marker="o",
                capsize=3,
                label=method,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        axis.set_xticks([5, 10, 15, 20])
        axis.set_xlabel("Patch budget K (includes center target)")
        axis.set_title(f"{NAMES[metric]} improvement vs Baseline-25")
        axis.set_ylabel("Positive means better than Baseline")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].remove()
    fig.suptitle("Mean paired improvement relative to original TRIPLEX")
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
