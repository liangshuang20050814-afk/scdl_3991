"""Render the BC1 patch-budget comparison as a publication-style table."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


METRICS = [
    ("test_ConcordanceCorrCoef", "CCC ↑", True),
    ("test_PearsonCorrCoef", "PCC(M) ↑", True),
    ("test_PearsonCorrCoef_hpg", "PCC(H) ↑", True),
    ("test_MeanSquaredError", "MSE ↓", False),
    ("test_MeanAbsoluteError", "MAE ↓", False),
]
METHODS = ["Baseline", "Random", "Similarity", "Diversity", "ST Hard"]
BUDGETS = {
    "Baseline": [25],
    "Random": [5, 10, 15, 20],
    "Similarity": [5, 10, 15, 20],
    "Diversity": [5, 10, 15, 20],
    "ST Hard": [5, 10, 15, 20],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--pcc-h-summary", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def build_lookup(summary_path, pcc_h_path):
    summary = pd.read_csv(summary_path)
    records = []
    for row in summary.itertuples(index=False):
        for metric, _, _ in METRICS:
            if metric == "test_PearsonCorrCoef_hpg":
                continue
            records.append({
                "method": row.method,
                "budget": int(row.budget),
                "metric": metric,
                "mean": getattr(row, f"{metric}_mean"),
                "std": getattr(row, f"{metric}_std"),
            })

    pcc_h = pd.read_csv(pcc_h_path)
    for row in pcc_h.itertuples(index=False):
        records.append({
            "method": row.method,
            "budget": int(row.budget),
            "metric": "test_PearsonCorrCoef_hpg",
            "mean": row.mean,
            "std": row.std_across_folds,
        })
    return pd.DataFrame(records).set_index(["method", "budget", "metric"])


def main():
    args = parse_args()
    lookup = build_lookup(args.summary, args.pcc_h_summary)
    rows = [(method, budget) for method in METHODS for budget in BUDGETS[method]]

    best = {}
    for metric, _, higher_is_better in METRICS:
        values = [float(lookup.loc[(method, budget, metric), "mean"]) for method, budget in rows]
        best[metric] = max(values) if higher_is_better else min(values)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(20, 16.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5, 0.965, "BC1 patch-budget comparison", ha="center", va="center",
        fontsize=26, fontweight="bold",
    )
    axis.text(
        0.5, 0.936,
        "Baseline uses 25 patches; K includes the central target patch",
        ha="center", va="center", fontsize=15, color="#4b5563",
    )

    left, right = 0.025, 0.975
    top, bottom = 0.905, 0.145
    boundaries = np.array([0.0, 0.18, 0.30, 0.44, 0.58, 0.72, 0.86, 1.0])
    boundaries = left + boundaries * (right - left)
    row_height = (top - bottom) / (len(rows) + 1)

    axis.add_patch(Rectangle(
        (left, top - row_height), right - left, row_height,
        facecolor="#f1f5f9", edgecolor="none",
    ))

    # Light group bands make the long table easier to scan while retaining
    # the sparse-rule layout of the K=20 comparison.
    group_spans = {}
    row_cursor = 0
    for group_index, method in enumerate(METHODS):
        start = row_cursor
        end = start + len(BUDGETS[method]) - 1
        group_spans[method] = (start, end)
        if method == "Baseline":
            color = "#dbeafe"
        elif group_index % 2 == 0:
            color = "#f8fafc"
        else:
            color = "white"
        y_top = top - row_height * (start + 1)
        y_bottom = top - row_height * (end + 2)
        axis.add_patch(Rectangle(
            (left, y_bottom), right - left, y_top - y_bottom,
            facecolor=color, edgecolor="none",
        ))
        row_cursor = end + 1

    rule_color = "#29323a"
    for x in boundaries:
        axis.plot([x, x], [bottom, top], color=rule_color, linewidth=1.25)
    axis.plot([left, right], [top, top], color=rule_color, linewidth=1.8)
    axis.plot(
        [left, right], [top - row_height, top - row_height],
        color=rule_color, linewidth=1.6,
    )
    for _, end in group_spans.values():
        y = top - row_height * (end + 2)
        axis.plot([left, right], [y, y], color=rule_color, linewidth=1.45)

    headers = ["Selection method", "Patch budget K", *[label for _, label, _ in METRICS]]
    for column, label in enumerate(headers):
        x = (boundaries[column] + boundaries[column + 1]) / 2
        axis.text(
            x, top - row_height / 2, label,
            ha="center", va="center", fontsize=14.2, fontweight="bold",
        )

    for method, (start, end) in group_spans.items():
        y_top = top - row_height * (start + 1)
        y_bottom = top - row_height * (end + 2)
        axis.text(
            (boundaries[0] + boundaries[1]) / 2,
            (y_top + y_bottom) / 2,
            "Baseline\n(original TRIPLEX)" if method == "Baseline" else method,
            ha="center", va="center", fontsize=14.5,
            fontweight="bold" if method == "Baseline" else "normal",
            linespacing=1.2,
        )

    for row_index, (method, budget) in enumerate(rows):
        y = top - row_height * (row_index + 1.5)
        axis.text(
            (boundaries[1] + boundaries[2]) / 2, y, str(budget),
            ha="center", va="center", fontsize=13.5,
            fontweight="bold" if method == "Baseline" else "normal",
        )
        for metric_index, (metric, _, _) in enumerate(METRICS):
            record = lookup.loc[(method, budget, metric)]
            mean = float(record["mean"])
            std = float(record["std"])
            is_best = np.isclose(mean, best[metric], rtol=0, atol=1e-12)
            column = metric_index + 2
            axis.text(
                (boundaries[column] + boundaries[column + 1]) / 2,
                y,
                f"{mean:.3f} ± {std:.3f}",
                ha="center", va="center", fontsize=12.4,
                fontweight="bold" if is_best else "normal",
                color="#111827",
            )

    axis.text(
        left, 0.101,
        "Values are mean ± population SD across 8 patient folds. Bold indicates the best mean for each metric.",
        ha="left", va="center", fontsize=12.5, color="#374151",
    )
    axis.text(
        left, 0.067,
        "Training seed is fixed at 2021; Random-K uses selection seed 1. PCC(H) uses the same Baseline-defined top-50 genes.",
        ha="left", va="center", fontsize=12.2, color="#374151",
    )
    axis.text(
        left, 0.034,
        "PCC(M): mean PCC across all 250 predicted genes. K=5 means one target patch plus up to four valid neighbors.",
        ha="left", va="center", fontsize=12.2, color="#374151",
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_prefix.with_suffix(".png"))
    print(output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
