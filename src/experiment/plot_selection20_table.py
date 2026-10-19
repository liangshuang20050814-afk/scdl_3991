"""Render the BC1 K=20 comparison as a publication-style table figure."""

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
ROWS = [
    ("Reference", "Baseline-25", "Baseline"),
    ("Static selection", "Random-20", "Random-20"),
    ("Static selection", "Similarity-20", "Similarity-20"),
    ("Static selection", "Diversity-20", "Diversity-20"),
    ("Learned selection", "Soft attention*", "Soft attention"),
    ("Learned selection", "ST Hard-20", "ST Hard-20"),
    ("Learned selection", "Two-stage Hard-20", "Two-stage Hard-20"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--pcc-h-summary", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = pd.read_csv(args.summary)
    pcc_h = pd.read_csv(args.pcc_h_summary)
    pcc_h["metric"] = "test_PearsonCorrCoef_hpg"
    pcc_h = pcc_h[["method", "metric", "mean", "std_across_folds"]]
    summary = pd.concat([summary, pcc_h], ignore_index=True)
    lookup = summary.set_index(["method", "metric"])

    best = {}
    for metric, _, higher in METRICS:
        values = [float(lookup.loc[(key, metric), "mean"]) for _, _, key in ROWS]
        best[metric] = max(values) if higher else min(values)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(20, 9.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.955,
        "BC1 neighbor-selection comparison at K = 20",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.915,
        "Original TRIPLEX Baseline uses 25 patches",
        ha="center",
        va="center",
        fontsize=15,
        color="#4b5563",
    )

    left, right = 0.025, 0.975
    top, bottom = 0.875, 0.185
    boundaries = np.array([0.0, 0.15, 0.355, 0.484, 0.613, 0.742, 0.871, 1.0])
    boundaries = left + boundaries * (right - left)
    row_height = (top - bottom) / (len(ROWS) + 1)

    # Header and reference shading.
    axis.add_patch(Rectangle(
        (left, top - row_height), right - left, row_height,
        facecolor="#f1f5f9", edgecolor="none",
    ))
    baseline_top = top - row_height
    axis.add_patch(Rectangle(
        (left, baseline_top - row_height), right - left, row_height,
        facecolor="#dbeafe", edgecolor="none",
    ))

    # Table rules, matching the sparse-rule style of the reference image.
    rule_color = "#29323a"
    for x in boundaries:
        axis.plot([x, x], [bottom, top], color=rule_color, linewidth=1.35)
    axis.plot([left, right], [top, top], color=rule_color, linewidth=1.8)
    axis.plot(
        [left, right], [top - row_height, top - row_height],
        color=rule_color, linewidth=1.6,
    )
    for after_row in (1, 4, 7):
        y = top - row_height * (after_row + 1)
        axis.plot([left, right], [y, y], color=rule_color, linewidth=1.5)

    headers = ["Selection type", "Method", *[label for _, label, _ in METRICS]]
    for column, label in enumerate(headers):
        x = (boundaries[column] + boundaries[column + 1]) / 2
        axis.text(
            x,
            top - row_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="bold",
        )

    group_spans = {
        "Reference": (0, 0),
        "Static selection": (1, 3),
        "Learned selection": (4, 6),
    }
    for group, (start, end) in group_spans.items():
        y_top = top - row_height * (start + 1)
        y_bottom = top - row_height * (end + 2)
        axis.text(
            (boundaries[0] + boundaries[1]) / 2,
            (y_top + y_bottom) / 2,
            group.replace(" ", "\n") if group != "Reference" else group,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            linespacing=1.25,
        )

    for row_index, (_, display_name, key) in enumerate(ROWS):
        y = top - row_height * (row_index + 1.5)
        axis.text(
            (boundaries[1] + boundaries[2]) / 2,
            y,
            display_name,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold" if key == "Baseline" else "normal",
        )
        for metric_index, (metric, _, _) in enumerate(METRICS):
            record = lookup.loc[(key, metric)]
            mean = float(record["mean"])
            std = float(record["std_across_folds"])
            is_best = np.isclose(mean, best[metric], rtol=0, atol=1e-12)
            column = metric_index + 2
            axis.text(
                (boundaries[column] + boundaries[column + 1]) / 2,
                y,
                f"{mean:.3f} ± {std:.3f}",
                ha="center",
                va="center",
            fontsize=13.2,
                fontweight="bold" if is_best else "normal",
                color="#111827",
            )

    axis.text(
        left,
        0.12,
        "Values are mean ± population SD across 8 patient folds. "
        "Bold indicates the best mean for each metric.",
        ha="left",
        va="center",
        fontsize=13,
        color="#374151",
    )
    axis.text(
        left,
        0.077,
        "Random-20 is averaged over five selection seeds within each fold. "
        "*Soft attention uses continuous gates over all valid patches and is not strict Hard-20.",
        ha="left",
        va="center",
        fontsize=12.5,
        color="#374151",
    )
    axis.text(
        left,
        0.037,
        "PCC(M): mean PCC across all 250 genes. PCC(H): mean PCC across the same Baseline-defined top-50 genes.",
        ha="left",
        va="center",
        fontsize=12.5,
        color="#374151",
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(output_prefix.with_suffix(".png"))
    print(output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
