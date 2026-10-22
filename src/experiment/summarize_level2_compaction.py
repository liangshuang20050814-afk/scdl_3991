"""Summarize BC1 Level 2 equivalence and efficiency benchmarks."""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("output/comparison/ST/andersson/level2_compact")
METHODS = ("random", "similarity", "diversity")
IMPLEMENTATIONS = ("mask", "level1", "level2")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate():
    groups = defaultdict(list)
    metadata = {}
    for method in METHODS:
        for row in read_csv(ROOT / f"benchmark_{method}.csv"):
            key = (
                method,
                row["implementation"],
                row["section"],
                row["phase"],
            )
            groups[key].append(row)
            metadata[key] = row

    summary = []
    for key, rows in sorted(groups.items()):
        method, implementation, section, phase = key
        for metric in (
            "milliseconds_per_step",
            "spots_per_second",
            "peak_allocated_mib",
        ):
            values = [float(row[metric]) for row in rows if row[metric] != ""]
            if not values:
                continue
            summary.append({
                "method": method,
                "implementation": implementation,
                "section": section,
                "phase": phase,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "median": statistics.median(values),
                "repeats": len(values),
                "neighbor_tokens": metadata[key]["neighbor_tokens"],
                "neighbor_payload_mib": metadata[key]["neighbor_payload_mib"],
                "total_payload_mib": metadata[key]["total_payload_mib"],
            })
    return summary


def lookup(summary, method, implementation, section, phase, metric):
    for row in summary:
        if (
            row["method"] == method
            and row["implementation"] == implementation
            and row["section"] == section
            and row["phase"] == phase
            and row["metric"] == metric
        ):
            return row
    raise KeyError((method, implementation, section, phase, metric))


def write_summary(summary):
    path = ROOT / "benchmark_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    comparisons = []
    for method in METHODS:
        for section, phase, metric in (
            ("dataloader", "load", "spots_per_second"),
            ("transfer", "cpu_to_gpu", "milliseconds_per_step"),
            ("model", "training", "spots_per_second"),
            ("model", "training", "peak_allocated_mib"),
            ("pipeline", "load_transfer_train", "spots_per_second"),
            ("pipeline", "load_transfer_train", "peak_allocated_mib"),
        ):
            level1 = lookup(summary, method, "level1", section, phase, metric)["mean"]
            level2 = lookup(summary, method, "level2", section, phase, metric)["mean"]
            comparisons.append({
                "method": method,
                "section": section,
                "phase": phase,
                "metric": metric,
                "level1_mean": level1,
                "level2_mean": level2,
                "level2_vs_level1_percent": (level2 / level1 - 1) * 100,
            })
    with (ROOT / "level2_vs_level1.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons


def grouped_panel(ax, summary, section, phase, metric, title, ylabel):
    x = np.arange(len(METHODS))
    width = 0.24
    colors = {"mask": "#8a94a6", "level1": "#5b8ff9", "level2": "#36b37e"}
    labels = {"mask": "Mask (25 loaded)", "level1": "Level 1 (25→5)", "level2": "Level 2 (5 loaded)"}
    for offset, implementation in enumerate(IMPLEMENTATIONS):
        rows = [
            lookup(summary, method, implementation, section, phase, metric)
            for method in METHODS
        ]
        values = [row["mean"] for row in rows]
        errors = [row["std"] for row in rows]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=colors[implementation],
            label=labels[implementation],
        )
    ax.set_xticks(x, [name.title() for name in METHODS])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    grouped_panel(
        axes[0, 0], summary, "dataloader", "load", "spots_per_second",
        "Data loading throughput", "spots / second (higher is better)",
    )
    grouped_panel(
        axes[0, 1], summary, "transfer", "cpu_to_gpu", "milliseconds_per_step",
        "CPU to GPU transfer", "milliseconds / batch (lower is better)",
    )
    grouped_panel(
        axes[1, 0], summary, "pipeline", "load_transfer_train", "spots_per_second",
        "End-to-end training pipeline", "spots / second (higher is better)",
    )
    grouped_panel(
        axes[1, 1], summary, "pipeline", "load_transfer_train", "peak_allocated_mib",
        "Peak GPU memory", "MiB (lower is better)",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("BC1 Level 2 physical K=5 token compaction", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ROOT / "level2_efficiency.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / "level2_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary, comparisons):
    equivalence = []
    for method in METHODS:
        equivalence.extend(read_csv(ROOT / f"equivalence_{method}.csv"))
    lines = [
        "# BC1 Level 2 compact-neighbor experiment",
        "",
        "## Correctness",
        "",
        "| Method | Level 1 tokens | Level 2 tokens | Max absolute logit difference | Exact within 1e-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in zip(METHODS, equivalence):
        lines.append(
            f"| {method.title()} | {row['level1_tokens']} | {row['level2_tokens']} | "
            f"{float(row['max_abs_logit_difference']):.3g} | {row['allclose_atol_1e-5']} |"
        )
    lines.extend([
        "",
        "All compact caches were also validated spot-by-spot against the original 25-token files.",
        "",
        "## Physical storage",
        "",
        "The original BC1 CIGAR neighbor cache is 757 MiB. Each K=5 method-specific cache is 134 MiB.",
        "",
        "## Efficiency comparison",
        "",
        "Mask means the original K=5 implementation: 25 tokens are loaded and unselected tokens are masked. Level 1 loads 25 and compacts to 5 inside the model. Level 2 loads only 5 from disk.",
        "",
        "Positive throughput percentages indicate faster Level 2; negative transfer/memory percentages indicate improvement.",
        "",
        "| Method | Loader vs Mask | Transfer vs Mask | End-to-end vs Mask | End-to-end vs Level 1 | Peak memory vs Mask |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        values = {}
        for section, phase, metric in (
            ("dataloader", "load", "spots_per_second"),
            ("transfer", "cpu_to_gpu", "milliseconds_per_step"),
            ("pipeline", "load_transfer_train", "spots_per_second"),
            ("pipeline", "load_transfer_train", "peak_allocated_mib"),
        ):
            mask = lookup(summary, method, "mask", section, phase, metric)["mean"]
            level1 = lookup(summary, method, "level1", section, phase, metric)["mean"]
            level2 = lookup(summary, method, "level2", section, phase, metric)["mean"]
            values[(section, metric, "mask")] = (level2 / mask - 1) * 100
            values[(section, metric, "level1")] = (level2 / level1 - 1) * 100
        lines.append(
            f"| {method.title()} | "
            f"{values[('dataloader', 'spots_per_second', 'mask')]:+.2f}% | "
            f"{values[('transfer', 'milliseconds_per_step', 'mask')]:+.2f}% | "
            f"{values[('pipeline', 'spots_per_second', 'mask')]:+.2f}% | "
            f"{values[('pipeline', 'spots_per_second', 'level1')]:+.2f}% | "
            f"{values[('pipeline', 'peak_allocated_mib', 'mask')]:+.2f}% |"
        )
    lines.extend([
        "",
        "Benchmarks use BC1 fold 0, batch size 128, physical GPU 2 (RTX A6000), and five repeats.",
        "Because Mask and Level 1 use the same input pipeline but showed timing variation, the Level 2-versus-Mask end-to-end comparison is the more conservative headline result.",
        "Prediction metrics are inherited from the numerically equivalent Level 1 inputs; no redundant 8-fold retraining was performed.",
    ])
    (ROOT / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    summary = aggregate()
    comparisons = write_summary(summary)
    plot(summary)
    write_report(summary, comparisons)
    print(ROOT / "comparison_report.md")


if __name__ == "__main__":
    main()
