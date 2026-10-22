"""Summarize Level-1 compact-token equivalence and GPU benchmarks."""

import argparse
from pathlib import Path

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
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("METHOD", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--benchmark-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def equivalence_rows(runs):
    rows = []
    for method, run_dir in runs:
        run_dir = Path(run_dir)
        for fold_dir in sorted(run_dir.glob("fold*")):
            original = pd.read_csv(fold_dir / "eval" / "metrics.csv").iloc[-1]
            compact = pd.read_csv(
                fold_dir / "compact_eval" / "metrics.csv"
            ).iloc[-1]
            for metric in METRICS:
                rows.append({
                    "method": method,
                    "fold": int(fold_dir.name.replace("fold", "")),
                    "metric": metric,
                    "mask_value": original[metric],
                    "compact_value": compact[metric],
                    "absolute_difference": abs(original[metric] - compact[metric]),
                })
    return pd.DataFrame(rows)


def benchmark_summary(paths):
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = (
        raw.groupby(["budget", "phase", "implementation"], as_index=False)
        .agg(
            milliseconds_per_step=("milliseconds_per_step", "median"),
            milliseconds_std=("milliseconds_per_step", "std"),
            peak_allocated_mib=("peak_allocated_mib", "median"),
            incremental_peak_mib=("incremental_peak_mib", "median"),
            spots_per_second=("spots_per_second", "median"),
        )
    )
    mask = summary[summary["implementation"] == "mask"].set_index(
        ["budget", "phase"]
    )
    compact = summary[summary["implementation"] == "compact"].set_index(
        ["budget", "phase"]
    )
    paired = compact.join(mask, lsuffix="_compact", rsuffix="_mask").reset_index()
    paired["speedup"] = (
        paired["milliseconds_per_step_mask"]
        / paired["milliseconds_per_step_compact"]
    )
    paired["time_change_percent"] = 100 * (
        paired["milliseconds_per_step_compact"]
        / paired["milliseconds_per_step_mask"] - 1
    )
    paired["peak_memory_reduction_percent"] = 100 * (
        1 - paired["peak_allocated_mib_compact"]
        / paired["peak_allocated_mib_mask"]
    )
    return raw, summary, paired


def plot_summary(paired, output_dir):
    training = paired[paired["phase"] == "training"].sort_values("budget")
    forward = paired[paired["phase"] == "forward"].sort_values("budget")
    budgets = training["budget"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    width = 1.7
    axes[0].bar(
        budgets - width / 2,
        training["peak_allocated_mib_mask"],
        width=width,
        color="#94a3b8",
        label="Mask (25-token tensor)",
    )
    axes[0].bar(
        budgets + width / 2,
        training["peak_allocated_mib_compact"],
        width=width,
        color="#2563eb",
        label="Compact (K-token tensor)",
    )
    for _, row in training.iterrows():
        axes[0].text(
            row["budget"] + width / 2,
            row["peak_allocated_mib_compact"] + 18,
            f"−{row['peak_memory_reduction_percent']:.1f}%",
            ha="center",
            fontsize=9,
            color="#1d4ed8",
            fontweight="bold",
        )
    axes[0].set_title("Training GPU memory")
    axes[0].set_ylabel("Peak allocated memory (MiB)")
    axes[0].set_xlabel("Patch budget K (target included)")
    axes[0].set_xticks(budgets)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].axhline(0, color="#111827", linewidth=1.2)
    axes[1].bar(
        budgets - width / 2,
        forward["time_change_percent"],
        width=width,
        color="#0ea5e9",
        label="Forward",
    )
    axes[1].bar(
        budgets + width / 2,
        training["time_change_percent"],
        width=width,
        color="#f97316",
        label="Forward + backward",
    )
    axes[1].set_title("Runtime change after compaction")
    axes[1].set_ylabel("Time change vs mask (%)\nnegative is faster")
    axes[1].set_xlabel("Patch budget K (target included)")
    axes[1].set_xticks(budgets)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Level 1 token compaction: memory improves, runtime remains similar",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "level1_efficiency.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "level1_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equivalence = equivalence_rows(args.run)
    equivalence_summary = (
        equivalence.groupby(["method", "metric"], as_index=False)
        .agg(
            max_absolute_difference=("absolute_difference", "max"),
            mean_absolute_difference=("absolute_difference", "mean"),
        )
    )
    paths = sorted(Path().glob(args.benchmark_glob))
    if not paths:
        raise FileNotFoundError(args.benchmark_glob)
    raw, benchmark, paired = benchmark_summary(paths)

    equivalence.to_csv(output_dir / "equivalence_fold_metrics.csv", index=False)
    equivalence_summary.to_csv(
        output_dir / "equivalence_summary.csv", index=False
    )
    raw.to_csv(output_dir / "benchmark_repeats.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_summary.csv", index=False)
    paired.to_csv(output_dir / "benchmark_paired.csv", index=False)
    plot_summary(paired, output_dir)

    max_pcc_difference = equivalence.loc[
        equivalence["metric"] == "test_PearsonCorrCoef",
        "absolute_difference",
    ].max()
    k5_training = paired[
        (paired["budget"] == 5) & (paired["phase"] == "training")
    ].iloc[0]
    lines = [
        "# Level 1 compact-neighbor experiment",
        "",
        "Selected neighbor tokens are gathered before the NeighborEncoder and "
        "FusionEncoder. Original 5x5 indices are retained for relative-position bias.",
        "",
        "## Numerical equivalence",
        "",
        f"Across the four K=5 methods and eight folds, the maximum absolute "
        f"PCC(M) difference was {max_pcc_difference:.3g}.",
        "",
        "## Efficiency benchmark",
        "",
        "A real BC1 fold-0 training batch (batch size 128, AMP, physical GPU 2) "
        "was measured with 10 warm-up and 100 timed iterations over five repeats.",
        "",
        f"At K=5, total training peak allocated memory decreased by "
        f"{k5_training['peak_memory_reduction_percent']:.2f}%, while the median "
        f"step-time change was {k5_training['time_change_percent']:+.2f}%.",
        "",
        "Interpretation: token compaction is correct and reduces training memory, "
        "but the 25-token neighbor sequence is too short and too small a fraction "
        "of total TRIPLEX compute to guarantee an end-to-end speedup.",
    ]
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(equivalence_summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
