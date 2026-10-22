"""Compare compact-trained K=5 models with mask-trained runs and Baseline-25."""

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "test_ConcordanceCorrCoef",
    "test_PearsonCorrCoef",
    "test_PearsonCorrCoef_hpg",
    "test_MeanSquaredError",
    "test_MeanAbsoluteError",
]
LOWER_IS_BETTER = {"test_MeanSquaredError", "test_MeanAbsoluteError"}
LABELS = {
    "test_ConcordanceCorrCoef": "CCC",
    "test_PearsonCorrCoef": "PCC(M)",
    "test_PearsonCorrCoef_hpg": "PCC(H)",
    "test_MeanSquaredError": "MSE",
    "test_MeanAbsoluteError": "MAE",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        metavar=("METHOD", "MASK_RUN", "COMPACT_RUN"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_run(run_dir, method, implementation):
    rows = []
    for fold in range(8):
        fold_dir = Path(run_dir) / f"fold{fold}"
        standard = pd.read_csv(fold_dir / "eval" / "metrics.csv").iloc[-1]
        pcch = pd.read_csv(fold_dir / "pcch" / "metrics.csv").iloc[-1]
        rows.append({
            "method": method,
            "implementation": implementation,
            "fold": fold,
            "test_ConcordanceCorrCoef": standard["test_ConcordanceCorrCoef"],
            "test_PearsonCorrCoef": standard["test_PearsonCorrCoef"],
            "test_PearsonCorrCoef_hpg": pcch["test_PearsonCorrCoef_hpg"],
            "test_MeanSquaredError": standard["test_MeanSquaredError"],
            "test_MeanAbsoluteError": standard["test_MeanAbsoluteError"],
        })
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_run(args.baseline_run, "Baseline-25", "baseline")
    for method, mask_run, compact_run in args.method:
        rows.extend(read_run(mask_run, method, "mask"))
        rows.extend(read_run(compact_run, method, "compact"))
    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "trained_fold_metrics.csv", index=False)

    summary = (
        folds.groupby(["method", "implementation"])[METRICS]
        .agg(["mean", lambda values: values.std(ddof=0)])
        .reset_index()
    )
    summary.columns = [
        "method", "implementation",
        *[
            f"{metric}_{stat}"
            for metric in METRICS
            for stat in ("mean", "std")
        ],
    ]
    summary.to_csv(output_dir / "trained_method_summary.csv", index=False)

    paired_rows = []
    for method, _, _ in args.method:
        method_data = folds[folds["method"] == method]
        mask = method_data[method_data["implementation"] == "mask"].set_index("fold")
        compact = method_data[method_data["implementation"] == "compact"].set_index("fold")
        for metric in METRICS:
            delta = compact[metric] - mask[metric]
            improvement = -delta if metric in LOWER_IS_BETTER else delta
            paired_rows.append({
                "method": method,
                "metric": LABELS[metric],
                "mean_compact_minus_mask": delta.mean(),
                "mean_improvement": improvement.mean(),
                "fold_win_rate": (improvement > 0).mean(),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_compact_vs_mask.csv", index=False)

    lines = [
        "# BC1 Level 1 compact-token training results",
        "",
        "All values are means ± population SD across eight patient-level folds. "
        "K=5 includes one target and four context patches.",
        "",
        "| Method | Implementation | CCC ↑ | PCC(M) ↑ | PCC(H) ↑ | MSE ↓ | MAE ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = ["Baseline-25", *[method for method, _, _ in args.method]]
    for method in order:
        implementations = ["baseline"] if method == "Baseline-25" else ["mask", "compact"]
        for implementation in implementations:
            row = summary[
                (summary["method"] == method)
                & (summary["implementation"] == implementation)
            ].iloc[0]
            cells = [
                f"{row[f'{metric}_mean']:.5f} ± {row[f'{metric}_std']:.5f}"
                for metric in METRICS
            ]
            lines.append(
                f"| {method} | {implementation} | " + " | ".join(cells) + " |"
            )
    (output_dir / "trained_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
