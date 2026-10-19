"""Collect fold and summary PCC(H) for the BC1 patch-budget experiment."""

import argparse
from pathlib import Path
import re

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", nargs=4, required=True,
        metavar=("METHOD", "K", "RUN_ID", "RUN_DIR"),
    )
    parser.add_argument("--eval-tag", default="pcch")
    parser.add_argument("--expected-folds", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def fold_number(path):
    return int(re.fullmatch(r"fold(\d+)", path.name).group(1))


def main():
    args = parse_args()
    rows = []
    seen = set()
    for method, budget_text, run_id, run_dir in args.run:
        budget = int(budget_text)
        key = (method, budget)
        if key in seen:
            raise ValueError(f"Duplicate method-budget pair: {key}")
        seen.add(key)
        fold_dirs = sorted(Path(run_dir).glob("fold*"), key=fold_number)
        if len(fold_dirs) != args.expected_folds:
            raise ValueError(f"Expected {args.expected_folds} folds in {run_dir}")
        for fold_dir in fold_dirs:
            metrics_path = fold_dir / args.eval_tag / "metrics.csv"
            metrics = pd.read_csv(metrics_path)
            column = "test_PearsonCorrCoef_hpg"
            row = metrics.dropna(subset=[column]).iloc[-1]
            rows.append({
                "method": method,
                "budget": budget,
                "run_id": run_id,
                "fold": fold_number(fold_dir),
                column: float(row[column]),
            })

    folds = pd.DataFrame(rows).sort_values(["method", "budget", "fold"])
    summary = (
        folds.groupby(["method", "budget"])["test_PearsonCorrCoef_hpg"]
        .agg(mean="mean", std_across_folds=lambda x: x.std(ddof=0))
        .reset_index()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "pcch_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "pcch_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
