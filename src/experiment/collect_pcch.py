"""Collect Baseline-defined PCC(H) from tagged BC1 evaluation runs."""

import argparse
from pathlib import Path
import re

import pandas as pd


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
    parser.add_argument(
        "--run", action="append", nargs=3, required=True,
        metavar=("METHOD", "RUN_ID", "RUN_DIR"),
    )
    parser.add_argument("--eval-tag", default="pcch")
    parser.add_argument("--expected-folds", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fold_number(path):
    return int(re.fullmatch(r"fold(\d+)", path.name).group(1))


def main():
    args = parse_args()
    rows = []
    for method, run_id, run_dir in args.run:
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
                "run_id": run_id,
                "fold": fold_number(fold_dir),
                column: float(row[column]),
            })

    raw = pd.DataFrame(rows)
    random = (
        raw[raw["method"] == "Random-20"]
        .groupby("fold", as_index=False)["test_PearsonCorrCoef_hpg"]
        .mean()
    )
    random["method"] = "Random-20"
    folded = pd.concat([
        raw[raw["method"] != "Random-20"][[
            "method", "fold", "test_PearsonCorrCoef_hpg"
        ]],
        random[["method", "fold", "test_PearsonCorrCoef_hpg"]],
    ], ignore_index=True)

    records = []
    for method in METHOD_ORDER:
        values = folded.loc[
            folded["method"] == method, "test_PearsonCorrCoef_hpg"
        ]
        if len(values) != args.expected_folds:
            raise ValueError(f"Expected {args.expected_folds} folded values for {method}")
        records.append({
            "method": method,
            "mean": values.mean(),
            "std_across_folds": values.std(ddof=0),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    raw.to_csv(output.with_name("pcch_raw_run_fold_metrics.csv"), index=False)
    folded.to_csv(output.with_name("pcch_fold_metrics.csv"), index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
