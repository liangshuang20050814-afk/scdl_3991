"""Create slide-ready figures that directly answer the two patch-selection questions."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


METHODS = ["Random", "Similarity", "Diversity", "ST Hard"]
COLORS = {
    "Random": "#4c78a8",
    "Similarity": "#f58518",
    "Diversity": "#54a24b",
    "ST Hard": "#e45756",
}
METRICS = [
    ("test_PearsonCorrCoef", "PCC(M) ↑", 1),
    ("test_PearsonCorrCoef_hpg", "PCC(H) ↑", 1),
    ("test_ConcordanceCorrCoef", "CCC ↑", 1),
    ("test_MeanSquaredError", "MSE ↓", -1),
    ("test_MeanAbsoluteError", "MAE ↓", -1),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-metrics", required=True)
    parser.add_argument("--pcch-fold-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=2021)
    return parser.parse_args()


def paired_improvements(folds, method, budget, metric, direction):
    baseline = (
        folds[folds["method"] == "Baseline"]
        .set_index("fold")[metric]
        .sort_index()
    )
    selected = (
        folds[(folds["method"] == method) & (folds["budget"] == budget)]
        .set_index("fold")[metric]
        .sort_index()
    )
    if not baseline.index.equals(selected.index):
        raise ValueError(f"Fold mismatch for {method}, K={budget}, {metric}")
    return direction * (selected - baseline).to_numpy(dtype=float)


def bootstrap_interval(values, samples, rng):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    return np.percentile(draws.mean(axis=1), [2.5, 97.5])


def save_figure(fig, output_dir, stem):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


def plot_context_sufficiency(folds, output_dir, samples, rng):
    fig, axis = plt.subplots(figsize=(12.5, 7.2))
    offsets = dict(zip(METHODS, [-0.9, -0.3, 0.3, 0.9]))
    contexts = [4, 9, 14, 19]

    for method in METHODS:
        means, lows, highs = [], [], []
        for context in contexts:
            values = paired_improvements(
                folds, method, context + 1, "test_PearsonCorrCoef", 1
            )
            low, high = bootstrap_interval(values, samples, rng)
            means.append(values.mean())
            lows.append(low)
            highs.append(high)
        means = np.asarray(means)
        errors = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        axis.errorbar(
            np.asarray(contexts) + offsets[method], means, yerr=errors,
            fmt="o", markersize=9, capsize=5, linewidth=2,
            color=COLORS[method], label=method,
        )

    axis.axhline(0, color="#111827", linestyle="--", linewidth=1.4)
    axis.axvspan(2.4, 5.6, color="#dbeafe", alpha=0.35)
    axis.text(
        4, axis.get_ylim()[1] if axis.get_ylim()[1] else 0.01,
        "Smallest tested budget", ha="center", va="top",
        fontsize=10.5, color="#1d4ed8",
    )
    axis.set_xticks(contexts, [f"{context}\n(K={context + 1})" for context in contexts])
    axis.set_xlabel("Number of selected context patches (target patch excluded)")
    axis.set_ylabel("Paired ΔPCC(M) vs Baseline-25  (positive = better)")
    fig.suptitle(
        "Question 1 — How many context patches are sufficient?",
        fontweight="bold", fontsize=17, y=0.985,
    )
    axis.text(
        0.5, 1.012,
        "Dots are 8-patient-fold means; bars are paired bootstrap 95% CIs",
        transform=axis.transAxes, ha="center", va="bottom", color="#4b5563",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=9, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    save_figure(fig, output_dir, "question1_context_sufficiency_technical")


def plot_context_sufficiency_simple(folds, pcch_folds, output_dir):
    """Audience-friendly PCC(M)/PCC(H) panels normalized to Baseline-25."""
    contexts = [4, 9, 14, 19]
    offsets = dict(zip(METHODS, [-0.9, -0.3, 0.3, 0.9]))
    display_names = {
        "Random": "Random selection",
        "Similarity": "Most similar",
        "Diversity": "Most diverse",
        "ST Hard": "Learned hard attention",
    }

    panels = [
        (folds, "test_PearsonCorrCoef", "PCC(M)", "Mean correlation across all 250 genes"),
        (pcch_folds, "test_PearsonCorrCoef_hpg", "PCC(H)", "Mean correlation across the top 50 genes"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), sharey=True)
    all_panel_values = []
    for axis, (data, metric, short_name, explanation) in zip(axes, panels):
        baseline_mean = data.loc[data["method"] == "Baseline", metric].mean()
        axis.axhspan(100, 105, color="#dcfce7", alpha=0.50)
        axis.axhspan(96, 100, color="#fff7ed", alpha=0.45)
        axis.axhline(100, color="#111827", linestyle="--", linewidth=1.7)
        panel_values = []
        for method in METHODS:
            relative_values = []
            x_values = []
            for context in contexts:
                mean_pcc = data.loc[
                    (data["method"] == method)
                    & (data["budget"] == context + 1),
                    metric,
                ].mean()
                relative = mean_pcc / baseline_mean * 100
                relative_values.append(relative)
                panel_values.append(relative)
                x_values.append(context + offsets[method])
            axis.scatter(
                x_values, relative_values,
                s=120, color=COLORS[method], edgecolor="white", linewidth=1.3,
                label=display_names[method], zorder=3,
            )
            for x_value, relative in zip(x_values, relative_values):
                axis.text(
                    x_value, relative + (0.20 if relative >= 100 else -0.24),
                    f"{relative:.1f}%",
                    ha="center", va="bottom" if relative >= 100 else "top",
                    fontsize=8.1, color=COLORS[method],
                )
        all_panel_values.extend(panel_values)
        axis.axvline(21.5, color="#cbd5e1", linewidth=1.2)
        axis.scatter(
            [24], [100], marker="*", s=260, color="#334155",
            edgecolor="white", linewidth=1.2, zorder=4,
            label="Full-context Baseline",
        )
        axis.text(
            24, 100.38, "100%", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#334155",
        )
        axis.set_xticks(
            [*contexts, 24], ["4", "9", "14", "19", "24\nFull"],
        )
        axis.set_xlim(1.7, 26.2)
        axis.set_title(f"{short_name}\n{explanation}", fontsize=13.5, fontweight="bold")
        axis.set_xlabel("Number of neighboring context patches", fontsize=11)
        axis.grid(axis="y", alpha=0.22)

    lower = min(97.6, min(all_panel_values) - 0.7)
    upper = max(104.8, max(all_panel_values) + 0.7)
    for axis in axes:
        axis.set_ylim(lower, upper)
    axes[0].set_ylabel(
        "Relative correlation performance\n(Baseline-25 = 100%)", fontsize=12
    )
    fig.suptitle(
        "Fewer context patches can preserve gene-expression prediction performance",
        fontsize=18, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.925,
        "The central target patch is always included; values are averages across 8 patient-level cross-validation folds",
        ha="center", va="bottom", fontsize=10.8, color="#4b5563",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=9.5,
        frameon=True, bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.91])
    save_figure(fig, output_dir, "question1_context_sufficiency")


def plot_strategy_scorecard(folds, pcch_folds, output_dir):
    combined = folds.merge(
        pcch_folds[["method", "budget", "fold", "test_PearsonCorrCoef_hpg"]],
        on=["method", "budget", "fold"], how="left", validate="one_to_one",
    )
    improvements = np.zeros((len(METHODS), len(METRICS)))
    wins = np.zeros_like(improvements)
    for row, method in enumerate(METHODS):
        for column, (metric, _, direction) in enumerate(METRICS):
            values = paired_improvements(combined, method, 5, metric, direction)
            improvements[row, column] = values.mean()
            wins[row, column] = (values > 0).sum()

    scale = np.max(np.abs(improvements), axis=0)
    scale[scale == 0] = 1
    normalized = improvements / scale
    fig, axis = plt.subplots(figsize=(12.5, 6.3))
    image = axis.imshow(normalized, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(METRICS)), [label for _, label, _ in METRICS])
    axis.set_yticks(range(len(METHODS)), METHODS)
    for row in range(len(METHODS)):
        for column in range(len(METRICS)):
            value = improvements[row, column]
            axis.text(
                column, row,
                f"{value:+.4f}\n{int(wins[row, column])}/8 folds",
                ha="center", va="center", fontsize=10.5,
                fontweight="bold" if value > 0 else "normal",
                color="white" if abs(normalized[row, column]) > 0.67 else "#111827",
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.78)
    colorbar.set_label("Column-normalized improvement (green = better)")
    axis.set_title(
        "Question 2 — Which selection strategy works best at 4 context patches (K=5)?",
        fontweight="bold", pad=13,
    )
    axis.set_xlabel("Cell text: paired mean improvement and patient-fold wins vs Baseline-25")
    fig.tight_layout()
    save_figure(fig, output_dir, "question2_strategy_scorecard_k5")


def plot_combined_answer(folds, output_dir):
    contexts = [4, 9, 14, 19]
    deltas = np.zeros((len(METHODS), len(contexts)))
    win_rates = np.zeros_like(deltas)
    for row, method in enumerate(METHODS):
        for column, context in enumerate(contexts):
            values = paired_improvements(
                folds, method, context + 1, "test_PearsonCorrCoef", 1
            )
            deltas[row, column] = values.mean()
            win_rates[row, column] = (values > 0).mean()

    limit = np.max(np.abs(deltas))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    for row, method in enumerate(METHODS):
        for column, context in enumerate(contexts):
            axis.scatter(
                column, row,
                s=500 + 1250 * win_rates[row, column],
                c=[deltas[row, column]], cmap="RdYlGn", norm=norm,
                edgecolor="#334155", linewidth=1.1,
            )
            axis.text(
                column, row,
                f"{deltas[row, column]:+.4f}\n{int(win_rates[row, column] * 8)}/8",
                ha="center", va="center", fontsize=9.5,
                fontweight="bold" if deltas[row, column] > 0 else "normal",
                color="white" if abs(deltas[row, column] / limit) > 0.58 else "#111827",
            )
    axis.set_xticks(
        range(len(contexts)),
        [f"{context} contexts\nK={context + 1}" for context in contexts],
    )
    axis.set_yticks(range(len(METHODS)), METHODS)
    axis.set_xlim(-0.65, len(contexts) - 0.35)
    axis.set_ylim(len(METHODS) - 0.55, -0.55)
    axis.grid(color="#cbd5e1", linewidth=0.8, alpha=0.7)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn")
    colorbar = fig.colorbar(scalar, ax=axis, shrink=0.78)
    colorbar.set_label("Paired mean ΔPCC(M) vs Baseline-25")
    axis.set_title(
        "Combined answer — Context budget × selection strategy",
        fontweight="bold", pad=13,
    )
    axis.set_xlabel(
        "Color/text = mean PCC change; bubble size and fraction = patient-fold win rate"
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "combined_context_strategy_answer")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.fold_metrics)
    pcch_folds = pd.read_csv(args.pcch_fold_metrics)
    rng = np.random.default_rng(args.bootstrap_seed)
    plot_context_sufficiency_simple(folds, pcch_folds, output_dir)
    plot_context_sufficiency(
        folds, output_dir, args.bootstrap_samples, rng
    )
    plot_strategy_scorecard(folds, pcch_folds, output_dir)
    plot_combined_answer(folds, output_dir)


if __name__ == "__main__":
    main()
