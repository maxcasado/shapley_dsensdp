"""
plot_noise_ablation.py
------------------------
Two-panel figure for the S2_S2VI Gaussian noise ablation: Shapley value and
Perceptual Score as a function of noise sigma, one line per modality.

Reads results/noise_ablation/noise_ablation_summary.csv (produced by
aggregate_noise_results.py) and writes
results/noise_ablation/fig_noise_ablation.{png,pdf}.

Usage:
    python -m scripts.noise.plot
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_ROOT = Path("results/noise_ablation")
SUMMARY_CSV = OUT_ROOT / "noise_ablation_summary.csv"

MODALITY_COLORS = {
    "S2_S2VI": "#4e79a7",
    "S1": "#f28e2b",
    "weather": "#e15759",
    "DEM": "#76b7b2",
}
MODALITY_ORDER = ["S2_S2VI", "S1", "weather", "DEM"]


def _plot_panel(ax, df, mean_col, std_col, ylabel, title):
    for modality in MODALITY_ORDER:
        sub = df[df["modality"] == modality].sort_values("sigma")
        if sub.empty:
            continue
        color = MODALITY_COLORS[modality]
        ax.plot(sub["sigma"], sub[mean_col], marker="o", color=color, label=modality, zorder=3)
        ax.fill_between(
            sub["sigma"],
            sub[mean_col] - sub[std_col],
            sub[mean_col] + sub[std_col],
            color=color, alpha=0.2, zorder=2,
        )
    ax.set_xlabel(r"Gaussian noise $\sigma$ (S2\_S2VI)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(SUMMARY_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    _plot_panel(axes[0], df, "phi_mean", "phi_std", r"Shapley value $\phi$", "Shapley value vs. noise")
    _plot_panel(axes[1], df, "ps_mean", "ps_std", "Perceptual Score (PS_raw)", "Perceptual Score vs. noise")
    axes[1].legend(title="Modality", loc="best")
    fig.tight_layout()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = OUT_ROOT / "fig_noise_ablation.png"
    pdf_path = OUT_ROOT / "fig_noise_ablation.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {png_path}")
    print(f"Saved -> {pdf_path}")


if __name__ == "__main__":
    main()
