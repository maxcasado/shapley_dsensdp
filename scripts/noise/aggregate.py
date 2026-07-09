"""
aggregate_noise_results.py
---------------------------
Aggregate the per-sigma Shapley/Perceptual-Score noise-ablation results
produced by run_noise_ablation.py into a single summary table.

Reads, for each results/noise_ablation/sigma_{sigma}/:
    shapley/shapley_run0_fold{k}.csv   (columns: run_id, fold_id, metric, view, shapley_value)
    ps/perceptual_score_fold{k}.csv    (columns: modality, ..., PS_raw, fold_id, ...)

Writes:
    results/noise_ablation/noise_ablation_summary.csv
        columns: sigma, modality, phi_mean, phi_std, ps_mean, ps_std
    (means/stds taken over the 5 folds)

Usage:
    python -m scripts.noise.aggregate
"""
import re
from pathlib import Path

import pandas as pd

OUT_ROOT = Path("results/noise_ablation")
PS_METRIC = "f1_weighted"


def _sigma_from_dirname(name: str) -> float:
    return float(re.match(r"sigma_(.+)", name).group(1))


def load_sigma_results(sigma_dir: Path) -> pd.DataFrame:
    sigma = _sigma_from_dirname(sigma_dir.name)

    shapley_files = sorted((sigma_dir / "shapley").glob("shapley_run*_fold*.csv"))
    df_shap = pd.concat([pd.read_csv(f) for f in shapley_files], ignore_index=True)
    df_shap = df_shap[df_shap["metric"] == PS_METRIC]
    phi = df_shap.groupby("view")["shapley_value"].agg(phi_mean="mean", phi_std="std")
    phi.index.name = "modality"

    ps_files = sorted((sigma_dir / "ps").glob("perceptual_score_fold*.csv"))
    df_ps = pd.concat([pd.read_csv(f) for f in ps_files], ignore_index=True)
    ps = df_ps.groupby("modality")["PS_raw"].agg(ps_mean="mean", ps_std="std")

    merged = phi.join(ps, how="outer").reset_index()
    merged.insert(0, "sigma", sigma)
    return merged


def main():
    sigma_dirs = [d for d in OUT_ROOT.glob("sigma_*") if d.is_dir()]
    sigma_dirs = sorted(sigma_dirs, key=lambda p: _sigma_from_dirname(p.name))

    rows = [
        load_sigma_results(d) for d in sigma_dirs
        if (d / "shapley").exists() and (d / "ps").exists()
    ]
    if not rows:
        print(f"No sigma results found under {OUT_ROOT}.")
        return

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["sigma", "modality"]).reset_index(drop=True)

    out_csv = OUT_ROOT / "noise_ablation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
