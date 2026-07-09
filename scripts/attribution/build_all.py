"""
scripts/attribution/build_all.py
--------------------------------
Single post-processing driver: from the v-dicts (the one source of truth),
produce EVERY attribution deliverable at once, each carrying the cross-fold
significance layer (mean ± std + |SNR|>2 gate, paired Wilcoxon for model
comparisons). Replaces the old run_comparison_analysis.py.

Reads (per model, first existing dir wins — legacy or regenerated):
    results/<model>/vdict/v_dict_fold{0..4}.pkl   (regenerated, format v2)
    results/<model>/eval[_geo_fixed]/v_dict_fold{0..4}.pkl   (legacy, format v1)
    results/<model>/perceptual_score_fold{0..4}.csv

Writes:
    results/attribution/<model>/attribution_{per_fold,summary}.csv
    results/attribution/<model>/coalition_values_{per_fold,summary}.csv
    results/attribution/geo_comparison_{per_fold,summary}.csv
    results/attribution/redundancy_com_{mean,std}.csv
    results/attribution/correlation_{summary,per_fold}.csv
    results/figures/fig3_phi_ps_marginal.*  fig4_scatter_phi_vs_ps.*
    results/figures/fig5_phi_decomposition.*  fig_geo_redistribution.*
    paper/tables/{attribution,coalition_values,geo_comparison}_<model>.tex  (with --paper)

Usage:
    python -m scripts.attribution.build_all
    python -m scripts.attribution.build_all --metric mcc          # needs v2 v-dicts
    python -m scripts.attribution.build_all --paper                # also emit LaTeX
"""
import argparse
from pathlib import Path

import pandas as pd

from shap_analysis.comparison import (
    load_v_dicts, compute_redundancy_matrix, correlation_analysis,
    build_shapley_table, add_phi_decomposition,
)
from shap_analysis.tables import (
    attribution_table, coalition_value_table, geo_comparison_table,
)
from shap_analysis import figures
from shap_analysis.stats import agg_frame
from shap_analysis.characteristic import DEFAULT_METRIC

ROOT     = Path(".")
RESULTS  = ROOT / "results"
ATTR_DIR = RESULTS / "attribution"
FIG_DIR  = RESULTS / "figures"
PAPER_TAB = ROOT / "paper" / "tables"
FOLD_IDS = [0, 1, 2, 3, 4]

MODELS = {
    "com": {
        "vdict_dirs":  [RESULTS / "com" / "vdict", RESULTS / "com" / "eval"],
        "ps_dir":      RESULTS / "com",
        "view_names":  ["S2_S2VI", "S1", "weather", "DEM"],
        "fixed_views": [],
    },
    "com_geo": {
        "vdict_dirs":  [RESULTS / "com_geo" / "vdict", RESULTS / "com_geo" / "eval_geo_fixed"],
        "ps_dir":      RESULTS / "com_geo",
        "view_names":  ["S2_S2VI", "S1", "weather", "DEM", "geo"],
        "fixed_views": ["geo"],
    },
}
COMMON_VIEWS = ["S2_S2VI", "S1", "weather", "DEM"]


def log(msg):
    print(msg, flush=True)


def _first_existing(dirs, fold_ids):
    for d in dirs:
        if all((Path(d) / f"v_dict_fold{f}.pkl").exists() for f in fold_ids):
            return Path(d)
    raise FileNotFoundError(f"no complete v_dict set in any of: {[str(d) for d in dirs]}")


def _load_ps(ps_dir, fold_ids):
    frames = []
    for f in fold_ids:
        p = Path(ps_dir) / f"perceptual_score_fold{f}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(args):
    metric = args.metric
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Characteristic function v = {metric}")

    vdicts = {}
    attr_summaries = {}
    attr_per_fold = {}

    for name, cfg in MODELS.items():
        log(f"\n=== {name} ===")
        vdir = _first_existing(cfg["vdict_dirs"], FOLD_IDS)
        log(f"  v-dicts <- {vdir}")
        vd = load_v_dicts(vdir, FOLD_IDS)
        vdicts[name] = vd
        ps = _load_ps(cfg["ps_dir"], FOLD_IDS)

        # --- main attribution table (ex-Table 2), significance-annotated -------
        at = attribution_table(vd, ps, cfg["view_names"], metric,
                               fixed_views=cfg["fixed_views"])
        out = ATTR_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        at["per_fold"].to_csv(out / "attribution_per_fold.csv", index=False)
        at["summary"].to_csv(out / "attribution_summary.csv", index=False)
        attr_summaries[name] = at["summary"]
        attr_per_fold[name] = at["per_fold"]
        log("  attribution_summary:\n" + at["summary"][
            ["modality", "phi_mean", "phi_std", "phi_resolved",
             "v_single_mean", "v_empty_mean"]].round(4).to_string(index=False))

        # --- coalition values (new table) -------------------------------------
        cv = coalition_value_table(vd, cfg["view_names"], metric)
        cv["per_fold"].to_csv(out / "coalition_values_per_fold.csv", index=False)
        cv["summary"].to_csv(out / "coalition_values_summary.csv", index=False)
        log(f"  coalition values: {len(cv['summary'])} coalitions "
            f"-> {out / 'coalition_values_summary.csv'}")

    # --- geo comparison (ex-Table 3) with paired Wilcoxon --------------------
    log("\n=== geo comparison (CoM vs CoM+geo) ===")
    gc = geo_comparison_table(vdicts["com"], vdicts["com_geo"], COMMON_VIEWS, metric)
    gc["per_fold"].to_csv(ATTR_DIR / "geo_comparison_per_fold.csv", index=False)
    gc["summary"].to_csv(ATTR_DIR / "geo_comparison_summary.csv", index=False)
    log(gc["summary"][["modality", "phi_com_mean", "phi_geo_mean", "delta_mean",
                       "delta_std", "wilcoxon_p", "all_same_sign"]].round(4).to_string(index=False))

    # --- redundancy (invariant to the v(∅) fix) + correlation ----------------
    mean_red, std_red = compute_redundancy_matrix(vdicts["com"], COMMON_VIEWS, metric, fixed_views=[])
    mean_red.to_csv(ATTR_DIR / "redundancy_com_mean.csv")
    std_red.to_csv(ATTR_DIR / "redundancy_com_std.csv")

    corr_rows, corr_folds = [], []
    for name, cfg in MODELS.items():
        free = [v for v in cfg["view_names"] if v not in cfg["fixed_views"]]
        ps = _load_ps(cfg["ps_dir"], FOLD_IDS)
        if not len(ps):
            continue
        shap_only = attr_per_fold[name].drop(
            columns=[c for c in ("PS_raw", "PS_model") if c in attr_per_fold[name].columns])
        res = correlation_analysis(shap_only, ps, free)
        res["per_fold"].insert(0, "model", name)
        corr_folds.append(res["per_fold"])
        corr_rows.append({"model": name, "overall_spearman": res["overall_spearman"],
                          "overall_pearson": res["overall_pearson"]})
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(ATTR_DIR / "correlation_summary.csv", index=False)
        pd.concat(corr_folds, ignore_index=True).to_csv(ATTR_DIR / "correlation_per_fold.csv", index=False)

    # --- figures (cross-fold error bars) -------------------------------------
    log("\n=== figures ===")
    figures.fig_phi_ps_marginal(attr_summaries["com"], COMMON_VIEWS,
                                FIG_DIR / "fig3_phi_ps_marginal", metric)
    figures.fig_scatter_phi_vs_ps(attr_per_fold["com"], COMMON_VIEWS,
                                  FIG_DIR / "fig4_scatter_phi_vs_ps")
    figures.fig_phi_decomposition(attr_summaries["com"], COMMON_VIEWS,
                                  FIG_DIR / "fig5_phi_decomposition")
    figures.fig_geo_redistribution(gc["summary"], COMMON_VIEWS,
                                   FIG_DIR / "fig_geo_redistribution")
    log(f"  figures -> {FIG_DIR}")

    # --- LaTeX (optional) — delegate to the paper/tables builders ------------
    if args.paper:
        log("\n=== paper tables (LaTeX) ===")
        from paper.tables import (
            build_table3_full_comparison, build_table_coalition_values,
            build_table_geo_comparison,
        )
        build_table3_full_comparison.main()
        build_table_coalition_values.main()
        build_table_geo_comparison.main()

    log(f"\nDone. Deliverables under {ATTR_DIR} and {FIG_DIR}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    help="characteristic function (f1_weighted, mcc, balanced_accuracy, ...); "
                         "non-f1 metrics require regenerated v2 v-dicts.")
    ap.add_argument("--paper", action="store_true", help="also emit LaTeX tables to paper/tables/")
    main(ap.parse_args())
