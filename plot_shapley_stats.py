"""
plot_shapley_stats.py
---------------------
Genere des histogrammes/barplots depuis les CSV de stats produits par infer_france.py.

Structure attendue dans --stats_dir :
    stats_global.csv
    stats_par_continent.csv
    stats_par_prediction.csv
    stats_continent_x_prediction.csv   (optionnel)

Usage :
    python plot_shapley_stats.py --stats_dir preds/france/stats
    python plot_shapley_stats.py --stats_dir preds/france/stats --out_dir preds/france/stats/plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Palette coherente avec plot_continent_map dans infer_france.py
CONTINENT_COLORS = {
    "Europe":           "#4e79a7",
    "Asia":             "#f28e2b",
    "Africa":           "#e15759",
    "North America":    "#76b7b2",
    "South America":    "#59a14f",
    "Oceania":          "#edc948",
    "Antarctica":       "#b07aa1",
    "Seven seas (open ocean)": "#9c755f",
    "Inconnu":          "#bab0ac",
    "Unknown":          "#bab0ac",
}

PRED_COLORS = {
    "Correct":   "#2ca02c",
    "Incorrect": "#d62728",
}

MODALITY_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                   "#59a14f", "#edc948", "#b07aa1", "#9c755f"]


# ==============================================================================
#  Helpers
# ==============================================================================

def _phi_cols(df):
    """Retourne les colonnes |phi|_mean_* dans l'ordre du CSV."""
    return [c for c in df.columns if c.startswith("|phi|_mean_")]


def _view_names(df):
    return [c.replace("|phi|_mean_", "") for c in _phi_cols(df)]


def _save(fig, path, verbose=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  -> {path}")


def _bar_with_errorbars(ax, x_pos, means, stds, colors, width=0.6):
    bars = ax.bar(x_pos, means, width=width, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(x_pos, means, yerr=stds,
                fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=3)
    return bars


# ==============================================================================
#  1. Global : barplot simple par modalite
# ==============================================================================

def plot_global(df_global, out_dir):
    views  = _view_names(df_global)
    means  = [df_global[f"|phi|_mean_{v}"].iloc[0] for v in views]
    stds   = [df_global[f"|phi|_std_{v}"].iloc[0]  for v in views]
    colors = MODALITY_COLORS[:len(views)]

    fig, ax = plt.subplots(figsize=(max(5, len(views) * 1.4), 4))
    x = np.arange(len(views))
    _bar_with_errorbars(ax, x, means, stds, colors)

    ax.set_xticks(x)
    ax.set_xticklabels(views, fontsize=11)
    ax.set_ylabel("mean(|φ|)", fontsize=11)
    ax.set_title(
        f"Importance Shapley globale  (n={int(df_global['n_points'].iloc[0])})",
        fontsize=13,
    )
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # valeurs au-dessus des barres
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom",
                fontsize=9, color="#333333")

    plt.tight_layout()
    _save(fig, out_dir / "hist_global.png")


# ==============================================================================
#  2. Par continent : barplot groupe + heatmap
# ==============================================================================

def plot_par_continent(df_cont, out_dir):
    views      = _view_names(df_cont)
    continents = df_cont.iloc[:, 0].tolist()   # premiere colonne = nom du continent
    n_cont     = len(continents)
    n_views    = len(views)

    # --- barplot groupe ---
    fig, ax = plt.subplots(figsize=(max(8, n_cont * 1.6), 5))
    width   = 0.8 / n_views
    x       = np.arange(n_cont)

    for vi, (view, color) in enumerate(zip(views, MODALITY_COLORS)):
        means = df_cont[f"|phi|_mean_{view}"].values
        stds  = df_cont[f"|phi|_std_{view}"].values
        offset = (vi - n_views / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width=width * 0.9,
                      color=color, label=view,
                      edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=2, linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ns = df_cont["n_points"].astype(int).tolist()
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(continents, ns)],
        fontsize=9,
    )
    ax.set_ylabel("mean(|φ|)", fontsize=11)
    ax.set_title("Importance Shapley par continent", fontsize=13)
    ax.legend(title="Modalite", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out_dir / "hist_par_continent.png")

    # --- heatmap continents x modalites ---
    matrix = np.array([
        [df_cont[f"|phi|_mean_{v}"].iloc[i] for v in views]
        for i in range(n_cont)
    ])

    fig, ax = plt.subplots(figsize=(max(5, n_views * 1.2), max(4, n_cont * 0.7)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="mean(|φ|)", fraction=0.03, pad=0.04)

    ax.set_xticks(np.arange(n_views))
    ax.set_xticklabels(views, fontsize=10)
    ax.set_yticks(np.arange(n_cont))
    ax.set_yticklabels(
        [f"{c}  (n={n})" for c, n in zip(continents, ns)],
        fontsize=9,
    )
    ax.set_title("Heatmap  mean(|φ|)  continents × modalites", fontsize=12)

    for i in range(n_cont):
        for j in range(n_views):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "#333333")

    plt.tight_layout()
    _save(fig, out_dir / "heatmap_continent_modalite.png")


# ==============================================================================
#  3. Par prediction (correct / incorrect)
# ==============================================================================

def _significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def plot_par_prediction(df_pred, out_dir, df_raw=None):
    """
    df_pred : CSV stats_par_prediction.csv
    df_raw  : CSV shapley_france.csv (points individuels),
              necessaire pour les analyses option 1 et 2.
    """
    views   = _view_names(df_pred)
    groups  = df_pred.iloc[:, 0].tolist()   # "Correct" / "Incorrect"
    n_grp   = len(groups)
    n_views = len(views)
    phi_cols = [f"{v}_shapley" for v in views]

    # --- barplot groupe (mean |phi| brut) ---
    fig, ax = plt.subplots(figsize=(max(5, n_views * 1.6), 4))
    width   = 0.8 / n_grp
    x       = np.arange(n_views)
    for gi, grp in enumerate(groups):
        color  = PRED_COLORS.get(grp, "#888888")
        means  = [df_pred[f"|phi|_mean_{v}"].iloc[gi] for v in views]
        stds   = [df_pred[f"|phi|_std_{v}"].iloc[gi]  for v in views]
        offset = (gi - n_grp / 2 + 0.5) * width
        n      = int(df_pred["n_points"].iloc[gi])
        ax.bar(x + offset, means, width=width * 0.9,
               color=color, label=f"{grp} (n={n})",
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="#333333", capsize=3, linewidth=1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(views, fontsize=11)
    ax.set_ylabel("mean(|φ|)", fontsize=11)
    ax.set_title("Importance Shapley brute : correct vs incorrect", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out_dir / "hist_par_prediction.png")

    # ---------------------------------------------------------------
    # Les deux options suivantes necessitent les phi individuels
    # ---------------------------------------------------------------
    if df_raw is None or "correct" not in df_raw.columns:
        print("  INFO : shapley_france.csv absent ou sans colonne 'correct',"
              " options 1 et 2 ignorees.")
        return

    missing_phi = [c for c in phi_cols if c not in df_raw.columns]
    if missing_phi:
        print(f"  INFO : colonnes phi manquantes dans le CSV brut : {missing_phi}")
        return

    df_c = df_raw[df_raw["correct"] == True]
    df_i = df_raw[df_raw["correct"] == False]

    # ---------------------------------------------------------------
    # Option 1 — phi normalise par point : part relative de chaque modalite
    # ---------------------------------------------------------------
    def _norm_phi(df_sub):
        """Calcule |phi_i| / sum_j |phi_j| pour chaque point."""
        abs_phi = df_sub[phi_cols].abs()
        row_sum = abs_phi.sum(axis=1).replace(0, np.nan)
        return abs_phi.div(row_sum, axis=0)

    norm_c = _norm_phi(df_c)
    norm_i = _norm_phi(df_i)

    mean_norm_c = norm_c.mean()
    mean_norm_i = norm_i.mean()

    fig, ax = plt.subplots(figsize=(max(5, n_views * 1.6), 4))
    width = 0.8 / 2
    x     = np.arange(n_views)
    for gi, (grp, means, n) in enumerate([
        ("Correct",   mean_norm_c.values, len(df_c)),
        ("Incorrect", mean_norm_i.values, len(df_i)),
    ]):
        offset = (gi - 1 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=PRED_COLORS[grp], label=f"{grp} (n={n})",
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(views, fontsize=11)
    ax.set_ylabel("mean( |φ_i| / Σ|φ_j| )", fontsize=10)
    ax.set_title(
        "Part relative de chaque modalite (phi normalise par point)",
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out_dir / "hist_phi_normalise.png")

    # ---------------------------------------------------------------
    # Option 2 — Mann-Whitney U sur les distributions de phi_i
    # ---------------------------------------------------------------
    print("\n  Mann-Whitney U  (correct vs incorrect)")
    print(f"  {'Modalite':<14}  {'U-stat':>10}  {'p-value':>10}  sig")
    print("  " + "-" * 44)

    mw_results = []
    for v, col in zip(views, phi_cols):
        a = df_c[col].abs().dropna().values
        b = df_i[col].abs().dropna().values
        u_stat, p_val = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = _significance_stars(p_val)
        print(f"  {v:<14}  {u_stat:>10.1f}  {p_val:>10.4f}  {stars}")
        mw_results.append({"modalite": v, "U_stat": u_stat, "p_value": p_val,
                           "significance": stars,
                           "mean_|phi|_correct": a.mean(),
                           "mean_|phi|_incorrect": b.mean()})

    df_mw = pd.DataFrame(mw_results)
    mw_csv = out_dir / "mannwhitney_correct_vs_incorrect.csv"
    df_mw.to_csv(mw_csv, index=False)
    print(f"  -> {mw_csv}")

    # barplot p-values avec seuils
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(max(8, n_views * 2), 4),
                                      gridspec_kw={"wspace": 0.4})
    x = np.arange(n_views)

    # U-stat (plus grand = distributions plus separees)
    colors_u = [MODALITY_COLORS[i % len(MODALITY_COLORS)] for i in range(n_views)]
    ax_u.bar(x, df_mw["U_stat"], color=colors_u,
             edgecolor="white", linewidth=0.5, zorder=2)
    ax_u.set_xticks(x)
    ax_u.set_xticklabels(views, fontsize=10)
    ax_u.set_ylabel("U statistic", fontsize=10)
    ax_u.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_u.set_axisbelow(True)

    # p-values avec échelle linéaire
    p_values = df_mw["p_value"]
    colors_p = ["#d62728" if p < 0.05 else "#aec7e8"
                for p in df_mw["p_value"]]

    ax_p.bar(x, p_values, color=colors_p, edgecolor="white", linewidth=0.5, zorder=2)

    # Affiche la valeur uniquement si p est très proche de 0
    for xi, p_val in enumerate(p_values):
        if p_val < 1e-100:  # seuil pour considérer comme "nulle"
            ax_p.text(xi, max(p_values) * 0.95, f"{p_val:.2e}", 
                    ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    ax_p.set_xticks(x)
    ax_p.set_xticklabels(views, fontsize=10)
    ax_p.set_ylabel("p-value", fontsize=10)
    ax_p.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_p.set_axisbelow(True)

    fig.suptitle("Mann-Whitney U : correct vs incorrect", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir / "mannwhitney_correct_vs_incorrect.png")

    # ============================================================  
    # PLOT DES DISTRIBUTIONS DES SHAPLEY VALUES (correct vs incorrect)  
    # ============================================================  
    print("\n  Création des plots de distribution des valeurs de Shapley...")
        
    n_cols = min(3, n_views)    
    n_rows = (n_views + n_cols - 1) // n_cols   
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 *   n_rows))
    if n_views == 1:
        axes = np.array([axes]) 
    axes = axes.flatten()   
        
    for idx, (v, col) in enumerate(zip(views, phi_cols)):   
        ax = axes[idx]  
            
        # Récupérer les valeurs absolues    
        correct_vals = df_c[col].abs().dropna().values  
        incorrect_vals = df_i[col].abs().dropna().values    
            
        # Tracer les densités (KDE) 
        ax.hist(correct_vals, bins=50, density=True, alpha=0.5,     
                color='#2ca02c', label=f'Correct (n={len(correct_vals)  :,})', edgecolor='none')
        ax.hist(incorrect_vals, bins=50, density=True, alpha=0.5,   
                color='#d62728', label=f'Incorrect (n={len(incorrect_vals):,})', edgecolor='none')
            
        # Optionnel : ajouter les densités lissées (KDE)    
        try:    
            from scipy.stats import gaussian_kde    
            if len(correct_vals) > 1:   
                kde_correct = gaussian_kde(correct_vals)    
                x_grid = np.linspace(0, max(correct_vals.max(), incorrect_vals.max()), 200)
                ax.plot(x_grid, kde_correct(x_grid), color='#2ca02c',   linewidth=2, alpha=0.7)
            if len(incorrect_vals) > 1: 
                kde_incorrect = gaussian_kde(incorrect_vals)    
                x_grid = np.linspace(0, max(correct_vals.max(), incorrect_vals.max()), 200)
                ax.plot(x_grid, kde_incorrect(x_grid), color='#d62728'  , linewidth=2, alpha=0.7)
        except: 
            pass  # KDE optionnel, on ignore si erreur  
            
        # Ajouter les moyennes  
        mean_c = correct_vals.mean()    
        mean_i = incorrect_vals.mean()  
        ax.axvline(mean_c, color='#2ca02c', linestyle='--', linewidth=  1.5, alpha=0.8)
        ax.axvline(mean_i, color='#d62728', linestyle='--', linewidth=  1.5, alpha=0.8)
            
        # Ajouter les médianes  
        median_c = np.median(correct_vals)  
        median_i = np.median(incorrect_vals)    
        ax.axvline(median_c, color='#2ca02c', linestyle=':', linewidth  =1.5, alpha=0.6)
        ax.axvline(median_i, color='#d62728', linestyle=':', linewidth  =1.5, alpha=0.6)
            
        # Légende avec stats    
        ax.set_title(f'{v}\nU = {df_mw.iloc[idx]["U_stat"]:.0f}, p = {  df_mw.iloc[idx]["p_value"]:.2e}', 
                    fontsize=10)   
        ax.set_xlabel('|Shapley value|', fontsize=9)    
        ax.set_ylabel('Densité', fontsize=9)    
        ax.legend(loc='upper right', fontsize=8)    
        ax.grid(True, alpha=0.3, linestyle='--')    
            
        # Option : échelle log si les données sont très asymétriques    
        # ax.set_xscale('log')  
        
    # Cacher les axes inutilisés    
    for idx in range(n_views, len(axes)):   
        axes[idx].set_visible(False)    
        
    fig.suptitle('Distribution des |Shapley values| : Correct vs Incor  rect', fontsize=14, fontweight='bold')
    plt.tight_layout()  
    _save(fig, out_dir / "shapley_distributions_correct_vs_incorrect.png")
    plt.close(fig)  
        
    # Optionnel : boxplots pour chaque modalité 
    print("  Création des boxplots...") 
    fig, ax = plt.subplots(figsize=(max(8, n_views * 1.5), 6))  
        
    # Préparer les données pour les boxplots    
    box_data = []   
    box_labels = [] 
    box_colors = [] 
    for v, col in zip(views, phi_cols): 
        correct_vals = df_c[col].abs().dropna().values  
        incorrect_vals = df_i[col].abs().dropna().values    
        box_data.extend([correct_vals, incorrect_vals]) 
        box_labels.extend([f'{v}\nCorrect', f'{v}\nIncorrect']) 
        box_colors.extend(['#2ca02c', '#d62728'])   
        
    # Créer les boxplots    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,     
                    showfliers=False,  # optionnel : masquer les outli  ers si trop nombreux
                    medianprops=dict(linewidth=1.5, color='black')) 
        
    # Colorier les boîtes   
    for patch, color in zip(bp['boxes'], box_colors):   
        patch.set_facecolor(color)  
        patch.set_alpha(0.7)    
        
    ax.set_ylabel('|Shapley value|', fontsize=10)   
    ax.set_xlabel('Modalité', fontsize=10)  
    ax.set_title('Distribution des |Shapley values| (boxplots)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)    
    plt.xticks(rotation=45, ha='right', fontsize=9) 
    plt.tight_layout()  
    _save(fig, out_dir / "shapley_boxplots_correct_vs_incorrect.png")   
    plt.close(fig)  
        
    print(f"  -> {out_dir}/shapley_distributions_correct_vs_incorrect.png")
    print(f"  -> {out_dir}/shapley_boxplots_correct_vs_incorrect.png")  
        

# ==============================================================================
#  4. Croise continent x prediction
# ==============================================================================

def plot_continent_x_prediction(df_cx, out_dir):
    views = _view_names(df_cx)

    # La colonne "group" contient "Continent / Correct" ou "Continent / Incorrect"
    group_col = df_cx.columns[0]
    df_cx = df_cx.copy()
    df_cx[["continent", "prediction"]] = (
        df_cx[group_col].str.split(" / ", expand=True)
    )

    continents  = sorted(df_cx["continent"].unique())
    predictions = sorted(df_cx["prediction"].unique())   # ["Correct", "Incorrect"]

    for view in views:
        col = f"|phi|_mean_{view}"
        if col not in df_cx.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(continents) * 1.5), 4))
        x      = np.arange(len(continents))
        n_pred = len(predictions)
        width  = 0.8 / n_pred

        for pi, pred in enumerate(predictions):
            sub    = df_cx[df_cx["prediction"] == pred].set_index("continent")
            means  = [sub.loc[c, col] if c in sub.index else 0.0 for c in continents]
            stds   = [sub.loc[c, f"|phi|_std_{view}"] if c in sub.index else 0.0
                      for c in continents]
            offset = (pi - n_pred / 2 + 0.5) * width
            color  = PRED_COLORS.get(pred, "#888888")
            ax.bar(x + offset, means, width=width * 0.9,
                   color=color, label=pred,
                   edgecolor="white", linewidth=0.4, zorder=2)
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", color="#333333", capsize=2,
                        linewidth=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(continents, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("mean(|φ|)", fontsize=11)
        ax.set_title(
            f"Importance Shapley  [{view}]  par continent × prediction",
            fontsize=12,
        )
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        _save(fig, out_dir / f"hist_cx_{view}.png")

    # --- heatmap (continent x modalite) pour chaque prediction ---
    for pred in predictions:
        sub    = df_cx[df_cx["prediction"] == pred].set_index("continent")
        matrix = np.array([
            [sub.loc[c, f"|phi|_mean_{v}"] if c in sub.index else np.nan
             for v in views]
            for c in continents
        ])

        fig, ax = plt.subplots(figsize=(max(5, len(views) * 1.2),
                                        max(3, len(continents) * 0.65)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="mean(|φ|)", fraction=0.03, pad=0.04)
        ax.set_xticks(np.arange(len(views)))
        ax.set_xticklabels(views, fontsize=10)
        ax.set_yticks(np.arange(len(continents)))
        ax.set_yticklabels(continents, fontsize=9)
        ax.set_title(
            f"Heatmap  mean(|φ|)  —  {pred}",
            fontsize=12,
        )
        for i, c in enumerate(continents):
            for j, v in enumerate(views):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if val > np.nanmax(matrix) * 0.6 else "#333333")
        plt.tight_layout()
        _save(fig, out_dir / f"heatmap_cx_{pred.lower()}.png")


# ==============================================================================
#  Main
# ==============================================================================

def main(args):
    stats_dir = Path(args.stats_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(name):
        p = stats_dir / name
        if not p.exists():
            print(f"  AVERTISSEMENT : {p} introuvable, ignore.")
            return None
        return pd.read_csv(p)

    raw_csv   = stats_dir.parent / "shapley_france.csv"
    df_raw    = None
    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv)
        print(f"CSV brut charge : {raw_csv}  ({len(df_raw)} points)")
    else:
        print(f"AVERTISSEMENT : {raw_csv} introuvable, options 1 et 2 desactivees.")

    print(f"\nLecture des stats depuis : {stats_dir}")
    df_global = _load("stats_global.csv")
    df_cont   = _load("stats_par_continent.csv")
    df_pred   = _load("stats_par_prediction.csv")
    df_cx     = _load("stats_continent_x_prediction.csv")

    print(f"\nGeneration des plots dans : {out_dir}")

    if df_global is not None:
        print("\n[1/4] Global")
        plot_global(df_global, out_dir)

    if df_cont is not None:
        print("\n[2/4] Par continent")
        plot_par_continent(df_cont, out_dir)

    if df_pred is not None:
        print("\n[3/4] Par prediction")
        plot_par_prediction(df_pred, out_dir, df_raw=df_raw)

    if df_cx is not None:
        print("\n[4/4] Croise continent x prediction")
        plot_continent_x_prediction(df_cx, out_dir)


    print("\nTermine.")
    print(f"\nFichiers generes dans {out_dir}/:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Histogrammes des stats Shapley produits par infer_france.py"
    )
    p.add_argument(
        "--stats_dir", "-s",
        default="preds/france/stats",
        help="Dossier contenant les CSV de stats (defaut: preds/france/stats)",
    )
    p.add_argument(
        "--out_dir", "-o",
        default=None,
        help="Dossier de sortie des plots (defaut: --stats_dir/plots)",
    )
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path(args.stats_dir) / "plots")
    return args


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as e:
        import traceback
        print(f"\nERREUR : {e}\n{traceback.format_exc()}")
        sys.exit(1)