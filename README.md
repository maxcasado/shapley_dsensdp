# DSensDp — Multi-Sensor Earth Observation with Decision-Level Fusion

Fork of [fmenat/DSensDp](https://github.com/fmenat/DSensDp). This version adds Shapley-based modality analysis, per-point spatial explainability, and a refactored codebase organised by concern.

---

## Architecture — one source, everything derived

The v-dicts are the **single source of truth**: the only artefact that needs model
inference. Everything else (tables, figures) is pure post-processing over them, with
the cross-fold **significance layer** (mean ± std, `|SNR|>2` gate, paired Wilcoxon)
baked in so nothing is ever reported as a bare mean. Because the v-dicts store the
per-coalition predictions (format v2), the **characteristic function** (`f1_weighted`,
`mcc`, `balanced_accuracy`, …) is a post-processing flag — no re-inference to change it.

```
scripts/vdict/build_vdicts.py  ──►  results/<model>/vdict/v_dict_fold{0..4}.pkl   [SOURCE]
                                                │
                                     scripts/attribution/build_all.py              [1 driver]
                                                ▼
                          results/attribution/**  +  results/figures/**  +  paper/tables/**
```

### Repository layout

```
DSensDp/
├── scripts/                    # thin CLIs, grouped by function — run as `python -m scripts.<group>.<name>`
│   ├── train/                  #   train_single · train_multi · train_geo_transfer
│   ├── vdict/                  #   build_vdicts (SOURCE OF TRUTH) · subset_metrics
│   ├── attribution/            #   build_all (tables+figs, significance) · perceptual_score
│   ├── spatial/                #   explain · spatial_stats · point_spatial_stats · compare_stats · compare_maps
│   ├── noise/                  #   run_ablation · aggregate · plot · injection
│   ├── representation/         #   branch_cka  (CKA, invariant to the Shapley fix)
│   ├── legacy/                 #   evaluate · script_koppen_shapley · script_shapley_viz · run_comparison_analysis
│   └── build_paper.py          #   orchestrator: `--plan` prints the full ordered pipeline
│
├── shap_analysis/              # post-processing library (the heavy logic)
│   ├── shapley.py              #   cooperative-game formula · run_subset_inference (v2) · scalar_vdict (v1/v2 reader)
│   ├── characteristic.py       #   CHAR_FUNCS registry (metric = flag) · analytical baselines v(∅)
│   ├── stats.py                #   significance: SNR_THRESHOLD, agg, agg_frame, fmt, wilcoxon_paired
│   ├── comparison.py           #   per-fold Shapley tables · redundancy · correlation
│   ├── tables.py               #   attribution / coalition-value / geo-comparison builders
│   ├── figures.py              #   figures with cross-fold error bars (shared palette)
│   └── spatial_shapley.py      #   per-point Shapley + SII (characteristic = raw logit)
│
├── paper/tables/               # LaTeX/CSV formatters, read results/attribution/**
├── config/  data/  code/       # configs · data prep · inherited model code (do not modify)
├── utils/                      # metrics · checkpoint · geo · debug
└── visualize/                  # maps · plot_shapley_stats
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.10, PyTorch 2.x, PyTorch Lightning 1.9.

Optional dependencies for geospatial export and continent assignment:
```bash
pip install geopandas shapely geodatasets
```

---

## Configuration

All scripts take a YAML config file via `-s`. See `config/dsensdp_ex.yaml` for a complete example.

Key fields:

```yaml
input_dir_folder:  path/to/data
output_dir_folder: res_out
data_name:         cropharvest_binary

task_type: classification   # classification | multilabel

save_weights:  true         # save model weights at each fold
save_test_set: true         # save test indices/labels at each fold

experiment:
  kfolds: 5
  runs:   1

training:
  batch_size: 128
  ...
```

---

## Usage

> Run everything from the repo root as a module (`python -m scripts.<group>.<name>`),
> so the `code.` / `shap_analysis.` / `utils.` imports resolve.

### Training

```bash
# Multi-sensor fusion (DSensDp)
python -m scripts.train.train_multi -s config/dsensdp_ex.yaml

# Single-input fusion
python -m scripts.train.train_single -s config/dsensdp_ex.yaml
```

At each fold, saves:
- `res_out/weights/<method>/model_run{r}_fold{k}.pt`
- `res_out/test_sets/<method>/test_indices_run{r}_fold{k}.npy`
- `res_out/test_sets/<method>/test_labels_run{r}_fold{k}.npy`
- `res_out/metadata/<data>/<method>/metadata_runs.csv`

---

### Build the v-dicts (source of truth) & the attribution deliverables

```bash
# 1) Regenerate the v-dicts (format v2, predictions stored) — needs checkpoints + GPU
python -m scripts.vdict.build_vdicts -s config/com_average.yaml \
    --fold_ids 0 1 2 3 4 --shapley --out_dir results/com/vdict
python -m scripts.vdict.build_vdicts -s config/com_geo.yaml \
    --fold_ids 0 1 2 3 4 --shapley --fixed_views geo --out_dir results/com_geo/vdict

# 2) Everything else — tables (mean ± std + SNR gate), figures, LaTeX — no GPU
python -m scripts.attribution.build_all --paper
#   --metric mcc | balanced_accuracy    # swap the characteristic function (needs v2 v-dicts)

# Full ordered pipeline (incl. the GPU steps: noise ablation, spatial maps):
python -m scripts.build_paper --plan
```

`build_all` also works on the legacy v1 v-dicts (`results/<model>/eval*/`) — it just
can't swap the metric there. Plain per-fold metrics/Shapley for one checkpoint:

```bash
python -m scripts.vdict.build_vdicts -s config/dsensdp_ex.yaml -r 0 -f 0 --shapley
```

---

### Spatial explanation

Computes per-point Shapley values within a geographic bounding box and generates spatial maps.

```bash
# Single fold
python -m scripts.spatial.explain -s config/dsensdp_ex.yaml -f 0

# All folds combined (recommended — each point evaluated by the model that didn't train on it)
python -m scripts.spatial.explain -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4

# Restrict to a region (France)
python -m scripts.spatial.explain -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 \
    --lon_min -5.5 --lon_max 9.5 --lat_min 41.0 --lat_max 51.5 \
    --out_dir preds/france

# Export as GeoJSON
python -m scripts.spatial.explain -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 --geo_format geojson
```

Output directory structure (`preds/spatial/` by default):

```
preds/spatial/
├── shapley_spatial.csv       # one row per point: lon, lat, phi per view, SII per pair
├── shapley_spatial.geojson   # (if --geo_format geojson)
├── maps_shapley/             # one map per modality + combined map
├── maps_gt_pred/             # ground truth, predicted score, correct/incorrect
├── maps_interactions/        # SII order-2 maps, interaction graph, interaction matrix
└── stats/
    ├── stats_global.csv
    ├── stats_par_continent.csv
    ├── stats_par_prediction.csv
    └── map_continents.png
```

#### Plot statistics from an existing CSV

```bash
python visualize/plot_shapley_stats.py \
    --stats_dir preds/spatial/stats \
    --out_dir   preds/spatial/stats/plots
```

---

### Aggregate evaluation across methods

Reads saved predictions from `res_out/pred/` and computes metrics across all runs and methods.

```bash
python -m scripts.legacy.evaluate -s config/eval_ex.yaml
```

---

## Outputs overview

| Directory | Content |
|---|---|
| `res_out/weights/` | Model checkpoints (`.pt`) per run and fold |
| `res_out/test_sets/` | Test indices, labels and identifiers per fold |
| `res_out/pred/` | Raw predictions saved during training |
| `res_out/metadata/` | Training metadata CSV (epochs, metrics, timing) |
| `preds/` | Evaluation and explanation outputs |