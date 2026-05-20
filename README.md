# DSensDp — Multi-Sensor Earth Observation with Decision-Level Fusion

Fork of [fmenat/DSensDp](https://github.com/fmenat/DSensDp). This version adds Shapley-based modality analysis, per-point spatial explainability, and a refactored codebase organised by concern.

---

## Project structure

```
DSensDp/
├── train_multi.py      # Train a multi-sensor fusion model (DSensDp)
├── train_single.py     # Train a single-input fusion model
├── eval.py             # Evaluate a saved checkpoint on its test set
├── explain.py          # Per-point spatial Shapley maps over a bounding box
├── evaluate.py         # Aggregate metrics across runs/methods (inherited)
│
├── config/             # YAML experiment configs
├── data/               # Dataset preparation scripts
├── code/               # Inherited model and training code (do not modify)
│
├── utils/
│   ├── metrics.py      # compute_metrics, get_random_baseline_metrics
│   ├── checkpoint.py   # build_model, load_test_data, resolve_weights_path
│   ├── geo.py          # build_coords, assign_continents
│   └── debug.py        # inspect_data_structure
│
├── shap_analysis/
│   ├── shapley.py          # Aggregate Shapley (metric-based, used in train loop)
│   └── spatial_shapley.py  # Per-point Shapley + Shapley Interaction Index (SII)
│
└── visualize/
    ├── maps.py              # Spatial maps: Shapley, GT/pred, interactions
    └── plot_shapley_stats.py # Histograms and stats from explain.py CSV output
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

### Training

```bash
# Multi-sensor fusion (DSensDp)
python train_multi.py -s config/dsensdp_ex.yaml

# Single-input fusion
python train_single.py -s config/dsensdp_ex.yaml
```

At each fold, saves:
- `res_out/weights/<method>/model_run{r}_fold{k}.pt`
- `res_out/test_sets/<method>/test_indices_run{r}_fold{k}.npy`
- `res_out/test_sets/<method>/test_labels_run{r}_fold{k}.npy`
- `res_out/metadata/<data>/<method>/metadata_runs.csv`

---

### Evaluation

Loads a checkpoint and evaluates it on its saved test set.

```bash
# Metrics only
python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0

# With aggregate Shapley values (one value per view per metric)
python eval.py -s config/dsensdp_ex.yaml -r 0 -f 0 --shapley

# With explicit checkpoint path
python eval.py -s config/dsensdp_ex.yaml -w res_out/weights/Dec_avg-SD-ignore-Plus/model_run0_fold0.pt
```

Outputs:
- `preds/metrics_run{r}_fold{f}.csv`
- `preds/shapley_run{r}_fold{f}.csv` (if `--shapley`)

---

### Spatial explanation

Computes per-point Shapley values within a geographic bounding box and generates spatial maps.

```bash
# Single fold
python explain.py -s config/dsensdp_ex.yaml -f 0

# All folds combined (recommended — each point evaluated by the model that didn't train on it)
python explain.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4

# Restrict to a region (France)
python explain.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 \
    --lon_min -5.5 --lon_max 9.5 --lat_min 41.0 --lat_max 51.5 \
    --out_dir preds/france

# Export as GeoJSON
python explain.py -s config/dsensdp_ex.yaml --fold_ids 0 1 2 3 4 --geo_format geojson
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
python evaluate.py -s config/eval_ex.yaml
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