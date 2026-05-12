import yaml
from eval import run_inference, shapley_bbox_per_point, _build_model, _load_test_data
import argparse
import numpy as np

# Charger config et checkpoint
with open("config/dsensdp_ex.yaml") as f:
    config = yaml.safe_load(f)

import torch
checkpoint = torch.load("/home/casado/DsensDp/DSensDp/res_out/weights/Dec_avg-SD-ignore-Plus/model_run0_fold0.pt", map_location="cpu")
ckpt_config = checkpoint.get("config", config)

# Reconstruire modele et test set
method  = _build_model(checkpoint, ckpt_config)
data_te = _load_test_data(checkpoint, ckpt_config)

# Recuperer les coordonnees (a adapter selon ton dataset)
coords = ...   # np.array (n_samples, 2) -> [lon, lat]

# Lancer
df = shapley_bbox_per_point(
    method      = method,
    data_te     = data_te,
    ckpt_config = ckpt_config,
    coords      = coords,
    lon_min=-3.8, lon_max=7.5,
    lat_min=43.1, lat_max=50.3,
    batch_size  = 32,
    out_csv     = "preds/shapley_bbox.csv",
)