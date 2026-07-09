import os
import torch
import xarray as xray
import numpy as np
from typing import List, Union, Dict

from .views_structure import Dataset_MultiView


def create_dataloader(dataset_pytorch, batch_size=32, train=True, parallel_processes=2, **args_loader):
    cpu_count = len(os.sched_getaffinity(0))
    return torch.utils.data.DataLoader(
        dataset_pytorch,
        batch_size=batch_size,
        num_workers=int(cpu_count/parallel_processes),
        shuffle=train,
        pin_memory=True,
        drop_last=False, #train,
        **args_loader
    )


def xray_to_dataviews(xray_data: xray.Dataset, views_used: List[str]=[]) -> Dataset_MultiView:
    all_possible_index = xray_data.coords["identifier"].values
    
    dataviews = Dataset_MultiView()    
    dataviews.train_identifiers = list(all_possible_index[xray_data["train_mask"].values])
    dataviews.val_identifiers = list(all_possible_index[~xray_data["train_mask"].values])
    dataviews.target_names = xray_data.attrs["target_names"]
    dataviews.view_names = xray_data.attrs["view_names"]

    dataviews.identifiers_target = dict(zip(all_possible_index, xray_data["target"]))
    # ── Positional encoding (geo) — computed from coords on the fly ──────────
    import xarray as xr
    lon_rad = xray_data["coords"].values[:, 0] * np.pi / 180
    lat_rad = xray_data["coords"].values[:, 1] * np.pi / 180
    # Fourier features — 64 frequencies log-spaced (Vaswani et al. 2017)
    # omega_k = 10000^(-2k/128), k = 0..63 -> 128 features total
    d = 128
    k = np.arange(d // 4, dtype=np.float32)          # 0..31
    omega = 1.0 / (10000.0 ** (2 * k / d))            # (32,)
    geo_arr = np.concatenate([
        np.sin(np.outer(lat_rad, omega)),              # (N, 32)
        np.cos(np.outer(lat_rad, omega)),              # (N, 32)
        np.sin(np.outer(lon_rad, omega)),              # (N, 32)
        np.cos(np.outer(lon_rad, omega)),              # (N, 32)
    ], axis=1).astype(np.float32)                      # (N, 128)
    geo_xr = xr.DataArray(
        geo_arr,
        dims=["identifier", "geo_features"],
        coords={"identifier": all_possible_index}
    )

    # Add geo to view_names so get_view_shapes can find it
    views_requested = views_used if len(views_used) > 0 else dataviews.view_names
    if "geo" in views_requested and "geo" not in dataviews.view_names:
        dataviews.view_names.append("geo")

    # Expand composite view names (e.g. S2_S2VI -> S2, S2VI)
    expanded_views = []
    for v in (dataviews.view_names if len(views_used) == 0 else views_used):
        if v == "geo":
            expanded_views.append("geo")
        elif "_" in v:
            expanded_views.extend(v.split("_"))
        else:
            expanded_views.append(v)

    for view_n in expanded_views:
        dataviews.views_data_ident2indx[view_n] = dict(zip(all_possible_index, np.arange(len(all_possible_index))))
        dataviews.views_data[view_n] = geo_xr if view_n == "geo" else xray_data[view_n]
    
    return dataviews

def load_structure(path: str, file_name: str, load_memory: bool = False, views_used: List[str]=[]) -> Dataset_MultiView:
    data  = xray.open_dataset(f"{path}/{file_name}.nc", engine="h5netcdf")
    if load_memory:
        data = data.load()
    dataset_structure =  xray_to_dataviews(data, views_used=views_used)
    data.close()
    return dataset_structure
