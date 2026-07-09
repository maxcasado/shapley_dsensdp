"""
noise_injection.py
-------------------
Gaussian noise injection wrapper for the S2_S2VI modality.

Monkey-patches `Dataset_MultiView.normalize_w_stats`
(code/datasets/views_structure.py) so that i.i.d. N(0, sigma^2) noise is added
to the raw "S2" and "S2VI" views (the two raw views concatenated into the
composite "S2_S2VI" modality, see code/datasets/utils.py:xray_to_dataviews)
right after z-score normalization. All other modalities (S1, weather, DEM)
are untouched.

No existing file is modified: the patch is installed once at import time and
is applied at the dataloader call site (`__getitem__` -> `normalize_w_stats`),
so it transparently affects both training and inference (full-set forward
passes, Shapley coalition forward passes, and Perceptual Score permutation
forward passes alike), as long as this module has been imported first.

sigma == 0.0 is a no-op (identical behavior to the unpatched method).
"""
import numpy as np

from code.datasets.views_structure import Dataset_MultiView

NOISY_RAW_VIEWS = ("S2", "S2VI")  # raw views composing the S2_S2VI modality

_original_normalize_w_stats = Dataset_MultiView.normalize_w_stats
_state = {"sigma": 0.0, "rng": np.random.default_rng(0)}


def _normalize_w_stats_with_noise(self, data, view_name):
    out = _original_normalize_w_stats(self, data, view_name)
    if _state["sigma"] > 0 and view_name in NOISY_RAW_VIEWS:
        noise = _state["rng"].normal(0.0, _state["sigma"], size=out.shape)
        out = out + noise.astype(out.dtype, copy=False)
    return out


Dataset_MultiView.normalize_w_stats = _normalize_w_stats_with_noise


def set_noise_sigma(sigma: float, seed: int = 0):
    """Activate sigma>0 i.i.d. Gaussian noise on S2_S2VI for every subsequent
    __getitem__ call (train and inference) until this is called again."""
    _state["sigma"] = float(sigma)
    _state["rng"] = np.random.default_rng(seed)
