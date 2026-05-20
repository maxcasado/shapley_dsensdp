"""
Lightweight debug helpers for inspecting Dataset_MultiView objects.
Not intended for production use; import only during development.
"""
import numpy as np


def inspect_data_structure(data_views, name: str = "") -> None:
    """
    Print a concise summary of a Dataset_MultiView object.

    Args:
        data_views: A Dataset_MultiView instance.
        name:       Optional label shown in the header.
    """
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"Samples : {len(data_views)}")
    print(f"Views   : {data_views.view_names}")

    labels = data_views.get_all_labels()
    print(f"Labels shape: {labels.shape}")
    if len(np.unique(labels)) <= 10:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Distribution: {dict(zip(unique, counts))}")

    print("Available attributes:")
    for attr in ["data", "views", "X", "features", "samples"]:
        if hasattr(data_views, attr):
            obj = getattr(data_views, attr)
            print(f"  - {attr}: {type(obj)}")
            if isinstance(obj, dict):
                print(f"    Keys: {list(obj.keys())}")
            elif hasattr(obj, "shape"):
                print(f"    Shape: {obj.shape}")

    print("First sample (per view):")
    for view_name in data_views.view_names:
        if hasattr(data_views, "data") and isinstance(data_views.data, dict) and view_name in data_views.data:
            sample = data_views.data[view_name][0]
            print(f"  {view_name}: {sample if not hasattr(sample, 'shape') else f'shape={sample.shape}'}")