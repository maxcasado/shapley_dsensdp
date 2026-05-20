"""
utils/geo.py
------------
Geographic utilities: coordinate alignment and continent assignment.
"""

import numpy as np
import xarray as xr


# ── Coordinate loading ────────────────────────────────────────────────────────

def build_coords(data_te, nc_path: str) -> np.ndarray:
    """
    Align dataset identifiers with geographic coordinates from a NetCDF file.

    Args:
        data_te:  Dataset_MultiView instance (provides get_all_identifiers()).
        nc_path:  Path to the .nc file containing 'identifier' and 'coords'
                  variables. 'coords' must have shape (n_total, 2) -> [lon, lat].

    Returns:
        np.ndarray of shape (n_samples, 2) with [longitude, latitude] per sample,
        aligned with data_te sample order.

    Raises:
        KeyError: if any identifier in data_te is absent from the .nc file.
    """
    print(f"Loading coordinates from {nc_path}...", flush=True)
    ds        = xr.open_dataset(nc_path)
    ids_nc    = ds["identifier"].values
    coords_nc = ds["coords"].values          # (n_total, 2) -> [lon, lat]
    id2idx    = {id_: i for i, id_ in enumerate(ids_nc)}

    ids_te  = data_te.get_all_identifiers()
    missing = [id_ for id_ in ids_te if id_ not in id2idx]
    if missing:
        raise KeyError(f"{len(missing)} identifiers missing from .nc: {missing[:5]}")

    coords = np.array([coords_nc[id2idx[id_]] for id_ in ids_te])
    print(f"  Coordinates aligned: {coords.shape}", flush=True)
    return coords


# ── Continent assignment ──────────────────────────────────────────────────────

# Bounding boxes ordered from most specific to broadest.
# Used as fallback when geopandas is unavailable.
_CONTINENT_BBOX = [
    ("Antarctique",     -180, -90,  180, -60),
    ("Oceanie",          110, -50,  180,  10),
    ("Amerique du Nord", -170,  15,  -50,  85),
    ("Amerique du Sud",   -82, -57,  -34,  15),
    ("Europe",            -25,  35,   45,  72),
    ("Afrique",           -20, -35,   55,  38),
    ("Asie",               25,  -5,  180,  80),
]


def assign_continents(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Assign a continent name to each (lon, lat) point.

    Tries geopandas + naturalearth for exact point-in-polygon matching.
    Falls back to approximate bounding boxes if geopandas is unavailable.

    Args:
        lons: Array of longitudes.
        lats: Array of latitudes.

    Returns:
        Array of continent name strings, "Inconnu" for unmatched points.
    """
    # ── Option 1: exact point-in-polygon via geopandas ────────────────────────
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from pathlib import Path as _Path

        _NE_SHP = _Path.home() / ".cache" / "naturalearth" / "ne_110m_admin_0_countries.shp"
        if not _NE_SHP.exists():
            raise FileNotFoundError(
                f"{_NE_SHP} not found. Download with:\n"
                "  python -c \"import urllib.request,zipfile,io,pathlib; "
                "data=urllib.request.urlopen('https://naciscdn.org/naturalearth/110m/cultural/"
                "ne_110m_admin_0_countries.zip').read(); "
                "dest=pathlib.Path.home()/'.cache'/'naturalearth'; dest.mkdir(parents=True,exist_ok=True); "
                "zipfile.ZipFile(io.BytesIO(data)).extractall(dest)\""
            )

        world  = gpd.read_file(_NE_SHP)[["CONTINENT", "geometry"]].rename(columns={"CONTINENT": "continent"})
        world  = world[world["continent"].notna()].copy()
        pts    = gpd.GeoDataFrame(
            {"geometry": [Point(lo, la) for lo, la in zip(lons, lats)]},
            crs="EPSG:4326",
        )
        joined     = gpd.sjoin(pts, world, how="left", predicate="within")
        continents = joined["continent"].fillna("Inconnu").values
        print("  Continents assigned via naturalearth (point-in-polygon).", flush=True)
        return continents

    except Exception as e:
        print(f"  geopandas unavailable ({e}), falling back to bounding boxes.", flush=True)

    # ── Option 2: approximate bounding boxes ──────────────────────────────────
    continents = np.full(len(lons), "Inconnu", dtype=object)
    for name, lon_min, lat_min, lon_max, lat_max in _CONTINENT_BBOX:
        mask = (
            (lons >= lon_min) & (lons <= lon_max) &
            (lats >= lat_min) & (lats <= lat_max) &
            (continents == "Inconnu")
        )
        continents[mask] = name
    return continents