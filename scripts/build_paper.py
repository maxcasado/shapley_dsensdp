"""
scripts/build_paper.py
----------------------
One orchestrator that (re)builds every paper deliverable from the single source
(the v-dicts). It runs the pure post-processing steps that need no GPU, and
prints — in order — the upstream commands that DO need a GPU (v-dict
regeneration, the noise-ablation retraining, and the spatial re-inference) so the
full pipeline is reproducible from one place.

    python -m scripts.build_paper           # run the no-GPU post-processing chain
    python -m scripts.build_paper --plan    # only print the ordered command plan

Pipeline (source of truth -> everything):

  [UPSTREAM — GPU, run once, regenerates the source]
    1. v-dicts (format v2, predictions stored so the characteristic function is a
       post-processing flag):
         python -m scripts.vdict.build_vdicts -s config/com_average.yaml \\
             --fold_ids 0 1 2 3 4 --shapley --out_dir results/com/vdict
         python -m scripts.vdict.build_vdicts -s config/com_geo.yaml \\
             --fold_ids 0 1 2 3 4 --shapley --fixed_views geo --out_dir results/com_geo/vdict
    2. Perceptual Score (per fold):
         python -m scripts.attribution.perceptual_score -s config/com_average.yaml \\
             --fold_ids 0 1 2 3 4 --out_dir results/com --n_perm 10
         python -m scripts.attribution.perceptual_score -s config/com_geo.yaml \\
             --fold_ids 0 1 2 3 4 --fixed_views geo --out_dir results/com_geo --n_perm 10
    3. Noise ablation (the ONLY retraining; Fig 7 — 7 sigma x 5 folds x model):
         python -m scripts.noise.run_ablation
         python -m scripts.noise.aggregate
         python -m scripts.noise.plot
    4. Spatial Shapley (re-inference through the corrected routine; spatial maps):
         python -m scripts.spatial.explain -s config/com_average.yaml --fold_ids 0 1 2 3 4
         python -m scripts.spatial.explain -s config/com_geo.yaml --fold_ids 0 1 2 3 4 --fixed_views geo
         python -m scripts.spatial.compare_maps --csv_a preds_com/spatial/shapley_spatial.csv \\
             --csv_b preds_com_geo_fixed/spatial/shapley_spatial.csv \\
             --name_a com --name_b com_geo --out results/compare_shapley_maps --resolution 2.0

  [DOWNSTREAM — no GPU, run here]
    5. Attribution tables + figures (significance baked in) + paper LaTeX:
         python -m scripts.attribution.build_all --paper
    6. Dataset / performance tables:
         python paper/tables/build_table1_dataset_stats.py
         python paper/tables/build_table2_performance.py
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_module(mod, argv=None):
    print(f"\n$ python -m {mod} {' '.join(argv or [])}", flush=True)
    old = sys.argv
    sys.argv = [mod] + (argv or [])
    try:
        runpy.run_module(mod, run_name="__main__")
    finally:
        sys.argv = old


def _run_script(path, argv=None):
    print(f"\n$ python {path} {' '.join(argv or [])}", flush=True)
    old = sys.argv
    sys.argv = [str(path)] + (argv or [])
    try:
        runpy.run_path(str(ROOT / path), run_name="__main__")
    finally:
        sys.argv = old


def main(args):
    if args.plan:
        print(__doc__)
        return

    print("=== DOWNSTREAM post-processing (no GPU) ===")
    # 5. attribution tables + figures + LaTeX (delegates to paper/tables builders)
    _run_module("scripts.attribution.build_all", ["--paper"])

    # 6. dataset + performance tables (best-effort; skip if inputs missing)
    for script in ("paper/tables/build_table1_dataset_stats.py",
                   "paper/tables/build_table2_performance.py"):
        try:
            _run_script(script)
        except Exception as exc:  # inputs may be absent in a fresh checkout
            print(f"  (skipped {script}: {type(exc).__name__}: {exc})")

    print("\nDownstream done. For the GPU upstream steps, see: "
          "python -m scripts.build_paper --plan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild paper deliverables from the v-dicts.")
    ap.add_argument("--plan", action="store_true",
                    help="only print the ordered command plan (incl. GPU upstream steps)")
    main(ap.parse_args())
