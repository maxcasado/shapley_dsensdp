import yaml
import argparse
import os
import sys
import time
import gc
import copy
import math
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from code.training.learn_pipeline import MultiFusion_train
from code.training.utils import assign_multifusion_name, output_name, assign_labels_weights
from code.datasets.views_structure import Dataset_MultiView
from code.datasets.utils import create_dataloader, load_structure


def _shapley_values(view_names, v_dict):
    n = len(view_names)
    shapley = {}
    for view in view_names:
        others = [v for v in view_names if v != view]
        phi = 0.0
        for size in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, size):
                S = frozenset(S_tuple)
                # Vérifier que les coalitions existent dans v_dict
                if S not in v_dict or (S | {view}) not in v_dict:
                    raise KeyError(f"Missing coalition: S={S} or SU{{view}}={S|{view}} not in v_dict")
                s = len(S)
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi += weight * (v_dict[S | {view}] - v_dict[S])
        shapley[view] = phi
    return shapley


def _compute_metrics(y_true, y_pred_proba, task_type):
    task = (task_type or "").lower()
    if task == "multilabel":
        y_pred = (y_pred_proba > 0.5).astype(int)
        print("Warning: using 0.5 threshold for multilabel classification, consider tuning this threshold for better performance.")
    else:
        y_pred = y_pred_proba.argmax(axis=-1)
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro",    zero_division=0),
    }

def main_run(config_file, just_return_first_model=False):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    runs_seed = config_file["experiment"].get("runs_seed", [])
    if len(runs_seed) == 0:
        runs = config_file["experiment"].get("runs", 1)
        runs_seed = [np.random.randint(50000) for _ in range(runs)]
    
    BS = config_file["training"]["batch_size"]
    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    if config_file.get("task_type", "").lower() == "classification":
        config_file["training"]["loss_args"]["name"] = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "multilabel":
        config_file["training"]["loss_args"]["name"] = "bce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    method_name = assign_multifusion_name(config_file["training"],config_file["method"], more_info_str=config_file.get("additional_method_name", ""))

    if "train" in data_name:
        print("train in data name AAAAA"*150)
        data_views_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_views_tr.load_stats(input_dir_folder, data_name)
        data_views_te = load_structure(input_dir_folder, data_name.replace("train", "test"), load_memory=config_file.get("load_memory", False))
        data_views_te.load_stats(input_dir_folder, data_name)
        try:
            data_views_va = load_structure(input_dir_folder, data_name.replace("train", "val"), load_memory=config_file.get("load_memory", False))
            data_views_va.load_stats(input_dir_folder, data_name)
        except:
            data_views_va = data_views_te
            print("No validation set found")
        kfolds = 1
    else:
        data_views_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_views_tr.load_stats(input_dir_folder, data_name)
        kfolds = config_file["experiment"].get("kfolds", 2)

    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    metadata_r = {"epoch_runs":[], "full_prediction_time":[], "training_time":[], "best_score":[],
                  **{k: [] for k in metric_keys}}
    for r,r_seed in enumerate(runs_seed):
        np.random.seed(r_seed)
        if kfolds != 1:
            indexs_ = data_views_tr.get_all_identifiers() 
            np.random.shuffle(indexs_)
            indexs_runs = np.array_split(indexs_, kfolds)
        for k in range(kfolds):
            print(f"******************************** Executing model on run {r+1} and kfold {k+1}")
            
            if kfolds != 1:
                data_views_tr.set_val_mask(indexs_runs[k])
                data_views_te = copy.deepcopy(data_views_tr)
                data_views_te.set_data_mode(train=False)
                data_views_va = data_views_te
            data_views_tr.set_additional_info(**config_file["experiment"].get("preprocess"))
            data_views_va.set_additional_info(**config_file["experiment"].get("preprocess"))
            data_views_te.set_additional_info(**config_file["experiment"].get("preprocess"))
            print(f"Training with {len(data_views_tr)} samples and validating on {len(data_views_te)}")

            if config_file.get("task_type", "").lower() in ["classification", "multilabel"]:
                assign_labels_weights(config_file, data_views_tr)

            start_aux = time.time()
            method, trainer = MultiFusion_train(data_views_tr, val_data=data_views_va,run_id=r,fold_id=k,method_name=method_name, **config_file)
            if just_return_first_model:
                return method
            metadata_r["training_time"].append(time.time()-start_aux)
            metadata_r["epoch_runs"].append(trainer.callbacks[0].stopped_epoch)
            metadata_r["best_score"].append(trainer.callbacks[0].best_score.cpu())
            print("Training done")
                        
            #store original predictions to calculate error-difference
            pred_time_Start = time.time()
            outputs_te = method.transform(create_dataloader(data_views_te, batch_size=BS, train=False), out_norm=output_name(config_file["task_type"]), not_return_repre=True)
            metadata_r["full_prediction_time"].append(time.time()-pred_time_Start)
            data_save_te = Dataset_MultiView([outputs_te["prediction"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{method_name}", ind_views=True, xarray=False)

            y_true = data_views_te.get_all_labels()
            metrics = _compute_metrics(y_true, outputs_te["prediction"], config_file.get("task_type"))
            for k_m, v_m in metrics.items():
                metadata_r[k_m].append(v_m)

            if config_file.get("compute_shapley", False):
                view_names = config_file["experiment"]["preprocess"]["view_names"]
                method.set_missing_info(None, **config_file["training"].get("missing_method", {}))
                baseline_metrics = { # under the assumption of random predictions and balanced classes
                    "accuracy" : 0.5,
                    "f1_macro" : 0.5,
                    "f1_weighted" : 0.5,
                    "precision_macro" : 0.5,
                    "recall_macro" : 0.5,
                }
                v_dict = {frozenset(): baseline_metrics} #empty set as baseline
                v_dict[frozenset(view_names)] = metrics #full set
                full_str = "_".join(sorted(view_names))
                for k_m, v_m in metrics.items():
                    col = f"shapley_{full_str}_{k_m}"
                    if col not in metadata_r:
                        metadata_r[col] = []
                    metadata_r[col].append(v_m)

                # run all 14 strict subsets (size 1 to n-1)
                for size in range(1, len(view_names)):
                    for subset_tuple in itertools.combinations(view_names, size):
                        subset_list = list(subset_tuple)
                        combi_str = "_".join(sorted(subset_list))
                        args_fwd = {"inference_views": subset_list, "missing_method": method.missing_method}
                        out_sub = method.transform(
                            create_dataloader(data_views_te, batch_size=BS, train=False),
                            out_norm=output_name(config_file["task_type"]),
                            args_forward=args_fwd, perc_forward=1.0, not_return_repre=True
                        )
                        sub_metrics = _compute_metrics(y_true, out_sub["prediction"], config_file.get("task_type"))
                        v_dict[frozenset(subset_list)] = sub_metrics
                        for k_m, v_m in sub_metrics.items():
                            col = f"shapley_{combi_str}_{k_m}"
                            if col not in metadata_r:
                                metadata_r[col] = []
                            metadata_r[col].append(v_m)

                # compute and store Shapley values per metric
                for metric_name in metric_keys:
                    v_scalar = {s: v_dict[s][metric_name] for s in v_dict}
                    sv = _shapley_values(view_names, v_scalar)
                    for view, phi in sv.items():
                        col = f"shapley_value_{view}_{metric_name}"
                        if col not in metadata_r:
                            metadata_r[col] = []
                        metadata_r[col].append(phi)

            for v in outputs_te:
                if ":prediction" in v and "aggregated" in outputs_te[v]:
                    print(f"Warning, overwritting based on auxiliar predictions from {v}")
                    data_save_te = Dataset_MultiView([outputs_te[v]["aggregated"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/pred_aux/{data_name}/{method_name}", ind_views=True, xarray=False) #overwrite previous pred

            if config_file.get("args_forward") and config_file["args_forward"].get("list_testing_views"): 
                for (test_views, percentages) in config_file["args_forward"].get("list_testing_views"):
                    for perc_missing in percentages:
                        print("Inference with the following views ",test_views, " and percentage missing ",perc_missing)
                        if "missing_method" in config_file["args_forward"]:
                            args_forward = {"inference_views":test_views, **{k:v for k,v in config_file["args_forward"].items() if k!= "list_testing_views"}}
                        else:
                            method.set_missing_info(None, **config_file["training"].get("missing_method", {}))
                            args_forward = {"inference_views":test_views, "missing_method": method.missing_method}
                            
                        pred_time_Start = time.time()
                        outputs_te = method.transform(create_dataloader(data_views_te, batch_size=config_file['args_forward'].get("batch_size", BS), train=False), out_norm=output_name(config_file["task_type"]), args_forward=args_forward, perc_forward=perc_missing, not_return_repre=True)
                        views_perc_key = f"{'_'.join(test_views)}_{perc_missing*100:.0f}"
                        if f"{views_perc_key}_prediction_time" not in metadata_r:
                            metadata_r[f"{views_perc_key}_prediction_time"] = []
                        metadata_r[f"{views_perc_key}_prediction_time"].append(time.time()-pred_time_Start)

                        fwd_metrics = _compute_metrics(y_true, outputs_te["prediction"], config_file.get("task_type"))
                        for k_m, v_m in fwd_metrics.items():
                            fwd_key = f"{views_perc_key}_{k_m}"
                            if fwd_key not in metadata_r:
                                metadata_r[fwd_key] = []
                            metadata_r[fwd_key].append(v_m)
                        print(f"  Metrics {views_perc_key}: " + "  ".join(f"{k_m}={v_m:.4f}" for k_m, v_m in fwd_metrics.items()))

                        aux_name = assign_multifusion_name(config_file["training"],config_file["method"], forward_views=test_views, perc=perc_missing,
                                                        more_info_str=config_file.get("additional_method_name", ""))
                        ## EXTRA -- PREDICTIONS ##
                        data_save_te = Dataset_MultiView([outputs_te["prediction"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                        data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_views=True, xarray=False)

                        for v in outputs_te:
                            if ":prediction" in v and "aggregated" in outputs_te[v]: #in case of multi-loss they are used as auxiliar for prediction
                                print(f"Warning, overwritting based on auxiliar predictions from {v}")
                                data_save_te = Dataset_MultiView([outputs_te[v]["aggregated"]], identifiers=data_views_te.get_all_identifiers(), view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                                data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_views=True, xarray=False) #overwrite previous pred
                        print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {aux_name} finished...")
            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {method_name} finished...")
    Path(f"{output_dir_folder}/metadata/{data_name}/{method_name}").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv")
    print("Epochs for %s runs on average for %.2f epochs +- %.3f"%(method_name,np.mean(metadata_r["epoch_runs"]),np.std(metadata_r["epoch_runs"])))
    print(f"Finished whole execution of {len(runs_seed)} runs in {time.time()-start_time:.2f} secs")
    return metadata_r


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)
    
    main_run(config_file)
