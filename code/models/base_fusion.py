import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

torch.set_float32_matmul_precision('high')

from .core_fusion import _BaseViewsLightning
from .losses import get_loss_by_name
from .missing_utils import augment_random_missing
from .utils import stack_all, object_to_list, collate_all_list, detach_all, map_encoders

class MVFusionMissing(_BaseViewsLightning):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 weights_loss_variations: dict = {},
                 znorm_logits: bool = False,
                 **kwargs,
                 ):
        super(MVFusionMissing, self).__init__(**kwargs)
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")

        view_encoders = map_encoders(view_encoders, view_names=view_names)
        self.view_encoders = nn.ModuleDict(view_encoders)
        self.view_names = list(self.view_encoders.keys())
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
        self.N_views = len(self.view_encoders)
        self.weights_loss_variations = weights_loss_variations

        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.missing_as_aug = False

        # Normalization of logits before fusion (DSensDp+ eq. 3)
        # K=2  : L2 normalization (per-sample z-score is degenerate for K=2)
        # K>2  : per-sample z-score as in the paper
        self.znorm_logits = znorm_logits

    @staticmethod
    def _znorm(x: torch.Tensor) -> torch.Tensor:
        K = x.shape[-1]
        if K == 2:
            # z-score is degenerate at K=2 (collapses every output to ±1,
            # saturating the CE and killing gradients — verified empirically).
            # L2 preserves direction without crushing magnitude.
            return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        mu  = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-3)
        return (x - mu) / std

    def set_missing_info(self, aug_status, name:str="impute", where:str ="", value_fill=None, random_perc = 0,**kwargs):
        self.missing_as_aug = aug_status
        if name == "impute":
            where = "input" if where == "" else where
            value_fill = 0.0 if type(value_fill) == type(None) else value_fill
        elif name == "adapt":
            where = "feature" if where == "" else where
            value_fill = torch.nan if type(value_fill) == type(None) else value_fill
        elif name == "ignore":
            pass
        self.missing_method = {"name": name, "where": where, "value_fill": value_fill}
        self.random_perc = random_perc

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            inference_views: list = [],
            missing_method: dict = {},
            ) -> Dict[str, torch.Tensor]:
        inference_views = self.view_names if len(inference_views) == 0 else inference_views

        zs_views = {}
        for v_name in self.view_names:
            forward_f = True
            if v_name in inference_views and v_name in views:
                data_forward = views[v_name]
            else:
                if missing_method.get("where") == "input":
                    data_forward = torch.ones_like(views[v_name]) * missing_method["value_fill"]
                elif missing_method.get("where") == "feature":
                    forward_f = False
                    value_fill = torch.nan if missing_method["value_fill"] == "nan" else missing_method["value_fill"]
                    zs_views[v_name] = torch.ones(
                        self.view_encoders[v_name].get_output_size(), device=self.device
                    ).repeat(list(views.values())[0].shape[0], 1) * value_fill
                elif missing_method.get("name") == "ignore":
                    forward_f = False
                else:
                    raise Exception("Inference with few number of views (missing) but no missing method *where* was indicated")

            if forward_f:
                zs_views[v_name] = self.view_encoders[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate: bool = True,
            not_return_repre: bool = False,
            out_norm: str = "",
            inference_views: list = [],
            missing_method: dict = {},
            forward_from_representation: bool = False,
            ) -> Dict[str, torch.Tensor]:
        # encoders
        if forward_from_representation:
            out_zs_views = {"views:rep": views}
        else:
            out_zs_views = self.forward_encoders(
                views, inference_views=inference_views, missing_method=missing_method)

        # merge function
        if len(inference_views) != 0 and missing_method.get("name") == "ignore":
            views_data = [out_zs_views["views:rep"][v]
                          for v in self.view_names if v in inference_views]
        else:
            views_data = [out_zs_views["views:rep"][v] for v in self.view_names]

        # Normalization before fusion — DSensDp+ eq. 3
        # K=2: L2 norm; K>2: per-sample z-score
        if self.znorm_logits:
            views_data = [self._znorm(v) for v in views_data]

        views_available_ohv = (
            torch.ones(self.N_views) if len(inference_views) == 0
            else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        )
        out_z_e = self.fusion_module(views_data, views_available=views_available_ohv.bool())

        # prediction head
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return_dic = {"prediction": self.apply_norm_out(out_y, out_norm)}
        if not_return_repre:
            return_dic["last_layer"] = out_y
            return dict(**return_dic, **out_z_e)
        elif intermediate:
            return_dic["last_layer"] = out_y
            return dict(**return_dic, **out_zs_views, **out_z_e)
        else:
            return return_dic

    def prepare_batch(self, batch: dict, return_target=True) -> list:
        views_dict, views_target = batch["views"], batch["target"]
        if return_target:
            if type(self.criteria) == torch.nn.CrossEntropyLoss:
                views_target = views_target.squeeze().to(torch.long)
            else:
                views_target = views_target.to(torch.float32)
        else:
            views_target = None
        return views_dict, views_target

    def loss_batch(self, batch: dict) -> dict:
        views_dict, views_target = self.prepare_batch(batch)

        if self.missing_as_aug and self.training:
            views_targets = views_target
            out_dic_full = self(views_dict)  # full forward — encodes all sensors once

            full_pred  = out_dic_full["prediction"]
            # Raw per-sensor logits (pre-znorm) reused for both KD and missing dropout
            pred_views = out_dic_full["views:rep"]

            # Per-sample Bernoulli sensor dropout (paper eq. 5, α = random_perc)
            all_logits = torch.stack(
                [self._znorm(pred_views[v]) if self.znorm_logits else pred_views[v]
                 for v in self.view_names],
                dim=1,
            )  # (B, S, K) — z-normed, consistent with forward()
            B, S, K = all_logits.shape
            alpha = self.random_perc
            mask = (torch.rand(B, S, device=all_logits.device) > alpha).float()
            empty = mask.sum(dim=1) == 0
            if empty.any():
                idx = torch.randint(0, S, (int(empty.sum()),), device=all_logits.device)
                mask[empty, idx] = 1.0
            weights = mask / mask.sum(dim=1, keepdim=True)
            missing_pred = (all_logits * weights.unsqueeze(-1)).sum(dim=1)  # (B, K)

            loss_full = self.criteria(full_pred, views_target)
            return_dic = {
                "objective": (
                    self.weights_loss_variations.get("main", 1) * self.criteria(missing_pred, views_target) +
                    self.weights_loss_variations.get("full", 0) * loss_full
                ),
                "loss_full": loss_full,
            }

        else:
            views_targets = views_target
            out_dic    = self(views_dict)
            full_pred  = out_dic["prediction"]
            pred_views = out_dic["views:rep"]
            loss_full  = self.criteria(full_pred, views_targets)
            return_dic = {
                "objective": self.weights_loss_variations.get("main", 1) * loss_full,
                "loss_full": loss_full,
            }

        if len(self.weights_loss_variations) != 0 and self.training:
            temp = self.weights_loss_variations.get("cross_temp", 1)

            # full_pred is already the fused normalized average (ŷ_full, eq. 4)
            if type(self.criteria) == torch.nn.CrossEntropyLoss:
                full_pred_detach = nn.functional.log_softmax(
                    full_pred / temp, dim=-1).detach()
            elif type(self.criteria) == torch.nn.BCEWithLogitsLoss:
                full_pred_detach = nn.functional.sigmoid(
                    full_pred / temp).detach()

            for v in self.view_names:
                # ŷ_s = normalize(G_s(X_s))  — paper eq. 3
                # Apply same normalization as in forward() to get ŷ_s
                pred_v_normed = (
                    self._znorm(pred_views[v]) if self.znorm_logits
                    else pred_views[v]
                )

                # L(y, ŷ_s) — individual cross-entropy on normalized logits
                if self.weights_loss_variations.get("individual", 0) != 0:
                    return_dic[v] = self.criteria(pred_v_normed, views_target)
                    return_dic["objective"] += (
                        self.weights_loss_variations["individual"] *
                        return_dic[v] / len(self.view_names)
                    )

                # L_kd(ŷ_full, ŷ_s; τ) — KL divergence on normalized logits (eq. 8)
                if self.weights_loss_variations.get("individual_sd", 0) != 0:
                    if type(self.criteria) == torch.nn.BCEWithLogitsLoss:
                        return_dic[v + "-sd"] = (
                            (temp ** 2) *
                            nn.functional.binary_cross_entropy_with_logits(
                                pred_v_normed / temp, full_pred_detach)
                        )
                    else:
                        pred_v_log = nn.functional.log_softmax(
                            pred_v_normed / temp, dim=-1)
                        return_dic[v + "-sd"] = (
                            (temp ** 2) *
                            nn.functional.kl_div(
                                pred_v_log, full_pred_detach,
                                log_target=True, reduction="batchmean")
                        )
                    return_dic["objective"] += (
                        self.weights_loss_variations["individual_sd"] *
                        return_dic[v + "-sd"] / len(self.view_names)
                    )

        if self.missing_as_aug and self.training:
            return_dic["full"]    = self.criteria(out_dic_full["prediction"], views_targets)
            return_dic["missing"] = self.criteria(missing_pred, views_targets)

        if self.training and not hasattr(self, "_dbg_loss"):
            print("DBG loss terms:",
                  "miss", float(return_dic["missing"]),
                  "full", float(return_dic["full"]),
                  "objective", float(return_dic["objective"]), flush=True)
            print("DBG full_pred stats:", float(full_pred.mean()), float(full_pred.std()),
                  "min", float(full_pred.min()), "max", float(full_pred.max()), flush=True)
            print("DBG missing_pred stats:", float(missing_pred.mean()), float(missing_pred.std()),
                  "min", float(missing_pred.min()), "max", float(missing_pred.max()), flush=True)
            print("DBG view_names:", self.view_names, flush=True)
            self._dbg_loss = True

        return return_dic

    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate: bool = True,
            not_return_repre: bool = False,
            out_norm: str = "",
            device: str = "",
            args_forward: dict = {},
            perc_forward: float = 1,
            **kwargs
            ) -> dict:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        missing_forward = False
        self.eval()
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, _ = self.prepare_batch(batch)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)

                if perc_forward == 1:
                    missing_forward = True
                elif perc_forward != 0:
                    if np.random.rand() < perc_forward:
                        missing_forward = True

                if missing_forward:
                    outputs_ = self(views_dict, intermediate=intermediate,
                                   not_return_repre=not_return_repre,
                                   out_norm=out_norm, **args_forward)
                    missing_forward = False
                else:
                    outputs_ = self(views_dict, intermediate=intermediate,
                                   not_return_repre=not_return_repre,
                                   out_norm=out_norm)

                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_)
                else:
                    collate_all_list(outputs, outputs_)
        self.train()
        return stack_all(outputs)