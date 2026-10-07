#!/usr/bin/env python
"""Step 13 - external validation with all model components frozen.

No retraining, refitting or recalibration on the external cohort:

* image probability = mean over all trained image models (seeds x folds) of a
  branch (ResNet: on the external tiles; FM: on the combined adapted features);
* fusion probability = mean over all stored fusion fold heads (seeds x folds);
* decision threshold = Youden threshold of the internal seed-aggregated
  out-of-fold predictions of the same model.

Reports per branch and head: AUROC of every individual fold head (mean +/- SD)
and of the ensemble (with CI), plus the threshold-based metrics, confusion
matrices and ROC / PR curves.

Prerequisites: steps 1, 5 and 6 with ``--cohort external``.
"""
import argparse

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from src.config import add_config_args, config_from_args, work_path
from src.data.clinical import FEATURE_NAMES, load_clinical
from src.data.datasets import (FeatureBagDataset, TileBagDataset, resnet_transforms,
                                  single_bag_collate)
from src.evaluation.metrics import metrics_with_ci
from src.fusion.late_fusion import load_heads
from src.models.attention_mil import FeatureAttentionMIL, ResNetAttentionMIL
from src.pipeline import (BRANCH_LABELS, features_dir, fusion_dir, load_patch_index,
                             patches_dir, patient_predictions)
from src.utils import get_device, get_logger

log = get_logger()


@torch.no_grad()
def resnet_probs(cfg, patients, device):
    rc = cfg["resnet_mil"]
    ds = TileBagDataset(patients, load_patch_index(cfg, "external"), patches_dir(cfg, "external"),
                        resnet_transforms(rc["input_size"], False))
    probs = []
    for seed in cfg["splits"]["seeds"]:
        for k in range(cfg["splits"]["n_folds"]):
            ck = torch.load(work_path(cfg, "resnet_mil", f"seed{seed}", f"fold{k}", "model_best.pt"),
                            map_location="cpu", weights_only=False)
            m = ResNetAttentionMIL(rc["backbone"], False, rc["attn_dim"], rc["attn_dropout"],
                                   rc["attn_temperature"], rc["chunk_size"])
            m.load_state_dict(ck["model"]); m.to(device).eval()
            p = {}
            for bag, _, pid, _ in DataLoader(ds, batch_size=1, collate_fn=single_bag_collate):
                p[pid] = float(torch.sigmoid(m(bag.to(device))[0]))
            probs.append(pd.Series(p))
    return pd.concat(probs, axis=1).mean(axis=1)


@torch.no_grad()
def fm_probs(cfg, patients, device):
    mc = cfg["fm_mil"]
    ds = FeatureBagDataset(patients, features_dir(cfg, mc["feature_set"], "external"))
    probs = []
    for seed in cfg["splits"]["seeds"]:
        for k in range(cfg["splits"]["n_folds"]):
            ck = torch.load(work_path(cfg, "fm_mil", f"seed{seed}", f"fold{k}", "model_best.pt"),
                            map_location="cpu", weights_only=False)
            m = FeatureAttentionMIL(ck["feat_dim"], mc["proj_dim"], mc["attn_dim"], mc["dropout"])
            m.load_state_dict(ck["model"]); m.to(device).eval()
            p = {pid: float(torch.sigmoid(m(f.to(device))[0]))
                 for f, _, pid in DataLoader(ds, batch_size=1, collate_fn=single_bag_collate)}
            probs.append(pd.Series(p))
    return pd.concat(probs, axis=1).mean(axis=1)


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--branches", nargs="+", default=["resnet", "fm"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    out = work_path(cfg, "external", "results")
    out.mkdir(parents=True, exist_ok=True)

    ext = load_clinical(cfg["paths"]["external_clinical_table"], cfg)
    log.info(f"external cohort: {len(ext)} labelled patients ({int(ext.label.sum())} responders)")
    rows, fold_rows, curves = [], [], {}
    for branch in args.branches:
        img = (resnet_probs if branch == "resnet" else fm_probs)(cfg, ext[["patient_id", "label"]], device)
        d = ext.assign(image_prob=ext["patient_id"].map(img)).dropna(subset=["image_prob"])
        d.drop(columns=[]).to_csv(out / f"{branch}_external_inputs.csv", index=False)
        y = d["label"].to_numpy()
        for head in cfg["external"]["heads"]:
            hs = []
            for seed in cfg["splits"]["seeds"]:
                hs += load_heads(fusion_dir(cfg, branch, seed) / "heads", f"fusion_{head}")
            per_head = np.stack([h.predict_proba(d) for h in hs])
            prob = per_head.mean(0)
            thr = float(patient_predictions(cfg, branch, f"fusion_{head}")["threshold"].iloc[0])
            head_aucs = [roc_auc_score(y, p) for p in per_head]
            r = metrics_with_ci(y, prob, thr, cfg)
            rows.append({"branch": branch, "head": head, "auroc_fold_heads_mean": np.mean(head_aucs),
                         "auroc_fold_heads_sd": np.std(head_aucs), **r})
            fold_rows += [{"branch": branch, "head": head, "model": i, "auroc": a}
                          for i, a in enumerate(head_aucs)]
            curves[(branch, head)] = (y, prob)
            pd.DataFrame({"patient_id": d["patient_id"], "label": y, "image_prob": d["image_prob"],
                          "fusion_prob": prob, "pred": (prob >= thr).astype(int)}
                         ).to_csv(out / f"{branch}_fusion_{head}_predictions.csv", index=False)
            log.info(f"{branch}/{head}: AUROC={r['auroc']:.3f} "
                     f"({r['auroc_lo']:.2f}-{r['auroc_hi']:.2f}) at thr={thr:.3f}")
    pd.DataFrame(rows).to_csv(out / "external_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "external_auroc_per_fold_head.csv", index=False)

    for branch in args.branches:
        for kind in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            for head in cfg["external"]["heads"]:
                y, p = curves[(branch, head)]
                if kind == "roc":
                    f, t, _ = roc_curve(y, p)
                    ax.plot(f, t, lw=1.8, label=f"{head} (AUROC {roc_auc_score(y, p):.2f})")
                else:
                    pr, rc, _ = precision_recall_curve(y, p)
                    ax.plot(rc, pr, lw=1.8, label=f"{head} (AP {average_precision_score(y, p):.2f})")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "--", c="grey", lw=0.8)
                ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
            else:
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"External - {BRANCH_LABELS[branch]} fusion", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out / f"{branch}_external_{kind}.png", dpi=300)
            fig.savefig(out / f"{branch}_external_{kind}.pdf")
            plt.close(fig)
    log.info(f"external validation results -> {out}")


if __name__ == "__main__":
    main()
