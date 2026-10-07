#!/usr/bin/env python
"""Step 7 - foundation-model branch: gated-attention MIL head on frozen
(DINO-adapted) CONCH+UNI tile embeddings. Image-only - no clinical input.

Per seed and outer fold: trained on ``train`` patients, checkpoint selected on
the inner ``val`` patients, applied to the held-out ``test`` fold. Loss: focal
loss with class weighting and label smoothing; AdamW with cosine schedule;
class-balanced sampling; bag augmentation (tile shuffling + feature noise).

    python scripts/07_train_fm_mil.py --config cfg.yaml [--seed 42 --fold 0]
"""
import argparse

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import add_config_args, config_from_args, dump_config, work_path
from src.data.datasets import FeatureBagDataset, balanced_sampler, single_bag_collate
from src.data.splits import fold_patients
from src.models.attention_mil import FeatureAttentionMIL, FocalLoss
from src.pipeline import features_dir, load_split, oof_path
from src.utils import get_device, get_logger, seed_everything

log = get_logger()


@torch.no_grad()
def predict(model, ds, device, keep_attention=False):
    model.eval()
    rows, att = [], []
    for feats, label, pid in DataLoader(ds, batch_size=1, collate_fn=single_bag_collate):
        logit, a = model(feats.to(device))
        rows.append((pid, int(label), float(torch.sigmoid(logit))))
        if keep_attention:
            att.append((pid, a.cpu().numpy()))
    return pd.DataFrame(rows, columns=["patient_id", "label", "image_prob"]), att


def auroc(df):
    return roc_auc_score(df["label"], df["image_prob"]) if df["label"].nunique() == 2 else float("nan")


def train_fold(cfg, seed, fold, split, fdir, device):
    mc = cfg["fm_mil"]
    out = work_path(cfg, "fm_mil", f"seed{seed}", f"fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed + fold)
    tr = FeatureBagDataset(fold_patients(split, fold, "train"), fdir, True, mc["feature_noise_std"])
    va = FeatureBagDataset(fold_patients(split, fold, "val"), fdir)
    te = FeatureBagDataset(fold_patients(split, fold, "test"), fdir)
    feat_dim = tr[0][0].shape[1]
    model = FeatureAttentionMIL(feat_dim, mc["proj_dim"], mc["attn_dim"], mc["dropout"]).to(device)
    n_pos = sum(tr.labels()); n_neg = len(tr) - n_pos
    pos_weight = n_pos / max(n_neg, 1)
    crit = FocalLoss(mc["focal_gamma"], alpha=pos_weight / (1 + pos_weight))
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mc["epochs"], eta_min=mc["min_lr"])
    loader = DataLoader(tr, batch_size=1, sampler=balanced_sampler(tr.labels()),
                        collate_fn=single_bag_collate)
    ls = mc["label_smoothing"]
    best, best_epoch, stale = -1.0, 0, 0
    log.info(f"[seed {seed} fold {fold}] train={len(tr)} val={len(va)} test={len(te)} dim={feat_dim}")
    for epoch in range(1, mc["epochs"] + 1):
        model.train()
        losses = []
        for feats, label, _ in loader:
            target = torch.tensor(label * (1 - ls) + 0.5 * ls, device=device)
            logit, _ = model(feats.to(device))
            loss = crit(logit, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), mc["grad_clip"])
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_auc = auroc(predict(model, va, device)[0])
        improved = val_auc == val_auc and val_auc > best
        if improved:
            best, best_epoch, stale = val_auc, epoch, 0
            torch.save({"model": model.state_dict(), "feat_dim": feat_dim, "epoch": epoch,
                        "val_auroc": val_auc, "config": mc}, out / "model_best.pt")
        else:
            stale += 1
        if epoch % 10 == 0 or improved:
            log.info(f"  epoch {epoch:3d} loss={np.mean(losses):.4f} val_AUROC={val_auc:.4f}"
                     f"{' *' if improved else ''}")
        if stale >= mc["patience"]:
            log.info(f"  early stop at epoch {epoch}")
            break
    if not (out / "model_best.pt").exists():
        torch.save({"model": model.state_dict(), "feat_dim": feat_dim, "epoch": epoch,
                    "val_auroc": float("nan"), "config": mc}, out / "model_best.pt")
    model.load_state_dict(torch.load(out / "model_best.pt", weights_only=False)["model"])
    pred, att = predict(model, te, device, keep_attention=True)
    pred["fold"] = fold
    pred.to_csv(out / "test_predictions.csv", index=False)
    torch.save({pid: a for pid, a in att}, out / "test_attention.pt")
    log.info(f"  best epoch {best_epoch}; test AUROC={auroc(pred):.4f}")


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    fdir = features_dir(cfg, cfg["fm_mil"]["feature_set"])
    seeds = [args.seed] if args.seed is not None else cfg["splits"]["seeds"]
    n_folds = cfg["splits"]["n_folds"]
    folds = [args.fold] if args.fold is not None else list(range(n_folds))
    dump_config(cfg, work_path(cfg, "fm_mil", "config_used.yaml"))
    for seed in seeds:
        split = load_split(cfg, seed)
        for fold in folds:
            train_fold(cfg, seed, fold, split, fdir, device)
        files = [work_path(cfg, "fm_mil", f"seed{seed}", f"fold{k}", "test_predictions.csv")
                 for k in range(n_folds)]
        if all(f.exists() for f in files):
            oof = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files])
            oof.to_csv(oof_path(cfg, "fm", seed), index=False)
            log.info(f"seed {seed}: pooled OOF AUROC = {auroc(oof):.4f} (n={len(oof)})")


if __name__ == "__main__":
    main()
