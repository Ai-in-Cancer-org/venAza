#!/usr/bin/env python
"""Step 3 - custom CNN branch: end-to-end ResNet-34 gated-attention MIL.

For each seed and outer fold the model is trained on the ``train`` patients,
early-stopped / checkpoint-selected on the inner ``val`` patients and finally
applied to the held-out ``test`` fold. The test-fold predictions of all folds
form the out-of-fold (OOF) image probabilities used for late fusion.

Training: patient-level BCE loss, Adam, ReduceLROnPlateau on validation AUROC,
gradient clipping, mixed precision, class-balanced patient sampling; the CNN
encoder is frozen for the first ``warmup_frozen_epochs`` epochs.

    python scripts/03_train_resnet_mil.py --config cfg.yaml               # all seeds/folds
    python scripts/03_train_resnet_mil.py --config cfg.yaml --seed 42 --fold 0
"""
import argparse
import copy

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import add_config_args, config_from_args, dump_config, work_path
from src.data.datasets import (TileBagDataset, balanced_sampler, resnet_transforms,
                                  single_bag_collate)
from src.data.splits import fold_patients
from src.models.attention_mil import ResNetAttentionMIL
from src.pipeline import load_patch_index, load_split, oof_path, patches_dir
from src.utils import get_device, get_logger, seed_everything

log = get_logger()


def build_model(rc):
    return ResNetAttentionMIL(rc["backbone"], rc["pretrained"], rc["attn_dim"],
                              rc["attn_dropout"], rc["attn_temperature"], rc["chunk_size"])


@torch.no_grad()
def predict(model, ds, device, keep_attention=False):
    model.eval()
    probs, attn_rows = [], []
    loader = DataLoader(ds, batch_size=1, collate_fn=single_bag_collate,
                        num_workers=0, shuffle=False)
    for bag, label, pid, paths in loader:
        logit, attn = model(bag.to(device))
        probs.append((pid, int(label), float(torch.sigmoid(logit))))
        if keep_attention:
            attn_rows += [(pid, p, float(a)) for p, a in zip(paths, attn.cpu().numpy())]
    df = pd.DataFrame(probs, columns=["patient_id", "label", "image_prob"])
    att = pd.DataFrame(attn_rows, columns=["patient_id", "patch_path", "attention"])
    return df, att


def auroc(df):
    return roc_auc_score(df["label"], df["image_prob"]) if df["label"].nunique() == 2 else float("nan")


def train_fold(cfg, seed, fold, split, tiles, device):
    rc = cfg["resnet_mil"]
    out = work_path(cfg, "resnet_mil", f"seed{seed}", f"fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed + fold)
    root = patches_dir(cfg)
    ds_tr = TileBagDataset(fold_patients(split, fold, "train"), tiles, root,
                           resnet_transforms(rc["input_size"], True), rc["max_train_tiles"])
    ds_va = TileBagDataset(fold_patients(split, fold, "val"), tiles, root,
                           resnet_transforms(rc["input_size"], False))
    ds_te = TileBagDataset(fold_patients(split, fold, "test"), tiles, root,
                           resnet_transforms(rc["input_size"], False))
    log.info(f"[seed {seed} fold {fold}] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
    loader = DataLoader(ds_tr, batch_size=1, sampler=balanced_sampler(ds_tr.labels()),
                        collate_fn=single_bag_collate, num_workers=rc["num_workers"])

    model = build_model(rc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=rc["plateau_factor"],
                                                       patience=rc["plateau_patience"])
    use_amp = bool(rc["use_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(rc["epochs"]):
        model.set_encoder_trainable(epoch >= rc["warmup_frozen_epochs"])
        model.train()
        losses = []
        for bag, label, _, _ in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logit, _ = model(bag.to(device))
                loss = crit(logit.view(1), torch.tensor([label], device=device))
            scaler.scale(loss).backward()
            if rc["grad_clip"]:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), rc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        val_auc = auroc(predict(model, ds_va, device)[0])
        if val_auc == val_auc:
            sched.step(val_auc)
        improved = val_auc == val_auc and val_auc > best_auc + rc["early_stop_min_delta"]
        log.info(f"  epoch {epoch:3d} loss={np.mean(losses):.4f} val_AUROC={val_auc:.4f}"
                 f" lr={opt.param_groups[0]['lr']:.2e}{' *' if improved else ''}")
        if improved:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= rc["early_stop_patience"]:
                log.info(f"  early stop at epoch {epoch}")
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "epoch": best_epoch, "val_auroc": best_auc,
                "config": rc}, out / "model_best.pt")

    test_pred, test_attn = predict(model, ds_te, device, keep_attention=True)
    test_pred["fold"] = fold
    test_pred.to_csv(out / "test_predictions.csv", index=False)
    test_attn.to_csv(out / "test_attention.csv", index=False)
    log.info(f"  [seed {seed} fold {fold}] best epoch {best_epoch}, test AUROC={auroc(test_pred):.4f}")
    return test_pred


def collect_oof(cfg, seed, n_folds):
    parts = []
    for k in range(n_folds):
        p = work_path(cfg, "resnet_mil", f"seed{seed}", f"fold{k}", "test_predictions.csv")
        if not p.exists():
            return None
        parts.append(pd.read_csv(p, dtype={"patient_id": str}))
    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(oof_path(cfg, "resnet", seed), index=False)
    return oof


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    tiles = load_patch_index(cfg)
    seeds = [args.seed] if args.seed is not None else cfg["splits"]["seeds"]
    n_folds = cfg["splits"]["n_folds"]
    folds = [args.fold] if args.fold is not None else list(range(n_folds))
    dump_config(cfg, work_path(cfg, "resnet_mil", "config_used.yaml"))
    for seed in seeds:
        split = load_split(cfg, seed)
        for fold in folds:
            train_fold(cfg, seed, fold, split, tiles, device)
        oof = collect_oof(cfg, seed, n_folds)
        if oof is not None:
            log.info(f"seed {seed}: pooled OOF AUROC = {auroc(oof):.4f} (n={len(oof)})")


if __name__ == "__main__":
    main()
