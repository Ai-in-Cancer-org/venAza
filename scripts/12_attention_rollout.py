#!/usr/bin/env python
"""Step 12 - qualitative explainability: attention rollout of the adapted
CONCH and UNI encoders (averaged), shown

  * for the tiles with the highest MIL attention of a patient, and
  * as a spatial reconstruction of each ROI (tiles rejected by QC in grey).

MIL attention comes from the foundation-model MIL head of the fold in which the
patient was held out (first seed). By default the most confident true
positives and true negatives of the FM fusion model are visualised.

    python scripts/12_attention_rollout.py --config cfg.yaml [--patients ID1 ID2]
"""
import argparse

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import add_config_args, config_from_args, work_path
from src.explainability.rollout import AttentionRollout, overlay
from src.models.foundation import fm_transform, load_encoder
from src.pipeline import (adapted_ckpt, load_patch_index, load_split, patches_dir,
                             patient_predictions)
from src.utils import get_device, get_logger

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--n_per_group", type=int, default=2)
    ap.add_argument("--head", default="lr")
    ap.add_argument("--encoders", nargs="+", default=["conch", "uni"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    ex = cfg["explainability"]
    device = get_device()
    out = work_path(cfg, "rollout")
    out.mkdir(parents=True, exist_ok=True)

    pred = patient_predictions(cfg, "fm", f"fusion_{args.head}")
    if args.patients:
        chosen = pred[pred["patient_id"].isin(args.patients)]
    else:
        tp = pred[pred.outcome == "TP"].nlargest(args.n_per_group, "prob")
        tn = pred[pred.outcome == "TN"].nsmallest(args.n_per_group, "prob")
        chosen = pd.concat([tp, tn])

    tf = fm_transform(cfg["foundation"]["tile_size"])
    rollouts = [AttentionRollout(load_encoder(e, cfg, adapted_ckpt(cfg, e), device),
                                 ex["rollout_head_fusion"]) for e in args.encoders]
    root = patches_dir(cfg)
    index = load_patch_index(cfg, kept_only=False)
    seed = cfg["splits"]["seeds"][0]
    split = load_split(cfg, seed)
    test_fold = split[split.role == "test"].set_index("patient_id")["fold"]

    def rollout_map(path):
        x = tf(Image.open(path)).unsqueeze(0).to(device)
        return np.mean([r(x) for r in rollouts], axis=0)

    for _, row in chosen.iterrows():
        pid, group = row["patient_id"], row["outcome"]
        pdir = out / f"{group}_{pid}"
        pdir.mkdir(exist_ok=True)
        fold = int(test_fold[pid])
        att = torch.load(work_path(cfg, "fm_mil", f"seed{seed}", f"fold{fold}", "test_attention.pt"),
                         weights_only=False).get(pid)
        tiles = index[(index.patient_id == pid) & (index.keep == 1)].sort_values("patch_path")
        tiles = tiles.assign(mil_attention=np.asarray(att) if att is not None else np.nan)

        # (a) highest-attention tiles
        top = tiles.nlargest(ex["rollout_top_tiles"], "mil_attention")
        n = len(top)
        fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)
        for j, (_, t) in enumerate(top.iterrows()):
            img = np.asarray(Image.open(root / t.patch_path).convert("RGB"))
            axes[0, j].imshow(img)
            axes[1, j].imshow(overlay(img, rollout_map(root / t.patch_path), ex["rollout_alpha"]))
            axes[0, j].set_title(f"attn {t.mil_attention:.3f}", fontsize=7)
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle(f"{group} - fusion probability {row['prob']:.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(pdir / "top_tiles_rollout.png", dpi=200)
        plt.close(fig)

        # (b) ROI reconstruction
        for roi, g in index[index.patient_id == pid].groupby("roi"):
            ps = cfg["patch_extraction"]["patch_size"]
            h, w = (g.row.max() + 1) * ps, (g.col.max() + 1) * ps
            canvas = np.full((h, w, 3), 200, np.uint8)
            for _, t in g.iterrows():
                if t.keep == 1:
                    img = np.asarray(Image.open(root / t.patch_path).convert("RGB"))
                    canvas[t.y:t.y + ps, t.x:t.x + ps] = overlay(img, rollout_map(root / t.patch_path),
                                                                 ex["rollout_alpha"])
            Image.fromarray(canvas).save(pdir / f"roi_{roi}_rollout.png")
        log.info(f"{group} {pid}: rollout figures in {pdir}")
    for r in rollouts:
        r.remove()


if __name__ == "__main__":
    main()
