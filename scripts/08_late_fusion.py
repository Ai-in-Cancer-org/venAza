#!/usr/bin/env python
"""Step 8 - late multimodal fusion and the clinical/genetic-only baseline.

For every seed:
  * clinical baseline : 16 clinical/genetic features             (branch "clinical")
  * ResNet fusion     : ResNet OOF image probability + 16 features (branch "resnet")
  * FM fusion         : FM OOF image probability + 16 features     (branch "fm")
each with logistic regression (L2), elastic net, random forest and XGBoost,
trained/evaluated on the same outer folds as the image models. Fold heads are
saved for frozen application to external data.
"""
import argparse

import _common  # noqa: F401
import pandas as pd

from src.config import add_config_args, config_from_args
from src.data.clinical import FEATURE_NAMES
from src.data.splits import test_fold_of
from src.fusion.late_fusion import run_cv_fusion, save_heads
from src.pipeline import fusion_dir, load_cohort, load_oof, load_split, oof_path
from src.utils import get_logger, save_json
from sklearn.metrics import roc_auc_score

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--branches", nargs="+", default=["clinical", "resnet", "fm"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    cohort = load_cohort(cfg)
    heads = cfg["fusion"]["heads"]

    for seed in cfg["splits"]["seeds"]:
        folds = test_fold_of(load_split(cfg, seed))
        base = folds.merge(cohort[["patient_id"] + FEATURE_NAMES], on="patient_id", how="inner")
        for branch in args.branches:
            out = fusion_dir(cfg, branch, seed)
            if branch == "clinical":
                table, cols, cond = base.copy(), FEATURE_NAMES, "clinical_only"
            else:
                if not oof_path(cfg, branch, seed).exists():
                    log.info(f"[seed {seed}] no OOF predictions for '{branch}' - skipped")
                    continue
                oof = load_oof(cfg, branch, seed)[["patient_id", "image_prob"]]
                table = base.merge(oof, on="patient_id", how="inner")
                if len(table) != len(base):
                    log.info(f"[seed {seed}] {branch}: {len(base) - len(table)} patient(s) "
                             f"without an image probability are excluded")
                cols, cond = ["image_prob"] + FEATURE_NAMES, "fusion"
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "table.csv", index=False)
            summary = {}
            if branch != "clinical":
                img = table[["patient_id", "label", "fold"]].assign(prob=table["image_prob"])
                img.to_csv(out / "oof_image_only.csv", index=False)
                summary["image_only"] = roc_auc_score(img["label"], img["prob"])
            for h in heads:
                pred, fitted = run_cv_fusion(table, cols, h, cfg)
                pred.to_csv(out / f"oof_{cond}_{h}.csv", index=False)
                save_heads(fitted, out / "heads", f"{cond}_{h}")
                summary[f"{cond}_{h}"] = roc_auc_score(pred["label"], pred["prob"])
            save_json({"feature_cols": cols, "pooled_auroc": summary}, out / "summary.json")
            log.info(f"[seed {seed}] {branch}: " +
                     ", ".join(f"{k}={v:.3f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
