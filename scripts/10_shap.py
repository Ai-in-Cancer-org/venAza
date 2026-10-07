#!/usr/bin/env python
"""Step 10 - SHAP feature attributions of the fusion models (log-odds scale).

Per seed and fold, a linear SHAP explainer (correlation-dependent perturbation)
of the logistic-regression fusion head explains the held-out patients; values
are averaged per patient across seeds. Outputs a beeswarm plot and the mean
SHAP value per feature for true positives and true negatives.
"""
import argparse

import _common  # noqa: F401
import numpy as np
import pandas as pd

from src.config import add_config_args, config_from_args, work_path
from src.data.clinical import DISPLAY_NAMES
from src.explainability.shap_analysis import fold_shap, plot_beeswarm, plot_group_means
from src.fusion.late_fusion import load_heads
from src.pipeline import BRANCH_LABELS, fusion_dir, patient_predictions
from src.utils import get_logger, load_json, save_json

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--branches", nargs="+", default=["fm", "resnet"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    ex = cfg["explainability"]
    head = ex["shap_head"]
    if head not in ("lr", "elasticnet"):
        raise ValueError("Linear SHAP requires a linear fusion head (lr / elasticnet)")
    out = work_path(cfg, "shap")
    out.mkdir(parents=True, exist_ok=True)
    for branch in args.branches:
        frames, bases = [], []
        for s in cfg["splits"]["seeds"]:
            d = fusion_dir(cfg, branch, s)
            if not (d / "table.csv").exists():
                continue
            table = pd.read_csv(d / "table.csv", dtype={"patient_id": str})
            cols = load_json(d / "summary.json")["feature_cols"]
            heads = load_heads(d / "heads", f"fusion_{head}")
            sv, base = fold_shap(table, heads, cols, ex["shap_perturbation"])
            frames.append(sv); bases.append(base)
        if not frames:
            log.info(f"{branch}: no fusion results - skipped")
            continue
        shap_df = pd.concat(frames).groupby("patient_id", sort=False).mean().reset_index()
        shap_df.to_csv(out / f"shap_values_{branch}_{head}.csv", index=False)
        preds = patient_predictions(cfg, branch, f"fusion_{head}")
        groups = shap_df["patient_id"].map(preds.set_index("patient_id")["outcome"]).values
        title = f"{BRANCH_LABELS[branch]} fusion ({head.upper()})"
        plot_beeswarm(shap_df, cols, DISPLAY_NAMES, out / f"beeswarm_{branch}_{head}", title)
        means = plot_group_means(shap_df, groups, cols, DISPLAY_NAMES,
                                 out / f"mean_shap_TP_TN_{branch}_{head}", title)
        means.to_csv(out / f"mean_shap_TP_TN_{branch}_{head}.csv")
        save_json({"base_value_mean_over_folds_and_seeds": float(np.mean(bases)),
                   "n_patients": len(shap_df)}, out / f"base_value_{branch}_{head}.json")
        log.info(f"{branch}: SHAP for {len(shap_df)} patients -> {out}")


if __name__ == "__main__":
    main()
