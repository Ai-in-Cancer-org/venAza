#!/usr/bin/env python
"""Step 9 - performance tables and ROC / PR figures.

For every branch and model (image only, clinical only, fusion x 4 heads):
  * per-seed pooled out-of-fold metrics (mean +/- SD across seeds);
  * seed-aggregated per-patient predictions (probabilities averaged over seeds;
    fold labels of the first seed are used for the per-fold rows/curves);
  * per-fold table with each fold's Youden threshold + pooled row with
    confidence intervals at the pooled Youden threshold;
  * confusion matrix, ROC and precision-recall curves (folds coloured, pooled
    out-of-fold estimate in bold black).
"""
import argparse

import _common  # noqa: F401
import numpy as np
import pandas as pd

from src.config import add_config_args, config_from_args, work_path
from src.evaluation.metrics import (METRICS, outcome_groups, per_fold_table,
                                       point_metrics, youden_threshold)
from src.evaluation.plots import roc_pr_figure
from src.pipeline import BRANCH_LABELS, evaluation_dir, fusion_dir
from src.utils import get_logger

log = get_logger()


def models_for(branch, heads):
    if branch == "clinical":
        return [f"clinical_only_{h}" for h in heads]
    return ["image_only"] + [f"fusion_{h}" for h in heads]


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--branches", nargs="+", default=["clinical", "resnet", "fm"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    seeds = cfg["splits"]["seeds"]
    heads = cfg["fusion"]["heads"]
    overview = []
    for branch in args.branches:
        out = evaluation_dir(cfg, branch)
        out.mkdir(parents=True, exist_ok=True)
        for model in models_for(branch, heads):
            per_seed = []
            for s in seeds:
                f = fusion_dir(cfg, branch, s) / f"oof_{model}.csv"
                if f.exists():
                    per_seed.append(pd.read_csv(f, dtype={"patient_id": str}).assign(seed=s))
            if not per_seed:
                continue
            seed_rows = []
            for d in per_seed:
                thr = youden_threshold(d["label"], d["prob"])
                seed_rows.append({"seed": d["seed"].iloc[0], **point_metrics(d["label"], d["prob"], thr)})
            seed_df = pd.DataFrame(seed_rows)
            seed_df.to_csv(out / f"{model}_per_seed.csv", index=False)

            first = per_seed[0][["patient_id", "label", "fold"]]
            mean_prob = pd.concat(per_seed).groupby("patient_id")["prob"].mean()
            agg = first.assign(prob=first["patient_id"].map(mean_prob).values)
            table = per_fold_table(agg, cfg)
            table.to_csv(out / f"{model}_metrics.csv", index=False)
            thr = float(table.loc[table["split"] == "Pooled", "threshold"].iloc[0])
            agg["threshold"] = thr
            agg["pred"] = (agg["prob"] >= thr).astype(int)
            agg["outcome"] = outcome_groups(agg["label"], agg["prob"], thr)
            agg.to_csv(out / f"{model}_patient_predictions.csv", index=False)
            pooled = table.iloc[-1]
            cm = pd.DataFrame([[pooled["TP"], pooled["FN"]], [pooled["FP"], pooled["TN"]]],
                              index=["GT responder", "GT non-responder"],
                              columns=["Pred responder", "Pred non-responder"])
            cm.to_csv(out / f"{model}_confusion_matrix.csv")
            roc_pr_figure(agg, f"{BRANCH_LABELS[branch]} - {model}", out / "figures" / model)
            overview.append({
                "branch": branch, "model": model, "n_seeds": len(per_seed),
                **{f"{m}_seed_mean": seed_df[m].mean() for m in ("auroc", "ap")},
                **{f"{m}_seed_sd": seed_df[m].std(ddof=0) for m in ("auroc", "ap")},
                **{f"pooled_{m}": pooled[m] for m in METRICS},
                **{f"pooled_{m}_ci": f"{pooled[m + '_lo']:.3f}-{pooled[m + '_hi']:.3f}"
                   for m in METRICS},
                "pooled_threshold": thr})
    ov = pd.DataFrame(overview)
    ov.to_csv(work_path(cfg, "evaluation", "overview.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 8):
        log.info("\n" + ov[["branch", "model", "auroc_seed_mean", "auroc_seed_sd",
                            "pooled_auroc", "pooled_auroc_ci"]].to_string(index=False))


if __name__ == "__main__":
    main()
