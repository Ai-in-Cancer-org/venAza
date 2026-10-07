#!/usr/bin/env python
"""Step 11 - statistical analyses.

1. Univariable logistic regression (odds ratio for CR/CRi) for every
   clinical/genetic variable and both out-of-fold image probabilities
   (ELN 2024 categories with intermediate risk as reference).
2. Multivariable logistic regression: each image probability adjusted for the
   genetic covariates given by --adjust (default NPM1, IDH1/2, RAS).
3. Comparison of clinical variables between TP / TN / FP / FN of a fusion model
   (Fisher-Freeman-Halton exact test for binary, Kruskal-Wallis for continuous).
4. Concordance of the two fusion models (Pearson r per ground-truth class,
   scatter plot, table of discordant patients).
"""
import argparse

import _common  # noqa: F401
import pandas as pd

from src.analysis.statistics import error_group_analysis, logit_or
from src.config import add_config_args, config_from_args, work_path
from src.data.clinical import MUTATIONS
from src.evaluation.plots import concordance_figure
from src.pipeline import load_cohort, patient_predictions
from src.utils import get_logger

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--head", default="lr", help="fusion head for error/concordance analyses")
    ap.add_argument("--adjust", nargs="+", default=["NPM1", "IDH1_2", "RAS"])
    ap.add_argument("--error_branch", default="fm")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = work_path(cfg, "statistics")
    out.mkdir(parents=True, exist_ok=True)

    c = load_cohort(cfg)
    for b in ("fm", "resnet"):
        try:
            img = patient_predictions(cfg, b, "image_only").set_index("patient_id")["prob"]
            c[f"image_prob_{b}"] = c["patient_id"].map(img)
        except FileNotFoundError:
            log.info(f"no image-only predictions for {b}")
    c["ELN2024_fav_vs_int"] = (c["eln2024"] == "favorable").astype(float)
    c["ELN2024_adv_vs_int"] = (c["eln2024"] == "adverse").astype(float)
    img_cols = [x for x in ("image_prob_fm", "image_prob_resnet") if x in c]

    # 1) univariable
    uni = [logit_or(c, "label", [v]) for v in MUTATIONS + ["age_years", "blasts_frac"] + img_cols]
    uni.append(logit_or(c, "label", ["ELN2024_fav_vs_int", "ELN2024_adv_vs_int"]))
    uni = pd.concat(uni).sort_values("OR", ascending=False)
    uni.to_csv(out / "univariable_odds_ratios.csv", index=False)
    log.info("Univariable odds ratios:\n" + uni.round(3).to_string(index=False))

    # 2) multivariable
    multi = [logit_or(c, "label", [v] + args.adjust).assign(model=f"{v} + {'+'.join(args.adjust)}")
             for v in img_cols]
    if multi:
        pd.concat(multi).to_csv(out / "multivariable_odds_ratios.csv", index=False)

    # 3) error groups
    try:
        pred = patient_predictions(cfg, args.error_branch, f"fusion_{args.head}")
        d = c.merge(pred[["patient_id", "outcome"]], on="patient_id")
        binary = list(MUTATIONS)
        if d["sex"].notna().any() and d["sex"].nunique() == 2:
            d["sex_male"] = (d["sex"] == "M").astype(float)
            binary.append("sex_male")
        err = error_group_analysis(d, "outcome", binary, ["blasts_frac", "age_years"])
        err.to_csv(out / f"error_groups_{args.error_branch}_fusion_{args.head}.csv", index=False)
    except FileNotFoundError as e:
        log.info(str(e))

    # 4) concordance FM vs ResNet fusion
    try:
        a = patient_predictions(cfg, "fm", f"fusion_{args.head}")
        b = patient_predictions(cfg, "resnet", f"fusion_{args.head}")
    except FileNotFoundError:
        log.info("concordance analysis needs both fusion models - skipped")
        return
    m = a[["patient_id", "label", "prob", "threshold", "outcome"]].merge(
        b[["patient_id", "prob", "threshold", "outcome"]], on="patient_id", suffixes=("_a", "_b"))
    stats = concordance_figure(m, m["threshold_a"].iloc[0], m["threshold_b"].iloc[0],
                               "FM", "ResNet", out / f"concordance_fusion_{args.head}")
    stats.to_csv(out / f"concordance_fusion_{args.head}.csv", index=False)
    dis = m[m["outcome_a"] != m["outcome_b"]].merge(
        c[["patient_id", "eln2024", "age_years"] + MUTATIONS], on="patient_id")
    dis["mutations"] = dis[MUTATIONS].apply(lambda r: ", ".join(k for k in MUTATIONS if r[k] == 1), axis=1)
    dis = dis.rename(columns={"outcome_a": "FM_pred", "outcome_b": "ResNet_pred"})
    dis[["label", "FM_pred", "ResNet_pred", "eln2024", "age_years", "mutations"]].reset_index(
        drop=True).to_csv(out / f"discordant_cases_fusion_{args.head}.csv", index_label="case")
    log.info(f"statistics written to {out}")


if __name__ == "__main__":
    main()
