#!/usr/bin/env python
"""Step 2 - build the analysis cohort and the cross-validation splits.

* reads the clinical table, maps best response to CR/CRi (1) vs. non-response (0)
  and encodes the 16 clinical/genetic fusion features;
* keeps patients that have a label and at least one QC-passed patch;
* writes one stratified split table per seed (outer test fold + inner validation);
* writes a descriptive cohort table.
"""
import argparse

import _common  # noqa: F401
import numpy as np
import pandas as pd

from src.config import add_config_args, config_from_args, ensure_dir, work_path
from src.data.clinical import MUTATIONS, load_clinical
from src.data.splits import make_splits
from src.pipeline import load_patch_index, split_path
from src.utils import get_logger

log = get_logger()


def describe(c: pd.DataFrame) -> pd.DataFrame:
    n = len(c)
    rows = [("N", str(n))]

    def med(s, scale=1.0):
        s = s.dropna() * scale
        return f"{s.median():.0f} ({s.quantile(.25):.0f}-{s.quantile(.75):.0f})"

    rows.append(("Age, years, median (IQR)", med(c["age_years"])))
    if c["sex"].notna().any():
        for s, cnt in c["sex"].value_counts().items():
            rows.append((f"Sex {s}, n (%)", f"{cnt}/{n} ({100 * cnt / n:.1f}%)"))
    rows.append(("BM blasts, %, median (IQR)", med(c["blasts_frac"], 100)))
    for m in MUTATIONS:
        k = int(c[m].sum())
        rows.append((f"{m}, n (%)", f"{k}/{n} ({100 * k / n:.1f}%)"))
    for col, name in (("eln2024", "ELN2024 less-intensive"), ("swog", "SWOG")):
        for lvl in ["favorable", "intermediate", "adverse", None]:
            k = int((c[col] == lvl).sum()) if lvl else int(c[col].isna().sum())
            rows.append((f"{name} {lvl or 'unknown'}, n (%)", f"{k}/{n} ({100 * k / n:.1f}%)"))
    k = int(c["label"].sum())
    rows.append(("CR/CRi, n (%)", f"{k}/{n} ({100 * k / n:.1f}%)"))
    return pd.DataFrame(rows, columns=["parameter", "value"])


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)

    clin = load_clinical(cfg["paths"]["clinical_table"], cfg)
    tiles = load_patch_index(cfg)
    with_tiles = set(tiles["patient_id"].unique())
    no_tiles = sorted(set(clin["patient_id"]) - with_tiles)
    if no_tiles:
        log.info(f"{len(no_tiles)} labelled patient(s) without QC-passed patches excluded")
    no_label = sorted(with_tiles - set(clin["patient_id"]))
    if no_label:
        log.info(f"{len(no_label)} patient folder(s) without a usable label excluded")
    cohort = clin[clin["patient_id"].isin(with_tiles)].reset_index(drop=True)
    cohort.to_csv(work_path(cfg, "cohort.csv"), index=False)
    log.info(f"Cohort: {len(cohort)} patients ({int(cohort['label'].sum())} responders)")

    ensure_dir(work_path(cfg, "splits"))
    sp = cfg["splits"]
    for seed in sp["seeds"]:
        s = make_splits(cohort, sp["n_folds"], sp["inner_val_frac"], seed)
        s.to_csv(split_path(cfg, seed), index=False)
        log.info(f"  seed {seed}: " + ", ".join(
            f"fold {k}: test={int(((s.fold == k) & (s.role == 'test')).sum())}"
            for k in range(sp["n_folds"])))
    ensure_dir(work_path(cfg, "tables"))
    describe(cohort).to_csv(work_path(cfg, "tables", "cohort_characteristics.csv"), index=False)


if __name__ == "__main__":
    main()
