"""Locations of pipeline artefacts and loaders shared by the scripts.

Layout below ``paths.work_dir`` (internal cohort unless noted)::

    patches/<patient>/<roi>/{kept,rejected}/patch_<row>_<col>.png
    patches/patch_index.csv                one row per patch (QC result)
    cohort.csv                             labels + encoded clinical features
    splits/seed<S>.csv                     fold / role per patient
    resnet_mil/seed<S>/fold<k>/model_best.pt
    resnet_mil/seed<S>/oof_predictions.csv
    dino/<encoder>/adapted_encoder_final.pt
    features/<conch|uni|combined>/<patient>.pt
    fm_mil/seed<S>/fold<k>/model_best.pt
    fm_mil/seed<S>/oof_predictions.csv
    fusion/<resnet|fm|clinical>/seed<S>/{oof_*.csv, heads/*.joblib, table.csv}
    evaluation/<branch>/...                tables + figures (seed-aggregated)
    external/...                           same structure for the external cohort
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import work_path

BRANCH_DIRS = {"resnet": "resnet_mil", "fm": "fm_mil"}
BRANCH_LABELS = {"resnet": "ResNet-34 MIL", "fm": "Foundation model (CONCH+UNI) MIL",
                 "clinical": "Clinical/genetic only"}


def cohort_root(cfg, cohort: str = "internal") -> Path:
    return work_path(cfg) if cohort == "internal" else work_path(cfg, "external")


def patches_dir(cfg, cohort="internal") -> Path:
    return cohort_root(cfg, cohort) / "patches"


def load_patch_index(cfg, cohort="internal", kept_only=True) -> pd.DataFrame:
    p = patches_dir(cfg, cohort) / "patch_index.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/01_extract_patches.py first")
    df = pd.read_csv(p, dtype={"patient_id": str})
    if kept_only:
        df = df[df["keep"] == 1]
    return df.sort_values(["patient_id", "patch_path"]).reset_index(drop=True)


def load_cohort(cfg) -> pd.DataFrame:
    p = work_path(cfg, "cohort.csv")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/02_prepare_cohort.py first")
    return pd.read_csv(p, dtype={"patient_id": str})


def split_path(cfg, seed: int) -> Path:
    return work_path(cfg, "splits", f"seed{seed}.csv")


def load_split(cfg, seed: int) -> pd.DataFrame:
    return pd.read_csv(split_path(cfg, seed), dtype={"patient_id": str})


def features_dir(cfg, feature_set: str, cohort="internal") -> Path:
    return cohort_root(cfg, cohort) / "features" / feature_set


def adapted_ckpt(cfg, encoder: str) -> Path:
    return work_path(cfg, "dino", encoder, "adapted_encoder_final.pt")


def oof_path(cfg, branch: str, seed: int) -> Path:
    return work_path(cfg, BRANCH_DIRS[branch], f"seed{seed}", "oof_predictions.csv")


def load_oof(cfg, branch: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(oof_path(cfg, branch, seed), dtype={"patient_id": str})


def fusion_dir(cfg, branch: str, seed: int) -> Path:
    return work_path(cfg, "fusion", branch, f"seed{seed}")


def evaluation_dir(cfg, branch: str) -> Path:
    return work_path(cfg, "evaluation", branch)


def patient_predictions(cfg, branch: str, model: str) -> pd.DataFrame:
    """Seed-aggregated per-patient predictions written by 09_evaluate.py.

    ``model`` is e.g. ``image_only``, ``clinical_only_lr`` or ``fusion_xgb``.
    """
    p = evaluation_dir(cfg, branch) / f"{model}_patient_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/09_evaluate.py first")
    return pd.read_csv(p, dtype={"patient_id": str})
