"""Clinical / genetic table: response labels and the 16-dim fusion feature vector.

Feature vector (order is fixed and used everywhere, incl. external validation):

    ELN 2024 less-intensive risk  one-hot (favorable, intermediate, adverse)   3
    SWOG cytogenetic category     one-hot (favorable, intermediate, unfavorable) 3
    binarised mutations NPM1, IDH1/2, TP53, ASXL1, RAS, PTPN11, RUNX1, FLT3-ITD 8
    age at diagnosis              scaled to [0, 1]                              1
    bone marrow blasts            fraction in [0, 1]                            1

Patients whose risk category cannot be assigned (e.g. SWOG "unknown") are
encoded with the all-zero indicator vector, i.e. treated as the reference level
rather than dropped or imputed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..utils import normalize_pid

RISK_LEVELS = ["favorable", "intermediate", "adverse"]
MUTATIONS = ["NPM1", "IDH1_2", "TP53", "ASXL1", "RAS", "PTPN11", "RUNX1", "FLT3_ITD"]
FEATURE_NAMES = (["ELN2024_fav", "ELN2024_int", "ELN2024_adv",
                  "SWOG_fav", "SWOG_int", "SWOG_unf"]
                 + MUTATIONS + ["age", "blasts"])
DISPLAY_NAMES = {
    "image_prob": "Image probability", "ELN2024_fav": "ELN2024 favorable",
    "ELN2024_int": "ELN2024 intermediate", "ELN2024_adv": "ELN2024 adverse",
    "SWOG_fav": "SWOG favorable", "SWOG_int": "SWOG intermediate",
    "SWOG_unf": "SWOG unfavorable", "NPM1": "NPM1", "IDH1_2": "IDH1/2", "TP53": "TP53",
    "ASXL1": "ASXL1", "RAS": "RAS", "PTPN11": "PTPN11", "RUNX1": "RUNX1",
    "FLT3_ITD": "FLT3-ITD", "age": "Age", "blasts": "BM blasts",
}


def _norm_text(v) -> str:
    return " ".join(str(v).strip().lower().split()) if pd.notna(v) else ""


def map_risk(v) -> str | None:
    """Map free-text risk categories onto favorable / intermediate / adverse."""
    s = _norm_text(v)
    if not s:
        return None
    if "unfav" in s or "adverse" in s or s.startswith("poor") or s == "high":
        return "adverse"
    if "fav" in s or s == "low":
        return "favorable"
    if "int" in s:
        return "intermediate"
    return None


def parse_binary(v) -> float:
    s = _norm_text(v).upper()
    return 1.0 if s in {"Y", "YES", "1", "1.0", "TRUE", "POS", "POSITIVE", "+", "MUT", "MUTATED"} else 0.0


def parse_blasts(v) -> float:
    """Blast count as a fraction in [0, 1]; accepts 0.35, 35, '35%' or '30-40%'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        f = float(v)
        return f if f <= 1.0 else f / 100.0
    except (TypeError, ValueError):
        pass
    s = str(v).replace("%", "").strip()
    m = re.match(r"(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)", s)
    if m:
        mid = (float(m.group(1)) + float(m.group(2))) / 2.0
        return mid / 100.0 if mid > 1.0 else mid
    try:
        f = float(s)
        return f / 100.0 if f > 1.0 else f
    except ValueError:
        return np.nan


def map_response(v, cfg: dict):
    s = _norm_text(v)
    pos = {x.lower() for x in cfg["clinical"]["responder_labels"]}
    neg = {x.lower() for x in cfg["clinical"]["non_responder_labels"]}
    if s in pos:
        return 1
    if s in neg:
        return 0
    return None


def read_table(path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_clinical(path, cfg: dict, require_label: bool = True) -> pd.DataFrame:
    """Read the clinical table and return canonical columns.

    Output columns: patient_id, label, eln2024, swog, age_years, blasts_frac,
    sex, the 8 mutation columns (0/1) and the 16 encoded features.
    """
    raw = read_table(path)
    cols = cfg["clinical"]["columns"]
    muts = cfg["clinical"]["mutations"]
    missing = [c for k, c in cols.items() if k != "sex" and c not in raw.columns]
    missing += [c for c in muts.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"Clinical table lacks columns {missing}. "
                       f"Adapt `clinical.columns` / `clinical.mutations` in the config.")

    out = pd.DataFrame({"patient_id": raw[cols["patient_id"]].map(normalize_pid)})
    out["response_raw"] = raw[cols["response"]].astype(str)
    out["label"] = raw[cols["response"]].map(lambda v: map_response(v, cfg))
    out["eln2024"] = raw[cols["eln2024"]].map(map_risk)
    out["swog"] = raw[cols["swog"]].map(map_risk)
    out["age_years"] = pd.to_numeric(raw[cols["age"]], errors="coerce")
    out["blasts_frac"] = raw[cols["blasts"]].map(parse_blasts)
    out["sex"] = raw[cols["sex"]].astype(str).str.strip().str.upper().str[:1] \
        if cols.get("sex") in raw.columns else np.nan
    for canon, src in muts.items():
        out[canon] = raw[src].map(parse_binary)

    feats = encode_features(out, cfg)
    # mutation columns are already 0/1 and shared between both views
    out = pd.concat([out, feats.drop(columns=MUTATIONS)], axis=1)
    if require_label:
        unknown = out.loc[out["label"].isna(), "response_raw"].value_counts()
        if len(unknown):
            print("Excluded (no CR/CRi vs non-response label):",
                  ", ".join(f"'{k}' x{v}" for k, v in unknown.items()))
        out = out[out["label"].notna()].copy()
        out["label"] = out["label"].astype(int)
    if out["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient identifiers in the clinical table")
    return out.reset_index(drop=True)


def encode_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """16-dim feature matrix (see module docstring) as a DataFrame."""
    x = pd.DataFrame(index=df.index)
    for prefix, col, short in (("ELN2024", "eln2024", None), ("SWOG", "swog", None)):
        for lvl, tag in zip(RISK_LEVELS, ["fav", "int", "adv" if prefix == "ELN2024" else "unf"]):
            x[f"{prefix}_{tag}"] = (df[col] == lvl).astype(float)
    for m in MUTATIONS:
        x[m] = df[m].astype(float)
    age_scale = float(cfg["clinical"].get("age_scale", 100.0))
    x["age"] = (df["age_years"] / age_scale).clip(0, 1)
    x["blasts"] = df["blasts_frac"].clip(0, 1)
    # continuous features: missing values are set to the cohort median
    for c in ("age", "blasts"):
        if x[c].isna().any():
            x[c] = x[c].fillna(x[c].median())
    return x[FEATURE_NAMES]
