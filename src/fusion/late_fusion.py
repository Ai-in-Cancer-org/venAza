"""Two-stage late multimodal fusion.

Stage 1 (upstream): every patient receives an *out-of-fold* image probability
from the image model that never saw them.

Stage 2 (here): a tabular classifier predicts response from
``[image_prob] + 16 clinical/genetic features`` (or the 16 features alone for
the clinical-only baseline). It is trained and evaluated on exactly the same
outer folds as the image models, so no head ever sees the labels of the
patients it is evaluated on. Fitted fold heads (incl. the scaler) are stored so
they can be applied, frozen, to an external cohort.
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sklearn

LINEAR_HEADS = {"lr", "elasticnet"}
# scikit-learn >= 1.8 infers the elastic-net penalty from l1_ratio
_SKLEARN_GE_1_8 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def make_head(name: str, cfg: dict):
    f = cfg["fusion"]
    rs = f.get("random_state", 42)
    if name == "lr":                     # L2-regularised logistic regression
        p = f["lr"]
        return LogisticRegression(C=p["C"], class_weight="balanced",
                                  max_iter=p["max_iter"], random_state=rs)
    if name == "elasticnet":
        p = f["elasticnet"]
        kw = {} if _SKLEARN_GE_1_8 else {"penalty": "elasticnet"}
        return LogisticRegression(C=p["C"], solver="saga", l1_ratio=p["l1_ratio"],
                                  class_weight="balanced", max_iter=p["max_iter"],
                                  random_state=rs, **kw)
    if name == "rf":
        p = f["rf"]
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      class_weight="balanced", random_state=rs, n_jobs=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        p = f["xgb"]
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"],
                             scale_pos_weight=p["scale_pos_weight"], eval_metric="logloss",
                             random_state=rs, n_jobs=-1)
    raise ValueError(f"Unknown fusion head '{name}'")


class FittedHead:
    """A fold head: optional StandardScaler + classifier, with fixed column order."""

    def __init__(self, name, feature_cols, scaler, model):
        self.name, self.feature_cols, self.scaler, self.model = name, list(feature_cols), scaler, model

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_cols].to_numpy(dtype=float)
        return self.scaler.transform(x) if self.scaler is not None else x

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(df))[:, 1]


def run_cv_fusion(table: pd.DataFrame, feature_cols: list[str], head: str, cfg: dict):
    """Out-of-fold fusion predictions using the ``fold`` column of ``table``.

    ``table`` holds one row per patient with ``patient_id, label, fold`` and the
    feature columns. Returns (predictions DataFrame, list of FittedHead per fold).
    """
    out = table[["patient_id", "label", "fold"]].copy()
    out["prob"] = np.nan
    heads = []
    for k in sorted(table["fold"].unique()):
        tr, te = table["fold"] != k, table["fold"] == k
        x_tr = table.loc[tr, feature_cols].to_numpy(dtype=float)
        y_tr = table.loc[tr, "label"].to_numpy(dtype=int)
        scaler = StandardScaler().fit(x_tr) if head in LINEAR_HEADS else None
        model = make_head(head, cfg)
        model.fit(scaler.transform(x_tr) if scaler is not None else x_tr, y_tr)
        fh = FittedHead(head, feature_cols, scaler, model)
        out.loc[te, "prob"] = fh.predict_proba(table.loc[te])
        heads.append((int(k), fh))
    return out, heads


def save_heads(heads, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, fh in heads:
        joblib.dump(fh, out_dir / f"{prefix}_fold{k}.joblib")


def load_heads(directory: Path, prefix: str) -> list[FittedHead]:
    files = sorted(Path(directory).glob(f"{prefix}_fold*.joblib"))
    if not files:
        raise FileNotFoundError(f"No fitted heads '{prefix}_fold*' in {directory}")
    return [joblib.load(f) for f in files]
