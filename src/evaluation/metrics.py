"""Patient-level performance metrics.

* Decision thresholds: Youden's J (argmax of TPR - FPR) on the evaluated set.
* Threshold-free: AUROC, average precision (AP).
* At the threshold: accuracy, F1, sensitivity, specificity, PPV, NPV.
* Confidence intervals: percentile bootstrap (default; ``n_bootstrap``
  resamples of patients with replacement from the pooled out-of-fold
  predictions, threshold held fixed) or Wilson score intervals for the
  proportion metrics (``ci_method: wilson``; AUROC/AP always bootstrap).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRICS = ["auroc", "ap", "accuracy", "f1", "sensitivity", "specificity", "ppv", "npv"]


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.where(np.isinf(thr), 1.0, thr)
    return float(thr[np.argmax(tpr - fpr)])


def confusion(y, p, thr):
    y = np.asarray(y, int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def _div(a, b):
    return a / b if b else np.nan


def point_metrics(y, p, thr) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    tp, tn, fp, fn = confusion(y, p, thr)
    two = len(np.unique(y)) == 2
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    ppv, npv = _div(tp, tp + fp), _div(tn, tn + fn)
    return {
        "auroc": roc_auc_score(y, p) if two else np.nan,
        "ap": average_precision_score(y, p) if two else np.nan,
        "accuracy": (tp + tn) / len(y),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def wilson(k, n, level=0.95):
    from scipy.stats import norm
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.5 + level / 2)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(y, p, thr, n_boot=2000, level=0.95, seed=0) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    rng = np.random.RandomState(seed)
    vals = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        pm = point_metrics(y[idx], p[idx], thr)
        for m in METRICS:
            vals[m].append(pm[m])
    a = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {m: tuple(np.nanpercentile(v, [a, 100 - a])) if v else (np.nan, np.nan)
                for m, v in vals.items()}


def metrics_with_ci(y, p, thr=None, cfg: dict | None = None) -> dict:
    ev = (cfg or {}).get("evaluation", {})
    n_boot, level = ev.get("n_bootstrap", 2000), ev.get("ci_level", 0.95)
    method, seed = ev.get("ci_method", "bootstrap"), ev.get("bootstrap_seed", 0)
    if thr is None:
        thr = youden_threshold(y, p)
    pm = point_metrics(y, p, thr)
    ci = bootstrap_ci(y, p, thr, n_boot, level, seed)
    if method == "wilson":
        tp, tn, fp, fn = pm["TP"], pm["TN"], pm["FP"], pm["FN"]
        ci.update({"accuracy": wilson(tp + tn, len(y), level),
                   "sensitivity": wilson(tp, tp + fn, level),
                   "specificity": wilson(tn, tn + fp, level),
                   "ppv": wilson(tp, tp + fp, level), "npv": wilson(tn, tn + fn, level)})
    row = {"n": len(y), "threshold": thr}
    for m in METRICS:
        row[m] = pm[m]
        row[f"{m}_lo"], row[f"{m}_hi"] = ci[m]
    row.update({k: pm[k] for k in ("TP", "TN", "FP", "FN")})
    return row


def per_fold_table(pred: pd.DataFrame, cfg: dict, prob_col: str = "prob") -> pd.DataFrame:
    """Rows: every fold (own Youden threshold) + pooled out-of-fold (with CIs)."""
    rows = []
    for k in sorted(pred["fold"].unique()):
        d = pred[pred["fold"] == k]
        thr = youden_threshold(d["label"], d[prob_col]) if d["label"].nunique() == 2 else 0.5
        r = point_metrics(d["label"], d[prob_col], thr)
        rows.append({"split": f"Fold {k}", "n": len(d), "threshold": thr, **r})
    pooled = metrics_with_ci(pred["label"], pred[prob_col], None, cfg)
    rows.append({"split": "Pooled", **pooled})
    return pd.DataFrame(rows)


def outcome_groups(y, p, thr) -> np.ndarray:
    y = np.asarray(y, int)
    yhat = (np.asarray(p) >= thr).astype(int)
    return np.select([(y == 1) & (yhat == 1), (y == 0) & (yhat == 0),
                      (y == 0) & (yhat == 1), (y == 1) & (yhat == 0)],
                     ["TP", "TN", "FP", "FN"], "NA")


def format_ci(v, lo, hi, nd=3) -> str:
    return f"{v:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"
