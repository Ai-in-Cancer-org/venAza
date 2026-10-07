"""SHAP attributions for the logistic-regression fusion head.

Within each cross-validation fold a linear SHAP explainer with
correlation-dependent perturbation (background = that fold's scaled training
data) explains the *held-out* patients of that fold. Because the folds
partition the cohort, every patient is explained exactly once by a model that
never saw them, giving one attribution vector per patient in log-odds of
response. The reported baseline is the mean expected value of the fold
explainers. With several seeds, attributions are averaged per patient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _explainer(model, background, perturbation: str):
    import shap
    if perturbation == "correlation_dependent":
        return shap.LinearExplainer(model, shap.maskers.Impute(background))
    return shap.LinearExplainer(model, shap.maskers.Independent(background))


def fold_shap(table: pd.DataFrame, heads, feature_cols, perturbation="correlation_dependent"):
    """``heads``: list of FittedHead in fold order matching ``table['fold']``."""
    rows, base_values = [], []
    for k, fh in enumerate(heads):
        tr, te = table["fold"] != k, table["fold"] == k
        bg = fh.transform(table.loc[tr])
        xt = fh.transform(table.loc[te])
        ex = _explainer(fh.model, bg, perturbation)
        sv = np.asarray(ex.shap_values(xt))
        if sv.ndim == 3:            # some versions return [n, f, classes]
            sv = sv[..., 1]
        ev = np.atleast_1d(ex.expected_value)
        base_values.append(float(ev[-1]))
        d = pd.DataFrame(sv, columns=feature_cols)
        d.insert(0, "patient_id", table.loc[te, "patient_id"].values)
        xs = pd.DataFrame(xt, columns=[f"x_{c}" for c in feature_cols])
        rows.append(pd.concat([d, xs], axis=1))
    return pd.concat(rows, ignore_index=True), float(np.mean(base_values))


def plot_beeswarm(shap_df, feature_cols, display, out_stem, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    exp = shap.Explanation(values=shap_df[feature_cols].to_numpy(),
                           data=shap_df[[f"x_{c}" for c in feature_cols]].to_numpy(),
                           feature_names=[display.get(c, c) for c in feature_cols])
    shap.plots.beeswarm(exp, max_display=len(feature_cols), show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_group_means(shap_df, groups, feature_cols, display, out_stem, title=""):
    """Mean SHAP value per feature for true positives vs true negatives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = shap_df.assign(group=groups)
    means = pd.DataFrame({g: d.loc[d["group"] == g, feature_cols].mean()
                          for g in ("TP", "TN")})
    order = means.abs().max(axis=1).sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.35 * len(order) + 1.5), sharey=True)
    for ax, g, lab in zip(axes, ("TP", "TN"), ("True positives (responders)",
                                                "True negatives (non-responders)")):
        v = means.loc[order, g]
        ax.barh([display.get(c, c) for c in order], v,
                color=["#d62728" if x > 0 else "#1f77b4" for x in v])
        ax.axvline(0, c="black", lw=0.6)
        ax.set_title(f"{lab}  (n={int((d['group'] == g).sum())})", fontsize=9)
        ax.set_xlabel("mean SHAP value (log-odds)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    plt.close(fig)
    return means
