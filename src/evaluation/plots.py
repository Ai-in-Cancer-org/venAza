"""Figures: per-fold + pooled ROC / PR curves and model concordance."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def _style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right" if "ROC" in title else "lower left", fontsize=7, frameon=False)


def roc_pr_figure(pred: pd.DataFrame, title: str, out_stem: Path, prob_col="prob") -> None:
    """Coloured curve per fold, bold black pooled out-of-fold curve."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    folds = sorted(pred["fold"].unique()) if "fold" in pred else []
    cmap = plt.get_cmap("tab10")
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        for i, k in enumerate(folds):
            d = pred[pred["fold"] == k]
            if d["label"].nunique() < 2:
                continue
            if kind == "roc":
                f, t, _ = roc_curve(d["label"], d[prob_col])
                ax.plot(f, t, color=cmap(i), lw=1, alpha=0.8,
                        label=f"Fold {k} (AUROC {roc_auc_score(d['label'], d[prob_col]):.2f})")
            else:
                pr, rc, _ = precision_recall_curve(d["label"], d[prob_col])
                ax.plot(rc, pr, color=cmap(i), lw=1, alpha=0.8,
                        label=f"Fold {k} (AP {average_precision_score(d['label'], d[prob_col]):.2f})")
        y, p = pred["label"], pred[prob_col]
        if kind == "roc":
            f, t, _ = roc_curve(y, p)
            ax.plot(f, t, color="black", lw=2.2, label=f"Pooled OOF (AUROC {roc_auc_score(y, p):.2f})")
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
            ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
            _style(ax, f"ROC - {title}")
        else:
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color="black", lw=2.2, label=f"Pooled OOF (AP {average_precision_score(y, p):.2f})")
            ax.axhline(np.mean(y), ls="--", color="grey", lw=0.8)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            _style(ax, f"PR - {title}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}_{kind}.{ext}", dpi=300)
        plt.close(fig)


def concordance_figure(df: pd.DataFrame, thr_a: float, thr_b: float, name_a: str,
                       name_b: str, out_stem: Path) -> pd.DataFrame:
    """Scatter of two models' probabilities, one panel per ground-truth class.

    Filled circles: both models agree at their Youden thresholds; crosses: only
    model A correct; triangles: only model B correct.
    """
    from scipy.stats import pearsonr
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (lab, title) in zip(axes, [(1, "Responders (CR/CRi)"), (0, "Non-responders")]):
        d = df[df["label"] == lab]
        ca = ((d["prob_a"] >= thr_a).astype(int) == lab)
        cb = ((d["prob_b"] >= thr_b).astype(int) == lab)
        agree = ca == cb
        ax.scatter(d.loc[agree, "prob_b"], d.loc[agree, "prob_a"], s=18, c="#4c72b0", alpha=0.7,
                   label="Agreement")
        m = ~agree & ca
        ax.scatter(d.loc[m, "prob_b"], d.loc[m, "prob_a"], marker="x", s=40, c="#c44e52",
                   label=f"{name_a} correct")
        m = ~agree & cb
        ax.scatter(d.loc[m, "prob_b"], d.loc[m, "prob_a"], marker="^", s=40, c="#55a868",
                   label=f"{name_b} correct")
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        ax.axhline(thr_a, c="grey", lw=0.5, ls=":"); ax.axvline(thr_b, c="grey", lw=0.5, ls=":")
        r, pv = pearsonr(d["prob_a"], d["prob_b"]) if len(d) > 2 else (np.nan, np.nan)
        n_dis = int((~agree).sum())
        ax.set_title(f"{title}\nPearson r={r:.3f} (p={pv:.2g}); disagreements={n_dis}", fontsize=9)
        ax.set_xlabel(f"{name_b} probability"); ax.set_ylabel(f"{name_a} probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7, frameon=False)
        stats.append({"class": title, "n": len(d), "pearson_r": r, "p_value": pv,
                      "disagreements": n_dis, f"only_{name_a}_correct": int((~agree & ca).sum()),
                      f"only_{name_b}_correct": int((~agree & cb).sum())})
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    plt.close(fig)
    return pd.DataFrame(stats)
