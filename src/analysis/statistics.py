"""Statistical analyses on top of the out-of-fold predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import kruskal


def logit_or(df: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.DataFrame:
    """Odds ratios (95% CI, Wald p) from a logistic regression via statsmodels."""
    import warnings

    import statsmodels.api as sm
    d = df[[outcome] + predictors].dropna()
    x = sm.add_constant(d[predictors].astype(float), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(d[outcome].astype(float), x).fit(disp=0, maxiter=200)
            ci = res.conf_int()
            rows = [{"variable": p, "n": len(d), "OR": np.exp(res.params[p]),
                     "OR_lo": np.exp(ci.loc[p, 0]), "OR_hi": np.exp(ci.loc[p, 1]),
                     "p_value": res.pvalues[p], "note": ""} for p in predictors]
    except Exception as e:  # singular design / (quasi-)perfect separation
        rows = [{"variable": p, "n": len(d), "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan,
                 "p_value": np.nan, "note": f"not estimable ({type(e).__name__})"}
                for p in predictors]
    return pd.DataFrame(rows)


def _log_table_prob(t: np.ndarray) -> float:
    """log P(table | margins) under the multivariate hypergeometric null."""
    r, c, n = t.sum(1), t.sum(0), t.sum()
    return (gammaln(r + 1).sum() + gammaln(c + 1).sum() - gammaln(n + 1)
            - gammaln(t + 1).sum())


def fisher_freeman_halton(groups, values, n_perm: int = 20000, seed: int = 0) -> float:
    """Fisher's exact test generalised to r x c tables (Monte-Carlo p-value).

    For 2x2 tables the exact scipy implementation is used.
    """
    from scipy.stats import fisher_exact
    groups, values = np.asarray(groups), np.asarray(values)
    gl, vl = np.unique(groups), np.unique(values)
    if len(vl) < 2 or len(gl) < 2:
        return np.nan

    def table(g):
        return np.array([[np.sum((g == a) & (values == b)) for b in vl] for a in gl])

    obs = table(groups)
    if obs.shape == (2, 2):
        return float(fisher_exact(obs)[1])
    lp_obs = _log_table_prob(obs)
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_perm):
        if _log_table_prob(table(rng.permutation(groups))) <= lp_obs + 1e-9:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def error_group_analysis(df: pd.DataFrame, group_col: str, binary_cols, continuous_cols,
                         order=("TP", "TN", "FP", "FN")) -> pd.DataFrame:
    """Compare clinical variables across TP / TN / FP / FN prediction groups."""
    rows = []
    for c in binary_cols:
        d = df[[group_col, c]].dropna()
        row = {"variable": c, "test": "Fisher-Freeman-Halton"}
        for g in order:
            s = d.loc[d[group_col] == g, c]
            row[g] = f"{int(s.sum())}/{len(s)} ({100 * s.mean():.1f}%)" if len(s) else "-"
        row["p_value"] = fisher_freeman_halton(d[group_col], d[c])
        rows.append(row)
    for c in continuous_cols:
        d = df[[group_col, c]].dropna()
        row = {"variable": c, "test": "Kruskal-Wallis"}
        samples = []
        for g in order:
            s = d.loc[d[group_col] == g, c]
            samples.append(s.values)
            row[g] = (f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"
                      if len(s) else "-")
        samples = [s for s in samples if len(s)]
        row["p_value"] = kruskal(*samples).pvalue if len(samples) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
