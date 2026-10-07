"""Patient-level stratified cross-validation splits shared by all branches.

One split table per seed. For every outer fold ``k`` each patient has a role:
``test`` (held-out fold k), ``val`` (inner validation, used only for early
stopping / checkpoint selection) or ``train``. Image models of both branches
and the fusion stage all read the same table, so every model sees exactly the
same data distribution and no model ever accesses the labels of its held-out
patients.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def make_splits(patients: pd.DataFrame, n_folds: int, val_frac: float, seed: int) -> pd.DataFrame:
    pl = patients[["patient_id", "label"]].drop_duplicates().reset_index(drop=True)
    if pl["label"].value_counts().min() < n_folds:
        raise ValueError("Smallest class has fewer patients than folds")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(pl["patient_id"], pl["label"])):
        tr = pl.iloc[tr_idx].reset_index(drop=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + fold)
        tr2, va = next(sss.split(tr["patient_id"], tr["label"]))
        for idx, role in ((tr2, "train"), (va, "val")):
            for _, r in tr.iloc[idx].iterrows():
                rows.append((seed, fold, r["patient_id"], int(r["label"]), role))
        for _, r in pl.iloc[te_idx].iterrows():
            rows.append((seed, fold, r["patient_id"], int(r["label"]), "test"))
    return pd.DataFrame(rows, columns=["seed", "fold", "patient_id", "label", "role"])


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str})
    return df


def fold_patients(split: pd.DataFrame, fold: int, role: str) -> pd.DataFrame:
    s = split[(split["fold"] == fold) & (split["role"] == role)]
    return s[["patient_id", "label"]].reset_index(drop=True)


def test_fold_of(split: pd.DataFrame) -> pd.DataFrame:
    """patient_id -> outer fold in which the patient is held out."""
    return split[split["role"] == "test"][["patient_id", "label", "fold"]].reset_index(drop=True)
