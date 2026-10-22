from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PROCESSED, SEED

# A's val/test products, frozen from the first content-relabelled build: StratifiedGroupKFold reshuffles
# almost every fold when a single row changes, and the wave-2 export skipped exactly these products.
HELDOUT_GROUPS = PROCESSED / "A_heldout_groups.csv"


def frozen_group_split(df: pd.DataFrame, path=HELDOUT_GROUPS) -> pd.DataFrame:
    """Assign val/test by the frozen product list (everything else is train); fall back to group_split."""
    if not Path(path).exists():
        return group_split(df)
    held = pd.read_csv(path).set_index("group_id")["split"]
    return df.reset_index(drop=True).assign(split=lambda d: d["group_id"].map(held).fillna("train"))


def group_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(df, df["label"], df["group_id"])]
    split = np.full(len(df), "train", dtype=object)
    split[folds[1]] = "val"
    split[folds[0]] = "test"
    df["split"] = split
    return df


def add_extra_to_train(split_df: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Put train-only extra rows into train, dropping any whose group (product) is already in val/test."""
    held = set(split_df.loc[split_df["split"] != "train", "group_id"])
    extra = extra[~extra["group_id"].isin(held)].assign(split="train")
    return pd.concat([split_df, extra], ignore_index=True)


def _take(train: pd.DataFrame, label: int, n: int, seed: int) -> pd.DataFrame:
    """n rows of one class, original-source rows first (extra rows only fill the remainder)."""
    rows = train[train["label"] == label]
    extra = rows["orig_split"].eq("train_extra") if "orig_split" in rows else pd.Series(False, index=rows.index)
    # shuffled separately, so the original rows picked do not depend on how much extra data exists
    return pd.concat([rows[~extra].sample(frac=1, random_state=seed),
                      rows[extra].sample(frac=1, random_state=seed)]).head(n)


def match_train_size(a: pd.DataFrame, b: pd.DataFrame, cap: dict | None = None,
                     seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give A and B train the same count per class: min(A, B, cap) for each class. Val/test are untouched."""
    ta, tb = a[a["split"] == "train"], b[b["split"] == "train"]
    have_a, have_b = ta["label"].value_counts(), tb["label"].value_counts()
    target = {k: min(int(have_a.get(k, 0)), int(have_b.get(k, 0)), (cap or {}).get(k, 10**9)) for k in (0, 1, 2)}
    out = []
    for df, train in ((a, ta), (b, tb)):
        parts = [_take(train, k, n, seed) for k, n in target.items()]
        out.append(pd.concat(parts + [df[df["split"] != "train"]]).reset_index(drop=True))
    return out[0], out[1]


def balanced_subset(test_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    n = int(test_df["label"].value_counts().min())
    return (test_df.groupby("label", group_keys=False)
            .sample(n=n, random_state=seed)
            .reset_index(drop=True))


def _sample_counts(df: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    """Exactly counts[k] random rows of each class k."""
    return pd.concat([df[df["label"] == k].sample(n=int(n), random_state=seed) for k, n in counts.items()])


def match_split_counts(a: pd.DataFrame, b: pd.DataFrame, splits=("val", "test"), seed: int = SEED) -> pd.DataFrame:
    """Sample A's given splits down to B's count per class, so both domains have the same split sizes."""
    out = [a[~a["split"].isin(splits)]]
    for s in splits:
        part = a[a["split"] == s]
        want = b.loc[b["split"] == s, "label"].value_counts().to_dict()
        have = part["label"].value_counts()
        if short := {k: (int(have.get(k, 0)), n) for k, n in want.items() if have.get(k, 0) < n}:
            raise ValueError(f"A {s} lacks rows to match B per class (have, need): {short}")
        out.append(_sample_counts(part, want, seed))
    return pd.concat(out).reset_index(drop=True)


def _folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    fold = np.zeros(len(df), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(sgkf.split(df, df["label"], df["group_id"])):
        fold[idx] = i
    return df.assign(fold=fold)


def full_split(a: pd.DataFrame, b: pd.DataFrame, shares=None, n_folds: int = 10,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-split both domains from scratch so every split has the same size and class counts in A and B.

    Each domain is cut into group-stratified folds (one fold = test, one = val, the rest = train), then every
    split is sampled to shares[split] * T rows with the class prior of min(A, B) per class. The fold pair and T
    are chosen to make T as large as possible; rows left over in the chosen folds are dropped (moving them
    would leak a product/restaurant across splits)."""
    shares = shares or {"train": 0.8, "val": 0.1, "test": 0.1}
    pool = pd.concat([a["label"].value_counts(), b["label"].value_counts()], axis=1).min(axis=1)
    prior = pool / pool.sum()
    fa, fb = _folds(a, n_folds, seed), _folds(b, n_folds, seed)

    def best_pair(f: pd.DataFrame) -> tuple[float, int, int]:
        c = pd.crosstab(f["fold"], f["label"]).reindex(columns=prior.index, fill_value=0)
        best = (-1.0, 0, 1)
        for te in c.index:
            for va in c.index:
                if te == va:
                    continue
                parts = {"test": c.loc[te], "val": c.loc[va], "train": c.drop([te, va]).sum()}
                t = min(parts[s][k] / (shares[s] * prior[k]) for s in parts for k in prior.index)
                best = max(best, (t, int(te), int(va)))
        return best

    (ta, te_a, va_a), (tb, te_b, va_b) = best_pair(fa), best_pair(fb)
    total = min(ta, tb)
    counts = {s: {int(k): int(prior[k] * shares[s] * total) for k in prior.index} for s in shares}
    out = []
    for f, te, va in ((fa, te_a, va_a), (fb, te_b, va_b)):
        f = f.assign(split=np.where(f["fold"] == te, "test", np.where(f["fold"] == va, "val", "train")))
        out.append(pd.concat([_sample_counts(f[f["split"] == s], counts[s], seed) for s in shares])
                   .drop(columns="fold").reset_index(drop=True))
    return out[0], out[1]
