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


def balanced_subset(test_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    n = int(test_df["label"].value_counts().min())
    return (test_df.groupby("label", group_keys=False)
            .sample(n=n, random_state=seed)
            .reset_index(drop=True))


def _sample_counts(df: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    """Exactly counts[k] random rows of each class k."""
    return pd.concat([df[df["label"] == k].sample(n=int(n), random_state=seed) for k, n in counts.items()])


def _folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    fold = np.zeros(len(df), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(sgkf.split(df, df["label"], df["group_id"])):
        fold[idx] = i
    return df.assign(fold=fold)


def full_split(a: pd.DataFrame, b: pd.DataFrame, shares=None, n_folds: int = 10,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-split both domains from scratch: each keeps its own class prior (natural ratio), and every split has the
    same size in A and B (the larger domain is sampled down).

    Each domain is cut into group-stratified folds (one fold = test, one = val, the rest = train), then every
    split is sampled to shares[split] * T rows with that domain's class prior. The fold pair and T are chosen to
    make T as large as possible; rows left over in the chosen folds are dropped (moving them would leak a
    product/restaurant across splits)."""
    shares = shares or {"train": 0.8, "val": 0.1, "test": 0.1}
    priors = [d["label"].value_counts() / len(d) for d in (a, b)]
    fa, fb = _folds(a, n_folds, seed), _folds(b, n_folds, seed)

    def best_pair(f: pd.DataFrame, prior: pd.Series) -> tuple[float, int, int]:
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

    (ta, te_a, va_a), (tb, te_b, va_b) = best_pair(fa, priors[0]), best_pair(fb, priors[1])
    total = min(ta, tb)
    counts = [{s: {int(k): int(p[k] * shares[s] * total) for k in p.index} for s in shares} for p in priors]
    for s in shares:  # per-class flooring can leave sizes off by a row or two: trim the majority class
        n = min(sum(c[s].values()) for c in counts)
        for c, p in zip(counts, priors):
            c[s][int(p.idxmax())] -= sum(c[s].values()) - n
    out = []
    for f, te, va, cnt in ((fa, te_a, va_a, counts[0]), (fb, te_b, va_b, counts[1])):
        f = f.assign(split=np.where(f["fold"] == te, "test", np.where(f["fold"] == va, "val", "train")))
        out.append(pd.concat([_sample_counts(f[f["split"] == s], cnt[s], seed) for s in shares])
                   .drop(columns="fold").reset_index(drop=True))
    return out[0], out[1]
