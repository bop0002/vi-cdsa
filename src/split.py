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
