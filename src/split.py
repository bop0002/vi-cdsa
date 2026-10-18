import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED


def group_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(df, df["label"], df["group_id"])]
    split = np.full(len(df), "train", dtype=object)
    split[folds[1]] = "val"
    split[folds[0]] = "test"
    df["split"] = split
    return df


def match_train_size(a: pd.DataFrame, b: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Sample A's train so each class has exactly B's train count (same size and label prior)."""
    target = b.loc[b["split"] == "train", "label"].value_counts()
    train = a[a["split"] == "train"]
    have = train["label"].value_counts().reindex(target.index, fill_value=0)
    short = {int(k): (int(have[k]), int(n)) for k, n in target.items() if have[k] < n}
    if short:
        raise ValueError(f"A train lacks rows to match B per class (have, need): {short}")
    parts = [train[train["label"] == k].sample(n=int(n), random_state=seed) for k, n in target.items()]
    return pd.concat(parts + [a[a["split"] != "train"]]).reset_index(drop=True)


def balanced_subset(test_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    n = int(test_df["label"].value_counts().min())
    return (test_df.groupby("label", group_keys=False)
            .sample(n=n, random_state=seed)
            .reset_index(drop=True))
