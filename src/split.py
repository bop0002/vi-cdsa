import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

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
    n = int((b["split"] == "train").sum())
    train = a[a["split"] == "train"]
    if n < len(train):
        train, _ = train_test_split(train, train_size=n, stratify=train["label"], random_state=seed)
    return pd.concat([train, a[a["split"] != "train"]]).reset_index(drop=True)


def balanced_subset(test_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    n = int(test_df["label"].value_counts().min())
    return (test_df.groupby("label", group_keys=False)
            .sample(n=n, random_state=seed)
            .reset_index(drop=True))
