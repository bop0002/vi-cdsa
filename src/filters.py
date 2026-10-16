import re

import pandas as pd

from src.config import MIN_WORDS
from src.text_norm import clean_text

VN_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
_SEP = re.compile(r"[\s,]+")


def dedup_key(t: str) -> str:
    return _SEP.sub(" ", t).strip()


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text_raw"].map(clean_text)
    df["n_words"] = df["text_clean"].map(lambda t: len(_SEP.sub(" ", t).split()))
    df["key"] = df["text_clean"].map(dedup_key)
    return df


def apply_row_filters(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df["n_words"] >= MIN_WORDS) & df["text_clean"].str.contains(VN_CHARS)
    return df[keep].reset_index(drop=True)


def dedup_with_conflicts(df: pd.DataFrame, priority_col: str | None = None,
                         priority: dict | None = None) -> pd.DataFrame:
    n_labels = df.groupby("key")["label"].transform("nunique")
    df = df[n_labels == 1]
    if priority_col:
        df = df.assign(_p=df[priority_col].map(priority)).sort_values(["_p", "id"]).drop(columns="_p")
    else:
        df = df.sort_values("id")
    return df.drop_duplicates("key").reset_index(drop=True)


def remove_cross_overlap(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = set(a["key"]) & set(b["key"])
    return (a[~a["key"].isin(shared)].reset_index(drop=True),
            b[~b["key"].isin(shared)].reset_index(drop=True))
