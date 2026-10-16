from pathlib import Path

import pandas as pd

from src.config import RAW, RES1_LABEL_MAP

RES1_FILES = {
    "train": "train-00000-of-00001.parquet",
    "val": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def rating_to_label(r: int) -> int:
    return 0 if r <= 2 else (1 if r == 3 else 2)


def load_book(path: Path = RAW / "book_comment_tiki.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["content"].notna()]
    out = pd.DataFrame({
        "id": "tiki_" + df["comment_id"].astype(str),
        "domain": "A",
        "source": "tiki_book",
        "group_id": "tiki_" + df["product_id"].astype(str),
        "rating": df["rating"].astype(int),
        "label": df["rating"].map(rating_to_label).astype(int),
        "text_raw": df["content"].astype(str),
        "orig_split": None,
    })
    return out.drop_duplicates("id").reset_index(drop=True)


def load_res1(d: Path = RAW / "res1") -> pd.DataFrame:
    parts = [pd.read_parquet(Path(d) / f).assign(orig_split=s) for s, f in RES1_FILES.items()]
    df = pd.concat(parts, ignore_index=True)
    ids = "res1_" + df["review_id"].astype(str)
    out = pd.DataFrame({
        "id": ids,
        "domain": "B",
        "source": "res1",
        "group_id": ids,
        "rating": -1,
        "label": df["label"].map(RES1_LABEL_MAP).astype(int),
        "text_raw": df["review"].astype(str),
        "orig_split": df["orig_split"],
    })
    return out.drop_duplicates("id").reset_index(drop=True)
