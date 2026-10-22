import json
from pathlib import Path

import pandas as pd

from src.config import RAW, RES1_LABEL_MAP

RES1_FILES = {
    "train": "train-00000-of-00001.parquet",
    "val": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


# Extra train-only data (2026-10-20): more Tiki book reviews (A) and Foody reviews (B, same site as res1).
EXTRA_SPLIT = "train_extra"
TIKI_BOOKS_DIR = RAW / "tiki_books"
FOODY_FILE = RAW / "foody" / "reviews.jsonl"


def rating_to_label(r: int) -> int:
    return 0 if r <= 2 else (1 if r == 3 else 2)


def foody_is_gray(r: float) -> bool:
    """Foody mixes a 10-point score with a 5-star scale (integer 1–5), so 3.0–7.5 cannot be trusted either way."""
    return 3 <= r <= 7.5


def foody_label(r: float) -> int:
    """Provisional label from the score; gray-zone rows are relabelled by content (see src.relabel)."""
    return 0 if r < 4 else (2 if r > 6 else 1)


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


def _book_frame(df: pd.DataFrame, source: str, orig_split) -> pd.DataFrame:
    df = df[df["content"].notna() & df["rating"].notna()]
    return pd.DataFrame({
        "id": "tiki_" + df["comment_id"].astype(str),
        "domain": "A",
        "source": source,
        "group_id": "tiki_" + df["product_id"].astype(str),
        "rating": df["rating"].astype(int),
        "label": df["rating"].astype(int).map(rating_to_label).astype(int),
        "text_raw": df["content"].astype(str),
        "orig_split": orig_split,
    })


def load_book_extra(d: Path = TIKI_BOOKS_DIR) -> pd.DataFrame:
    """Train-only Tiki book reviews: the 1–4★ crawl (reviews.jsonl) plus the book part of the earlier
    multi-category crawl (snapshot_books.parquet, content = "<title>. <body>"; the auto title is stripped later)."""
    d = Path(d)
    parts = []
    if (d / "reviews.jsonl").exists():
        parts.append(pd.read_json(d / "reviews.jsonl", lines=True, dtype={"comment_id": str, "product_id": str}))
    if (d / "snapshot_books.parquet").exists():
        parts.append(pd.read_parquet(d / "snapshot_books.parquet"))
    if not parts:
        return _book_frame(pd.DataFrame(columns=["comment_id", "product_id", "rating", "content"]),
                           "tiki_book_extra", EXTRA_SPLIT)
    df = pd.concat([p[["comment_id", "product_id", "rating", "content"]] for p in parts], ignore_index=True)
    return _book_frame(df, "tiki_book_extra", EXTRA_SPLIT).drop_duplicates("id").reset_index(drop=True)


def load_foody(path: Path = FOODY_FILE) -> pd.DataFrame:
    path = Path(path)
    rows = [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []
    df = pd.DataFrame(rows, columns=["review_id", "res_id", "rating", "content"])
    df = df[df["content"].notna() & df["rating"].notna()]
    out = pd.DataFrame({
        "id": "foody_" + df["review_id"].astype(str),
        "domain": "B",
        "source": "foody",
        "group_id": "foody_" + df["res_id"].astype(str),
        "rating": df["rating"].astype(float),
        "label": df["rating"].astype(float).map(foody_label).astype(int),
        "text_raw": df["content"].astype(str),
        "orig_split": EXTRA_SPLIT,
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
