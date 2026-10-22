"""Collect the book reviews from the earlier multi-category Tiki crawl (cumulative snapshot files)
into data/raw/tiki_books/snapshot_books.parquet. Non-book categories are left out (Domain A = books).

    python -X utf8 scripts/extract_tiki_snapshot_books.py ../tiki_reviews
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "tiki_books" / "snapshot_books.parquet"


def main(src: str) -> int:
    df = pd.concat([pd.read_parquet(p) for p in sorted(Path(src).glob("tiki_reviews_*.parquet"))])
    df = df.drop_duplicates("review_id")
    book = df["category"].str.contains("sach|truyen", regex=True) & (df["category"] != "may-doc-sach")
    df = df[book].assign(product_id=lambda d: d["product_url"].str.extract(r"p(\d+)\.html")[0])
    df = df[df["product_id"].notna()]
    # the old scraper stored "<title>. <body>" (title only when there is no body); keep the body, like book_comment
    df["content"] = df["content"].str.split(". ", n=1, regex=False).str[1]
    df = df[df["content"].notna()]
    out = df.rename(columns={"review_id": "comment_id"})[["comment_id", "product_id", "rating", "content", "category"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} book reviews -> {OUT}")
    print(out["rating"].value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
