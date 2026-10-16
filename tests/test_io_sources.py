import pandas as pd
from src.io_sources import load_book, load_res1, rating_to_label


def test_rating_to_label():
    assert [rating_to_label(r) for r in [1, 2, 3, 4, 5]] == [0, 0, 1, 2, 2]


def test_load_book_maps_schema_and_drops_empty(tmp_path):
    p = tmp_path / "book.csv"
    pd.DataFrame({
        "product_id": [10, 10, 11],
        "comment_id": [1, 2, 3],
        "title": ["Cực kì hài lòng", "Bình thường", "Hài lòng"],
        "thank_count": [0, 0, 0],
        "customer_id": [7, 8, 9],
        "rating": [5, 3, 4],
        "content": ["Sách hay lắm nha", None, "Giao hàng nhanh, sách đẹp"],
    }).to_csv(p, index=False, encoding="utf-8-sig")
    df = load_book(p)
    assert list(df.id) == ["tiki_1", "tiki_3"]
    assert list(df.group_id) == ["tiki_10", "tiki_11"]
    assert list(df.label) == [2, 2]
    assert set(df.domain) == {"A"}
    assert "title" not in df.columns
    assert df.orig_split.isna().all()


def test_load_res1_remaps_labels_and_splits(tmp_path):
    for f, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        pd.DataFrame({"review_id": [1, 2, 3], "review": ["a b c"] * 3, "label": [0, 1, 2]}) \
            .assign(review_id=lambda x: x.review_id + {"train": 0, "val": 10, "test": 20}[split]) \
            .to_parquet(tmp_path / f"{f}-00000-of-00001.parquet")
    df = load_res1(tmp_path)
    assert len(df) == 9
    assert list(df[df.orig_split == "train"].label) == [0, 2, 1]
    assert set(df.orig_split) == {"train", "val", "test"}
    assert set(df.domain) == {"B"}
    assert (df.rating == -1).all()
    assert (df.group_id == df.id).all()
