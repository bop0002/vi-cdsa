import pandas as pd
import json

from src.io_sources import (foody_is_gray, foody_label, load_book, load_book_extra, load_foody, load_res1,
                            rating_to_label)


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


def test_foody_label_and_gray_range():
    assert [foody_label(r) for r in [1.0, 3.9, 4.0, 6.0, 6.1, 9.5]] == [0, 0, 1, 1, 2, 2]
    assert [foody_is_gray(r) for r in [2.9, 3.0, 5.0, 7.5, 7.6]] == [False, True, True, True, False]


def test_load_foody_is_train_extra_grouped_by_restaurant(tmp_path):
    p = tmp_path / "reviews.jsonl"
    rows = [{"review_id": 1, "res_id": 9, "rating": 8.4, "content": "ngon lắm"},
            {"review_id": 1, "res_id": 9, "rating": 8.4, "content": "ngon lắm"},
            {"review_id": 2, "res_id": 9, "rating": 2.0, "content": None}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    df = load_foody(p)
    assert list(df.id) == ["foody_1"] and list(df.group_id) == ["foody_9"]
    assert list(df.label) == [2] and set(df.orig_split) == {"train_extra"} and set(df.domain) == {"B"}


def test_load_book_extra_merges_crawl_and_snapshot(tmp_path):
    (tmp_path / "reviews.jsonl").write_text(json.dumps(
        {"comment_id": 5, "product_id": 7, "rating": 3, "title": "Bình thường", "content": "tạm được"}), encoding="utf-8")
    pd.DataFrame({"comment_id": ["5", "6"], "product_id": ["7", "8"], "rating": [3, 4],
                  "content": ["tạm được", "Hài lòng. sách ổn"]}).to_parquet(tmp_path / "snapshot_books.parquet")
    df = load_book_extra(tmp_path)
    assert sorted(df.id) == ["tiki_5", "tiki_6"]
    assert set(df.orig_split) == {"train_extra"} and set(df.source) == {"tiki_book_extra"}
    assert sorted(df.label) == [1, 2]
