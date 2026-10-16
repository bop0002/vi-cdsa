import pandas as pd
from src.filters import (add_text_columns, apply_row_filters, dedup_key,
                         dedup_with_conflicts, remove_cross_overlap)


def frame(texts, labels, splits=None):
    df = pd.DataFrame({"id": [f"x{i}" for i in range(len(texts))], "text_raw": texts, "label": labels})
    if splits:
        df["orig_split"] = splits
    return add_text_columns(df)


def test_dedup_key_ignores_commas():
    assert dedup_key("sách hay, giao nhanh") == dedup_key("sách hay giao nhanh")


def test_row_filters():
    df = frame(["Cực kì hài lòng", "tốt lắm", "very good product here", "Sách rất hay và ý nghĩa"], [2, 2, 2, 2])
    out = apply_row_filters(df)
    assert list(out.id) == ["x3"]


def test_dedup_drops_conflicts_and_duplicates():
    df = frame(["sách rất hay", "Sách rất hay!", "bìa bị rách nát", "bìa bị rách nát"], [2, 2, 0, 1])
    out = dedup_with_conflicts(df)
    assert list(out.key) == ["sách rất hay"]


def test_dedup_priority_keeps_train_copy():
    df = frame(["quán ăn rất ngon", "quán ăn rất ngon"], [2, 2], splits=["test", "train"])
    out = dedup_with_conflicts(df, priority_col="orig_split", priority={"train": 0, "val": 1, "test": 2})
    assert list(out.orig_split) == ["train"]


def test_remove_cross_overlap():
    a = frame(["sản phẩm rất tốt", "sách hay quá đi"], [2, 2])
    b = frame(["sản phẩm rất tốt", "món ăn rất ngon"], [2, 2])
    a2, b2 = remove_cross_overlap(a, b)
    assert list(a2.key) == ["sách hay quá đi"]
    assert list(b2.key) == ["món ăn rất ngon"]
