import pandas as pd
import pytest

from src.dataset import load_split


def _write(tmp_path, name, n=3):
    pd.DataFrame({"id": [f"x{i}" for i in range(n)], "label": [0, 1, 2][:n],
                  "text_ml": ["a b c"] * n, "text_dl": ["a b c"] * n}).to_parquet(tmp_path / f"{name}.parquet")


def test_load_split_reads_named_file(tmp_path):
    _write(tmp_path, "A_train")
    df = load_split("A", "train", processed_dir=tmp_path)
    assert len(df) == 3 and list(df.label) == [0, 1, 2]


def test_load_split_balanced_test(tmp_path):
    _write(tmp_path, "B_test_balanced")
    _write(tmp_path, "A_train_balanced")
    assert len(load_split("A", "train_balanced", processed_dir=tmp_path)) == 3
    assert len(load_split("B", "test_balanced", processed_dir=tmp_path)) == 3


@pytest.mark.parametrize("domain, split", [("C", "train"), ("A", "dev"), ("a", "train")])
def test_load_split_rejects_unknown(domain, split, tmp_path):
    with pytest.raises(ValueError):
        load_split(domain, split, processed_dir=tmp_path)
