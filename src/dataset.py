from pathlib import Path

import pandas as pd

from src.config import PROCESSED

DOMAINS = ("A", "B")
SPLITS = ("train", "train_balanced", "val", "test", "test_balanced")
VARIANTS = ("full", "original", "extra")


def load_split(domain: str, split: str, variant: str = "full", processed_dir: Path = PROCESSED) -> pd.DataFrame:
    """Đọc một tập đã xử lý, ví dụ load_split("A", "train") hoặc load_split("B", "test", variant="original").
    A = sách Tiki, B = nhà hàng. Các phiên bản: xem src/build_dataset.py."""
    if domain not in DOMAINS or split not in SPLITS or variant not in VARIANTS:
        raise ValueError(f"domain must be one of {DOMAINS}, split one of {SPLITS}, variant one of {VARIANTS}; "
                         f"got {domain!r}, {split!r}, {variant!r}")
    return pd.read_parquet(Path(processed_dir) / variant / f"{domain}_{split}.parquet")
