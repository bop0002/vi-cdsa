from pathlib import Path

import pandas as pd

from src.config import PROCESSED

DOMAINS = ("A", "B")
SPLITS = ("train", "val", "test", "test_balanced")


def load_split(domain: str, split: str, processed_dir: Path = PROCESSED) -> pd.DataFrame:
    """Đọc một tập đã xử lý, ví dụ load_split("A", "train"). A = Tiki book, B = res1."""
    if domain not in DOMAINS or split not in SPLITS:
        raise ValueError(f"domain must be one of {DOMAINS}, split one of {SPLITS}; got {domain!r}, {split!r}")
    return pd.read_parquet(Path(processed_dir) / f"{domain}_{split}.parquet")
