"""Load → filter → dedup → cross-domain overlap removal, shared by build_dataset and relabel."""
from typing import Callable

import pandas as pd

from src.filters import add_text_columns, apply_row_filters, dedup_with_conflicts, remove_cross_overlap
from src.io_sources import load_book, load_res1

LABEL_SOURCE = {"tiki_book": "rating", "res1": "dataset"}


def clean_sources(relabels: pd.DataFrame | None = None,
                  step: Callable[[str, pd.DataFrame, pd.DataFrame], None] = lambda *_: None):
    """Return cleaned (A, B). If `relabels` (id, new_label) is given, content-based labels override the originals."""
    from src.relabel import apply_relabels  # local import: relabel itself uses clean_sources

    a, b = load_book(), load_res1()
    a = a.assign(label_source=a["source"].map(LABEL_SOURCE))
    b = b.assign(label_source=b["source"].map(LABEL_SOURCE))
    step("load", a, b)
    if relabels is not None:
        a, b = apply_relabels(a, relabels), apply_relabels(b, relabels)
        step("relabel", a, b)
    a, b = apply_row_filters(add_text_columns(a)), apply_row_filters(add_text_columns(b))
    step("row_filters", a, b)
    a = dedup_with_conflicts(a)
    b = dedup_with_conflicts(b, priority_col="orig_split", priority={"train": 0, "val": 1, "test": 2})
    step("dedup", a, b)
    a, b = remove_cross_overlap(a, b)
    step("cross_overlap", a, b)
    return a, b
