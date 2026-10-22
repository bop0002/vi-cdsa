"""Load → filter → dedup → cross-domain overlap removal, shared by build_dataset and relabel."""
from typing import Callable

import pandas as pd

from src.filters import add_text_columns, apply_row_filters, dedup_with_conflicts, remove_cross_overlap
from src.io_sources import load_book, load_book_extra, load_foody, load_res1

LABEL_SOURCE = {"tiki_book": "rating", "res1": "dataset", "tiki_book_extra": "rating", "foody": "rating"}


def _clean(df: pd.DataFrame, relabels: pd.DataFrame | None) -> pd.DataFrame:
    from src.relabel import apply_relabels  # local import: relabel itself uses clean_sources

    df = df.assign(label_source=df["source"].map(LABEL_SOURCE))
    if relabels is not None:
        df = apply_relabels(df, relabels)
    return apply_row_filters(add_text_columns(df))


NEAR_DUP_WORDS = 8  # res1 holds trimmed/edited copies of Foody reviews, so exact keys miss some duplicates


def key_prefix(keys: pd.Series, n: int = NEAR_DUP_WORDS) -> pd.Series:
    return keys.str.split().str[:n].str.join(" ")


def _merge_extra(base: pd.DataFrame, extra: pd.DataFrame, other_keys: set) -> pd.DataFrame:
    """Append extra rows whose text is new to both domains (and not a near-copy of a base row);
    the base rows are never touched."""
    extra = dedup_with_conflicts(extra)
    extra = extra[~extra["id"].isin(set(base["id"]))]  # same review already in the base source
    extra = extra[~extra["key"].isin(set(base["key"]) | other_keys)]
    extra = extra[~key_prefix(extra["key"]).isin(set(key_prefix(base["key"])))]
    return pd.concat([base, extra], ignore_index=True)


def clean_sources(relabels: pd.DataFrame | None = None,
                  step: Callable[[str, pd.DataFrame, pd.DataFrame], None] = lambda *_: None,
                  extra: bool = True):
    """Return cleaned (A, B). If `relabels` (id, new_label) is given, content-based labels override the originals.

    The original sources go through exactly the same steps as before, so their splits do not move;
    the extra train-only sources (orig_split="train_extra") are cleaned on their own and appended last.
    """
    a, b = load_book(), load_res1()
    step("load", a, b)
    a, b = _clean(a, relabels), _clean(b, relabels)
    step("relabel+row_filters", a, b)
    a = dedup_with_conflicts(a)
    b = dedup_with_conflicts(b, priority_col="orig_split", priority={"train": 0, "val": 1, "test": 2})
    step("dedup", a, b)
    a, b = remove_cross_overlap(a, b)
    step("cross_overlap", a, b)
    if extra:
        ea, eb = _clean(load_book_extra(), relabels), _clean(load_foody(), relabels)
        step("extra_loaded", ea, eb)
        a, b = (_merge_extra(a, ea, set(b["key"]) | set(eb["key"])),
                _merge_extra(b, eb, set(a["key"]) | set(ea["key"])))
        step("with_extra", a, b)
    return a, b
