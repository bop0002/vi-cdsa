import re

import pandas as pd

from src.config import MIN_WORDS

LEFTOVER_TITLE = re.compile(r"^(rất không hài lòng|không hài lòng|cực k[ìỳ] hài lòng|hài lòng|bình thường)$")
SPLITS = ("train", "val", "test")


def _min_class(df: pd.DataFrame) -> int:
    return int(df["label"].value_counts().reindex([0, 1, 2], fill_value=0).min())


def run_checks(frames: dict, min_test: int = 250, min_train: int = 1000) -> list[str]:
    errs = []
    for name, df in frames.items():
        if (df["n_words"] < MIN_WORDS).any():
            errs.append(f"{name}: n_words < {MIN_WORDS}")
        if not df["label"].isin([0, 1, 2]).all():
            errs.append(f"{name}: invalid label")
        if df["text_clean"].str.match(LEFTOVER_TITLE).any():
            errs.append(f"{name}: auto-title text left")
        if df["key"].duplicated().any():
            errs.append(f"{name}: duplicate text")
    for d in "AB":
        keys = {s: set(frames[f"{d}_{s}"]["key"]) for s in SPLITS}
        for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
            if keys[x] & keys[y]:
                errs.append(f"{d}: overlap {x}/{y}")
        if _min_class(frames[f"{d}_test"]) < min_test:
            errs.append(f"{d}_test: class < {min_test}")
        if _min_class(frames[f"{d}_train"]) < min_train:
            errs.append(f"{d}_train: class < {min_train}")
        allf = pd.concat([frames[f"{d}_{s}"] for s in SPLITS])
        if len(allf) and allf["group_id"].value_counts(normalize=True).iloc[0] > 0.02:
            errs.append(f"{d}: group_id share > 2%")
    ka = set().union(*(frames[f"A_{s}"]["key"] for s in SPLITS))
    kb = set().union(*(frames[f"B_{s}"]["key"] for s in SPLITS))
    if ka & kb:
        errs.append(f"overlap A/B: {len(ka & kb)} texts")
    if len(frames["A_train"]) != len(frames["B_train"]):
        errs.append(f"train size A={len(frames['A_train'])} != B={len(frames['B_train'])}")
    return errs
