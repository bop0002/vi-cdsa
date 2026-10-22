"""Build Domain A (Tiki book) and Domain B (restaurant) in one of three variants.

    python -m src.build_dataset --variant full       # default; writes data/processed/full/
    python -m src.build_dataset --variant all        # full, original and extra

- full:     all cleaned data (original + crawled), re-split 80/10/10 by product/restaurant;
            A and B have the same class counts in every split (prior = min(A, B) per class).
- original: original sources only (Tiki CSV, res1 with its own split); A val/test sampled to B's
            class counts, so both domains are 80/10/10 with identical counts.
- extra:    original val/test (A larger than B) + crawled data in train only, train Neg:Neu:Pos = 2:1:2.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.checks import run_checks
from src.config import OUTPUT_COLUMNS, PROCESSED, SEED, TRAIN_CLASS_CAP
from src.prepare import clean_sources
from src.relabel import RELABEL_FILE, drop_unannotated_gray
from src.segment import segment, to_ml
from src.io_sources import EXTRA_SPLIT
from src.split import (add_extra_to_train, balanced_subset, frozen_group_split, full_split, match_split_counts,
                       match_train_size)

VARIANTS = ("full", "original", "extra")
# splits whose class counts must be identical in A and B
EQUAL_SPLITS = {"full": ("train", "val", "test"), "original": ("train", "val", "test"), "extra": ("train",)}


def dist(df: pd.DataFrame) -> dict:
    return {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}


def describe(df: pd.DataFrame) -> dict:
    return {"n": len(df), "labels": dist(df),
            "words": df["n_words"].describe()[["mean", "50%", "min", "max"]].round(1).to_dict()}


def write_audit_if_missing(df: pd.DataFrame, path: Path) -> bool:
    """Write the label-audit CSV only if absent (the user annotates it by hand)."""
    path = Path(path)
    if path.exists():
        print(f"audit exists, not overwritten: {path}")
        return False
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def load_relabels(path: Path = RELABEL_FILE) -> pd.DataFrame | None:
    """Content-based labels from the relabel step (docs/relabel_runbook.md), or None if not produced yet."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")[["id", "new_label", "note"]]


def split_variant(variant: str, relabels, step) -> tuple[pd.DataFrame, pd.DataFrame]:
    if variant == "original":
        a, b = clean_sources(relabels, step, extra=False)
        a, b = frozen_group_split(a), b.assign(split=b["orig_split"])
        a, b = match_train_size(a, b)
        return match_split_counts(a, b, ("val", "test")), b
    a, b = clean_sources(relabels, step)
    if variant == "full":
        a, b = drop_unannotated_gray(a, b)
        step("drop_unannotated", a, b)
        return full_split(a, b)
    a_extra = a["orig_split"].eq(EXTRA_SPLIT)
    a = add_extra_to_train(frozen_group_split(a[~a_extra]), a[a_extra])
    b = b.assign(split=b["orig_split"].replace(EXTRA_SPLIT, "train"))
    step("split", a[a["split"] == "train"], b[b["split"] == "train"])
    return match_train_size(a, b, cap=TRAIN_CLASS_CAP)


def build(variant: str) -> int:
    out_dir = PROCESSED / variant
    log = {"variant": variant, "seed": SEED, "steps": [], "splits": {}}

    def step(name, a, b):
        log["steps"].append({"step": name, "A": {"n": len(a), "labels": dist(a)},
                             "B": {"n": len(b), "labels": dist(b)}})
        print(f"{name:<20} A={len(a):>6} {dist(a)} | B={len(b):>6} {dist(b)}")

    relabels = load_relabels()
    log["relabel_file"] = str(RELABEL_FILE) if relabels is not None else None
    print(f"relabels: {len(relabels)} rows from {RELABEL_FILE}" if relabels is not None
          else "relabels: none (using original labels)")
    a, b = split_variant(variant, relabels, step)
    step("final", a, b)

    frames = {}
    for d, df in (("A", a), ("B", b)):
        df = df.copy()
        df["text_dl"] = df["text_clean"].map(segment)
        df["text_ml"] = df["text_dl"].map(to_ml)
        for s in ("train", "val", "test"):
            frames[f"{d}_{s}"] = df[df["split"] == s].reset_index(drop=True)
        frames[f"{d}_train_balanced"] = balanced_subset(frames[f"{d}_train"])
        frames[f"{d}_test_balanced"] = balanced_subset(frames[f"{d}_test"])

    errors = run_checks(frames, equal_splits=EQUAL_SPLITS[variant])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        log["splits"][name] = describe(df)
    if not errors:
        for name, df in frames.items():
            df[OUTPUT_COLUMNS].to_parquet(out_dir / f"{name}.parquet", index=False)
        for d in "AB":
            audit = pd.concat([frames[f"{d}_train"], frames[f"{d}_val"], frames[f"{d}_test"]])
            sample = (audit.groupby("label", group_keys=False).sample(n=100, random_state=SEED)
                      [["id", "label", "text_raw", "text_clean"]].assign(correct=""))
            write_audit_if_missing(sample, PROCESSED / f"label_audit_{d}.csv")
    else:
        print("checks failed: parquet and audit files NOT written")
    log["errors"] = errors
    (out_dir / "stats.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== {variant} -> {out_dir}")
    for name, s in log["splits"].items():
        print(f"{name:<18} n={s['n']:>6} labels={s['labels']}")
    if errors:
        print("CHECKS FAILED:", *errors, sep="\n  ")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="full")
    args = parser.parse_args(argv)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    return max(build(v) for v in variants)


if __name__ == "__main__":
    sys.exit(main())
