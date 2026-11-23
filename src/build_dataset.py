"""Build Domain A (Tiki book) and Domain B (restaurant) into data/processed/.

    python -X utf8 -m src.build_dataset

Each domain keeps its own (natural) class ratio and is re-split 80/10/10 by product/restaurant; the larger
domain is sampled down so A and B have the same size in every split.
A = original Tiki only (the Tiki crawl kept only 1–4★, which would skew A's ratio), B = res1 + Foody.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.checks import run_checks
from src.config import OUTPUT_COLUMNS, PROCESSED, SEED
from src.prepare import clean_sources
from src.relabel import RELABEL_FILE, drop_unannotated_gray
from src.segment import segment, to_ml
from src.io_sources import EXTRA_SPLIT
from src.split import balanced_subset, full_split


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


def split_domains(relabels, step) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = clean_sources(relabels, step)
    a, b = drop_unannotated_gray(a[a["orig_split"].ne(EXTRA_SPLIT)], b)
    step("pool", a, b)
    return full_split(a, b)


def build(out_dir: Path = PROCESSED) -> int:
    log = {"seed": SEED, "steps": [], "splits": {}}

    def step(name, a, b):
        log["steps"].append({"step": name, "A": {"n": len(a), "labels": dist(a)},
                             "B": {"n": len(b), "labels": dist(b)}})
        print(f"{name:<20} A={len(a):>6} {dist(a)} | B={len(b):>6} {dist(b)}")

    relabels = load_relabels()
    log["relabel_file"] = str(RELABEL_FILE) if relabels is not None else None
    print(f"relabels: {len(relabels)} rows from {RELABEL_FILE}" if relabels is not None
          else "relabels: none (using original labels)")
    a, b = split_domains(relabels, step)
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

    errors = run_checks(frames)
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
            write_audit_if_missing(sample, out_dir / f"label_audit_{d}.csv")
    else:
        print("checks failed: parquet and audit files NOT written")
    log["errors"] = errors
    (out_dir / "stats.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== {out_dir}")
    for name, s in log["splits"].items():
        print(f"{name:<18} n={s['n']:>6} labels={s['labels']}")
    if errors:
        print("CHECKS FAILED:", *errors, sep="\n  ")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    return build()


if __name__ == "__main__":
    sys.exit(main())
