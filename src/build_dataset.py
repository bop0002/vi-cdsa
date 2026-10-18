"""Build Domain A (Tiki book) and Domain B (res1). Run: python -m src.build_dataset"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.checks import run_checks
from src.config import OUTPUT_COLUMNS, PROCESSED, SEED
from src.prepare import clean_sources
from src.relabel import RELABEL_FILE
from src.segment import segment, to_ml
from src.split import balanced_subset, group_split, match_train_size


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
    return pd.read_csv(path, encoding="utf-8-sig")[["id", "new_label"]]


def main() -> int:
    log = {"seed": SEED, "steps": [], "splits": {}}

    def step(name, a, b):
        log["steps"].append({"step": name, "A": {"n": len(a), "labels": dist(a)},
                             "B": {"n": len(b), "labels": dist(b)}})
        print(f"{name:<20} A={len(a):>6} {dist(a)} | B={len(b):>6} {dist(b)}")

    relabels = load_relabels()
    log["relabel_file"] = str(RELABEL_FILE) if relabels is not None else None
    print(f"relabels: {len(relabels)} rows from {RELABEL_FILE}" if relabels is not None
          else "relabels: none (using original labels)")
    a, b = clean_sources(relabels, step)
    a = group_split(a)
    b = b.assign(split=b["orig_split"])
    a = match_train_size(a, b)
    step("match_train", a, b)

    frames = {}
    for d, df in (("A", a), ("B", b)):
        df = df.copy()
        df["text_dl"] = df["text_clean"].map(segment)
        df["text_ml"] = df["text_dl"].map(to_ml)
        for s in ("train", "val", "test"):
            frames[f"{d}_{s}"] = df[df["split"] == s].reset_index(drop=True)
        frames[f"{d}_test_balanced"] = balanced_subset(frames[f"{d}_test"])

    errors = run_checks(frames)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        log["splits"][name] = describe(df)
    if not errors:
        for name, df in frames.items():
            df[OUTPUT_COLUMNS].to_parquet(PROCESSED / f"{name}.parquet", index=False)
        for d in "AB":
            audit = pd.concat([frames[f"{d}_train"], frames[f"{d}_val"], frames[f"{d}_test"]])
            sample = (audit.groupby("label", group_keys=False).sample(n=100, random_state=SEED)
                      [["id", "label", "text_raw", "text_clean"]].assign(correct=""))
            write_audit_if_missing(sample, PROCESSED / f"label_audit_{d}.csv")
    else:
        print("checks failed: parquet and audit files NOT written")
    log["errors"] = errors
    (PROCESSED / "stats.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, s in log["splits"].items():
        print(f"{name:<18} n={s['n']:>6} labels={s['labels']}")
    if errors:
        print("CHECKS FAILED:", *errors, sep="\n  ")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
