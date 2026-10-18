"""Content-based relabeling of the "gray zone" (A 3★/4★, B Neutral). See docs/relabel_runbook.md.

Usage:
    python -X utf8 -m src.relabel export     # write blind batches to data/relabel/batches/
    python -X utf8 -m src.relabel validate   # check data/relabel/results/ against the batches
    python -X utf8 -m src.relabel merge      # write data/relabel/relabel_labels.csv
    python -X utf8 -m src.relabel qa         # agreement with the D3 audit + 100-row review sample
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED, ROOT, SEED

RELABEL_DIR = ROOT / "data" / "relabel"
BATCH_DIR = RELABEL_DIR / "batches"
RESULT_DIR = RELABEL_DIR / "results"
RELABEL_FILE = RELABEL_DIR / "relabel_labels.csv"
REVIEW_SAMPLE = RELABEL_DIR / "review_sample.csv"
BATCH_SIZE = 200


def select_gray(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Rows whose original label is unreliable per D3: A rated 3★ or 4★, B labelled Neutral."""
    return pd.concat([a[a["rating"].isin([3, 4])], b[b["label"] == 1]]).reset_index(drop=True)


def export_batches(gray: pd.DataFrame, out_dir: Path, batch_size: int = BATCH_SIZE) -> list[Path]:
    """Write blind JSONL batches ({id, text}); the original label is deliberately not exported."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for domain, g in gray.groupby("domain", sort=True):
        for i, start in enumerate(range(0, len(g), batch_size)):
            chunk = g.iloc[start:start + batch_size]
            path = out_dir / f"{domain}_{i:03d}.jsonl"
            lines = (json.dumps({"id": r.id, "text": " ".join(str(r.text_raw).split())}, ensure_ascii=False)
                     for r in chunk.itertuples())
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths.append(path)
    return paths


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_results(batch_dir: Path, result_dir: Path, partial: bool = False) -> list[str]:
    """Check each result file against its batch. partial=True ignores batches not annotated yet (pilot runs)."""
    errs = []
    for batch in sorted(Path(batch_dir).glob("*.jsonl")):
        res_path = Path(result_dir) / batch.name
        if not res_path.exists():
            if not partial:
                errs.append(f"{batch.name}: missing result file")
            continue
        try:
            rows = _read_jsonl(res_path)
        except json.JSONDecodeError as e:
            errs.append(f"{batch.name}: invalid JSON ({e})")
            continue
        want = {r["id"] for r in _read_jsonl(batch)}
        got = [r.get("id") for r in rows]
        if missing := sorted(want - set(got)):
            errs.append(f"{batch.name}: missing ids {missing[:5]}")
        if extra := sorted(set(got) - want):
            errs.append(f"{batch.name}: unexpected ids {extra[:5]}")
        if len(got) != len(set(got)):
            errs.append(f"{batch.name}: duplicate ids")
        if any(r.get("label") not in (0, 1, 2) for r in rows):
            errs.append(f"{batch.name}: invalid label (must be 0, 1 or 2)")
        if any(not str(r.get("note") or "").strip() for r in rows):
            errs.append(f"{batch.name}: empty note")
    return errs


def merge_results(gray: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    rows = [r for p in sorted(Path(result_dir).glob("*.jsonl")) for r in _read_jsonl(p)]
    res = pd.DataFrame(rows)[["id", "label", "note"]].rename(columns={"label": "new_label"})
    merged = gray[["id", "domain", "label"]].rename(columns={"label": "orig_label"}).merge(res, on="id")
    return merged[["id", "domain", "orig_label", "new_label", "note"]]


def apply_relabels(df: pd.DataFrame, relabels: pd.DataFrame) -> pd.DataFrame:
    """Override `label` with `new_label` by id and mark those rows label_source='content'."""
    if not relabels["new_label"].isin([0, 1, 2]).all():
        raise ValueError("new_label must be 0, 1 or 2")
    new = df["id"].map(relabels.set_index("id")["new_label"])
    hit = new.notna()
    out = df.copy()
    out.loc[hit, "label"] = new[hit].astype(int)
    out.loc[hit, "label_source"] = "content"
    out["label"] = out["label"].astype(int)
    return out


def qa_against_audit(merged: pd.DataFrame, audit: pd.DataFrame) -> dict:
    """Agreement with the D3 audit: kept label when audit said correct, changed it when audit said wrong."""
    m = merged.merge(audit[["id", "correct"]], on="id")
    agree = ((m["correct"] == 1) & (m["new_label"] == m["orig_label"])) | \
            ((m["correct"] == 0) & (m["new_label"] != m["orig_label"]))
    return {"n": int(len(m)), "agree": int(agree.sum()), "rate": round(float(agree.mean()), 3) if len(m) else 0.0}


def _gray_pool() -> pd.DataFrame:
    from src.prepare import clean_sources
    a, b = clean_sources()
    return select_gray(a, b)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "validate", "merge", "qa"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--partial", action="store_true", help="validate/qa only the batches annotated so far")
    args = parser.parse_args(argv)

    if args.command == "export":
        gray = _gray_pool()
        paths = export_batches(gray, BATCH_DIR, args.batch_size)
        print(f"exported {len(gray)} rows in {len(paths)} batches to {BATCH_DIR}")
        print(gray.groupby("domain").size().to_dict())
        return 0
    if args.command == "validate":
        errs = validate_results(BATCH_DIR, RESULT_DIR, partial=args.partial)
        done = len(list(RESULT_DIR.glob("*.jsonl"))) if RESULT_DIR.exists() else 0
        print(f"result files: {done}/{len(list(BATCH_DIR.glob('*.jsonl')))}")
        print("\n".join(errs) if errs else "ALL RESULTS VALID")
        return 1 if errs else 0
    if args.command == "merge":
        errs = validate_results(BATCH_DIR, RESULT_DIR)
        if errs:
            print("fix these first:", *errs, sep="\n  ")
            return 1
        merged = merge_results(_gray_pool(), RESULT_DIR)
        merged.to_csv(RELABEL_FILE, index=False, encoding="utf-8-sig")
        print(f"wrote {len(merged)} rows to {RELABEL_FILE}")
        print(pd.crosstab([merged["domain"], merged["orig_label"]], merged["new_label"]))
        return 0
    if args.command == "qa":
        errs = validate_results(BATCH_DIR, RESULT_DIR, partial=args.partial)
        if errs:
            print("fix these first:", *errs, sep="\n  ")
            return 1
        gray = _gray_pool()
        merged = merge_results(gray, RESULT_DIR)
        print(f"annotated rows: {len(merged)}/{len(gray)}")
        print(pd.crosstab([merged["domain"], merged["orig_label"]], merged["new_label"]))
        for d in "AB":
            audit = pd.read_csv(PROCESSED / f"label_audit_{d}.csv", encoding="utf-8-sig")
            print(f"{d}: agreement with D3 audit {qa_against_audit(merged[merged['domain'] == d], audit)}")
        texts = gray.set_index("id")["text_raw"]
        sample = (pd.concat([g.sample(n=min(50, len(g)), random_state=SEED) for _, g in merged.groupby("domain")])
                  .assign(text=lambda d: d["id"].map(texts), human_label=""))
        sample.to_csv(REVIEW_SAMPLE, index=False, encoding="utf-8-sig")
        print(f"review sample ({len(sample)} rows, fill human_label 0/1/2): {REVIEW_SAMPLE}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
