"""Optional runner: annotate data/relabel/batches/*.jsonl with any OpenAI-compatible chat API.

Works with OpenAI, DeepSeek, Gemini (OpenAI-compatible endpoint), OpenRouter, a local Ollama, ...
Already-finished batches (a result file exists) are skipped, so the script can be stopped and rerun.

    pip install openai
    export RELABEL_API_KEY=...        # never commit keys
    export RELABEL_BASE_URL=...       # e.g. https://api.deepseek.com  (omit for OpenAI)
    export RELABEL_MODEL=...          # e.g. deepseek-chat, gpt-4o-mini, gemini-2.0-flash
    python -X utf8 scripts/run_relabel_api.py --only A_000 A_001   # pilot
    python -X utf8 scripts/run_relabel_api.py                      # everything left
    python -X utf8 -m src.relabel validate
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data" / "relabel" / "batches"
RESULT_DIR = ROOT / "data" / "relabel" / "results"
PROMPT = (ROOT / "docs" / "relabel_prompt.md").read_text(encoding="utf-8")
CHUNK = 50  # items per request; small chunks keep cheap models from truncating or skipping lines


def parse_lines(text: str, want_ids: list[str]) -> list[dict] | None:
    """Parse JSONL output; return rows in input order, or None if any id is missing/invalid."""
    rows = {}
    for line in text.strip().strip("`").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") in want_ids and r.get("label") in (0, 1, 2) and str(r.get("note") or "").strip():
            rows[r["id"]] = {"id": r["id"], "label": r["label"], "note": r["note"]}
    return [rows[i] for i in want_ids] if len(rows) == len(want_ids) else None


def annotate_chunk(client, model: str, items: list[dict], retries: int = 4) -> list[dict]:
    user = "\n".join(json.dumps(it, ensure_ascii=False) for it in items)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}])
            rows = parse_lines(resp.choices[0].message.content or "", [it["id"] for it in items])
            if rows is not None:
                return rows
            print(f"  incomplete output, retry {attempt + 1}")
        except Exception as e:  # network / rate limit: back off and retry
            print(f"  error {type(e).__name__}: {e}; retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chunk starting at {items[0]['id']} failed after {retries} attempts")


def main() -> int:
    from openai import OpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="batch names without .jsonl, e.g. A_000 B_000")
    args = parser.parse_args()
    client = OpenAI(api_key=os.environ["RELABEL_API_KEY"], base_url=os.environ.get("RELABEL_BASE_URL"))
    model = os.environ["RELABEL_MODEL"]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    batches = sorted(BATCH_DIR.glob("*.jsonl"))
    if args.only:
        batches = [b for b in batches if b.stem in set(args.only)]
    for batch in batches:
        out = RESULT_DIR / batch.name
        if out.exists():
            continue
        items = [json.loads(l) for l in batch.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = []
        for i in range(0, len(items), CHUNK):
            rows += annotate_chunk(client, model, items[i:i + CHUNK])
        tmp = out.with_suffix(".part")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        tmp.replace(out)  # only complete batches get the final name
        print(f"{batch.name}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
