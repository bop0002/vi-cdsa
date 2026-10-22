"""Crawl extra 1–4★ Tiki book reviews (to grow Domain A's Negative/Neutral pool) into data/raw/tiki_books/.

Books are listed per subcategory of "Sách tiếng Việt" (316); for each product the review API is queried
with a star filter, so the 5★ majority is never downloaded. Resumable: finished products are skipped.

    python -X utf8 scripts/crawl_tiki_books.py --target-gray 15000
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "tiki_books"
API = "https://tiki.vn/api"
# subcategories of 316 "Sách tiếng Việt" (from the listing API's category filter)
CATEGORIES = [839, 846, 393, 870, 2527, 2321, 887, 2320, 897, 873, 879, 880, 881, 1084,
              861, 857, 875, 882, 876, 885, 868, 862, 6905]
SORTS = ("top_seller", "popular", "newest")
STARS = (1, 2, 3, 4)
DELAY = 0.4


# Tiki's BytePlus WAF answers with a JS page that sets the _wafchallengeid cookie; run that script in Node
# (same approach as tiki_reviews/tiki_review_scraper.py).
WAF_JS = r"""let m = require("fs").readFileSync(0, "utf8").match(/<script>([\s\S]*?)<\/script>/);
if (!m) process.exit(1);
let document = { cookie: '' };
let window = { location: { reload: () => {
    let c = document.cookie.match(/_wafchallengeid=([^;]+)/);
    if (c) console.log(c[1]);
    process.exit(0);
} } };
eval(m[1]);"""


def solve_waf(s, html: str) -> bool:
    node = shutil.which("node")
    if not node or "_wafchallengeid" not in html:
        return False
    try:
        token = subprocess.run([node, "-e", WAF_JS], input=html, capture_output=True, text=True,
                               encoding="utf-8", timeout=10).stdout.strip()
    except subprocess.TimeoutExpired:
        return False
    if re.fullmatch(r"[\w.\-]+", token or ""):
        s.cookies.set("_wafchallengeid", token, domain="tiki.vn")
        return True
    return False


def get_json(s, url: str, params: dict) -> dict | None:
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=30)
            if r.status_code == 200 and r.text.startswith("{"):
                return r.json()
            if solve_waf(s, r.text):
                print("waf challenge solved", file=sys.stderr)
                continue
            print(f"http {r.status_code} {url} {params} {r.text[:80]!r}", file=sys.stderr)
        except Exception as e:  # curl_cffi raises its own error types
            print(f"error {e} {url} {params}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return None


def list_products(s, path: Path, min_reviews: int) -> list[dict]:
    if path.exists():
        return [json.loads(l) for l in path.open(encoding="utf-8")]
    products = {}
    for cat in CATEGORIES:
        for sort in SORTS:
            for page in range(1, 51):
                d = get_json(s, f"{API}/personalish/v1/blocks/listings",
                             {"limit": 40, "category": cat, "page": page, "sort": sort})
                time.sleep(DELAY)
                if not d or not d.get("data"):
                    break
                for x in d["data"]:
                    if (x.get("review_count") or 0) >= min_reviews and "/316/" in (x.get("primary_category_path") or ""):
                        products[x["id"]] = {"product_id": x["id"], "name": x["name"], "category": cat,
                                             "review_count": x["review_count"]}
                if page >= d["paging"]["last_page"]:
                    break
            print(f"cat {cat} {sort}: {len(products)} products", file=sys.stderr)
    with path.open("w", encoding="utf-8") as f:
        for p in products.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return list(products.values())


def product_reviews(s, pid: int, max_pages: int) -> list[dict]:
    d = get_json(s, f"{API}/v2/reviews", {"product_id": pid, "page": 1, "limit": 1})
    time.sleep(DELAY)
    if not d:
        return []
    counts = {int(k): v.get("count", 0) for k, v in (d.get("stars") or {}).items()}
    rows = []
    for star in STARS:
        if not counts.get(star):
            continue
        for page in range(1, max_pages + 1):
            d = get_json(s, f"{API}/v2/reviews", {"product_id": pid, "page": page, "limit": 20,
                                                   "sort": f"score|desc,id|desc,stars|{star}"})
            time.sleep(DELAY)
            if not d or not d.get("data"):
                break
            rows += [{"comment_id": x["id"], "product_id": pid, "rating": x.get("rating"),
                      "title": x.get("title"), "content": x.get("content"),
                      "customer_id": x.get("customer_id"), "created_at": x.get("created_at")}
                     for x in d["data"] if x.get("rating") == star]
            if page >= (d.get("paging") or {}).get("last_page", 1):
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gray", type=int, default=15000, help="stop after this many 3★+4★ reviews")
    ap.add_argument("--min-reviews", type=int, default=10)
    ap.add_argument("--max-pages-per-star", type=int, default=5, help="20 reviews per page; spreads data over more books")
    ap.add_argument("--skip-products", type=Path, help="file of product ids to skip (e.g. books in A val/test)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s = requests.Session(impersonate="chrome124")
    s.get("https://tiki.vn/", timeout=30)
    products = sorted(list_products(s, OUT_DIR / "products.jsonl", args.min_reviews),
                      key=lambda p: -p["review_count"])
    out, done_path = OUT_DIR / "reviews.jsonl", OUT_DIR / "done_products.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    skip = set(args.skip_products.read_text().split()) if args.skip_products else set()
    gray = sum(1 for l in out.open(encoding="utf-8") if json.loads(l)["rating"] in (3, 4)) if out.exists() else 0
    print(f"{len(products)} products; {len(done)} done; {gray} gray reviews so far", file=sys.stderr)

    with out.open("a", encoding="utf-8") as f, done_path.open("a") as fd:
        for i, p in enumerate(products):
            if gray >= args.target_gray:
                break
            if str(p["product_id"]) in done or str(p["product_id"]) in skip:
                continue
            rows = product_reviews(s, p["product_id"], args.max_pages_per_star)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            fd.write(f"{p['product_id']}\n")
            f.flush(); fd.flush()
            gray += sum(r["rating"] in (3, 4) for r in rows)
            if i % 20 == 0:
                print(f"[{i}/{len(products)}] {p['product_id']} +{len(rows)} gray={gray}", file=sys.stderr)
    print(f"done: {gray} 3★/4★ reviews in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
