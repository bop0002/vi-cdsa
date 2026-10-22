"""Crawl extra restaurant reviews from Foody (same source as res1) into data/raw/foody/.

Lists places of each city, then fetches reviews of the places with the most reviews first
until --target reviews are collected. Resumable: places already in reviews.jsonl are skipped.

    python -X utf8 scripts/crawl_foody.py --target 50000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "foody"
BASE = "https://www.foody.vn"
CITIES = ("ho-chi-minh", "ha-noi")
# The list API stops after 20 pages (2000 places) per query, so each city is also listed per district.
# Ids found by probing districtId; None = whole city.
DISTRICTS = {
    "ho-chi-minh": [None, *range(2, 20)],
    "ha-noi": [None, *range(20, 30)],
}
PAGE_SIZE = 100
MAX_PAGES = 20
MAX_REVIEWS_PER_PLACE = 100  # the API stops returning items after 100
DELAY = 1.0


def session(city: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{BASE}/{city}",
    })
    s.get(f"{BASE}/{city}", timeout=30)  # sets the city cookie used by the list API
    return s


def get_json(s: requests.Session, path: str, params: dict) -> dict | None:
    for attempt in range(4):
        try:
            r = s.get(BASE + path, params={"t": int(time.time() * 1000), **params}, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"http {r.status_code} {path} {params}", file=sys.stderr)
        except (requests.RequestException, ValueError) as e:
            print(f"error {e} {path} {params}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return None


def list_places(city: str, path: Path) -> list[dict]:
    if path.exists():
        return [json.loads(l) for l in path.open(encoding="utf-8")]
    s, places = session(city), {}
    for district in DISTRICTS[city]:
        for page in range(1, MAX_PAGES + 1):
            d = get_json(s, "/__get/Place/HomeListPlace",
                         {"page": page, "lat": 0, "lon": 0, "count": PAGE_SIZE, "type": 1,
                          "districtId": district or ""})
            time.sleep(DELAY)
            if not d or not d.get("Items"):
                break
            for x in d["Items"]:
                places[x["Id"]] = {"res_id": x["Id"], "name": x["Name"], "city": city, "url": x.get("Url"),
                                   "avg_rating": x.get("AvgRating"), "total_reviews": x.get("TotalReviews") or 0}
            if page * PAGE_SIZE >= (d.get("Total") or 0):
                break
        print(f"{city} district {district}: {len(places)} places", file=sys.stderr)
    with path.open("w", encoding="utf-8") as f:
        for p in places.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return list(places.values())


def place_reviews(s: requests.Session, place: dict) -> list[dict]:
    rows, last = [], None
    while len(rows) < MAX_REVIEWS_PER_PLACE:
        d = get_json(s, "/__get/Review/ResLoadMore",
                     {"ResId": place["res_id"], "LastId": last or "", "Count": 10, "Type": 1,
                      "fromOwner": "", "isLatest": "true", "HasPicture": "false"})
        time.sleep(DELAY)
        if not d or not d.get("Items"):
            break
        rows += [{"review_id": x["Id"], "res_id": place["res_id"], "city": place["city"],
                  "rating": x.get("AvgRating"), "title": x.get("Title"), "content": x.get("Description"),
                  "created": x.get("CreatedDate")} for x in d["Items"]]
        if d.get("LastId") in (None, last) or len(rows) >= (d.get("Total") or 0):
            break
        last = d["LastId"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--min-reviews", type=int, default=5, help="skip places with fewer listed reviews")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    places = [p for c in CITIES for p in list_places(c, OUT_DIR / f"places_{c}.jsonl")]
    places = sorted((p for p in places if p["total_reviews"] >= args.min_reviews),
                    key=lambda p: -p["total_reviews"])
    out = OUT_DIR / "reviews.jsonl"
    done_path = OUT_DIR / "done_places.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    n = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    print(f"{len(places)} places with >= {args.min_reviews} reviews; {len(done)} done; {n} reviews so far",
          file=sys.stderr)

    sessions = {c: session(c) for c in CITIES}
    with out.open("a", encoding="utf-8") as f, done_path.open("a") as fd:
        for i, p in enumerate(places):
            if n >= args.target:
                break
            if str(p["res_id"]) in done:
                continue
            rows = place_reviews(sessions[p["city"]], p)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            fd.write(f"{p['res_id']}\n")
            f.flush(); fd.flush()
            n += len(rows)
            if i % 20 == 0:
                print(f"[{i}/{len(places)}] {p['city']} {p['res_id']} +{len(rows)} total={n}", file=sys.stderr)
    print(f"done: {n} reviews in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
