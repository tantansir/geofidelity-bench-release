"""
Probe Mapillary Graph API rate limits for the token in use.

We issue a controlled burst of search+meta+thumbnail requests against
Tokyo (image-dense, known to return results) and record response headers
and wall-clock timing. Output: outputs/mapillary_probe.json.

Why this matters: v3 targets ~25k images across 25 cities. Each image
costs ~3 API calls (search was already amortised per tile, but meta
+ thumbnail are per-image). Without an empirical rate-limit number we
cannot estimate whether the full run fits in one day on one token.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
import time
from datetime import datetime, timezone

import requests

import config


TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    raise RuntimeError("Set MAPILLARY_TOKEN before probing Mapillary API limits.")
BASE_URL = "https://graph.mapillary.com"
SESSION = requests.Session()
OUT_PATH = config.OUTPUT_DIR / "mapillary_probe.json"


RATE_HEADERS = [
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-app-usage",
    "x-ad-account-usage",
    "x-business-use-case-usage",
    "x-fb-request-id",
    "x-fb-trace-id",
    "retry-after",
]


def _collect_rate(resp: requests.Response) -> dict:
    return {k: resp.headers.get(k) for k in RATE_HEADERS
            if resp.headers.get(k) is not None}


def _search(lat: float, lon: float, radius_m: int = 800):
    r_deg = radius_m / 111000.0
    bbox = f"{lon - r_deg},{lat - r_deg},{lon + r_deg},{lat + r_deg}"
    return SESSION.get(
        f"{BASE_URL}/images",
        params={
            "access_token": TOKEN,
            "fields": "id,geometry,captured_at,is_pano,camera_type",
            "bbox": bbox,
            "limit": 100,
            "is_pano": "false",
        },
        timeout=30,
    )


def _meta(image_id: str):
    return SESSION.get(
        f"{BASE_URL}/{image_id}",
        params={"access_token": TOKEN, "fields": "thumb_1024_url"},
        timeout=15,
    )


def main(burst: int = 50):
    results: list[dict] = []
    t0 = time.time()

    # 1 search call first
    r = _search(35.6762, 139.6503, 800)
    print(f"search: http {r.status_code}")
    results.append({
        "call": "search",
        "status": r.status_code,
        "headers": _collect_rate(r),
        "elapsed_s": round(time.time() - t0, 3),
    })
    if r.status_code != 200:
        print("search failed, aborting probe")
        _save(results)
        return
    ids = [str(x["id"]) for x in r.json().get("data", [])][:burst]
    print(f"got {len(ids)} image ids, running {len(ids)} meta calls")

    for i, iid in enumerate(ids):
        t1 = time.time()
        rr = _meta(iid)
        rate = _collect_rate(rr)
        results.append({
            "call": "meta",
            "idx": i,
            "image_id": iid,
            "status": rr.status_code,
            "headers": rate,
            "elapsed_s": round(time.time() - t1, 3),
        })
        if rr.status_code == 429:
            print(f"hit 429 at call #{i}, stopping burst")
            break
        if rr.status_code != 200:
            print(f"call #{i}: http {rr.status_code}")
        # don't sleep — we want to see if we get throttled

    total = time.time() - t0
    summary = {
        "probe_time_utc": datetime.now(timezone.utc).isoformat(),
        "total_calls": len(results),
        "total_wall_s": round(total, 2),
        "throughput_rps": round(len(results) / max(total, 0.001), 2),
        "status_counts": _count_status(results),
        "final_headers": results[-1]["headers"] if results else {},
    }
    print(json.dumps(summary, indent=2))
    _save({"summary": summary, "calls": results})


def _count_status(results: list[dict]) -> dict:
    out: dict[int, int] = {}
    for r in results:
        s = r["status"]
        out[s] = out.get(s, 0) + 1
    return {str(k): v for k, v in out.items()}


def _save(payload) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", type=int, default=50)
    args = ap.parse_args()
    main(burst=args.burst)
