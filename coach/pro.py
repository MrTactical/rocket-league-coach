"""
Benchmark your positioning against a higher rank, using real replays.

    python coach/pro.py --rank grand-champion --count 30
    python coach/pro.py --show

Needs a ballchasing.com API key: sign in there with Steam and copy the key from
your profile, then either put it in coach/ballchasing.key or set BALLCHASING_KEY.
The key is gitignored -- it is yours, and it identifies your account.

WHY REPLAY FILES AND NOT THE STATS API. Ballchasing's per-replay stats are
aggregates: time in each third, average distance to the ball, percent behind
the ball. Useful, but they cannot answer the question this exists for -- where
the covering defender stands, as a fraction of the way from their net to the
ball, on attacks that were held. That needs frame data, so this downloads the
replay files and runs them through the same parser and the same measurement
used on your own matches. Comparing a number computed two different ways is
how you get a confident wrong answer.

Rate limits on the free tier are 2 list calls and 1 download per second, 200
downloads an hour. This sleeps accordingly and defaults to a small sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coach.shadow import shadow_stats, summarise  # noqa: E402
from coach.timeline import Match, parse  # noqa: E402

API = "https://ballchasing.com/api"
KEY_FILE = ROOT / "coach" / "ballchasing.key"
OUT = ROOT / "coach" / "pro-benchmark.json"
CACHE_DIR = ROOT / "coach" / ".pro-replays"

DOWNLOAD_SLEEP = 1.1        # free tier: 1 download/second
LIST_SLEEP = 0.6


def api_key():
    k = os.environ.get("BALLCHASING_KEY")
    if k:
        return k.strip()
    if KEY_FILE.is_file():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return None


def _get(path, params=None, binary=False, key=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit("ballchasing rejected the key (401). Check it.")
        if e.code == 429:
            raise SystemExit("rate limited (429). Wait a minute and retry.")
        raise SystemExit("ballchasing error %s on %s" % (e.code, path))
    return data if binary else json.loads(data)


def collect(rank, count, playlist="ranked-standard", key=None, verbose=True):
    """Download `count` replays at a rank and measure them. Returns raw stats."""
    CACHE_DIR.mkdir(exist_ok=True)
    listing = _get("/replays", {
        "playlist": playlist, "min-rank": rank, "max-rank": rank,
        "count": min(200, max(count * 2, count)), "sort-by": "replay-date",
    }, key=key)
    time.sleep(LIST_SLEEP)

    ids = [r["id"] for r in (listing.get("list") or [])]
    if verbose:
        print("found %d replays at %s" % (len(ids), rank))

    held, conceded, used = [], [], 0
    for rid in ids:
        if used >= count:
            break
        path = CACHE_DIR / (rid + ".replay")
        if not path.is_file():
            try:
                path.write_bytes(_get("/replays/%s/file" % rid, binary=True,
                                      key=key))
            except SystemExit:
                raise
            except Exception as e:
                if verbose:
                    print("  skip %s (%s)" % (rid[:8], str(e)[:40]))
                continue
            time.sleep(DOWNLOAD_SLEEP)
        try:
            m = Match(parse(path))
        except Exception as e:
            if verbose:
                print("  unparseable %s (%s)" % (rid[:8], str(e)[:40]))
            continue
        h, c = shadow_stats(m)
        if not h and not c:
            continue
        held += h
        conceded += c
        used += 1
        if verbose:
            print("  %2d/%d  %s  held %-5d conceded %-4d"
                  % (used, count, rid[:8], len(h), len(c)))

    return {"rank": rank, "playlist": playlist, "matches": used,
            "held": held, "conceded": conceded}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark against a higher rank.")
    ap.add_argument("--rank", default="grand-champion",
                    help="champion-1 ... grand-champion")
    ap.add_argument("--count", type=int, default=25, help="replays to sample")
    ap.add_argument("--playlist", default="ranked-standard")
    ap.add_argument("--show", action="store_true", help="print the saved result")
    args = ap.parse_args()

    if args.show:
        if not OUT.is_file():
            print("No benchmark yet. Run without --show.")
            return 1
        d = json.loads(OUT.read_text(encoding="utf-8"))
        for rank, v in d.items():
            print("%-18s matches %-4s  held %.2f (n=%d)  conceded %.2f (n=%d)"
                  % (rank, v.get("matches"), v["depth_held"], v["n_held"],
                     v["depth_conceded"], v["n_conceded"]))
        return 0

    key = api_key()
    if not key:
        print("No ballchasing API key.")
        print()
        print("  1. go to ballchasing.com and sign in with Steam")
        print("  2. copy the API key from your profile")
        print("  3. save it:   coach/ballchasing.key   (one line, gitignored)")
        print("     or set the BALLCHASING_KEY environment variable")
        return 1

    raw = collect(args.rank, args.count, args.playlist, key)
    if not raw["matches"]:
        print("No usable replays came back.")
        return 1

    stats = summarise(raw["held"], raw["conceded"])
    stats["matches"] = raw["matches"]
    stats["playlist"] = raw["playlist"]
    stats["fetched"] = time.strftime("%Y-%m-%d")

    saved = {}
    if OUT.is_file():
        try:
            saved = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
    saved[args.rank] = stats
    OUT.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    print()
    print("%s, %d matches" % (args.rank, raw["matches"]))
    print("  covering defender depth, fraction from own net to ball")
    print("    attacks held      %.2f   (n=%d)"
          % (stats["depth_held"], stats["n_held"]))
    print("    attacks conceded  %.2f   (n=%d)"
          % (stats["depth_conceded"], stats["n_conceded"]))
    print("  lateral, share of the ball's x")
    print("    held              %.2f" % stats["lateral_held"])
    print("    conceded          %.2f" % stats["lateral_conceded"])
    print()
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
