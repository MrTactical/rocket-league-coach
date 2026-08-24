"""
Record an MMR reading.

    python coach/mmr.py 1083                 # 3v3, today
    python coach/mmr.py 1083 --duo 1108      # both playlists
    python coach/mmr.py 1083 --date 2026-08-20
    python coach/mmr.py --show               # print the history

Ranks are not in replay files. They live only on Psyonix's servers, and the
public trackers return 403 to automated requests, so there is no honest way to
scrape them -- this is a thirty-second manual step after a session, and it is
what turns a pile of match stats into "am I climbing or sliding".

Your profile: rocketleague.tracker.network/rocket-league/profile/steam/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

RANK = Path(__file__).resolve().parent / "rank.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Record an MMR reading.")
    ap.add_argument("mmr", nargs="?", type=int, help="your 3v3 MMR")
    ap.add_argument("--duo", type=int, help="your 2v2 MMR")
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    ap.add_argument("--note", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not RANK.is_file():
        print("No %s yet." % RANK)
        return 1
    d = json.loads(RANK.read_text(encoding="utf-8"))
    hist = d.setdefault("history", [])

    if args.show or args.mmr is None:
        if not hist:
            print("no readings yet")
            return 0
        print("%-12s %8s %8s  %s" % ("date", "3v3", "2v2", "note"))
        prev = None
        for h in sorted(hist, key=lambda x: x.get("date", "")):
            cur = h.get("mmr_3v3")
            delta = ""
            if prev is not None and cur is not None:
                delta = " (%+d)" % (cur - prev)
            print("%-12s %8s%-7s %6s  %s"
                  % (h.get("date", "?"), cur if cur is not None else "-", delta,
                     h.get("mmr_2v2", "-"), h.get("note", "")))
            if cur is not None:
                prev = cur
        return 0

    when = args.date or date.today().isoformat()
    hist = [h for h in hist if h.get("date") != when]     # one row per day
    row = {"date": when, "mmr_3v3": args.mmr}
    if args.duo:
        row["mmr_2v2"] = args.duo
    if args.note:
        row["note"] = args.note
    hist.append(row)
    d["history"] = sorted(hist, key=lambda x: x.get("date", ""))

    # Keep the headline playlist figures in step with the newest reading.
    for pl in d.get("playlists", []):
        if pl.get("main") and args.mmr:
            pl["mmr"] = args.mmr
            if pl.get("season_peak") and args.mmr > pl["season_peak"]:
                pl["season_peak"] = args.mmr
        elif "Doubles" in (pl.get("name") or "") and args.duo:
            pl["mmr"] = args.duo

    RANK.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("recorded %s: 3v3 %d%s" % (when, args.mmr,
                                     (", 2v2 %d" % args.duo) if args.duo else ""))
    if len(d["history"]) > 1:
        first, last = d["history"][0], d["history"][-1]
        a, b = first.get("mmr_3v3"), last.get("mmr_3v3")
        if a and b:
            print("  %+d since %s" % (b - a, first["date"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
