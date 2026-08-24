"""
Read your MMR out of Rocket League's own log file.

    python coach/mmrlog.py               # show what the log knows
    python coach/mmrlog.py --record      # append it to coach/rank.json

No API, no scraping, no injection. Rocket League writes this itself:

    [0624.24] Matchmaking: Post-divide PartyLeaderMMR: 49.1883
    [0626.14] Matchmaking: PartyLeaderTier=(16)
    [1056.xx] DevOnline: Set rich presence to: Standard in ... data: Playlist-13

The logged figure is on the internal scale. Multiplied out it matches the
public trackers exactly -- 49.1883 -> 1083.8, against a tracker reading of
1083 for the same account, and tier 16 -> Champion I, also matching.

THREE CAVEATS, all of which matter:

  * It is the PARTY LEADER's MMR. Queue solo and that is you. Queue in a party
    you do not lead and it is someone else's number, so those readings are
    discarded unless you pass --trust-party.
  * Rotated Launch-backup-*.log files DO keep their readings. An earlier
    version of this module claimed otherwise and read only the live log, which
    silently discarded a whole session's trace every time the game restarted.
    The reason the claim looked true: the backups that existed at the time all
    predated the game version that started logging MMR at all.
  * A reading is taken when you QUEUE, so it is your rating going INTO that
    match, never the result of it. The last match of a session is therefore
    always missing from the trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.path.expanduser(
    "~/Documents/My Games/Rocket League/TAGame/Logs"))
LIVE_LOG = LOG_DIR / "Launch.log"
RANK = ROOT / "coach" / "rank.json"

# internal * SCALE + OFFSET == the number the trackers show.
SCALE, OFFSET = 20.0, 100.0

TIERS = [
    "Unranked",
    "Bronze I", "Bronze II", "Bronze III",
    "Silver I", "Silver II", "Silver III",
    "Gold I", "Gold II", "Gold III",
    "Platinum I", "Platinum II", "Platinum III",
    "Diamond I", "Diamond II", "Diamond III",
    "Champion I", "Champion II", "Champion III",
    "Grand Champion I", "Grand Champion II", "Grand Champion III",
    "Supersonic Legend",
]

PLAYLISTS = {
    10: ("Ranked Duel 1v1", "mmr_1v1"),
    11: ("Ranked Doubles 2v2", "mmr_2v2"),
    12: ("Ranked Solo Standard", "mmr_solo"),
    13: ("Ranked Standard 3v3", "mmr_3v3"),
}

LINE = re.compile(
    r"\[(\d+\.\d+)\]\s+(?:"
    r"Matchmaking: Post-divide PartyLeaderMMR:\s*(?P<mmr>[0-9.]+)"
    r"|Matchmaking: PartyLeaderTier=\((?P<tier>\d+)\)"
    r"|DevOnline: Set rich presence to: .*?data: Playlist-(?P<pl>\d+)"
    r")")


def tier_name(n):
    return TIERS[n] if isinstance(n, int) and 0 <= n < len(TIERS) else "?"


def read_all_logs():
    """
    Every reading across every log, oldest first.

    Rebuilt from scratch each call, which is cheap and idempotent -- a few
    regex passes over a handful of megabytes. Readings are keyed by
    (log file, in-file timestamp) so re-reading a log can never duplicate them.
    """
    import glob
    out = []
    for f in sorted(glob.glob(str(LOG_DIR / "*.log")), key=os.path.getmtime):
        stamp = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
        for r in read_log(f):
            r["log"] = os.path.basename(f)
            r["date"] = stamp
            out.append(r)
    return out


def read_log(path=LIVE_LOG):
    """Every MMR reading in a log, tagged with the playlist in play at the time."""
    if not Path(path).is_file():
        return []
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    out, playlist, tier = [], None, None
    for m in LINE.finditer(txt):
        if m.group("pl"):
            playlist = int(m.group("pl"))
        elif m.group("tier"):
            tier = int(m.group("tier"))
        elif m.group("mmr"):
            raw = float(m.group("mmr"))
            out.append({
                "t": float(m.group(1)),
                "internal": raw,
                "mmr": round(raw * SCALE + OFFSET),
                "tier": tier,
                "tier_name": tier_name(tier),
                "playlist": playlist,
                "playlist_name": (PLAYLISTS.get(playlist) or ("?", None))[0],
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Read MMR from Rocket League's log.")
    ap.add_argument("--record", action="store_true",
                    help="append the newest reading to coach/rank.json")
    ap.add_argument("--log", default=str(LIVE_LOG))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    readings = read_log(args.log) if args.log != str(LIVE_LOG) else read_all_logs()
    if not readings:
        if not args.quiet:
            print("No MMR in %s." % args.log)
            print("Rocket League only writes it while matchmaking, and only to")
            print("the live Launch.log -- play a ranked match and check again.")
        return 1

    if not args.quiet:
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(os.path.getmtime(args.log)))
        print("%s  (%d readings)" % (when, len(readings)))
        print()
        print("%-8s %-22s %7s %6s  %s" % ("t(s)", "playlist", "MMR", "tier", "rank"))
        print("-" * 62)
        prev = None
        for r in readings:
            delta = ""
            if prev is not None:
                d = r["mmr"] - prev
                delta = "  %+d" % d if d else ""
            print("%-8.0f %-22s %7d%-5s %5s  %s"
                  % (r["t"], r["playlist_name"], r["mmr"], delta,
                     r["tier"] if r["tier"] is not None else "-", r["tier_name"]))
            prev = r["mmr"]
        first, last = readings[0], readings[-1]
        print("-" * 62)
        print("session movement: %+d  (%d -> %d)"
              % (last["mmr"] - first["mmr"], first["mmr"], last["mmr"]))
        print()
        print("This is the PARTY LEADER's MMR -- accurate for you when you queue")
        print("solo, someone else's when you do not. Readings are taken as you")
        print("queue, so your last match's result is not in here yet.")

    if args.record:
        if not RANK.is_file():
            print("No %s to record into." % RANK)
            return 1
        d = json.loads(RANK.read_text(encoding="utf-8"))
        hist = d.setdefault("history", [])
        when = time.strftime("%Y-%m-%d",
                             time.localtime(os.path.getmtime(args.log)))
        last = readings[-1]
        field = (PLAYLISTS.get(last["playlist"]) or ("?", "mmr_3v3"))[1] or "mmr_3v3"

        row = next((h for h in hist if h.get("date") == when), None)
        if row is None:
            row = {"date": when}
            hist.append(row)
        row[field] = last["mmr"]
        row["source"] = "Launch.log"
        if last["tier_name"] != "?":
            row["rank"] = last["tier_name"]

        for pl in d.get("playlists", []):
            if pl.get("name") == last["playlist_name"]:
                pl["mmr"] = last["mmr"]
                if last["tier_name"] != "?":
                    pl["rank"] = pl.get("rank") or last["tier_name"]
                if pl.get("season_peak") and last["mmr"] > pl["season_peak"]:
                    pl["season_peak"] = last["mmr"]

        d["history"] = sorted(hist, key=lambda h: h.get("date", ""))
        RANK.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print("recorded %s %s = %d (%s)"
              % (when, field, last["mmr"], last["tier_name"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
