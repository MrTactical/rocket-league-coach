"""
You and your regular team-mate, measured as a unit.

Every other metric here treats a player alone. This one asks the questions
that only exist for a pair: does the team win more with this person in it,
where do the two of you want the same job, and what is the third slot actually
missing.

WHAT THIS CAN AND CANNOT SAY. Win rate with a team-mate is confounded -- you
queue with a friend at particular times, in particular moods, against whatever
the matchmaker produced. A pairing that wins more is not proof the pairing
causes it. What is much harder to explain away is the SHAPE: two players both
ahead of the ball at the same moment is a measurement of the same instant, not
a correlation across matches, and it is the state the possession split showed
costs goals.

    python coach/stack.py --with TEAMMATE
    python coach/stack.py --list --min 5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coach.profile_export import _key, load_cache  # noqa: E402

LOG = logging.getLogger("stack")

MIN_MATCHES = 5          # below this a win rate is noise, not a signal


def _won(entry):
    sc = entry.get("score") or [0, 0]
    t = entry.get("team")
    if t not in (0, 1) or len(sc) < 2:
        return None
    return (sc[0] > sc[1]) if t == 0 else (sc[1] > sc[0])


def _players(entry):
    return ((entry.get("sections") or {}).get("lobby") or {}).get("data") or {}


def collect(cache, me, team_size=None):
    """Group the player's matches by which team-mates were in them."""
    me_key = _key(me)
    rows = []
    for e in cache.get("entries", {}).values():
        if not e.get("sections"):
            continue
        if team_size and e.get("team_size") != team_size:
            continue
        lob = _players(e)
        pls = lob.get("players") or {}
        mine = {k: v for k, v in pls.items() if v.get("mine")}
        if not any(_key(k) == me_key for k in mine):
            continue
        w = _won(e)
        if w is None:
            continue
        mates = sorted(k for k in mine if _key(k) != me_key)
        rows.append({"entry": e, "won": w, "mates": mates,
                     "players": pls, "clashes": lob.get("clashes") or []})
    return rows


def by_mate(rows, me):
    """Win rate and shared-shape stats, per team-mate."""
    me_key = _key(me)
    out = {}
    for r in rows:
        for mate in r["mates"]:
            k = _key(mate)
            d = out.setdefault(k, {"name": mate, "n": 0, "w": 0,
                                   "my_exposed": [], "their_exposed": [],
                                   "my_first": [], "their_first": [],
                                   "both_up": 0, "clashes": 0})
            d["n"] += 1
            d["w"] += int(r["won"])
            mine = next((v for kk, v in r["players"].items()
                         if _key(kk) == me_key), None)
            theirs = next((v for kk, v in r["players"].items()
                           if _key(kk) == k), None)
            if mine and theirs:
                d["my_exposed"].append(mine.get("exposed") or 0.0)
                d["their_exposed"].append(theirs.get("exposed") or 0.0)
                d["my_first"].append(mine.get("first_man_rel") or 1.0)
                d["their_first"].append(theirs.get("first_man_rel") or 1.0)
            for c in r["clashes"]:
                if mate in c:
                    d["clashes"] += 1
                    if "ahead of the ball" in c:
                        d["both_up"] += 1
    return out


def third_slot(rows, me, mate):
    """
    What the team is missing when it is just the two of you.

    Compares matches where the third player was a random fill against the
    lobby population on each axis, to say what the slot actually needs rather
    than what would be nice to have.
    """
    me_key, mate_key = _key(me), _key(mate)
    gaps = {"airborne": [], "first_man_rel": [], "exposed": [], "speed": [],
            "boost_held": []}
    n = 0
    for r in rows:
        if not any(_key(m) == mate_key for m in r["mates"]):
            continue
        third = [v for k, v in r["players"].items()
                 if v.get("mine") and _key(k) not in (me_key, mate_key)]
        if len(third) != 1:
            continue
        opp = [v for v in r["players"].values() if not v.get("mine")]
        if not opp:
            continue
        n += 1
        t = third[0]
        for axis in gaps:
            tv, ov = t.get(axis), [o.get(axis) for o in opp if o.get(axis) is not None]
            if tv is not None and ov:
                gaps[axis].append(tv - statistics.median(ov))
    if not n:
        return None
    return {"n": n, "gaps": {a: statistics.median(v) for a, v in gaps.items() if v}}


def report(cache, me, mate=None, team_size=3, min_n=MIN_MATCHES):
    rows = collect(cache, me, team_size)
    if not rows:
        LOG.error("no %s matches found for %r", "%dv%d" % (team_size, team_size), me)
        return 1

    stats = by_mate(rows, me)
    solo_n = sum(1 for r in rows if not r["mates"])
    overall = sum(1 for r in rows if r["won"])

    print("%s -- %d matches %dv%d, %d won (%.0f%%)"
          % (me, len(rows), team_size, team_size, overall,
             100.0 * overall / len(rows)))
    if solo_n:
        print("  %d with no recognised team-mate" % solo_n)
    print()

    ranked = sorted((d for d in stats.values() if d["n"] >= min_n),
                    key=lambda d: -d["n"])
    if not ranked:
        print("No team-mate reaches %d matches. Lower --min to see more."
              % min_n)
        return 0

    print("%-24s %5s %6s   %-13s %-13s %s"
          % ("team-mate", "games", "won", "your exposed", "their exposed",
             "both ahead"))
    for d in ranked:
        print("%-24s %5d %5.0f%%   %-13s %-13s %d of %d"
              % (d["name"][:24], d["n"], 100.0 * d["w"] / d["n"],
                 "%.1f%%" % statistics.median(d["my_exposed"] or [0]),
                 "%.1f%%" % statistics.median(d["their_exposed"] or [0]),
                 d["both_up"], d["n"]))

    best = ranked[0]
    target = _key(mate) if mate else _key(best["name"])
    pick = next((d for d in stats.values() if _key(d["name"]) == target), None)
    if pick is None or pick["n"] < min_n:
        return 0

    print()
    print("--- %s, %d matches ---" % (pick["name"], pick["n"]))
    mine = statistics.median(pick["my_first"] or [1.0])
    theirs = statistics.median(pick["their_first"] or [1.0])
    if mine > 1.15 and theirs > 1.15:
        print("  Both of you go for the ball more than an even share "
              "(%.2fx and %.2fx). One of you has to give it up." % (mine, theirs))
    elif abs(mine - theirs) < 0.12:
        print("  You take near-identical shares of the ball (%.2fx vs %.2fx) "
              "-- no one is the designated first man." % (mine, theirs))
    else:
        lead = "you" if mine > theirs else pick["name"]
        print("  %s is clearly first man (%.2fx vs %.2fx), which is a "
              "working split." % (lead, max(mine, theirs), min(mine, theirs)))

    if pick["n"]:
        share = 100.0 * pick["both_up"] / pick["n"]
        print("  Both of you ahead of the ball with the opponent on it: "
              "flagged in %d of %d matches (%.0f%%)."
              % (pick["both_up"], pick["n"], share))
        if share > 25:
            print("    That is the 2.78x concede state, and it is a pairing "
                  "problem rather than an individual one.")

    ts = third_slot(rows, me, pick["name"])
    if ts:
        print()
        print("  Third slot, across %d matches with a random fill:" % ts["n"])
        labels = {"airborne": "air time", "first_man_rel": "ball share",
                  "exposed": "caught upfield", "speed": "speed",
                  "boost_held": "boost held"}
        for axis, gap in sorted(ts["gaps"].items(), key=lambda kv: -abs(kv[1])):
            direction = "above" if gap > 0 else "below"
            print("    %-16s %+.2f %s the opponents' median"
                  % (labels.get(axis, axis), gap, direction))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyse you and a regular team-mate as a unit.")
    ap.add_argument("--me", help="your in-game name (default: the cached one)")
    ap.add_argument("--with", dest="mate", help="focus on this team-mate")
    ap.add_argument("--cache", default=str(ROOT / "coach" / ".replay-cache.json"))
    ap.add_argument("--team-size", type=int, default=3, choices=(2, 3))
    ap.add_argument("--min", type=int, default=MIN_MATCHES,
                    help="minimum matches before a team-mate is listed")
    ap.add_argument("--list", action="store_true",
                    help="just list team-mates and win rates")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    cache = load_cache(args.cache)
    if cache is None:
        return 2
    me = args.me or cache.get("player")
    if not me:
        LOG.error("no player given and none cached -- pass --me")
        return 2

    return report(cache, me, None if args.list else args.mate,
                  args.team_size, args.min)


if __name__ == "__main__":
    sys.exit(main())
