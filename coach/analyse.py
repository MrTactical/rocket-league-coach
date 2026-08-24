"""
Full match analyser. Runs every metric family over your replays.

    python coach/analyse.py                    # your most recent match
    python coach/analyse.py --last 5           # aggregate the last N
    python coach/analyse.py --player Name
    python coach/analyse.py --json out.json    # machine-readable, for the page

Each metric family lives in coach/metrics/ and exposes TITLE, compute(),
render() and tips(). Modules are discovered, not registered, so adding a new
one is a matter of dropping in a file.

A module that raises is reported and skipped rather than killing the run: a
broken boost analyser should not cost you the rotation analysis.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Player names are arbitrary Unicode and the Windows console is cp1252 by
# default. One lobby containing a name with an Extended Arabic-Indic digit
# killed a 93-replay run at match 80 with UnicodeEncodeError -- after all the
# work, and with no JSON written. Never let a name choice cost a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from coach.timeline import load, recent_replays  # noqa: E402

# Order the report reads best in. Anything not listed still runs, at the end.
PREFERRED = ["kickoffs", "touches", "mechanics", "boost",
             "positioning", "rotation", "recovery", "scoreline",
             "fifties", "whiffs", "demos", "indecision", "goals", "lobby"]


def discover():
    import coach.metrics as pkg
    found = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module("coach.metrics.%s" % mod.name)
        except Exception as e:
            print("  ! could not import metrics/%s: %s" % (mod.name, e))
            continue
        if all(hasattr(m, a) for a in ("TITLE", "compute", "render", "tips")):
            found[mod.name] = m
        else:
            print("  ! metrics/%s does not implement the contract, skipped"
                  % mod.name)
    ordered = [(k, found[k]) for k in PREFERRED if k in found]
    ordered += [(k, v) for k, v in sorted(found.items()) if k not in PREFERRED]
    return ordered


def pick_player(matches, explicit):
    """
    Whoever recorded the replays, named as they are NOW.

    A plain vote across every match elects the name you used most, which for a
    long history is the name you used years ago -- the page came out titled
    "MrTactical ^-^" because 86 replays from 2023 out-voted 7 from today.
    Match.resolve() matches names tolerantly, so the current spelling still
    finds the older ones.
    """
    if explicit:
        return explicit
    for m in sorted(matches, key=lambda x: x.date or "", reverse=True):
        if m.name:
            return m.name
    from collections import Counter
    votes = Counter()
    for m in matches:
        for n in m.teams:
            votes[n] += 1
    return votes.most_common(1)[0][0] if votes else None


def header(m, who):
    line = "  %s  --  %s" % (who, m.date or "?")
    meta = "%dv%d" % (m.team_size, m.team_size)
    mine = m.teams.get(who)
    if mine is not None and m.score:
        us, them = (m.score[mine], m.score[1 - mine])
        verdict = "WON" if us > them else ("LOST" if us < them else "DREW")
        meta += "   %s %d-%d" % (verdict, us, them)
    if m.forfeit:
        meta += "   (FORFEIT -- short and skewed)"
    return line, "  " + meta


def analyse_match(m, who, modules, echo=True):
    """
    Run every metric family over one match and return its payload entry.

    Split out of main() so the folder watcher can analyse a single new replay
    without re-parsing the whole history.
    """
    entry = {"date": m.date, "team_size": m.team_size,
             "team": m.teams.get(who),
             "resolved_name": who,
             "score": list(m.score), "forfeit": m.forfeit,
             "duration": m.duration(),
             "stat_line": m.stat_line(who), "sections": {}}
    tips_out = []
    for key, mod in modules:
        try:
            res = mod.compute(m, who)
            lines = mod.render(res)
            tips = mod.tips(res, m, who) or []
        except Exception:
            if echo:
                print()
                print("  %s -- FAILED" % getattr(mod, "TITLE", key))
                print("    " + traceback.format_exc().strip().splitlines()[-1])
            continue
        if echo:
            print()
            print("  " + mod.TITLE)
            print("  " + "-" * 62)
            for ln in lines:
                print(ln)
        entry["sections"][key] = {
            "title": mod.TITLE, "lines": lines, "tips": tips,
            "data": _jsonable(res),
        }
        tips_out.extend(tips)
    return entry, tips_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Full replay analysis.")
    ap.add_argument("--player")
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--json", help="also write the results as JSON")
    ap.add_argument("--replay", help="analyse one specific .replay file")
    args = ap.parse_args()

    paths = [args.replay] if args.replay else recent_replays(args.last)
    if not paths:
        print("No replays found. Save one by holding Backspace in a match.")
        return 1

    matches = []
    for p in paths:
        try:
            matches.append(load(p))
        except Exception as e:
            print("skipped %s: %s" % (os.path.basename(p), str(e)[:120]))
    if not matches:
        print("Nothing parsed.")
        return 1

    who_raw = pick_player(matches, args.player)
    modules = discover()
    if not modules:
        print("No metric modules found in coach/metrics/.")
        return 1

    payload = {"player": who_raw, "matches": []}
    all_tips = []

    for m in matches:
        who = m.resolve(who_raw)
        if who is None:
            print("skipped %s: %r not in this match"
                  % (os.path.basename(m.path or "?"), who_raw))
            continue

        h1, h2 = header(m, who)
        print()
        print("=" * 68)
        print(h1)
        print(h2)
        print("=" * 68)

        entry, tips = analyse_match(m, who, modules)
        all_tips.extend(tips)

        payload["matches"].append(entry)

    if all_tips:
        print()
        print("=" * 68)
        print("  WHAT TO WORK ON")
        print("=" * 68)
        for t in all_tips:
            print("  * %s" % t)
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


# Bulk series -- every speed sample, every touch, every segment -- are working
# data for the module that produced them, not results. Ninety-three matches of
# them is a hundred megabytes of JSON that nothing reads.
BULK_LIMIT = 40


def _jsonable(o, depth=0):
    """Drop anything json cannot represent, and prune bulk series."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if str(k).startswith("_"):
                continue          # module-private working data
            out[str(k)] = _jsonable(v, depth + 1)
        return out
    if isinstance(o, (list, tuple)):
        if len(o) > BULK_LIMIT:
            # Keep the shape and the count, drop the body.
            return {"__pruned": len(o)}
        return [_jsonable(v, depth + 1) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
