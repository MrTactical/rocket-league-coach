"""
Watch for new replays and keep the coaching page current.

    python coach/watch.py                  # run it and leave it running
    python coach/watch.py --auto           # idle until Rocket League starts
    python coach/watch.py --once           # catch up on anything new, then exit
    python coach/watch.py --interval 20    # poll every 20s (default 15)
    python coach/watch.py --out page.html

--auto is the set-and-forget mode: it sleeps cheaply until Rocket League is
running, watches actively while you play, does one final sweep after you quit
(to catch the last replay you saved), then goes back to sleep. Start it once at
login and never think about it again.

Leave it running while you play. Every time Rocket League writes a replay --
you press Backspace at the end of a match -- it analyses that one match and
rebuilds the page. Nothing else re-parses.

Incremental on purpose. Re-analysing the whole history takes about three
minutes; a single new match takes about a second. The cache is keyed by file
path plus size plus mtime, so a replay that changes on disk is re-analysed and
one that has not is never touched again.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from coach.analyse import analyse_match, discover  # noqa: E402
from coach.page import build  # noqa: E402
from coach.mmrlog import LIVE_LOG, PLAYLISTS, read_all_logs  # noqa: E402
from coach.timeline import DEMOS, load, save_replay_hint  # noqa: E402
from coach.viewer import build_track  # noqa: E402

# Beside the exe when frozen; the bundle dir is a temp folder that vanishes on
# exit, so nothing writable can live there.
#
# Running from source these belong in coach/ -- but the REPORT does not. It
# lives at the project root, which is where the launcher, the README and the
# browser tab all expect it. Sending it to coach/ meant the watcher happily
# rebuilt a file nobody had open, so the page "never updated" while updating
# perfectly every match.
FROZEN = getattr(sys, "frozen", False)
DATA = Path(sys.executable).parent if FROZEN else ROOT / "coach"
REPORT = (DATA if FROZEN else ROOT) / "coach-report.html"
CACHE = DATA / ".replay-cache.json"
RANK = DATA / "rank.json"
REFRESH_SECS = 30    # how often the written page re-loads itself


GAME_PROCESS = "RocketLeague.exe"
IDLE_POLL = 20.0          # how often to check whether the game has started


def game_running():
    """Is Rocket League up? Falls back to tasklist if psutil is missing."""
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            if (proc.info.get("name") or "").lower() == GAME_PROCESS.lower():
                return True
        return False
    except ImportError:
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq " + GAME_PROCESS, "/NH"],
                capture_output=True, text=True, timeout=15).stdout
        except Exception:
            return True        # can't tell -- assume yes rather than go blind
        return GAME_PROCESS.lower() in (out or "").lower()


def auto_loop(cache, modules, out_path, verbose, interval):
    """
    Sleep until Rocket League runs, watch while it does, sweep once after.

    The trailing sweep matters: the replay for your last match is written as
    you leave it, and if the watcher stopped the moment the process died that
    game would be missed until the next session.
    """
    was_up = False
    print("  waiting for %s -- checking every %.0fs" % (GAME_PROCESS, IDLE_POLL))
    try:
        while True:
            up = game_running()
            if up and not was_up:
                print()
                print("  Rocket League started -- watching every %.0fs" % interval)
                sweep(cache, modules, out_path, verbose)
            elif not up and was_up:
                print()
                print("  Rocket League closed -- final sweep")
                sweep(cache, modules, out_path, verbose)
                print("  back to sleep; play again and I will pick it up")
            elif up:
                sweep(cache, modules, out_path, verbose)
            was_up = up
            time.sleep(interval if up else IDLE_POLL)
    except KeyboardInterrupt:
        print()
        print("stopped")


def resolve_identity(explicit=None):
    """
    Work out which player is you, once, before analysing anything.

    Only recent replays carry a `PlayerName` header property -- 7 of 93 here.
    The other 86 have you in the roster and nothing marking you as the
    recorder, so a per-file "whose replay is this" check skips almost your
    entire history. Newest-first, because the name you use now should win over
    the one you used three years ago; the roster vote is the fallback.

    Reads headers only. No network-stream parse, so this costs milliseconds.
    """
    if explicit:
        return explicit
    from collections import Counter
    from coach.replay import read_header

    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime,
                   reverse=True)
    votes = Counter()
    for f in files:
        try:
            h = read_header(f)
        except Exception:
            continue
        if h.get("PlayerName"):
            return h["PlayerName"]
        for pl in (h.get("PlayerStats") or []):
            if pl.get("Name"):
                votes[pl["Name"]] += 1
    return votes.most_common(1)[0][0] if votes else None


def key_for(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.basename(path), st.st_size, int(st.st_mtime))


def load_cache():
    if CACHE.is_file():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            print("cache was unreadable, starting fresh")
    return {"entries": {}, "player": None}


def save_cache(cache):
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(CACHE)


def current_name(matches, cached):
    """The name you use NOW -- the newest replay wins, not the most frequent."""
    for m in sorted(matches, key=lambda x: x.get("date") or "", reverse=True):
        if m.get("resolved_name"):
            return m["resolved_name"]
    return cached


def refresh_mmr():
    """
    Pull MMR out of Rocket League's own log.

    Costs nothing and needs no API: the game writes PartyLeaderMMR while
    matchmaking. Returns (changed, latest_reading) so the caller can mention it
    only when it actually moved.
    """
    readings = read_all_logs()
    if not readings:
        return False, None
    rank_path = RANK
    try:
        d = json.loads(rank_path.read_text(encoding="utf-8"))
    except Exception:
        d = {"source": "Launch.log", "playlists": [], "history": []}

    when = time.strftime("%Y-%m-%d",
                         time.localtime(os.path.getmtime(LIVE_LOG)))
    last = readings[-1]
    field = (PLAYLISTS.get(last["playlist"]) or ("?", "mmr_3v3"))[1] or "mmr_3v3"

    hist = d.setdefault("history", [])
    row = next((h for h in hist if h.get("date") == when), None)
    if row is None:
        row = {"date": when}
        hist.append(row)
    changed = row.get(field) != last["mmr"]
    row[field] = last["mmr"]
    row["source"] = "Launch.log"
    if last["tier_name"] != "?":
        row["rank"] = last["tier_name"]

    # The within-session trace, so the page can show movement game by game
    # rather than only day by day.
    # The whole accumulated trace, across every log we can still see, not just
    # the current one. Restarting the game used to wipe this back to a single
    # reading.
    d["session"] = {
        "date": when,
        "playlist": last["playlist_name"],
        "readings": [r["mmr"] for r in readings],
        "spans": sorted({r.get("date") for r in readings if r.get("date")}),
    }
    for pl in d.get("playlists", []):
        if pl.get("name") == last["playlist_name"]:
            pl["mmr"] = last["mmr"]
            if pl.get("season_peak") and last["mmr"] > pl["season_peak"]:
                pl["season_peak"] = last["mmr"]

    d["history"] = sorted(hist, key=lambda h: h.get("date", ""))
    rank_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return changed, last


def rebuild(cache, out_path):
    # Entries recorded only so a file is never retried are bookkeeping, not
    # matches. Counting them gave a header reading "93 matches - 4W / 3L".
    real = [e for e in cache["entries"].values()
            if not e.get("absent") and not e.get("unreadable")
            and e.get("sections")]
    payload = {
        "player": cache.get("player"),
        "matches": sorted(real, key=lambda m: m.get("date") or ""),
    }
    payload["player"] = current_name(payload["matches"], payload["player"])
    payload["track"] = cache.get("track")
    # The 'be here' marker takes its position from measurement, not a constant.
    # Both files are optional: without them the marker falls back and says so.
    if payload["track"]:
        me = ROOT / "coach" / "shadow-me.json"
        pro = ROOT / "coach" / "pro-benchmark.json"
        sh = {}
        if me.is_file():
            try:
                d = json.loads(me.read_text(encoding="utf-8"))
                sh = {"depth": d.get("depth_held"),
                      "lateral": d.get("lateral_held"),
                      "depth_conceded": d.get("depth_conceded"),
                      "n_held": d.get("n_held"), "n_conceded": d.get("n_conceded"),
                      "matches": d.get("matches")}
            except Exception:
                sh = {}
        if sh and pro.is_file():
            try:
                pd = json.loads(pro.read_text(encoding="utf-8"))
                best = sorted(pd.items())[-1] if pd else None
                if best:
                    sh["pro"] = {"rank": best[0],
                                 "depth": best[1].get("depth_held"),
                                 "matches": best[1].get("matches")}
            except Exception:
                pass
        if sh:
            payload["track"]["shadow"] = sh
    rank_path = RANK
    if rank_path.is_file():
        try:
            payload["rank"] = json.loads(rank_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("  rank.json unreadable (%s), continuing without it" % e)
    Path(out_path).write_text(build(payload, refresh=REFRESH_SECS),
                              encoding="utf-8")
    return len(payload["matches"])


def headline(entry):
    """One line worth reading the moment a match finishes."""
    sl = entry.get("stat_line") or {}
    sc = entry.get("score") or [0, 0]
    team = entry.get("team")
    if team in (0, 1):
        us, them = sc[team], sc[1 - team]
        verdict = "WON" if us > them else "LOST" if us < them else "DREW"
        head = "%s %d-%d" % (verdict, us, them)
    else:
        head = "%d-%d" % (sc[0], sc[1])
    rot = ((entry.get("sections") or {}).get("rotation") or {}).get("data") or {}
    con = (((entry.get("sections") or {}).get("goals") or {})
           .get("data") or {}).get("conceded") or {}
    ts = entry.get("team_size", 0)
    bits = ["%dv%d %s" % (ts, ts, head),
            "%s pts" % sl.get("Score", 0),
            "%s goals" % sl.get("Goals", 0)]
    if rot.get("last_back") is not None:
        bits.append("deepest %.0f%%" % rot["last_back"])
    if con.get("dist_own"):
        bits.append("conceded from %.0f uu" % con["dist_own"])
    return "  ".join(bits)


def _sec(entry, key):
    return ((entry.get("sections") or {}).get(key) or {}).get("data") or {}


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def baseline(cache, key, getter, exclude_date, limit=12):
    """
    Your own recent average for one metric, so a number can be read as better
    or worse than usual rather than in a vacuum. Same playlist only -- a 2v2
    rotation share is not a baseline for a 3v3 one.
    """
    rows = [e for e in cache["entries"].values()
            if e.get("sections") and e.get("date") != exclude_date]
    rows.sort(key=lambda e: e.get("date") or "")
    vals = []
    for e in rows[-limit:]:
        try:
            v = getter(_sec(e, key), e)
        except Exception:
            v = None
        if isinstance(v, (int, float)):
            vals.append(v)
    return _avg(vals)


def post_match(entry, cache):
    """The bit you read straight after a game: what happened, and versus usual."""
    sl = entry.get("stat_line") or {}
    sc = entry.get("score") or [0, 0]
    team = entry.get("team")
    ts = entry.get("team_size", 0)
    if team in (0, 1):
        us, them = sc[team], sc[1 - team]
        verdict = "WON" if us > them else "LOST" if us < them else "DREW"
    else:
        us, them, verdict = sc[0], sc[1], "PLAYED"

    print()
    print("  " + "=" * 62)
    print("  %s %d-%d   %dv%d   %s" % (verdict, us, them, ts, ts,
                                       entry.get("date") or ""))
    print("  " + "=" * 62)
    print("  %s points   %s goals   %s assists   %s saves   %s shots"
          % (sl.get("Score", 0), sl.get("Goals", 0), sl.get("Assists", 0),
             sl.get("Saves", 0), sl.get("Shots", 0)))

    date = entry.get("date")
    METRICS = [
        ("recovery", "heading home when conceding", "%+.0f%%",
         lambda d, e: _avg(d.get("share") or [])),
        ("rotation", "deepest of your team", "%.0f%%",
         lambda d, e: d.get("last_back")),
        ("rotation", "double committed", "%.1f%%",
         lambda d, e: d.get("double_commit")),
        ("boost", "boost held", "%.0f",
         lambda d, e: d.get("avg_held")),
        ("touches", "ball speed off touch", "%.0f uu/s",
         lambda d, e: d.get("avg_exit")),
        ("goals", "distance out when conceding", "%.0f uu",
         lambda d, e: (d.get("conceded") or {}).get("dist_own")),
    ]
    print()
    print("  %-30s %12s %14s" % ("", "this match", "your last 12"))
    print("  " + "-" * 58)
    for key, label, fmt, get in METRICS:
        try:
            now = get(_sec(entry, key), entry)
        except Exception:
            now = None
        if not isinstance(now, (int, float)):
            continue
        was = baseline(cache, key, get, date)
        if was is None:
            print("  %-30s %12s %14s" % (label, fmt % now, "-"))
            continue
        # For these two, lower is better; for the rest, higher.
        lower_better = label in ("double committed", "distance out when conceding")
        better = (now < was) if lower_better else (now > was)
        mark = "  better" if abs(now - was) > abs(was) * 0.05 and better else (
            "  worse" if abs(now - was) > abs(was) * 0.05 else "  same")
        print("  %-30s %12s %14s%s" % (label, fmt % now, fmt % was, mark))

    tips = []
    for sec_ in (entry.get("sections") or {}).values():
        tips.extend(sec_.get("tips") or [])
    if tips:
        print()
        print("  WHERE TO IMPROVE")
        print("  " + "-" * 58)
        for t in tips:
            words, line = t.split(), "   *"
            for w in words:
                if len(line) + len(w) + 1 > 72:
                    print(line)
                    line = "    "
                line += " " + w
            print(line)
    print()


def sweep(cache, modules, out_path, verbose):
    # MMR moves when you queue, not when a replay lands, so it is checked
    # before the no-new-replays early return -- otherwise a session where you
    # forget to save any replays records no rating movement at all.
    moved, last = refresh_mmr()
    if moved and last:
        print("  ~ MMR now %d (%s, %s)"
              % (last["mmr"], last["tier_name"], last["playlist_name"]))

    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime)
    known = set(cache["entries"])
    fresh = [f for f in files if key_for(f) not in known]
    if not fresh:
        if moved:
            rebuild(cache, out_path)
        return 0

    done = 0
    absent = []
    for f in fresh:
        name = os.path.basename(f)
        try:
            m = load(f)
        except Exception as e:
            # Remember the failure so a permanently unparseable replay is not
            # retried on every single poll for the rest of the session.
            cache["entries"][key_for(f)] = {
                "date": "", "team_size": 0, "score": [0, 0], "sections": {},
                "stat_line": {}, "unreadable": str(e)[:120]}
            print("  ! %s could not be parsed: %s" % (name, str(e)[:90]))
            continue

        who = m.resolve(cache["player"]) if cache.get("player") else None
        if who is None and m.name:
            who = m.resolve(m.name) or m.name
        if who is None:
            # Some saved replays are of matches you were not in. Record that
            # once so the file is never re-parsed, and count them rather than
            # printing a line per file.
            cache["entries"][key_for(f)] = {
                "date": m.date or "", "team_size": m.team_size,
                "score": list(m.score), "sections": {}, "stat_line": {},
                "absent": True}
            absent.append(name)
            continue

        try:
            entry, _ = analyse_match(m, who, modules, echo=verbose)
        except Exception:
            print("  ! %s failed during analysis:" % name)
            print("    " + traceback.format_exc().strip().splitlines()[-1])
            continue

        cache["entries"][key_for(f)] = entry
        try:
            cache["track"] = build_track(m, who)
        except Exception:
            cache["track"] = None
        done += 1
        print("  + %s   %s" % (entry.get("date") or name, headline(entry)))
        post_match(entry, cache)

    if absent:
        print("  . %d replay(s) skipped -- you are not a player in them"
              % len(absent))
    if done or absent:
        save_cache(cache)
    if done:
        total = rebuild(cache, out_path)
        print("  page rebuilt from %d matches -> %s" % (total, out_path))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep the coaching page current.")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="seconds between checks (default 15)")
    ap.add_argument("--once", action="store_true",
                    help="catch up on new replays, then exit")
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--verbose", action="store_true",
                    help="print the full analysis for each new match")
    ap.add_argument("--reset", action="store_true",
                    help="throw away the cache and re-analyse everything")
    ap.add_argument("--player", help="your in-game name, if detection is wrong")
    ap.add_argument("--auto", action="store_true",
                    help="idle until Rocket League is running, then watch")
    args = ap.parse_args()
    # A packaged build is double-clicked, not invoked with flags, so the
    # set-and-forget mode is the only sensible default there.
    if getattr(sys, "frozen", False) and not args.once:
        args.auto = True

    if args.reset and CACHE.is_file():
        CACHE.unlink()
        print("cache cleared")

    if not DEMOS.is_dir():
        print("No replay folder at %s" % DEMOS)
        return 1

    cache = load_cache()
    cache["player"] = resolve_identity(args.player) or cache.get("player")
    if not cache["player"]:
        print("Could not work out which player is you -- pass --player.")
        return 1
    modules = discover()
    if not modules:
        print("No metric modules in coach/metrics/.")
        return 1

    print("watching %s" % DEMOS)
    print("  you are %r" % cache["player"])
    print("  %d replays already seen, %d metric families loaded"
          % (len(cache["entries"]), len(modules)))
    print("  " + save_replay_hint())
    print()

    if args.auto:
        auto_loop(cache, modules, args.out, args.verbose, args.interval)
        return 0

    n = sweep(cache, modules, args.out, args.verbose)
    if args.once:
        if not n:
            print("  nothing new")
        return 0

    print("  polling every %.0fs -- ctrl-C to stop" % args.interval)
    try:
        while True:
            time.sleep(args.interval)
            sweep(cache, modules, args.out, args.verbose)
    except KeyboardInterrupt:
        print()
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
