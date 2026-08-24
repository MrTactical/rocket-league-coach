"""
Read your own play out of your replay history.

    python coach/form.py                 # auto-detects you, all replays
    python coach/form.py --player Name   # if auto-detection picks wrong
    python coach/form.py --last 20       # only the most recent N

Header data only, so this is stat-line coaching, not positional coaching. What
it CAN see: finishing, defensive involvement, playmaking, how you rate against
the lobby you were actually in, and how all of that has moved over time. What
it cannot see: rotation, positioning, boost economy -- those live in the
replay's network stream and need a heavier parser.

Everything is measured RELATIVE TO THE LOBBY. An absolute score of 300 means
nothing on its own; 300 when the lobby averaged 250 means something, and the
same 300 when the lobby averaged 400 means something very different.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach.replay import read_header  # noqa: E402

DEMOS = Path(os.path.expanduser("~/Documents/My Games/Rocket League/TAGame/Demos"))


def num(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def load_matches(limit=None):
    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime)
    if limit:
        files = files[-limit:]
    out = []
    for f in files:
        try:
            h = read_header(f)
        except Exception:
            continue
        ps = h.get("PlayerStats") or []
        if len(ps) < 2:
            continue
        out.append((h, ps))
    return out


def detect_player(matches):
    """The name in PlayerName, or failing that whoever appears most often."""
    votes = Counter()
    for h, ps in matches:
        n = h.get("PlayerName")
        if n:
            votes[n] += 3          # the recording client's own name
        for p in ps:
            if p.get("Name"):
                votes[p["Name"]] += 1
    return votes.most_common(1)[0][0] if votes else None


def rows_for(matches, who):
    out = []
    for h, ps in matches:
        me = next((p for p in ps if p.get("Name") == who), None)
        if me is None:
            continue
        team = num(me.get("Team"), -1)
        mates = [p for p in ps if num(p.get("Team"), -2) == team and p is not me]
        opps = [p for p in ps if num(p.get("Team"), -2) not in (team, -2)]
        if not opps:
            continue
        lobby = [p for p in ps if p is not me]
        out.append({
            "date": h.get("Date", "?"),
            "map": h.get("MapName", "?"),
            "mode": h.get("MatchType", "?"),
            "size": num(h.get("TeamSize"), max(1, len(ps) // 2)),
            # TotalSecondsPlayed is a newer field -- present in 1 of 87 real
            # replays here. NumFrames/RecordFPS is in all of them.
            "secs": (float(h.get("TotalSecondsPlayed") or 0.0)
                     or num(h.get("NumFrames")) / (float(h.get("RecordFPS") or 30.0))),
            "score": num(me.get("Score")),
            "goals": num(me.get("Goals")),
            "assists": num(me.get("Assists")),
            "saves": num(me.get("Saves")),
            "shots": num(me.get("Shots")),
            "team_goals": sum(num(p.get("Goals")) for p in ps
                              if num(p.get("Team"), -2) == team),
            "opp_goals": sum(num(p.get("Goals")) for p in ps
                             if num(p.get("Team"), -2) not in (team, -2)),
            "lobby_score": st.mean([num(p.get("Score")) for p in lobby]) if lobby else 0,
            "mate_score": st.mean([num(p.get("Score")) for p in mates]) if mates else None,
        })
    return out


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def block(title):
    print()
    print(title)
    print("-" * max(46, len(title)))


def report(who, rows):
    n = len(rows)
    wins = sum(1 for r in rows if r["team_goals"] > r["opp_goals"])
    draws = sum(1 for r in rows if r["team_goals"] == r["opp_goals"])
    goals = sum(r["goals"] for r in rows)
    shots = sum(r["shots"] for r in rows)
    saves = sum(r["saves"] for r in rows)
    assists = sum(r["assists"] for r in rows)
    mins = sum(r["secs"] for r in rows) / 60.0 or 1.0

    print("=" * 62)
    print("  %s  --  %d matches, %.0f minutes played" % (who, n, mins))
    print("=" * 62)

    block("Results")
    print("  record            %d W  %d L  %d D   (%.0f%% wins)"
          % (wins, n - wins - draws, draws, pct(wins, n)))
    print("  goals for/against %d / %d  (%+d)"
          % (sum(r["team_goals"] for r in rows), sum(r["opp_goals"] for r in rows),
             sum(r["team_goals"] - r["opp_goals"] for r in rows)))

    block("Your output, per match")
    for label, tot in (("goals", goals), ("assists", assists),
                       ("saves", saves), ("shots", shots)):
        print("  %-16s %5.2f" % (label, tot / n))
    print("  %-16s %5.1f%%   (%d goals from %d shots)"
          % ("shooting", pct(goals, shots), goals, shots))
    print("  %-16s %5.1f   (over %.0f minutes)"
          % ("shots / min", shots / mins, mins))

    block("Against the lobby you were actually in")
    diffs = [r["score"] - r["lobby_score"] for r in rows]
    above = sum(1 for d in diffs if d > 0)
    print("  your score        %.0f avg" % st.mean([r["score"] for r in rows]))
    print("  lobby average     %.0f avg" % st.mean([r["lobby_score"] for r in rows]))
    print("  difference        %+.0f   (above the lobby in %d of %d, %.0f%%)"
          % (st.mean(diffs), above, n, pct(above, n)))
    mate_rows = [r for r in rows if r["mate_score"] is not None]
    if mate_rows:
        md = [r["score"] - r["mate_score"] for r in mate_rows]
        print("  vs your teammates %+.0f   (out-scored them in %.0f%% of matches)"
              % (st.mean(md), pct(sum(1 for d in md if d > 0), len(md))))

    block("Style: where your score comes from")
    # Rough attribution using Rocket League's own scoring weights.
    atk = goals * 100 + shots * 20 + assists * 50
    dfn = saves * 50
    total = atk + dfn or 1
    print("  attacking         %5.0f%%  (goals, shots, assists)" % pct(atk, total))
    print("  defending         %5.0f%%  (saves)" % pct(dfn, total))
    if pct(dfn, total) > 40:
        print("  -> you carry a lot of the defensive load for your team")
    elif pct(dfn, total) < 15:
        print("  -> almost no defensive contribution; you live up the pitch")

    block("Trend: earlier half vs most recent half")
    half = max(1, n // 2)
    old, new = rows[:half], rows[-half:]

    def avg(rs, k):
        return st.mean([r[k] for r in rs]) if rs else 0.0

    for k in ("score", "goals", "assists", "saves", "shots"):
        o, w = avg(old, k), avg(new, k)
        if w > o * 1.05:
            arrow = "up"
        elif w < o * 0.95:
            arrow = "down"
        else:
            arrow = "flat"
        print("  %-16s %7.2f -> %7.2f   %s" % (k, o, w, arrow))
    ow = pct(sum(1 for r in old if r["team_goals"] > r["opp_goals"]), len(old))
    nw = pct(sum(1 for r in new if r["team_goals"] > r["opp_goals"]), len(new))
    print("  %-16s %6.0f%% -> %6.0f%%" % ("win rate", ow, nw))

    block("Playlists")
    sizes = Counter("%dv%d" % (r["size"], r["size"]) for r in rows)
    for mode, c in sizes.most_common():
        sel = [r for r in rows if "%dv%d" % (r["size"], r["size"]) == mode]
        w = pct(sum(1 for r in sel if r["team_goals"] > r["opp_goals"]), len(sel))
        print("  %-8s %3d matches   %3.0f%% wins   %+.0f vs lobby"
              % (mode, c, w, st.mean([r["score"] - r["lobby_score"] for r in sel])))

    block("Pointers")
    tips = []
    sh = pct(goals, shots)
    if shots and sh < 25:
        tips.append(
            "Shooting %.0f%% (%d from %d). Under about 30%% usually means shots "
            "from bad angles or with no power behind them. Fewer, better shots "
            "beats more shots." % (sh, goals, shots))
    elif sh > 45:
        tips.append(
            "Shooting %.0f%% is high, which means you only shoot when it is "
            "clearly on. If the goal count is still low you are being too "
            "selective -- take more." % sh)
    if saves / n > 2.5:
        tips.append(
            "%.1f saves a match is a lot. Either your team leaves you exposed, "
            "or you rotate back too hard and end up doing their job."
            % (saves / n))
    if assists / n < 0.3 and goals / n > 0.8:
        tips.append(
            "Plenty of goals, almost no assists (%.2f a match). You finish "
            "rather than create, and in 3s that gets read quickly."
            % (assists / n))
    if st.mean(diffs) > 40:
        tips.append(
            "You out-score the lobby by %+.0f on average. You are the best "
            "player in most of your games, which is itself a ranking signal."
            % st.mean(diffs))
    elif st.mean(diffs) < -30:
        tips.append(
            "You are %+.0f against the lobby average. These games sit above "
            "your current level -- good for improving, expect losses."
            % st.mean(diffs))
    # Trend-based pointers. These matter more than the absolute numbers: a
    # weakness you are growing into is worth more attention than one you have
    # always had.
    def avg2(rs, k):
        return st.mean([r[k] for r in rs]) if rs else 0.0

    a_old, a_new = avg2(old, "assists"), avg2(new, "assists")
    g_old, g_new = avg2(old, "goals"), avg2(new, "goals")
    s_old, s_new = avg2(old, "score"), avg2(new, "score")

    if a_old > 0.15 and a_new < a_old * 0.7 and g_new > g_old:
        tips.append(
            "Your assists have fallen %.2f -> %.2f while goals rose %.2f -> %.2f. "
            "You have shifted from creating to finishing. In 2s that is often "
            "correct, but check you are not taking the ball off a teammate who "
            "already had the better touch." % (a_old, a_new, g_old, g_new))

    if s_new > s_old * 1.05 and nw < ow:
        tips.append(
            "This is the one worth acting on. Your individual output is UP "
            "(score %.0f -> %.0f) but your win rate is DOWN (%.0f%% -> %.0f%%). "
            "More output with worse results is almost never mechanics -- it is "
            "usually positioning: winning more of the ball but in the wrong "
            "places, and leaving the net open doing it."
            % (s_old, s_new, ow, nw))

    if not tips:
        tips.append(
            "Nothing stands out as a weakness in the stat lines. The next "
            "useful signal is positional, which needs the network-stream "
            "parser.")

    for t in tips:
        print("  * " + t)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Coach yourself from your replay history.")
    ap.add_argument("--player", help="your in-game name (auto-detected otherwise)")
    ap.add_argument("--last", type=int, help="only the most recent N matches")
    args = ap.parse_args()

    if not DEMOS.is_dir():
        print("No replay folder at %s" % DEMOS)
        return 1

    matches = load_matches(args.last)
    if not matches:
        print("No readable replays in %s" % DEMOS)
        print("Save one by holding Backspace at the end of a match.")
        return 1

    who = args.player or detect_player(matches)
    rows = rows_for(matches, who)
    if not rows:
        print("No matches found for %r." % who)
        names = Counter(p.get("Name") for _, ps in matches for p in ps)
        print("Names seen: %s" % ", ".join(str(x) for x, _ in names.most_common(12)))
        return 1

    report(who, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
