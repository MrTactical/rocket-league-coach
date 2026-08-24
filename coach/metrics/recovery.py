"""
Do you actually go home when the ball gets behind you?

This exists because of one measurement. Across 18 conceded goals, in the four
seconds before the ball crossed the line:

    you             0% of your up/down-pitch movement was homeward,
                    and you finished 273 uu FURTHER from your own net
    your teammates  +19% homeward, closing 1347 uu in the same moments
    opponents       +23% homeward when they were the ones conceding

So it is not what conceding looks like in general -- teammates and opponents
both retreat, and in those exact seconds. The gap is 1620 uu of recovery, and
it belongs to one player.

The headline "you concede from 3737 uu out" was misleading on its own: the
average distance is unremarkable, and only 2 of 18 concedes had you in the
opponent's half. The problem is not where you are. It is that you do not turn.

Measured two ways, because either alone can mislead:
  * RETREAT SHARE -- of your movement along the pitch, how much points home.
    Immune to how fast you were going.
  * GROUND RECOVERED -- how much closer to your net you actually got.
    Immune to pointing the right way while stationary.
"""

from __future__ import annotations

import math

TITLE = "Recovery -- do you turn and go home"

WINDOW = 4.0        # seconds before a goal to judge
MAX_GAP = 0.5       # ignore sample gaps (goal replays, pauses)
GOOD_SHARE = 25.0   # teammates and opponents both sit near +20%


def _homeward(seg, name, own_y, sgn):
    """(share of movement that is homeward, ground recovered) over a segment."""
    goalward = total = 0.0
    for k in range(len(seg) - 1):
        c = seg[k]["cars"].get(name)
        if not c:
            continue
        dt = seg[k + 1]["t"] - seg[k]["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        goalward += (-c["vel"][1] * sgn) * dt
        total += abs(c["vel"][1]) * dt
    if total <= 0.0:
        return None, None
    first = seg[0]["cars"].get(name)
    last = next((seg[i]["cars"].get(name) for i in range(len(seg) - 1, -1, -1)
                 if seg[i]["cars"].get(name)), None)
    if not first or not last:
        return 100.0 * goalward / total, None
    closed = abs(first["pos"][1] - own_y) - abs(last["pos"][1] - own_y)
    return 100.0 * goalward / total, closed


def compute(match, who):
    r = {"ok": False, "n": 0, "share": [], "closed": [],
         "mate_share": [], "mate_closed": [], "away": 0,
         "goalside_start": 0, "goalside_end": 0, "boost": []}

    team = match.teams.get(who)
    if team is None or not match.goals_meta or not match.samples:
        return r
    sgn = match.attack_sign(who)
    own_y = -5120.0 * sgn
    n_frames = len(match.frames) or 1

    for g in match.goals_meta:
        if g.get("PlayerTeam") == team:
            continue                       # only goals against us
        idx = int(len(match.samples) * (g.get("frame", 0) / n_frames))
        idx = max(0, min(len(match.samples) - 1, idx))
        t0 = match.samples[idx]["t"]
        j = idx
        while j > 0 and t0 - match.samples[j]["t"] < WINDOW:
            j -= 1
        seg = match.samples[j:idx + 1]
        if len(seg) < 5 or who not in seg[0]["cars"]:
            continue

        share, closed = _homeward(seg, who, own_y, sgn)
        if share is None:
            continue
        r["n"] += 1
        r["share"].append(share)
        if closed is not None:
            r["closed"].append(closed)
        if share < 0:
            r["away"] += 1

        me0, ball0 = seg[0]["cars"][who], seg[0]["ball"]
        if (me0["pos"][1] * sgn) < (ball0[1] * sgn):
            r["goalside_start"] += 1
        meL = seg[-1]["cars"].get(who)
        if meL and (meL["pos"][1] * sgn) < (seg[-1]["ball"][1] * sgn):
            r["goalside_end"] += 1
        if me0.get("boost") is not None:
            r["boost"].append(me0["boost"])

        for nm in seg[0]["cars"]:
            if nm == who or match.teams.get(nm) != team:
                continue
            ms, mc = _homeward(seg, nm, own_y, sgn)
            if ms is not None:
                r["mate_share"].append(ms)
            if mc is not None:
                r["mate_closed"].append(mc)

    r["ok"] = r["n"] > 0
    return r


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def render(res):
    if not res.get("ok"):
        return ["  no conceded goals in this match"]
    n = res["n"]
    return [
        "  goals conceded          %5d" % n,
        "  movement heading home   %+5.0f %%   (your teammates %+.0f%%)"
        % (_mean(res["share"]), _mean(res["mate_share"])),
        "  ground recovered        %+5.0f uu  (your teammates %+.0f uu)"
        % (_mean(res["closed"]), _mean(res["mate_closed"])),
        "  drove AWAY from your net %4d of %d" % (res["away"], n),
        "  goal-side 4 s before     %4d of %d" % (res["goalside_start"], n),
        "  goal-side at the goal    %4d of %d" % (res["goalside_end"], n),
        "  boost you had            %4.0f      not a resource problem"
        % _mean(res["boost"]),
    ]


def tips(res, match, who):
    if not res.get("ok") or res["n"] < 2:
        return []
    out = []
    share = _mean(res["share"])
    closed = _mean(res["closed"])
    mate_closed = _mean(res["mate_closed"])

    mate = _mean(res["mate_share"])
    if share < GOOD_SHARE:
        # Only invoke the teammates when they actually did better. Quoting a
        # worse number as if it were the standard reads as nonsense -- "only
        # 18%, against -49% for your teammates" argues against itself.
        if mate > share + 5.0:
            out.append(
                "In the 4 s before a goal, only %.0f%% of your movement is "
                "toward your own net, against %.0f%% for your teammates in the "
                "very same seconds. The trigger to drill: the moment the ball "
                "gets behind you, point the car at your NET, not at the ball."
                % (share, mate))
        else:
            out.append(
                "In the 4 s before a goal, only %.0f%% of your movement is "
                "toward your own net -- under the %.0f%% that separates players "
                "who recover from players who watch. (Your teammates were no "
                "better here at %.0f%%, so this one was a team collapse.) The "
                "trigger to drill: ball gets behind you, point at your NET."
                % (share, GOOD_SHARE, mate))
    if closed < 0 < mate_closed:
        out.append(
            "You finish those four seconds %.0f uu FURTHER from your net while "
            "your teammates close %.0f uu. You are not being caught out of "
            "position -- you are declining to come back."
            % (-closed, mate_closed))
    if res["away"] >= max(2, res["n"] // 3):
        out.append(
            "On %d of %d concedes you were mostly driving away from your own "
            "goal as it went in. Boost is not the issue: you had %.0f."
            % (res["away"], res["n"], _mean(res["boost"])))
    if res["goalside_start"] >= res["n"] * 0.6 and res["goalside_end"] <= res["n"] * 0.4:
        out.append(
            "You were goal-side on %d of %d concedes four seconds out, and "
            "only %d of %d by the time it went in. You are losing the position "
            "you already had, not failing to reach it."
            % (res["goalside_start"], res["n"], res["goalside_end"], res["n"]))
    return out
