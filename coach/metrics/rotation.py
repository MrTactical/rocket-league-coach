"""
Rotation and team shape.

In 3s the biggest single rank determinant is not mechanics, it is whether the
three of you take turns. This module measures the taking of turns.

Everything here is TIME weighted, not frame weighted. Samples arrive at about
26 Hz but the interval is irregular and it collapses during busy play, so
counting frames quietly over-weights scrambles -- exactly the moments when
everybody is near the ball -- and flatters the rotation numbers.

Dead time is thrown away first. A five minute match yields ~483 s of samples
but only ~335 s of football: the rest is the kickoff countdown, the frozen
ball after a goal, and the replay gap. During those stretches the ball sits in
the net at 2700 uu/s while six cars drive in circles, which is noise in every
metric below.

Definitions, because "last man" means two different things to two players:

  first man       closest teammate to the ball, by ground (x, y) distance.
                  Ground distance, not 3D: the player underneath a ball 1800
                  uu in the air is the one contesting it, and 3D distance says
                  he is the furthest away.
  last man back   the DEEPEST teammate, the one nearest his own goal line.
                  Reported alongside "third to the ball", which is a different
                  question and usually, but not always, the same player.
  rotation cycle  first man -> last man back -> first man again. One complete
                  loop. Ball chasing produces first-man spells with no
                  last-man spell in between.
"""

from __future__ import annotations

import math
import statistics as st

TITLE = "Rotation and team shape"

# Shared with coach/positional.py so the two agree.
NEAR_BALL = 1800.0      # committed to the ball
CROWD_DIST = 1200.0     # two cars this close are in each other's way
GOAL_Y = 5120.0

DT_CAP = 0.20           # a longer gap than this is a hole in the recording
MIN_SPELL = 0.75        # a role held briefer than this is a flicker, not a spell
SEG_GAP = 0.50          # break the timeline where play stops for this long
MIN_SEG = 0.75          # ignore slivers of play left either side of dead time
FROZEN = 0.75           # ball state unchanged this long is stale, not still


# ---------------------------------------------------------------- helpers

def _fd(a, b):
    """Ground distance. Height is deliberately ignored: see the docstring."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vlen(v):
    try:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    except Exception:
        return 0.0


def _wquant(vals, wts, q):
    """Weighted quantile. None on empty input."""
    if not vals:
        return None
    pairs = sorted(zip(vals, wts))
    total = sum(wts)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total:
            return v
    return pairs[-1][0]


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def _live_mask(samples):
    """
    Flag the samples where the ball is actually in play.

    Three kinds of dead time, all found from runs of identical ball state (the
    parser carries the last replicated value forward, so a ball that has
    stopped being replicated looks frozen):

      * parked on the centre spot        -- kickoff countdown
      * frozen while nominally at speed  -- the goal, and the celebration after
      * frozen inside the goal mouth     -- same, for replays with no velocity

    A ball genuinely asleep on the floor mid-pitch stays in play: it is frozen,
    but it is slow and it is not on the centre spot.
    """
    n = len(samples)
    live = [True] * n
    i = 0
    while i < n:
        j = i
        try:
            here = samples[i]["ball"]
            while j + 1 < n and samples[j + 1]["ball"] == here:
                j += 1
            end_t = samples[j + 1]["t"] if j + 1 < n else samples[j]["t"]
            dur = end_t - samples[i]["t"]
            spd = _vlen(samples[i].get("ball_vel") or (0.0, 0.0, 0.0))
            centre = (abs(here[0]) < 40.0 and abs(here[1]) < 40.0
                      and abs(here[2] - 92.75) < 40.0)
            in_net = abs(here[1]) > GOAL_Y - 120.0
            if dur >= FROZEN and (centre or in_net or spd > 200.0):
                for k in range(i, j + 1):
                    live[k] = False
        except Exception:
            pass
        i = j + 1
    return live


def _segments(samples, live):
    """Contiguous stretches of live play, as (start, end) index pairs."""
    segs = []
    cur = None
    for i, ok in enumerate(live):
        if not ok:
            if cur:
                segs.append(tuple(cur))
                cur = None
            continue
        if cur is None:
            cur = [i, i]
        elif samples[i]["t"] - samples[cur[1]]["t"] > SEG_GAP:
            segs.append(tuple(cur))
            cur = [i, i]
        else:
            cur[1] = i
    if cur:
        segs.append(tuple(cur))
    return [s for s in segs
            if samples[s[1]]["t"] - samples[s[0]]["t"] >= MIN_SEG]


# ---------------------------------------------------------------- compute

def compute(match, who):
    try:
        return _compute(match, who)
    except Exception as exc:                    # never take the whole report down
        return {"ok": False, "who": who, "n_mates": 0,
                "reason": "rotation failed: %s" % (exc,)}


def _compute(match, who):
    out = {"ok": False, "who": who, "n_mates": 0, "reason": "",
           "play_s": 0.0, "span_s": 0.0}

    samples = list(getattr(match, "samples", None) or [])
    try:
        name = match.resolve(who) if who else None
    except Exception:
        name = None
    if name is None and who in (getattr(match, "teams", None) or {}):
        name = who
    out["who"] = name or who
    if not samples:
        out["reason"] = "no frames in this replay"
        return out
    if name is None:
        out["reason"] = "player %r is not in this replay" % (who,)
        return out

    try:
        mates = list(match.mates(name))
    except Exception:
        mates = []
    try:
        sign = float(match.attack_sign(name))
    except Exception:
        sign = 1.0
    roster = len(mates) + 1
    nominal = getattr(match, "team_size", 0) or roster
    out.update({"n_mates": len(mates), "roster": roster,
                "team_size": nominal, "mates": mates})

    span = samples[-1]["t"] - samples[0]["t"]
    out["span_s"] = span

    dt = []
    for i in range(len(samples)):
        step = (samples[i + 1]["t"] - samples[i]["t"]
                if i + 1 < len(samples) else 0.0333)
        dt.append(min(max(step, 0.0), DT_CAP))

    live = _live_mask(samples)
    segs = _segments(samples, live)
    live_s = sum(dt[i] for a, b in segs for i in range(a, b + 1))
    fallback = False
    if live_s < 5.0 and span > 20.0:
        # Dead-time detection found nothing sane. Better to measure the whole
        # replay and say so than to report zeroes.
        fallback = True
        live = [True] * len(samples)
        segs = [(0, len(samples) - 1)]
    out["dead_time_fallback"] = fallback

    play = 0.0
    rank_t = {}
    last_t = 0.0
    dc_t = 0.0
    dc_by = dict((k, 0.0) for k in mates)
    md_v, md_w = [], []
    short_t = 0.0
    fm_t = 0.0
    nocover_t = 0.0
    team_n_w = 0.0                 # sum of dt * (own team present in frame)
    first_spells, last_spells = [], []
    cycles = 0
    resets = 0
    rechallenges = 0
    lags = []

    for a, b in segs:
        seq = []                   # confirmed spells this segment, in order
        runF = runL = None
        for i in range(a, b + 1):
            if not live[i]:
                continue
            s = samples[i]
            cars = s.get("cars") or {}
            me = cars.get(name)
            ball = s.get("ball")
            if me is None or me.get("pos") is None or ball is None:
                # Player is gone from this frame: close any open spell rather
                # than bridging across the hole.
                if runF is not None and runF[1] - runF[0] >= MIN_SPELL:
                    first_spells.append((runF[0], runF[1]))
                    seq.append(("F", runF[0], runF[1]))
                if runL is not None and runL[1] - runL[0] >= MIN_SPELL:
                    last_spells.append((runL[0], runL[1]))
                    seq.append(("L", runL[0], runL[1]))
                runF = runL = None
                continue

            t, step = s["t"], dt[i]
            p = me["pos"]
            mate_pos = [cars[k]["pos"] for k in mates
                        if k in cars and cars[k].get("pos") is not None]

            play += step
            team_n_w += step * (1 + len(mate_pos))
            if len(mate_pos) != len(mates):
                short_t += step

            my_bd = _fd(p, ball)
            rank = 1 + sum(1 for q in mate_pos if _fd(q, ball) < my_bd)
            rank_t[rank] = rank_t.get(rank, 0.0) + step

            depth = p[1] * sign
            is_last = bool(mate_pos) and all(q[1] * sign >= depth
                                             for q in mate_pos)
            if is_last:
                last_t += step

            if mate_pos:
                md_v.append(min(_fd(p, q) for q in mate_pos))
                md_w.append(step)
                crowded = False
                for k in mates:
                    c = cars.get(k)
                    q = c.get("pos") if c else None
                    if q is None:
                        continue
                    if (my_bd < NEAR_BALL and _fd(q, ball) < NEAR_BALL
                            and _fd(p, q) < CROWD_DIST):
                        dc_by[k] = dc_by.get(k, 0.0) + step
                        crowded = True
                if crowded:
                    dc_t += step

            if rank == 1:
                fm_t += step
                ball_depth = ball[1] * sign
                if not any(q[1] * sign < ball_depth for q in mate_pos):
                    nocover_t += step
                if runF is None:
                    runF = [t, t + step]
                else:
                    runF[1] = t + step
            elif runF is not None:
                if runF[1] - runF[0] >= MIN_SPELL:
                    first_spells.append((runF[0], runF[1]))
                    seq.append(("F", runF[0], runF[1]))
                runF = None

            if is_last:
                if runL is None:
                    runL = [t, t + step]
                else:
                    runL[1] = t + step
            elif runL is not None:
                if runL[1] - runL[0] >= MIN_SPELL:
                    last_spells.append((runL[0], runL[1]))
                    seq.append(("L", runL[0], runL[1]))
                runL = None

        if runF is not None and runF[1] - runF[0] >= MIN_SPELL:
            first_spells.append((runF[0], runF[1]))
            seq.append(("F", runF[0], runF[1]))
        if runL is not None and runL[1] - runL[0] >= MIN_SPELL:
            last_spells.append((runL[0], runL[1]))
            seq.append(("L", runL[0], runL[1]))

        # A cycle only counts inside one stretch of play. A goal and the
        # kickoff after it reset everybody, so first man before the goal and
        # last man after it is a restart, not a rotation.
        seq.sort(key=lambda x: x[1])
        collapsed = []
        for kind, s0, s1 in seq:
            if collapsed and collapsed[-1][0] == kind:
                continue
            collapsed.append((kind, s0, s1))
        for i in range(len(collapsed) - 2):
            if (collapsed[i][0] == "F" and collapsed[i + 1][0] == "L"
                    and collapsed[i + 2][0] == "F"):
                cycles += 1
        for i in range(len(seq) - 1):
            if seq[i][0] != "F":
                continue
            if seq[i + 1][0] == "L":
                resets += 1
                lags.append(max(0.0, seq[i + 1][1] - seq[i][2]))
            else:
                rechallenges += 1

    out["ok"] = play > 0.0
    if not out["ok"]:
        out["reason"] = "%s never appears in a live frame" % name
        return out

    fdur = [y - x for x, y in first_spells]
    ldur = [y - x for x, y in last_spells]
    mean_team = (team_n_w / play) if play else roster

    out.update({
        "play_s": play,
        "even_share": 100.0 / mean_team if mean_team else 100.0,
        "mean_team": mean_team,
        "rank_share": dict((k, _pct(v, play)) for k, v in rank_t.items()),
        "last_back": _pct(last_t, play),
        "short_handed": _pct(short_t, play),
        "mate_med": _wquant(md_v, md_w, 0.50),
        "mate_p10": _wquant(md_v, md_w, 0.10),
        "double_commit": _pct(dc_t, play),
        "dc_by_mate": dict((k, _pct(v, play)) for k, v in dc_by.items()),
        "fm_spells": len(first_spells),
        "fm_mean": st.mean(fdur) if fdur else 0.0,
        "fm_med": st.median(fdur) if fdur else 0.0,
        "fm_max": max(fdur) if fdur else 0.0,
        "lb_spells": len(last_spells),
        "lb_mean": st.mean(ldur) if ldur else 0.0,
        "cycles": cycles,
        "cycles_per_min": 60.0 * cycles / play if play else 0.0,
        "cycle_s": (play / cycles) if cycles else None,
        "reset_rate": (_pct(resets, resets + rechallenges)
                       if (resets + rechallenges) else None),
        "reset_lag": st.median(lags) if lags else None,
        "no_cover": _pct(nocover_t, fm_t),
        "fm_time": fm_t,
        "segments": len(segs),
    })
    return out


# ---------------------------------------------------------------- render

def _row(label, value, note=""):
    line = "  %-20s %8s" % (label, value)
    return (line + "  " + note).rstrip() if note else line


def render(result):
    r = result or {}
    if not r.get("ok"):
        return ["  %s" % (r.get("reason") or "no rotation data")]

    n = int(r.get("roster") or 1)
    even = r.get("even_share") or 0.0
    size = int(r.get("team_size") or n)
    lines = [_row("play analysed", "%.0f s" % r["play_s"],
                  "of %.0f s in the replay" % (r.get("span_s") or 0.0))]
    if r.get("dead_time_fallback"):
        lines.append("  (dead time not separable -- whole replay measured)")
    if not r.get("n_mates"):
        lines.append("  no teammates in this replay, rotation does not apply")
        return lines

    rs = r.get("rank_share") or {}
    labels = {1: "first to the ball", 2: "second to the ball",
              3: "third to the ball", 4: "fourth to the ball"}
    for k in range(1, n + 1):
        lines.append(_row(labels.get(k, "%dth to the ball" % k),
                          "%.1f %%" % rs.get(k, 0.0),
                          "%.1f %% is even for %dv%d" % (even, size, size)
                          if k == 1 else ""))
    lines.append(_row("last man back", "%.1f %%" % r["last_back"],
                      "deepest of the %d" % n))
    if r.get("short_handed", 0.0) > 2.0:
        lines.append(_row("short-handed", "%.1f %%" % r["short_handed"],
                          "of play with a mate missing"))

    lines.append(_row("first-man spell", "%.1f s" % r["fm_mean"],
                      "mean of %d, longest %.1f s"
                      % (r["fm_spells"], r["fm_max"])))
    if r.get("cycles"):
        lines.append(_row("rotation cycles", "%d" % r["cycles"],
                          "one full loop every %.0f s of play" % r["cycle_s"]))
    else:
        lines.append(_row("rotation cycles", "0", "never completed one"))
    if r.get("reset_rate") is not None:
        note = "of first-man spells"
        if r.get("reset_lag") is not None:
            note += ", median %.1f s to get there" % r["reset_lag"]
        lines.append(_row("reset to last man", "%.0f %%" % r["reset_rate"], note))
    lines.append(_row("no cover behind you", "%.0f %%" % r.get("no_cover", 0.0),
                      "of your time as first man"))

    if r.get("mate_med") is not None:
        lines.append(_row("nearest mate", "%.0f uu" % r["mate_med"], "median"))
    if r.get("mate_p10") is not None:
        lines.append(_row("nearest mate p10", "%.0f uu" % r["mate_p10"],
                          "the tightest tenth of play"))
    pairs = sorted(((v, k) for k, v in (r.get("dc_by_mate") or {}).items()),
                   reverse=True)
    tail = ", ".join("%s %.1f%%" % (k, v) for v, k in pairs if v >= 0.05)
    lines.append(_row("DOUBLE COMMITTED", "%.1f %%" % r["double_commit"], tail))
    return lines


# ---------------------------------------------------------------- tips

def tips(result, match, who):
    r = result or {}
    if not r.get("ok") or not r.get("n_mates"):
        return []
    play = r.get("play_s") or 0.0
    if play < 90.0:
        return []                      # too little football to judge shape

    out = []
    even = r.get("even_share") or 33.3
    size = int(r.get("team_size") or r.get("roster") or 3)
    fm = (r.get("rank_share") or {}).get(1, 0.0)
    lb = r.get("last_back", 0.0)
    dc = r.get("double_commit", 0.0)
    p10 = r.get("mate_p10")
    med = r.get("mate_med")
    pairs = sorted(((v, k) for k, v in (r.get("dc_by_mate") or {}).items()),
                   reverse=True)
    worst = pairs[0] if pairs else (0.0, "your teammate")

    # 1. Share of the ball.
    if fm > even * 1.55:
        out.append(
            "You are first to the ball %.1f%% of the time against an even "
            "%.0f%% for %dv%d, so you are playing most of the touches "
            "yourself. Pick a role per possession: either you challenge or "
            "you are the outlet, and let the other %d cycle through."
            % (fm, even, size, size, r["n_mates"]))
    elif fm > even * 1.22 and (dc > 10.0 or lb < even * 0.75):
        out.append(
            "You are first to the ball %.1f%% of the time (even is %.0f%%) "
            "and %s. That combination is ball chasing rather than leading the "
            "play: after a challenge, commit to leaving -- boost back past "
            "your teammate, not alongside him."
            % (fm, even,
               ("you are double committed for %.1f%% of play" % dc)
               if dc > 10.0 else
               ("you are last man only %.1f%% of it" % lb)))
    elif fm < even * 0.68:
        out.append(
            "You are first to the ball only %.1f%% of the time against an "
            "even %.0f%%, so you are waiting for the play to come to you. "
            "Take the 50/50s you are nearest; a passive first man forces your "
            "teammates to over-commit." % (fm, even))

    # 2. Double commit: the most expensive shape error there is.
    dc_limit = 12.0 if size <= 2 else 10.0
    if dc > dc_limit:
        out.append(
            "You are double committed %.1f%% of play -- you and a teammate "
            "both inside %d uu of the ball AND within %d uu of each other, "
            "worst with %s at %.1f%%. One touch beats both of you and the "
            "pitch is open behind. Target is under %.0f%%."
            % (dc, int(NEAR_BALL), int(CROWD_DIST), worst[1], worst[0],
               dc_limit * 0.7))
    elif p10 is not None and p10 < 750.0:
        out.append(
            "In the tightest tenth of the match you are within %.0f uu of a "
            "teammate, which is inside one car's turning circle. Even when it "
            "does not become a double commit, neither of you can choose "
            "freely at that range." % p10)
    elif med is not None and med < 1700.0:
        out.append(
            "Median distance to your nearest teammate is only %.0f uu. A "
            "%dv%d wants the team spread across the thirds; at that spacing "
            "you are sharing one." % (med, size, size))

    # 3. Depth: never back, or never leaving the back.
    if lb < even * 0.65:
        out.append(
            "You are the last man back only %.1f%% of the time against an "
            "even %.0f%%, so somebody else is covering every time you commit. "
            "That is a debt your teammates pay -- take your turn at the back "
            "post." % (lb, even))
    elif lb > even * 1.35 and fm < even * 1.1:
        out.append(
            "You are the deepest man %.1f%% of the time against an even "
            "%.0f%%, and first to the ball only %.1f%%. You are anchoring "
            "rather than rotating: once a teammate takes the challenge, "
            "follow him up as second man instead of holding the net."
            % (lb, even, fm))

    # 4. Does the rotation actually come round?
    cpm = r.get("cycles_per_min") or 0.0
    if cpm < 1.8 and play > 150.0:
        out.append(
            "You completed %d full rotation cycles in %.0f s of play, %.1f a "
            "minute. A healthy loop -- challenge, off to the back post, "
            "challenge again -- comes round every 15-25 s. Yours is not "
            "coming round." % (r.get("cycles", 0), play, cpm))
    reset = r.get("reset_rate")
    if reset is not None and reset < 35.0 and r.get("fm_spells", 0) >= 8:
        out.append(
            "Only %.0f%% of your first-man spells end with you getting all "
            "the way back to last man before you challenge again; the other "
            "%.0f%% are re-attacks from midfield, which is where counters "
            "come from." % (reset, 100.0 - reset))

    # 5. Committing with the net empty behind you.
    nc = r.get("no_cover", 0.0)
    if nc > 30.0 and r.get("fm_time", 0.0) > 45.0:
        out.append(
            "%.0f%% of your time as first man has no teammate goal-side of "
            "the ball behind you. Before committing, check the shadow: if "
            "nobody is home, delay and shepherd instead of challenging." % nc)

    if r.get("fm_max", 0.0) > 12.0 and r.get("fm_mean", 0.0) > 4.5:
        out.append(
            "Your average first-man spell is %.1f s and the longest ran "
            "%.0f s. Spells that long are you following the ball around the "
            "pitch; a challenge should be 2-3 s and then you are out."
            % (r["fm_mean"], r["fm_max"]))

    return out[:4]
