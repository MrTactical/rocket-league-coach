"""
Ball contact quality -- how often you touch the ball, and how well.

The replay never says "player X touched the ball". It says where every car and
the ball were, ~30 times a second, and it says which TEAM touched last. Turning
that into a touch list needs two signals agreeing:

  1. PROXIMITY. The player is inside ~300 uu of the ball centre (ball radius
     92.75 plus a car's reach plus replication slop) and is the closest car --
     or is inside 215 uu, which is close enough that a tie with a slightly
     nearer car still means both cars hit it, as happens on every kickoff.
  2. AN IMPULSE. The ball's velocity changed by more than gravity explains
     between two consecutive samples. Without this a car merely driving past a
     flying ball registers as a touch; with it, a keeper standing in the goal
     mouth while a shot flies past no longer gets credit for it.

Both are needed. Proximity alone merges a whole wall carry into one "touch";
impulses alone lose contacts whose velocity update landed in a replication gap.
So proximity defines a contact WINDOW and the impulses inside it split that
window into individual touches -- a five-push ground dribble scores five, which
is exactly the behaviour that makes "touches per minute" mean something.

Calibration on the sample 3v3 (MrTactical, 4-5): 95 of the 96 hit_team
handovers in the match are matched by a detected touch from the team that took
over, and the count of touches aimed inside the net reproduces the header's
shot count for four of the six players exactly (4 of 4 for MrTactical).

What this module is FOR: power and direction. A player who touches often and
softly is dribbling into traffic instead of clearing or shooting.
"""

from __future__ import annotations

import math

TITLE = "Ball contact quality"

# Soccar geometry, unreal units.
GOAL_Y = 5120.0
GOAL_MOUTH = 1786.0 / 2.0
CROSSBAR = 642.0
BALL_R = 92.75
GRAVITY = 650.0

# Detection.
PROX = 300.0        # outer contact radius, centre to centre
CLOSE = 215.0       # close enough to count even if another car is nearer
MERGE = 0.20        # gap that ends a contact window, seconds
IMPULSE = 110.0     # per-frame gravity-corrected dv that counts as a hit
REFRACT = 0.13      # impulses inside this window are one touch
FALLBACK_DV = 250.0  # velocity change needed when no single frame spikes
FALLBACK_WIN = 0.20
MAX_DT = 0.12       # samples further apart than this can't be differenced

# Judgement.
BIG = 1500.0        # "you hit through it"
WEAK = 800.0        # a dink
GROUND_Z = 200.0    # ball centre at or under this is a ground touch
AIR_Z = 500.0       # ball centre over this is an air touch
AERIAL_CAR_Z = 250.0
PRESSURE = 500.0    # an opponent this close makes the touch contested
CHAIN_GAP = 1.2     # consecutive touches inside this are one sequence


def _vlen(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _share(n, d):
    return 100.0 * n / d if d else 0.0


# -- play clock ----------------------------------------------------------


def play_seconds(match):
    """Seconds of LIVE play, from the match clock rather than the sample span.

    The sample span includes kickoff countdowns, goal replays and the warmup
    before the whistle -- 482s of samples for 337s of football in the sample
    match. A touch rate per minute of samples would flatter everyone equally
    and mean nothing. The clock ticks one per second while live and freezes
    otherwise, so a run of samples sharing a clock value is live only if it is
    short and the next value is one step away (down in regulation, up in
    overtime).
    """
    try:
        S = match.samples
    except Exception:
        return 0.0
    if len(S) < 2:
        return 0.0

    segs = []
    for s in S:
        c = s.get("clock")
        if segs and segs[-1][0] == c:
            segs[-1][2] = s["t"]
        else:
            segs.append([c, s["t"], s["t"]])

    live = 0.0
    for k, (c, t0, t1) in enumerate(segs):
        end = segs[k + 1][1] if k + 1 < len(segs) else t1
        nxt = segs[k + 1][0] if k + 1 < len(segs) else None
        span = end - t0
        if (c is not None and nxt is not None and abs(nxt - c) == 1
                and span <= 1.6):
            live += span

    if live < 30.0:
        # No usable clock (very short clip, or a replay that never ticked).
        live = sum(min(S[i + 1]["t"] - S[i]["t"], 0.5)
                   for i in range(len(S) - 1))
    return live


# -- detection -----------------------------------------------------------


def _prep(match):
    """One shared pass: nearest car per frame, and the ball's impulse per frame."""
    S = match.samples
    n = len(S)
    near = [(9e9, None)] * n
    imp = [0.0] * n
    for i, s in enumerate(S):
        try:
            b = s["ball"]
            best = (9e9, None)
            for nm, c in s["cars"].items():
                d = _dist(c["pos"], b)
                if d < best[0]:
                    best = (d, nm)
            near[i] = best
            if i:
                dt = s["t"] - S[i - 1]["t"]
                if 0.0 < dt <= MAX_DT:
                    a, bb = S[i - 1]["ball_vel"], s["ball_vel"]
                    imp[i] = _vlen((bb[0] - a[0], bb[1] - a[1],
                                    bb[2] - a[2] + GRAVITY * dt))
        except Exception:
            continue
    return near, imp


def detect(match, who, pre=None):
    """Every touch by `who`, oldest first. Never raises."""
    try:
        S = match.samples
    except Exception:
        return []
    n = len(S)
    if n < 2 or not who:
        return []
    near, imp = pre if pre else _prep(match)

    # Frames where this player is in contact range and is (near enough to) the
    # closest car.
    hits = []
    for i, s in enumerate(S):
        try:
            c = s["cars"].get(who)
            if not c:
                continue
            d = _dist(c["pos"], s["ball"])
            if d > PROX:
                continue
            if d > near[i][0] + 1e-9 and d > CLOSE:
                continue
            hits.append((i, d))
        except Exception:
            continue

    windows = []
    for i, d in hits:
        if windows and S[i]["t"] - S[windows[-1][-1][0]]["t"] <= MERGE:
            windows[-1].append((i, d))
        else:
            windows.append([(i, d)])

    out = []
    for w in windows:
        try:
            i0, i1 = w[0][0], w[-1][0]
            hi = min(n - 1, i1 + 2)          # let the ball settle two frames
            spikes = [k for k in range(i0, hi + 1) if imp[k] >= IMPULSE]
            clusters = []
            for k in spikes:
                if clusters and S[k]["t"] - S[clusters[-1][-1]]["t"] <= REFRACT:
                    clusters[-1].append(k)
                else:
                    clusters.append([k])

            if not clusters:
                # Nothing spiked -- maybe a soft carry whose change is spread
                # over several frames. Demand a real change over a short span,
                # and demand genuine contact distance, or a ball flying past a
                # parked car sneaks in.
                if min(x[1] for x in w) > CLOSE:
                    continue
                best, bi = 0.0, None
                for a in range(i0, hi + 1):
                    va, ta = S[a]["ball_vel"], S[a]["t"]
                    for b in range(a + 1, hi + 1):
                        dt = S[b]["t"] - ta
                        if dt > FALLBACK_WIN:
                            break
                        vb = S[b]["ball_vel"]
                        d = _vlen((vb[0] - va[0], vb[1] - va[1],
                                   vb[2] - va[2] + GRAVITY * dt))
                        if d > best:
                            best, bi = d, b
                if bi is None or best < FALLBACK_DV:
                    continue
                clusters = [[bi]]

            for cl in clusters:
                a, b = cl[0], cl[-1]
                v_pre = S[max(0, a - 1)]["ball_vel"]
                e = min(n - 1, b + 1)
                ei = max(range(b, e + 1), key=lambda k: _vlen(S[k]["ball_vel"]))
                rng = [k for k in range(max(i0, a - 2), min(i1, b) + 1)
                       if who in S[k]["cars"]]
                if not rng:
                    rng = [k for k in (a, b) if who in S[k]["cars"]]
                if not rng:
                    continue
                ci = min(rng, key=lambda k: _dist(S[k]["cars"][who]["pos"],
                                                  S[k]["ball"]))
                car = S[ci]["cars"][who]
                out.append({
                    "t": S[ci]["t"],
                    "i": ci,
                    "dist": _dist(car["pos"], S[ci]["ball"]),
                    "pre": _vlen(v_pre),
                    "exit": _vlen(S[ei]["ball_vel"]),
                    "evel": S[ei]["ball_vel"],
                    "ball": S[ci]["ball"],
                    "car": car["pos"],
                    "impulse": max(imp[k] for k in cl),
                })
        except Exception:
            continue

    ded = []
    for t in out:
        if ded and t["i"] == ded[-1]["i"]:
            continue
        ded.append(t)
    return ded


def _aimed_at_net(ball, vel, sign):
    """Would this touch, flown straight, cross the goal line inside the frame?

    A drag-free, bounce-free projection. Crude, but it reproduces the header's
    shot count closely (4 of 4 for the sample player), because what the game
    calls a shot is mostly "was it pointed at the net hard enough to get there".
    """
    try:
        gy = sign * GOAL_Y
        vy = vel[1]
        if vy * sign <= 100.0:
            return False
        tt = (gy - ball[1]) / vy
        if tt <= 0.0 or tt > 3.0:
            return False
        x = ball[0] + vel[0] * tt
        if abs(x) > GOAL_MOUTH + BALL_R:
            return False
        z = ball[2] + vel[2] * tt - 0.5 * GRAVITY * tt * tt
        # z below zero means it is on the floor by then, which still goes in.
        return z <= CROSSBAR + BALL_R
    except Exception:
        return False


# -- metrics -------------------------------------------------------------


def compute(match, who) -> dict:
    r = {
        "player": who, "ok": False, "touches": 0, "play_s": 0.0,
        "per_min": 0.0, "lobby_per_min": 0.0, "team_per_min": 0.0,
        "avg_exit": 0.0, "med_exit": 0.0, "peak_exit": 0.0, "peak_t": 0.0,
        "avg_added": 0.0, "big": 0, "big_pct": 0.0, "weak": 0, "weak_pct": 0.0,
        "fwd": 0, "fwd_pct": 0.0, "side": 0, "side_pct": 0.0,
        "back": 0, "back_pct": 0.0, "back_att": 0,
        "avg_z": 0.0, "ground": 0, "low": 0, "air": 0, "aerial": 0,
        "max_z": 0.0, "at_net": 0, "contested": 0, "contested_pct": 0.0,
        "seqs": 0, "longest": 0, "runs": 0, "run_touches": 0,
        "med_gap": 0.0, "killed": 0, "first_gap": 0.0,
    }
    try:
        if match is None or not who:
            return r
        S = getattr(match, "samples", None) or []
        if len(S) < 2:
            return r
        pre = _prep(match)
        ts = detect(match, who, pre)
        r["play_s"] = play_seconds(match)
        r["ok"] = True

        # Lobby context: how busy is everyone else? Cheap, the heavy pass is
        # already done and shared.
        try:
            mates = list(getattr(match, "teams", {}) or {})
            counts = {}
            for nm in mates:
                counts[nm] = len(detect(match, nm, pre)) if nm != who else len(ts)
            if counts and r["play_s"] > 0:
                per = 60.0 / r["play_s"]
                r["lobby_per_min"] = _mean([v * per for v in counts.values()])
                side = [v * per for nm, v in counts.items()
                        if match.teams.get(nm) == match.teams.get(who)]
                r["team_per_min"] = _mean(side)
        except Exception:
            pass

        r["touches"] = len(ts)
        if not ts:
            return r
        if r["play_s"] > 0:
            r["per_min"] = 60.0 * len(ts) / r["play_s"]

        try:
            sign = float(match.attack_sign(who))
        except Exception:
            sign = 1.0
        try:
            opps = list(match.opponents(who))
        except Exception:
            opps = []

        exits = [t["exit"] for t in ts]
        r["avg_exit"] = _mean(exits)
        r["med_exit"] = _median(exits)
        peak = max(ts, key=lambda t: t["exit"])
        r["peak_exit"] = peak["exit"]
        r["peak_t"] = peak["t"]
        r["avg_added"] = _mean([t["exit"] - t["pre"] for t in ts])
        r["big"] = sum(1 for e in exits if e >= BIG)
        r["weak"] = sum(1 for e in exits if e < WEAK)
        r["big_pct"] = _share(r["big"], len(ts))
        r["weak_pct"] = _share(r["weak"], len(ts))

        # Direction: cosine between the ball leaving and the line to their net.
        for t in ts:
            v, b = t["evel"], t["ball"]
            goal = (0.0, sign * GOAL_Y, CROSSBAR * 0.5)
            to = (goal[0] - b[0], goal[1] - b[1], goal[2] - b[2])
            n1, n2 = _vlen(v), _vlen(to)
            cos = 0.0 if n1 < 1e-6 or n2 < 1e-6 else (
                v[0] * to[0] + v[1] * to[1] + v[2] * to[2]) / (n1 * n2)
            if cos > 0.34:
                t["dir"] = "fwd"
            elif cos < -0.34:
                t["dir"] = "back"
            else:
                t["dir"] = "side"
            if t["dir"] == "back" and b[1] * sign > 0:
                r["back_att"] += 1
            if _aimed_at_net(b, v, sign):
                r["at_net"] += 1

        r["fwd"] = sum(1 for t in ts if t["dir"] == "fwd")
        r["side"] = sum(1 for t in ts if t["dir"] == "side")
        r["back"] = sum(1 for t in ts if t["dir"] == "back")
        r["fwd_pct"] = _share(r["fwd"], len(ts))
        r["side_pct"] = _share(r["side"], len(ts))
        r["back_pct"] = _share(r["back"], len(ts))

        zs = [t["ball"][2] for t in ts]
        r["avg_z"] = _mean(zs)
        r["max_z"] = max(zs)
        r["ground"] = sum(1 for z in zs if z <= GROUND_Z)
        r["low"] = sum(1 for z in zs if GROUND_Z < z < AIR_Z)
        r["air"] = sum(1 for z in zs if z >= AIR_Z)
        r["aerial"] = sum(1 for t in ts if t["ball"][2] >= AIR_Z
                          and t["car"][2] >= AERIAL_CAR_Z)

        # Killed momentum: arrived fast, left slow. Fine in defence, a wasted
        # attack anywhere else.
        r["killed"] = sum(1 for t in ts
                          if t["pre"] >= 1200.0 and t["exit"] < t["pre"] - 400.0)

        # Pressure at the moment of contact.
        for t in ts:
            try:
                cars = S[t["i"]]["cars"]
                near = min((_dist(cars[o]["pos"], t["ball"])
                            for o in opps if o in cars), default=None)
            except Exception:
                near = None
            t["opp"] = near
            if near is not None and near <= PRESSURE:
                r["contested"] += 1
        r["contested_pct"] = _share(r["contested"], len(ts))

        # Contact sequences: touches strung together without letting go.
        chains = []
        for t in ts:
            if chains and t["t"] - chains[-1][-1]["t"] <= CHAIN_GAP:
                chains[-1].append(t)
            else:
                chains.append([t])
        r["seqs"] = len(chains)
        r["longest"] = max(len(c) for c in chains)
        r["runs"] = sum(1 for c in chains if len(c) >= 4)
        r["run_touches"] = sum(len(c) for c in chains if len(c) >= 4)
        gaps = [ts[i + 1]["t"] - ts[i]["t"] for i in range(len(ts) - 1)]
        r["med_gap"] = _median(gaps)
        r["_chains"] = [len(c) for c in chains]
        r["_list"] = ts
    except Exception:
        pass
    return r


# -- output --------------------------------------------------------------


def render(result) -> list[str]:
    r = result or {}
    if not r.get("ok"):
        return ["  no usable frames for this player"]
    if not r.get("touches"):
        return ["  touches                      0  -- never got near the ball"]

    L = []
    L.append("  touches                  %5d   %.1f per min of play (lobby %.1f)"
             % (r["touches"], r["per_min"], r["lobby_per_min"]))
    L.append("  contact sequences        %5d   longest run %d touches, %.1fs apart typical"
             % (r["seqs"], r["longest"], r["med_gap"]))
    L.append("  ball speed off touch     %5.0f uu/s avg, %5.0f median"
             % (r["avg_exit"], r["med_exit"]))
    L.append("  hardest touch            %5.0f uu/s at %.0fs"
             % (r["peak_exit"], r["peak_t"]))
    L.append("  speed added per touch    %+5.0f uu/s avg" % r["avg_added"])
    L.append("  big hits (>%4.0f uu/s)    %5d   %3.0f%%" % (BIG, r["big"], r["big_pct"]))
    L.append("  weak dinks (<%3.0f uu/s)   %5d   %3.0f%%" % (WEAK, r["weak"], r["weak_pct"]))
    L.append("  sent at their goal       %5d   %3.0f%%" % (r["fwd"], r["fwd_pct"]))
    L.append("  sent sideways            %5d   %3.0f%%" % (r["side"], r["side_pct"]))
    L.append("  sent back at own goal    %5d   %3.0f%%   %d from the attacking half"
             % (r["back"], r["back_pct"], r["back_att"]))
    L.append("  aimed inside the net     %5d" % r["at_net"])
    L.append("  touch height             %5.0f uu avg, %.0f uu highest" % (r["avg_z"], r["max_z"]))
    L.append("  ground / low / air       %5d / %d / %d   (%d real aerials)"
             % (r["ground"], r["low"], r["air"], r["aerial"]))
    L.append("  contested (opp <%3.0f uu) %5d   %3.0f%%"
             % (PRESSURE, r["contested"], r["contested_pct"]))
    if r["killed"]:
        L.append("  killed a fast ball       %5d   arrived >1200, left 400+ slower" % r["killed"])
    return L


def tips(result, match, who) -> list[str]:
    r = result or {}
    out = []
    if not r.get("ok") or r.get("touches", 0) < 8:
        return out
    n = r["touches"]

    if r["weak_pct"] >= 30.0:
        out.append(
            "%d of your %d touches left the ball under %.0f uu/s (%.0f%%) -- that is a "
            "nudge, not a clear or a shot. Commit to the contact: get the nose "
            "through the ball instead of letting it roll off the side of the car."
            % (r["weak"], n, WEAK, r["weak_pct"]))

    if r["avg_exit"] < 1400.0:
        out.append(
            "Your average touch sends the ball out at only %.0f uu/s. A competent "
            "3s player averages nearer 1800. Add a flick or a boost tap into the "
            "contact so the ball actually travels somewhere your teammates can use."
            % r["avg_exit"])

    if r["peak_exit"] < 2200.0:
        out.append(
            "Your hardest touch all match was %.0f uu/s (at %.0fs). Nothing you hit "
            "was fast enough to beat a set keeper -- practise powershots off a "
            "moving approach rather than shooting from a standstill."
            % (r["peak_exit"], r["peak_t"]))

    if r["back_pct"] >= 30.0 and r["back_att"] >= 3:
        out.append(
            "%.0f%% of your touches (%d) sent the ball back toward your own goal, and "
            "%d of those were taken in the attacking half -- those are giveaways, not "
            "clears. When you are ahead of the ball, take the touch across the face "
            "of goal or into the corner instead of back into your own net's line."
            % (r["back_pct"], r["back"], r["back_att"]))

    if r["runs"] >= 3 and r["avg_exit"] < 1700.0:
        out.append(
            "You strung %d runs of 4+ consecutive touches (%d touches in total) while "
            "averaging only %.0f uu/s off the ball -- that is carrying it into traffic. "
            "Decide on the second touch: shoot it, or pass it wide and reset for boost."
            % (r["runs"], r["run_touches"], r["avg_exit"]))

    if r["contested_pct"] >= 65.0:
        out.append(
            "%.0f%% of your touches had an opponent inside %.0f uu. You are arriving "
            "into challenges rather than into space -- give the first man the 50/50 "
            "and take the second ball, which you can actually hit cleanly."
            % (r["contested_pct"], PRESSURE))

    if n >= 20 and r["air"] <= max(1, int(0.05 * n)):
        out.append(
            "Only %d of your %d touches were above %.0f uu. Every ball over head "
            "height is being conceded -- start contesting them, even just with a "
            "jump-and-block, or the opponents get free possession every time it "
            "goes up." % (r["air"], n, AIR_Z))

    if r["lobby_per_min"] > 0 and r["per_min"] >= 1.7 * r["lobby_per_min"]:
        out.append(
            "You took %.1f touches per minute against a lobby average of %.1f. That "
            "much of the ball in a %dv%d means you are taking your teammates' touches "
            "as well as your own -- after a touch that leaves the ball moving, "
            "peel off and let it run rather than chasing it down again."
            % (r["per_min"], r["lobby_per_min"], match.team_size, match.team_size))

    if r["killed"] >= max(4, int(0.25 * n)):
        out.append(
            "%d touches took a ball arriving above 1200 uu/s and left it 400+ uu/s "
            "slower. You are stopping the ball dead in front of you -- redirect the "
            "pace instead of absorbing it, and it stays out of the danger zone."
            % r["killed"])

    return out
