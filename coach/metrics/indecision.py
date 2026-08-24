"""
Indecision -- steering back and forth instead of committing.

`steer` is captured per frame and nothing used it. The signal is sign
reversals: a player who has decided is holding a line, a player who has not is
sawing left-right, and near the ball that costs the challenge.

Two zones, because the same wobble means different things in each. Reversals
out in open space are just driving. Reversals inside NEAR_BALL, with the ball
in play, are hesitation at the moment it matters.

REPORTED WITH A CAVEAT, and the caveat is the point: this has not been shown to
correlate with anything in this player's data yet. It is here to be watched
across a few sessions, not acted on today. A number that looks like a diagnosis
before it has earned it is worse than no number.
"""

from __future__ import annotations

import math

TITLE = "Indecision -- steering reversals"

NEAR_BALL = 2000.0
DEADZONE = 0.15      # ignore micro-corrections; only real steering counts
MAX_GAP = 0.5


def compute(match, who):
    r = {"ok": False, "near_secs": 0.0, "far_secs": 0.0,
         "near_flips": 0, "far_flips": 0, "lobby": {}}
    if not match.samples:
        return r

    state = {}
    for i in range(len(match.samples) - 1):
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        for name, car in s["cars"].items():
            st = state.setdefault(name, {"sign": 0, "near_s": 0.0,
                                         "far_s": 0.0, "near": 0, "far": 0})
            near = math.dist(car["pos"], s["ball"]) <= NEAR_BALL
            if near:
                st["near_s"] += dt
            else:
                st["far_s"] += dt
            steer = car.get("steer") or 0.0
            sign = 0 if abs(steer) < DEADZONE else (1 if steer > 0 else -1)
            if sign and st["sign"] and sign != st["sign"]:
                st["near" if near else "far"] += 1
            if sign:
                st["sign"] = sign

    for name, st in state.items():
        if st["near_s"] > 20.0:
            r["lobby"][name] = 60.0 * st["near"] / st["near_s"]
    me = state.get(who)
    if me:
        r.update(near_secs=me["near_s"], far_secs=me["far_s"],
                 near_flips=me["near"], far_flips=me["far"])
        r["ok"] = me["near_s"] > 20.0
    return r


def render(res):
    if not res.get("ok"):
        return ["  not enough time near the ball to judge"]
    near = 60.0 * res["near_flips"] / max(res["near_secs"], 1e-9)
    far = 60.0 * res["far_flips"] / max(res["far_secs"], 1e-9)
    peers = [v for n, v in res["lobby"].items() if abs(v - near) > 1e-9]
    peers.sort()
    out = [
        "  reversals near the ball %5.0f / min   (%.0f s spent there)"
        % (near, res["near_secs"]),
        "  reversals in open play  %5.0f / min" % far,
    ]
    if peers:
        med = peers[len(peers) // 2]
        out.append("  lobby median near ball  %5.0f / min" % med)
    out.append("  (unvalidated -- watch the trend, do not act on one match)")
    return out


def tips(res, match, who):
    # Deliberately silent. The metric has not been shown to predict anything
    # yet, and a tip is a claim. It earns tips once a few sessions of data show
    # it moving with something that matters.
    return []
