"""
Whiffs -- you went for it and did not touch it.

"Committed" is the hard part. Distance alone is useless: you pass close to the
ball constantly without going for it. The definition used here needs all three:

  1. inside REACH of the ball,
  2. actually CLOSING on it (velocity pointing at it, not past it),
  3. moving with intent -- above WALK, so drifting past does not count.

A whiff is then a commitment where the ball's velocity barely changes over the
next third of a second: you arrived and nothing happened to it.

Deliberately conservative. A missed number here is better than a false one --
"you whiffed 40 times" would be nonsense you would rightly ignore.
"""

from __future__ import annotations

import math

TITLE = "Whiffs -- committed and missed"

REACH = 450.0        # close enough that a touch was the intent
WALK = 700.0         # slower than this you were not going for it
CLOSING = 0.55       # cosine of velocity-to-ball angle; must be heading at it
SETTLE = 0.35        # seconds to see whether the ball reacted
CHANGED = 250.0      # ball velocity change that counts as a real touch
COOLDOWN = 1.2       # one commitment per approach, not one per frame


def _closing(car, ball_pos):
    to_ball = [ball_pos[k] - car["pos"][k] for k in range(3)]
    d = math.dist((0, 0, 0), to_ball)
    v = math.dist((0, 0, 0), car["vel"])
    if d < 1e-6 or v < 1e-6:
        return 0.0
    return sum(to_ball[k] * car["vel"][k] for k in range(3)) / (d * v)


def compute(match, who):
    r = {"ok": False, "commits": 0, "whiffs": 0, "speed": [], "height": []}
    if not match.samples:
        return r
    samples = match.samples
    last_t = -99.0

    for i, s in enumerate(samples):
        if s["t"] - last_t < COOLDOWN:
            continue
        me = s["cars"].get(who)
        if not me:
            continue
        if math.dist(me["pos"], s["ball"]) > REACH:
            continue
        if math.dist((0, 0, 0), me["vel"]) < WALK:
            continue
        if _closing(me, s["ball"]) < CLOSING:
            continue

        j = i
        while j < len(samples) - 1 and samples[j]["t"] - s["t"] < SETTLE:
            j += 1
        delta = math.dist(s["ball_vel"], samples[j]["ball_vel"])

        r["commits"] += 1
        last_t = s["t"]
        if delta < CHANGED:
            r["whiffs"] += 1
            r["speed"].append(math.dist((0, 0, 0), me["vel"]))
            r["height"].append(s["ball"][2])

    r["ok"] = r["commits"] > 0
    return r


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def render(res):
    if not res.get("ok"):
        return ["  no clear commitments to the ball detected"]
    rate = 100.0 * res["whiffs"] / res["commits"]
    out = [
        "  committed approaches    %5d" % res["commits"],
        "  whiffed                 %5d     %.0f %%" % (res["whiffs"], rate),
    ]
    if res["speed"]:
        out.append("  speed when you whiffed  %5.0f uu/s" % _mean(res["speed"]))
        out.append("  ball height then        %5.0f uu" % _mean(res["height"]))
    return out


def tips(res, match, who):
    if not res.get("ok") or res["commits"] < 6:
        return []
    rate = 100.0 * res["whiffs"] / res["commits"]
    out = []
    if rate > 25:
        out.append(
            "You whiffed %d of %d committed approaches (%.0f%%). Above about a "
            "quarter usually means arriving too fast to adjust -- the last "
            "car-length is where the touch is decided, and you are still at "
            "%.0f uu/s into it." % (res["whiffs"], res["commits"], rate,
                                    _mean(res["speed"])))
    if res["height"] and _mean(res["height"]) > 300:
        out.append(
            "Your whiffs happen at an average ball height of %.0f uu. Those are "
            "aerial or backboard touches; if they keep missing, the cheaper fix "
            "is to let the ball come down rather than meet it up there."
            % _mean(res["height"]))
    return out
