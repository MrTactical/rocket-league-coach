"""
Contested balls -- who wins the 50/50s.

A challenge is two cars from opposing teams both inside CONTEST_RADIUS of the
ball at the same instant. The winner is decided by where the ball is heading
one second later: if it is travelling toward the opponent's goal the challenger
from this side won it, and vice versa. Balls that end up going nowhere in
particular are neutral, which is a real and common outcome -- a 50/50 that
bounces straight up is not a win.

Deliberately not counted: any challenge starting within 3 s of a kickoff. The
opening contest of every kickoff is a 50/50 by construction and would swamp
the open-play ones, which are the coachable kind.
"""

from __future__ import annotations

import math

TITLE = "50/50s -- contested balls"

CONTEST_RADIUS = 700.0    # both cars this close to the ball
SETTLE = 1.0              # seconds later, see where the ball went
DECISIVE = 400.0          # ball speed toward a goal below this is neutral
COOLDOWN = 2.0            # do not re-count the same scramble every frame
KICKOFF_GUARD = 3.0


def compute(match, who):
    r = {"ok": False, "n": 0, "won": 0, "lost": 0, "neutral": 0,
         "speed_in": [], "opp_speed_in": []}
    team = match.teams.get(who)
    if team is None or not match.samples:
        return r
    sgn = match.attack_sign(who)
    samples = match.samples

    # Kickoff instants: ball parked at the centre spot.
    kickoffs = [s["t"] for s in samples
                if abs(s["ball"][0]) < 100 and abs(s["ball"][1]) < 100
                and math.dist((0, 0, 0), s["ball_vel"]) < 50]

    last_t = -99.0
    for i, s in enumerate(samples):
        if s["t"] - last_t < COOLDOWN:
            continue
        me = s["cars"].get(who)
        if not me:
            continue
        if any(abs(s["t"] - k) < KICKOFF_GUARD for k in kickoffs):
            continue
        ball = s["ball"]
        if math.dist(me["pos"], ball) > CONTEST_RADIUS:
            continue
        opp = [c for n, c in s["cars"].items()
               if match.teams.get(n) not in (None, team)
               and math.dist(c["pos"], ball) <= CONTEST_RADIUS]
        if not opp:
            continue

        # Where is the ball a second later?
        j = i
        while j < len(samples) - 1 and samples[j]["t"] - s["t"] < SETTLE:
            j += 1
        after = samples[j]
        toward = after["ball_vel"][1] * sgn      # +ve = toward their goal

        r["n"] += 1
        last_t = s["t"]
        r["speed_in"].append(math.dist((0, 0, 0), me["vel"]))
        r["opp_speed_in"].append(
            max(math.dist((0, 0, 0), c["vel"]) for c in opp))
        if toward > DECISIVE:
            r["won"] += 1
        elif toward < -DECISIVE:
            r["lost"] += 1
        else:
            r["neutral"] += 1

    r["ok"] = r["n"] > 0
    return r


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def render(res):
    if not res.get("ok"):
        return ["  no contested balls detected"]
    n = res["n"]
    return [
        "  contested balls         %5d" % n,
        "  won / neutral / lost    %2d / %2d / %2d   (%.0f%% won, %.0f%% lost)"
        % (res["won"], res["neutral"], res["lost"],
           _pct(res["won"], n), _pct(res["lost"], n)),
        "  your speed into them    %5.0f uu/s" % _mean(res["speed_in"]),
        "  theirs                  %5.0f uu/s" % _mean(res["opp_speed_in"]),
    ]


def tips(res, match, who):
    if not res.get("ok") or res["n"] < 4:
        return []
    out = []
    mine, theirs = _mean(res["speed_in"]), _mean(res["opp_speed_in"])
    win, loss = _pct(res["won"], res["n"]), _pct(res["lost"], res["n"])

    if theirs > mine + 200:
        out.append(
            "You arrive at contested balls %.0f uu/s slower than the opponent "
            "challenging you (%.0f vs %.0f). Speed decides 50/50s more than "
            "timing does -- either commit early enough to be at pace, or do not "
            "commit and shadow instead." % (theirs - mine, mine, theirs))
    if loss > win + 15:
        out.append(
            "You lose %.0f%% of contested balls and win %.0f%%. Losing the "
            "challenge puts the ball behind you, which is exactly the position "
            "the recovery numbers say you struggle to come back from."
            % (loss, win))
    return out
