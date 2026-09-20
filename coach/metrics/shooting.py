"""
Shooting accuracy, conversion, and what happens to the ball after you hit it.

Three questions people actually ask about themselves, and one of them cannot
be answered honestly by a replay.

SHOOTING ACCURACY and CONVERSION are fine. A shot is a touch that sends the
ball goalward hard enough to arrive, and it is ON TARGET when the drag-free
projection crosses the goal line inside the frame. That projection is
`touches._aimed_at_net`, which is not new and not guessed: it reproduces the
match header's own shot count for four of six players in the sample match, and
4 of 4 for the player this was built on. So:

    accuracy   = on target / shots
    conversion = goals / on target

PASSING ACCURACY cannot be measured, and this module does not claim to. A pass
is defined by intent -- you meant the ball to reach a team-mate -- and a replay
records only what happened. A deflection off a shin that lands at a team-mate's
feet is indistinguishable from a threaded pass, and a perfect pass that a
team-mate ignores is indistinguishable from a bad one.

What IS measurable is the outcome: who touched the ball next. That is reported
here as POSSESSION RETAINED, deliberately not called passing accuracy, because
the difference matters. It answers "when you give the ball up, who gets it"
-- useful, honest, and not the same question.
"""

from __future__ import annotations

import math

from coach.metrics.touches import _aimed_at_net, detect

TITLE = "Shooting and possession"

GOAL_Y = 5120.0
BALL_R = 92.75

# A touch has to be going somewhere to count as a shot attempt. Below this it
# is a pass, a dribble or a nudge, and counting it would drown the numerator.
SHOT_MIN_SPEED = 900.0

# ...and it has to be aimed goalward rather than merely fast.
SHOT_MIN_FORWARD = 250.0        # uu/s of goalward velocity

# ...and it has to arrive somewhere near the net. Without this a pass across
# the face, or a clear up the wing, counts as a shot attempt purely because it
# is fast and forward: the first version reported 11 attempts where the game
# counted 3 shots, and "missed wide by 2023 uu" -- which is not a miss, it is
# a ball that was never going near the goal.
SHOT_MAX_WIDE = 900.0           # uu beyond the post, at the goal line

# Only from the attacking side of halfway. A clear from your own box that
# happens to be pointed at the far net is not a shot attempt, and counting it
# flatters the denominator and wrecks accuracy.
SHOT_MIN_Y = -900.0             # relative to halfway, in the attacking sense

# How long to wait for the next touch by anyone before calling the ball dead.
NEXT_TOUCH_WINDOW = 6.0


def _sp(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def touch_timeline(match):
    """
    Every player's touches, merged and sorted.

    hit_team cannot answer "who touched it next": it names the TEAM, so a
    team-mate touching after you leaves it unchanged and is invisible. The
    first CHANGE after your touch is therefore always the opponent, which made
    "they got the ball" the only reachable outcome and reported 81-94%
    turnover for everyone. Detecting each player's touches separately is the
    only way to see a team-mate in the chain.
    """
    out = []
    for name in match.teams:
        try:
            for t in detect(match, name):
                out.append((t["t"], name))
        except Exception:
            continue
    out.sort()
    return out


def _next_toucher(timeline, after_t, me):
    """The first player other than `me` to touch, within the window."""
    for t, name in timeline:
        if t <= after_t + 0.12:
            continue
        if t - after_t > NEXT_TOUCH_WINDOW:
            return None
        if name != me:
            return name
    return None


def compute(match, who):
    r = {"ok": False, "shots": 0, "on_target": 0, "goals": 0,
         "retained": 0, "given_away": 0, "shot_speed": [], "misses": []}
    if not match.samples or who not in match.teams:
        return r

    team = match.teams[who]
    sign = match.attack_sign(who)

    try:
        touches = detect(match, who)
        timeline = touch_timeline(match)
    except Exception:
        return r
    if not touches:
        return r

    for t in touches:
        ball, ev = t["ball"], t["evel"]
        speed = _sp(ev)
        forward = ev[1] * sign

        # --- shot attempt? ---
        # Where the drag-free projection crosses the goal line, if it does.
        cross_x = cross_z = None
        gy = sign * GOAL_Y
        if abs(ev[1]) > 1e-6:
            tt = (gy - ball[1]) / ev[1]
            if 0.0 < tt <= 3.0:
                cross_x = ball[0] + ev[0] * tt
                cross_z = ball[2] + ev[2] * tt + 0.5 * -650.0 * tt * tt

        is_shot = (speed >= SHOT_MIN_SPEED
                   and forward >= SHOT_MIN_FORWARD
                   and (ball[1] * sign) >= SHOT_MIN_Y
                   and cross_x is not None
                   and abs(cross_x) <= 893.0 + SHOT_MAX_WIDE)
        if is_shot:
            r["shots"] += 1
            r["shot_speed"].append(speed)
            if _aimed_at_net(ball, ev, sign):
                r["on_target"] += 1
            else:
                # Where it went instead, so a miss has a direction.
                r["misses"].append({"wide": abs(cross_x) - 893.0,
                                    "high": (cross_z or 0.0) - 642.0})
        else:
            # --- not a shot: who got the ball next ---
            nxt = _next_toucher(timeline, t["t"], who)
            if nxt is not None:
                if match.teams.get(nxt) == team:
                    r["retained"] += 1
                else:
                    r["given_away"] += 1

    r["goals"] = sum(1 for g in match.goals_meta
                     if (g.get("PlayerName") or "") == who)
    r["ok"] = r["shots"] > 0 or (r["retained"] + r["given_away"]) > 0
    return r


def _pct(a, b):
    return (100.0 * a / b) if b else 0.0


def render(res):
    if not res.get("ok"):
        return ["  no touches to judge"]
    out = []
    shots, on = res["shots"], res["on_target"]
    out.append("  shot attempts        %5d" % shots)
    out.append("  on target            %5d   %.0f%% accuracy"
               % (on, _pct(on, shots)))
    out.append("  goals                %5d   %.0f%% of those on target"
               % (res["goals"], _pct(res["goals"], on)))
    if res["shot_speed"]:
        avg = sum(res["shot_speed"]) / len(res["shot_speed"])
        out.append("  average shot speed   %5.0f uu/s" % avg)

    wide = [m["wide"] for m in res["misses"] if m["wide"] > 0]
    high = [m["high"] for m in res["misses"] if m["high"] > 0]
    if wide:
        out.append("  missed wide          %5d   by %.0f uu on average"
                   % (len(wide), sum(wide) / len(wide)))
    if high:
        out.append("  missed high          %5d   by %.0f uu on average"
                   % (len(high), sum(high) / len(high)))

    kept, lost = res["retained"], res["given_away"]
    if kept + lost:
        out.append("")
        out.append("  possession after a non-shot touch")
        out.append("    your side kept it  %5d   %.0f%%"
                   % (kept, _pct(kept, kept + lost)))
        out.append("    they got it        %5d   %.0f%%"
                   % (lost, _pct(lost, kept + lost)))
        out.append("    (outcome, not intent -- a replay cannot tell a pass")
        out.append("     from a lucky deflection, so this is not 'passing")
        out.append("     accuracy' and is not labelled as one)")
    return out


def tips(res, match, who):
    if not res.get("ok"):
        return []
    out = []
    shots, on = res["shots"], res["on_target"]

    if shots >= 6 and _pct(on, shots) < 45.0:
        wide = [m["wide"] for m in res["misses"] if m["wide"] > 0]
        high = [m["high"] for m in res["misses"] if m["high"] > 0]
        where = ""
        if len(wide) > len(high) * 2 and wide:
            where = (" Almost all of them are wide rather than high, by %.0f uu "
                     "on average -- that is an aim problem, not a power one."
                     % (sum(wide) / len(wide)))
        elif len(high) > len(wide) * 2 and high:
            where = (" Almost all of them are going over, by %.0f uu on "
                     "average -- you are getting under the ball."
                     % (sum(high) / len(high)))
        out.append(
            "Only %.0f%% of your %d goalward strikes were actually on frame.%s"
            % (_pct(on, shots), shots, where))

    if on >= 5 and _pct(res["goals"], on) < 20.0:
        out.append(
            "You put %d shots on target and scored %d. At this rate the "
            "problem is not creating chances, it is what happens when you "
            "get one." % (on, res["goals"]))

    kept, lost = res["retained"], res["given_away"]
    if kept + lost >= 15 and _pct(lost, kept + lost) > 55.0:
        out.append(
            "After a touch that was not a shot, the other team gets the ball "
            "%.0f%% of the time (%d of %d). That is possession handed over, "
            "not passing -- but it is where your attacks are ending."
            % (_pct(lost, kept + lost), lost, kept + lost))
    return out
