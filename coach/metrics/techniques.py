"""
Named mechanics -- the ones players invented, not the ones the engine has.

mechanics.py measures what the car physically does: speed, air time, flips,
powerslide. This asks a different question: which of the community's named
techniques is this player actually using, and are the ones they reach for
paying off.

WHAT IS AND IS NOT DETECTABLE HERE. Replay frames arrive at roughly 10-30 Hz,
irregularly. That is plenty for anything defined by where a car is over a few
tenths of a second, and not enough for anything defined by its exact attitude
at the instant of a touch. So:

  detected      carry/dribble, flick, air dribble, ceiling play, wall play,
                wave dash, half-flip, air roll, flip reset, pinch, bump
  NOT detected  musty flick, tornado spin, stall, directional air roll style
                -- these differ from their neighbours only in car attitude
                during a window shorter than the gap between two frames, so
                any count would be a guess wearing a number's clothes.

Every threshold below is in unreal units and came from the geometry rather
than from taste: the ceiling is 2044, the side walls +/-4096, the back walls
+/-5120, a car is about 17 tall at rest and the ball's radius is 92.75.
"""

from __future__ import annotations

import math

TITLE = "Named mechanics"

CEILING = 2044.0
SIDE_X = 4096.0
BACK_Y = 5120.0
BALL_R = 92.75

MAX_GAP = 0.5           # s, ignore a jump between two distant frames

# A ball sitting on the roof: close in plan view, and roughly a ball's radius
# plus the car's height above it.
# Measured, not guessed: over 4 matches, frames with the ball within 250 uu
# of a low car have median plan-distance 182 and median dz 152. A ball
# actually sitting on the roof is dxy < ~110 (the ball's own radius is 92.75,
# so its centre can sit that far off the car's centre and still be on it) --
# 165 was letting in a ball merely passing nearby, which is why it never
# resolved into a run long enough to count and carries read as zero.
CARRY_XY = 110.0
CARRY_DZ = (55.0, 210.0)
CARRY_MIN = 0.25        # s, shorter than this is a bounce off the roof

AIR_DRIBBLE_Z = 320.0   # both car and ball clear of the floor
AIR_DRIBBLE_XY = 300.0
AIR_DRIBBLE_MIN = 0.60  # a real air dribble is carried, not just co-located

CEILING_Z = 1850.0      # attached to, or dropping off, the ceiling
WALL_Z = 220.0          # up a wall rather than driving along its base
WALL_NEAR = 220.0       # this close to a wall plane

WAVE_LAND_Z = 45.0      # landed
WAVE_AIR_Z = 165.0      # a wave dash is low: it is not a flip off a wall
WAVE_WINDOW = 0.55      # s from dodge to landing
WAVE_GAIN = 90.0        # uu/s of speed kept or gained through the landing

HALF_FLIP_TURN = 130.0  # deg of yaw change that counts as turning around
HALF_FLIP_WINDOW = 1.3  # s

AIR_ROLL_RATE = 2.2     # rad/s of total angular velocity while airborne
PINCH_GAIN = 1400.0     # uu/s the ball gains in a single step
BUMP_DIST = 190.0       # car-to-car
BUMP_SPEED = 500.0      # closing speed that makes it a bump, not a brush


def _sp(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _near_wall(pos):
    return (abs(pos[0]) > SIDE_X - WALL_NEAR
            or abs(pos[1]) > BACK_Y - WALL_NEAR)


def _turn(a, b):
    """Absolute yaw change, shortest way round."""
    return abs(((b - a + 540.0) % 360.0) - 180.0)


def compute(match, who):
    r = {"ok": False, "counts": {}, "seconds": {}, "live": 0.0,
         "flick_speed": [], "carry_len": []}
    if not match.samples or who not in match.teams:
        return r

    counts = dict.fromkeys(
        ["carry", "flick", "air_dribble", "wave_dash",
         "half_flip", "flip_reset", "pinch", "bump"], 0)
    # ceiling and wall are TIME, not events -- they live in `seconds` only.
    seconds = dict.fromkeys(["carry", "air_dribble", "ceiling", "wall",
                             "air_roll"], 0.0)

    # Dodges, so a wave dash or half-flip can be anchored to a real flip
    # rather than inferred from motion that merely looks like one.
    dodges = [e["t"] for e in match.events
              if e["kind"] == "dodge" and e.get("player") == who]
    counts["flip_reset"] = sum(1 for e in match.events
                               if e["kind"] == "flip_reset"
                               and e.get("player") == who)

    carry_run = air_run = 0.0
    live = 0.0
    prev_ball_sp = None
    touching = set()        # opponents already in contact, so one bump is one

    for i in range(len(match.samples) - 1):
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        me = s["cars"].get(who)
        if me is None:
            continue
        live += dt
        ball = s["ball"]
        pos = me["pos"]

        dxy = math.hypot(pos[0] - ball[0], pos[1] - ball[1])
        dz = ball[2] - pos[2]

        # --- carry / dribble: the ball riding on the roof ---
        if dxy < CARRY_XY and CARRY_DZ[0] < dz < CARRY_DZ[1]:
            carry_run += dt
        else:
            if carry_run >= CARRY_MIN:
                counts["carry"] += 1
                r["carry_len"].append(carry_run)
            carry_run = 0.0

        # --- air dribble: both off the floor, travelling together ---
        if pos[2] > AIR_DRIBBLE_Z and ball[2] > AIR_DRIBBLE_Z \
                and dxy < AIR_DRIBBLE_XY:
            air_run += dt
            seconds["air_dribble"] += dt
        else:
            if air_run >= AIR_DRIBBLE_MIN:
                counts["air_dribble"] += 1
            air_run = 0.0

        # --- surfaces ---
        if pos[2] > CEILING_Z:
            seconds["ceiling"] += dt
        if _near_wall(pos) and pos[2] > WALL_Z:
            seconds["wall"] += dt

        # --- air roll: spinning while genuinely airborne ---
        if pos[2] > 200.0 and _sp(me.get("ang") or (0.0, 0.0, 0.0)) > AIR_ROLL_RATE:
            seconds["air_roll"] += dt

        # --- pinch: the ball leaving far faster than it arrived, against a
        #     surface, with this player on it ---
        bsp = _sp(s["ball_vel"])
        if prev_ball_sp is not None and dxy < 300.0 \
                and bsp - prev_ball_sp > PINCH_GAIN \
                and (_near_wall(ball) or ball[2] < BALL_R + 60.0):
            counts["pinch"] += 1
        prev_ball_sp = bsp

        # --- bump: closing on an opponent fast, without a demolish ---
        # Counted on ENTRY only. Counting every frame the two cars were within
        # range reported 45 bumps a match, which is a frame counter wearing a
        # bump's name.
        now_touching = set()
        for n, c in s["cars"].items():
            if n == who or match.teams.get(n) == match.teams.get(who):
                continue
            if math.dist(pos, c["pos"]) < BUMP_DIST:
                now_touching.add(n)
                if n not in touching:
                    rel = math.dist((0, 0, 0),
                                    tuple(me["vel"][k] - c["vel"][k]
                                          for k in range(3)))
                    if rel > BUMP_SPEED:
                        counts["bump"] += 1
        touching = now_touching

    if carry_run >= CARRY_MIN:
        counts["carry"] += 1
        r["carry_len"].append(carry_run)
    seconds["carry"] = sum(r["carry_len"])

    # --- dodge-anchored techniques -------------------------------------
    idx = {s["t"]: s for s in match.samples}
    times = sorted(idx)

    def near(t):
        lo, hi = 0, len(times) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if times[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return idx[times[lo]]

    for dt_ in dodges:
        s0 = near(dt_)
        me0 = s0["cars"].get(who)
        if me0 is None:
            continue

        # WAVE DASH: dodge low to the ground, land almost at once, keep speed.
        if me0["pos"][2] < WAVE_AIR_Z:
            after = near(dt_ + WAVE_WINDOW)
            me1 = after["cars"].get(who)
            if me1 and me1["pos"][2] < WAVE_LAND_Z:
                if _sp(me1["vel"]) - _sp(me0["vel"]) > -WAVE_GAIN:
                    counts["wave_dash"] += 1

        # HALF-FLIP: a dodge that turns the car around. Only meaningful with a
        # real rotation channel -- velocity-derived heading cannot see it,
        # because the car is travelling backwards for most of the move.
        r0 = me0.get("rot")
        if r0:
            later = near(dt_ + HALF_FLIP_WINDOW)
            me1 = later["cars"].get(who)
            r1 = me1.get("rot") if me1 else None
            if r1 and _turn(r0[0], r1[0]) > HALF_FLIP_TURN \
                    and me0["pos"][2] < 200.0:
                counts["half_flip"] += 1

        # FLICK: a dodge taken while the ball was on the roof, and the ball
        # leaves meaningfully faster than the car was going.
        b0 = s0["ball"]
        if math.hypot(me0["pos"][0] - b0[0], me0["pos"][1] - b0[1]) < CARRY_XY \
                and CARRY_DZ[0] < b0[2] - me0["pos"][2] < CARRY_DZ[1]:
            after = near(dt_ + 0.6)
            gain = _sp(after["ball_vel"]) - _sp(s0["ball_vel"])
            if gain > 250.0:
                counts["flick"] += 1
                r["flick_speed"].append(_sp(after["ball_vel"]))

    r["counts"] = counts
    r["seconds"] = seconds
    r["live"] = live
    r["ok"] = live > 30.0
    return r


def _per10(n, live):
    return (n * 600.0 / live) if live else 0.0


def render(res):
    if not res.get("ok"):
        return ["  not enough live play to judge"]
    c, sec, live = res["counts"], res["seconds"], res["live"]
    out = ["  %-16s %7s   %s" % ("technique", "count", "per 10 min")]
    for key, label in (("carry", "carries"), ("flick", "flicks"),
                       ("air_dribble", "air dribbles"),
                       ("wave_dash", "wave dashes"),
                       ("half_flip", "half-flips"),
                       ("flip_reset", "flip resets"),
                       ("pinch", "pinches"), ("bump", "bumps")):
        out.append("  %-16s %7d   %.1f" % (label, c[key], _per10(c[key], live)))
    out.append("")
    for key, label in (("carry", "ball on your roof"),
                       ("air_dribble", "air dribbling"),
                       ("wall", "up a wall"), ("ceiling", "on the ceiling"),
                       ("air_roll", "air rolling")):
        out.append("  %-20s %5.1f s   %.1f%% of live play"
                   % (label, sec[key], 100.0 * sec[key] / live if live else 0.0))
    if res["carry_len"]:
        out.append("")
        out.append("  longest carry        %5.1f s" % max(res["carry_len"]))
    return out


def tips(res, match, who):
    """
    Deliberately sparse.

    A count of a technique is not evidence that using it more would help --
    that needs the same held-versus-conceded split the positional metrics get,
    and these have not had it. So this reports one thing it can actually
    stand behind: a mechanic used often enough to matter that is not working.
    """
    if not res.get("ok"):
        return []
    out = []
    c, live = res["counts"], res["live"]
    if c["carry"] >= 6 and c["flick"] == 0:
        out.append(
            "You put the ball on your roof %d times and never flicked it. A "
            "carry that ends by driving into someone hands them possession "
            "in front of your net -- either flick it or give it up earlier."
            % c["carry"])
    if res["seconds"]["wall"] > live * 0.06 and c["air_dribble"] == 0:
        out.append(
            "You spend %.0f%% of the match up the walls and never come off "
            "one with the ball. Wall time that does not become a touch is "
            "just time out of the play."
            % (100.0 * res["seconds"]["wall"] / live))
    return out
