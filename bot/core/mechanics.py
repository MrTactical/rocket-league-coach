"""
Per-mechanic timing and geometry, from established RLBot prior art.

These are the numbers that decide *which* way of hitting the ball is even
admissible, and how long each takes to set up. They are not our inventions:
the height bands, lead times and jump-duration curves are taken from Botimus
Prime (Darxeal/BotimusPrime), cross-checked against VirxERLU and Kamael, and
the constants against the RLBot wiki and RLUtilities.

The reason this module exists at all is a structural bug. Our solver used to
walk the ball prediction once and return the earliest reachable slice across
every technique. That sounds obviously right and is obviously wrong: a lofted
ball descends into ground range, and the descending low slice is ALWAYS earlier
than any aerial slice, so an aerial can never win. Measured over a real match,
the solver proposed an aerial on 3 ticks out of 7,200.

No established bot solves it that way. They run one earliest-touch search per
mechanic, each restricted to a hard height band and a minimum lead time, then
arbitrate between the surviving typed candidates. Because the bands are
disjoint, "just wait for it to land" is not a move the aerial search is capable
of making -- it is inadmissible rather than merely unattractive.
"""

from __future__ import annotations

from .vec import clamp

# --- ball-height bands, in uu of ball centre ------------------------------
#
# Bands overlap slightly on purpose: several techniques are legitimate for a
# ball at 280uu, and the arbiter picks between them on merit.
GROUND_MAX_Z = 200.0
DODGE_MAX_Z = 300.0
DOUBLE_JUMP_MIN_Z = 250.0
DOUBLE_JUMP_MAX_Z = 550.0
AERIAL_MIN_Z = 500.0
AERIAL_MAX_Z = 800.0
FAST_AERIAL_MIN_Z = 800.0
FAST_AERIAL_MAX_Z = 1800.0

# Lead time an aerial needs, interpolated across its band. A higher ball needs
# to be committed to earlier.
AERIAL_LEAD_LOW = 0.8
AERIAL_LEAD_HIGH = 1.5
FAST_AERIAL_LEAD_LOW = 1.3
FAST_AERIAL_LEAD_HIGH = 2.5

# A ground touch counts as "on a surface" only if the ball is actually settling
# onto it rather than passing through the band at speed.
GROUND_SURFACE_GAP = 120.0
GROUND_NORMAL_SPEED = 300.0


def range_map(v: float, a: float, b: float, c: float, d: float) -> float:
    """Linear interpolation of v from range [a,b] onto [c,d], unclamped ends."""
    if b - a == 0.0:
        return c
    return c + (v - a) / (b - a) * (d - c)


def jump_duration(ball_z: float, extra: float = 0.0) -> float:
    """
    Seconds of jump needed to bring the car's hitbox up to a ball at `ball_z`.

    Botimus Prime's `DodgeStrike.get_jump_duration`. The clamp caps it at 1.5s
    so an absurd height cannot ask for a jump the car does not have.
    """
    return 0.05 + clamp((ball_z - 92.0) / 500.0, 0.0, 1.5) + extra


# A double jump only exists inside the dodge window, and the second jump gains
# roughly 495uu in total. Beyond that the fitted curve keeps returning larger
# numbers -- 1.66s at z=550 -- describing a jump the car cannot perform.
DOUBLE_JUMP_MAX_TIME = 1.25


def double_jump_time_needed(height: float) -> float:
    """
    Seconds for a double jump to reach `height`.

    Botimus Prime's fitted cubic. Valid across the 250-550uu band; outside it
    the fit is not meaningful, which is why the band is enforced separately.
    Callers must also reject results above DOUBLE_JUMP_MAX_TIME: the curve will
    happily quote 1.66s for a 550uu ball, but the dodge window closes at 1.25s,
    so that jump does not exist.
    """
    h = height
    return (
        1.872348977e-8 * h * h * h
        - 1.126747937e-5 * h * h
        + 3.560647225e-3 * h
        - 7.446058499e-3
    )


def aerial_lead(ball_z: float) -> float:
    """Minimum lead time before an aerial to `ball_z` is worth starting."""
    if ball_z < FAST_AERIAL_MIN_Z:
        return range_map(ball_z, AERIAL_MIN_Z, AERIAL_MAX_Z,
                         AERIAL_LEAD_LOW, AERIAL_LEAD_HIGH)
    return range_map(ball_z, FAST_AERIAL_MIN_Z, FAST_AERIAL_MAX_Z,
                     FAST_AERIAL_LEAD_LOW, FAST_AERIAL_LEAD_HIGH)


# --- contact offsets ------------------------------------------------------
#
# How far the car's origin sits from the ball centre at contact. Botimus uses
# 165 for a normal dodge strike and 130 when forced to hit square on.
HIT_OFFSET = 165.0
HIT_OFFSET_PERPENDICULAR = 130.0

# Runway reserved ahead of a jump so the dodge lands ON the ball rather than in
# front of it: jump_duration seconds at roughly 1000 uu/s.
RUNWAY_PER_SECOND = 1000.0


# --- the arrive shift -----------------------------------------------------
#
# Botimus Prime's `Arrive`. The single most valuable idea in the prior art:
# drive at a point pulled BACK along the shot line, and require arrival there
# EARLIER than contact. Because the car must cover extra ground in less time,
# the speed controller commands more speed, so the car is still accelerating
# when it reaches the ball -- which is where the power in a touch comes from.
LERP_T = 0.56
SHIFT_SPEED_FACTOR = 1.6
TIME_SHIFT_FACTOR = 1.2

# Below this the car should not creep toward the ball at all.
MIN_USEFUL_SPEED = 300.0

# --- strike speed floor ---------------------------------------------------
#
# Arriving "on time" is not the goal; arriving with CLOSING SPEED is. A touch
# transfers momentum in proportion to the difference between car and ball
# velocity, so a car pacing a rolling ball imparts nothing however well timed
# it is. Measured in a real match: car 385 uu/s within 250uu of the ball, ball
# 448 uu/s -- the bot was jogging alongside it, nudging, about once a second.
#
# So the approach speed is floored at the ball's own speed plus a margin. Being
# early is a far smaller sin than being feeble: an early arrival still strikes
# the ball, it just strikes it sooner.
STRIKE_SPEED_MARGIN = 900.0
STRIKE_MIN_SPEED = 1250.0

# The Dodge manoeuvre needs about 0.16s from trigger to the flip firing, so the
# window has to open early enough for the flip to land ON the ball rather than
# after it. A floor of 0.28s made it reachable at all -- but only just: at
# 1400 uu/s that window is 390uu wide and the whole approach passes through it
# in a quarter of a second, so the bot managed roughly 0.2 dodges per minute.
# 0.45s gives the trigger room to actually catch.
DODGE_WINDOW_MIN = 0.45


def arrive_shift(distance: float, car_speed: float, additional: float,
                 turn_radius_now: float) -> float:
    """
    How far back along the shot line to aim.

    Decays continuously to zero as the car closes, so the aim point slides
    smoothly onto the true contact point rather than jumping.

    The turn-radius guard matters and is easy to miss: when the car is close,
    a shift large enough to matter puts the aim point inside the car's own
    turning circle, which is precisely what makes a bot orbit the ball. In that
    case take the shift away entirely and drive straight at the target.
    """
    shift = clamp(distance * LERP_T, 0.0, clamp(car_speed, 1500.0, 2300.0) * SHIFT_SPEED_FACTOR)
    if shift - additional * 0.5 < turn_radius_now * 1.1:
        return 0.0
    return shift + additional
