"""
Ground driving: steering, throttle, boost and powerslide.

A note on the steering sign, because it is the easiest thing in this codebase
to get backwards. `Mat3.left` is the axis that points along +Y when the car
sits at yaw=0. Working the reference ATBA bot's arithmetic through for a car
at the origin facing +X with a target at +Y, it commands *positive* steer --
and positive steer is right in Rocket League. So a positive local-Y offset
means positive steer. Do not flip this without re-deriving it.
"""

from __future__ import annotations

import math

from rlbot import flat

from ..core.constants import MAX_SPEED, SUPERSONIC_THRESHOLD, speed_for_curvature
from ..core.vec import Vec3, clamp

# Proportional gain on heading error. High enough to be crisp, low enough not
# to saw the wheel back and forth at speed.
STEER_GAIN = 3.0

# Above this heading error, powersliding turns tighter than steering alone.
SLIDE_ANGLE = 1.45
SLIDE_MIN_SPEED = 550.0

# Only burn boost when we are pointed close to where we want to go.
BOOST_ANGLE = 0.30


def local_angle_to(car, target: Vec3) -> float:
    """Signed heading error to `target`: positive means steer right."""
    local = car.ori.to_local(target - car.pos)
    return math.atan2(local.y, local.x)


def steer_toward(car, target: Vec3) -> float:
    return clamp(local_angle_to(car, target) * STEER_GAIN, -1.0, 1.0)


def drive_to(
    car,
    target: Vec3,
    target_speed: float = MAX_SPEED,
    allow_boost: bool = True,
    allow_slide: bool = True,
    respect_turn_limit: bool = True,
    min_turn_speed: float = 400.0,
) -> flat.ControllerState:
    """
    Drive at `target`, trying to arrive travelling at `target_speed`.

    Speed is normally capped by how tight the turn is: there is no point
    carrying 2000uu/s into a corner the car physically cannot hold.

    `min_turn_speed` is the floor that cap will not push below. The cap is
    derived from the curvature needed to *land on* the target, which explodes as
    the distance shrinks -- with the default floor a car closing on a contact
    point gets throttled to a crawl, which turns a shot into a nudge. Striking
    passes a high floor instead.

    `respect_turn_limit=False` removes the cap altogether. Use it almost never:
    a car asking for 2300 into a corner it cannot hold does not miss the corner,
    it orbits the target indefinitely.
    """
    c = flat.ControllerState()

    angle = local_angle_to(car, target)
    c.steer = clamp(angle * STEER_GAIN, -1.0, 1.0)

    dist = car.pos.flat_dist(target)

    # Cap speed to what this turn radius allows, but only when the turn is
    # tight enough and far enough away to matter.
    if respect_turn_limit and dist > 1.0 and abs(angle) > 0.15:
        # Chord geometry: curvature needed to arc onto the target.
        curvature = 2.0 * math.sin(abs(angle)) / max(dist, 1.0)
        turn_limit = speed_for_curvature(curvature)
        target_speed = min(target_speed, max(turn_limit, min_turn_speed))

    speed = car.forward_speed
    error = target_speed - speed

    if error > 40.0:
        c.throttle = 1.0
        c.boost = (
            allow_boost
            and abs(angle) < BOOST_ANGLE
            and speed < SUPERSONIC_THRESHOLD
            and error > 180.0
            and car.on_ground
        )
    elif error < -180.0:
        # Well over target speed: brake rather than coast.
        c.throttle = -1.0
    elif error < -40.0:
        c.throttle = 0.0
    else:
        c.throttle = clamp(error / 120.0, -1.0, 1.0)

    if allow_slide and abs(angle) > SLIDE_ANGLE and car.speed > SLIDE_MIN_SPEED:
        c.handbrake = True
        c.throttle = 1.0
        c.boost = False

    # If we are effectively stationary and pointing away, keep throttle on so
    # the car actually rotates instead of sitting still with the wheel turned.
    if abs(speed) < 60.0 and c.throttle == 0.0:
        c.throttle = 1.0

    return c


def arrive_at(
    car,
    target: Vec3,
    arrival_time: float,
    now: float,
    allow_boost: bool = True,
) -> flat.ControllerState:
    """
    Drive so as to reach `target` at `arrival_time` -- not before.

    Used for intercepts and for holding a defensive position: getting there
    early and sitting still is usually worse than arriving on the beat with
    speed already built up.
    """
    time_left = arrival_time - now
    dist = car.pos.flat_dist(target)

    if time_left <= 0.02:
        needed = MAX_SPEED
    else:
        needed = clamp(dist / time_left, 0.0, MAX_SPEED)

    return drive_to(car, target, needed, allow_boost=allow_boost)


def face_direction(car, direction: Vec3) -> float:
    """Steer value that points the car along `direction` (used when parked)."""
    return steer_toward(car, car.pos + direction.normalized() * 500.0)


def should_half_flip(car, target: Vec3) -> bool:
    """
    True when reversing direction by flipping beats simply turning around.

    Deliberately conservative. A half-flip IS a backwards dodge, so firing it
    loosely makes the car appear to backflip at random -- which is exactly what
    happened once grounded cars could dodge at all: 339 ticks of half-flip in a
    two-minute match, most of them right after a touch, when the next target
    happens to be behind the car.

    It is only worth it when the target is a long way behind AND the car is
    slow enough that turning would genuinely cost more time. At speed, turning
    is faster and keeps the wheels down.
    """
    angle = abs(local_angle_to(car, target))
    return (
        angle > 2.6
        and car.pos.flat_dist(target) > 1800.0
        and car.speed < 1100.0
        and car.on_ground
    )
