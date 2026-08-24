"""
Motion estimation: how far can a car get, and how soon.

The core trick is `ReachCurve`. Naively answering "can I reach this point by
time T?" for every one of ~360 ball-prediction slices means simulating the car
360 times per tick. Instead we simulate the car forward *once* per tick and
store cumulative distance over time; every slice then costs an array lookup.
"""

from __future__ import annotations

import math

from .constants import (
    BOOST_ACCEL,
    BOOST_USAGE_PER_SEC,
    GRAVITY,
    JUMP_HOLD_ACCEL,
    JUMP_IMPULSE,
    JUMP_MAX_HOLD_TIME,
    MAX_SPEED,
    MAX_SPEED_NO_BOOST,
    throttle_accel,
)

# Vertical speed a full held jump buys before boost does anything: the initial
# impulse plus the accelerating hold.
JUMP_LAUNCH_SPEED = JUMP_IMPULSE + JUMP_HOLD_ACCEL * JUMP_MAX_HOLD_TIME
from .vec import Vec3, clamp

# Resolution of the forward simulation. 1/60s over 6s = 360 samples, which
# comfortably covers the ball prediction horizon.
SIM_DT = 1.0 / 60.0
SIM_HORIZON = 6.0
SIM_STEPS = int(SIM_HORIZON / SIM_DT) + 1


class ReachCurve:
    """
    Cumulative ground distance a car can cover over time from its current
    speed and boost, assuming it drives straight and boosts greedily.
    """

    __slots__ = ("dist", "boost_spent", "speed")

    def __init__(self, speed: float, boost: float, allow_boost: bool = True):
        dist = [0.0] * SIM_STEPS
        spent = [0.0] * SIM_STEPS
        v = clamp(speed, 0.0, MAX_SPEED)
        b = boost
        d = 0.0
        used = 0.0

        for i in range(1, SIM_STEPS):
            boosting = allow_boost and b > 0.0 and v < MAX_SPEED
            a = throttle_accel(v)
            if boosting:
                a += BOOST_ACCEL
                burn = BOOST_USAGE_PER_SEC * SIM_DT
                b = max(0.0, b - burn)
                used += burn
            v = min(v + a * SIM_DT, MAX_SPEED if boosting else max(v, MAX_SPEED_NO_BOOST))
            d += v * SIM_DT
            dist[i] = d
            spent[i] = used

        self.dist = dist
        self.boost_spent = spent
        self.speed = speed

    def speed_at(self, t: float) -> float:
        """
        Speed reached after driving for time t.

        Differentiated from the same table `distance_at` reads, so the two can
        never disagree about the same drive.
        """
        if t <= 0.0:
            return clamp(self.speed, 0.0, MAX_SPEED)
        if t >= SIM_HORIZON:
            return (self.dist[-1] - self.dist[-2]) / SIM_DT
        i = int(t / SIM_DT)
        j = min(i + 1, SIM_STEPS - 1)
        if j == i:
            return (self.dist[-1] - self.dist[-2]) / SIM_DT
        return (self.dist[j] - self.dist[i]) / SIM_DT

    def distance_at(self, t: float) -> float:
        """Distance coverable within time t."""
        if t <= 0.0:
            return 0.0
        if t >= SIM_HORIZON:
            # Extrapolate at max speed beyond the simulated horizon.
            return self.dist[-1] + (t - SIM_HORIZON) * MAX_SPEED
        i = int(t / SIM_DT)
        frac = (t - i * SIM_DT) / SIM_DT
        lo = self.dist[i]
        hi = self.dist[min(i + 1, SIM_STEPS - 1)]
        return lo + (hi - lo) * frac

    def boost_used_by(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        i = min(int(t / SIM_DT), SIM_STEPS - 1)
        return self.boost_spent[i]

    def time_for(self, distance: float) -> float:
        """Earliest time this car can cover `distance`. inf if out of horizon."""
        if distance <= 0.0:
            return 0.0
        d = self.dist
        if distance > d[-1]:
            extra = (distance - d[-1]) / MAX_SPEED
            return SIM_HORIZON + extra
        # Cumulative distance is monotonic, so binary search.
        lo, hi = 0, SIM_STEPS - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if d[mid] < distance:
                lo = mid + 1
            else:
                hi = mid
        return lo * SIM_DT


def turn_penalty(car, target: Vec3) -> float:
    """
    Extra seconds lost turning to face `target` before driving at it.

    Cheap heuristic rather than an arc solve: the cost grows with both the
    angle and the current speed (a fast car carves a much wider arc). Beyond
    ~140 degrees the car would half-flip instead, which caps the cost.
    """
    to_target = (target - car.pos).flat()
    if to_target.length_sq() < 1.0:
        return 0.0
    fwd = car.ori.forward.flat().normalized()
    dirn = to_target.normalized()
    angle = math.acos(clamp(fwd.dot(dirn), -1.0, 1.0))

    speed_factor = 0.30 + 0.55 * (car.speed / MAX_SPEED)
    if angle > 2.44:  # ~140 deg: half-flip is faster than turning around
        return 0.55 + 0.25 * speed_factor
    return angle * speed_factor


def ground_time_to(car, target: Vec3, curve: ReachCurve) -> float:
    """Estimated seconds for `car` to drive to `target`."""
    dist = car.pos.flat_dist(target)
    return curve.time_for(dist) + turn_penalty(car, target)


def aerial_feasible(
    car, target: Vec3, time_available: float, gravity_z: float = GRAVITY
) -> tuple[bool, float]:
    """
    Can this car fly to `target` within `time_available`?

    Models the aerial as a single constant-thrust burn: solve for the average
    acceleration that closes the gap, then check it against what boost can
    actually deliver in the time left after rotating to face it.

    Returns (feasible, boost_required).
    """
    T = time_available
    if T <= 0.05:
        return False, 0.0

    delta = target - car.pos

    # Starting velocity for the ballistic solve. A car on the ground is about
    # to jump, and the jump is worth 583 uu/s upward before boost contributes
    # anything -- ignoring it meant solving as though the car had to fight
    # gravity from a standing start, leaving only ~342 uu/s^2 of net climb.
    # The earliest "feasible" ball at z=900 came out at 3.04s, longer than any
    # window the intercept solver ever offers, so aerials were never selected:
    # across 1,273 recorded touches the highest ball struck was 464uu and not
    # one was above the 480uu jump ceiling.
    v0 = car.vel
    if car.on_ground:
        v0 = v0 + Vec3(0.0, 0.0, JUMP_LAUNCH_SPEED)

    # delta = v0*T + 0.5*g*T^2 + 0.5*a*T^2  ->  solve for the thrust a.
    grav = Vec3(0.0, 0.0, gravity_z)
    needed = (delta - v0 * T - grav * (0.5 * T * T)) * (2.0 / (T * T))
    mag = needed.length()
    if mag < 1e-6:
        return True, 0.0

    # We cannot thrust for the whole window -- rotating to point at the thrust
    # vector eats into it. Jumping off the ground costs a little more.
    fwd = car.ori.forward
    angle = math.acos(clamp(fwd.dot(needed.normalized()), -1.0, 1.0))
    turn_time = 0.12 + 0.28 * (angle / math.pi)
    if car.on_ground:
        turn_time += 0.20  # jump + leave the ground

    usable = T - turn_time
    if usable <= 0.02:
        return False, 0.0

    # The same impulse must now be delivered over a shorter burn.
    required_accel = mag * T / usable
    if required_accel > BOOST_ACCEL:
        return False, 0.0

    boost_needed = (required_accel / BOOST_ACCEL) * usable * BOOST_USAGE_PER_SEC
    return boost_needed <= car.boost, boost_needed


def predict_landing(car, gravity_z: float = GRAVITY, max_time: float = 3.0) -> tuple[Vec3, float]:
    """
    Where and when an airborne car returns to the floor, ignoring walls.
    Used by the recovery controller to pre-orient for landing.
    """
    z = car.pos.z
    vz = car.vel.z
    # z + vz*t + 0.5*g*t^2 = 17 (approx resting ride height)
    a = 0.5 * gravity_z
    b = vz
    c = z - 17.0
    disc = b * b - 4 * a * c
    if disc < 0 or abs(a) < 1e-9:
        return car.pos.copy(), max_time
    t = (-b - math.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b + math.sqrt(disc)) / (2 * a)
    t = clamp(t, 0.0, max_time)
    landing = Vec3(car.pos.x + car.vel.x * t, car.pos.y + car.vel.y * t, 17.0)
    return landing, t
