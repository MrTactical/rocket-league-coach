"""
Intercept solving: where and when can a given car meet the ball, and how?

This runs one earliest-touch search PER MECHANIC rather than a single search
across all of them. See `mechanics.py` for why that distinction is the whole
ballgame: a single earliest-touch scan can never choose an aerial, because a
lofted ball descends into ground range and the descending low slice is always
earlier than any aerial slice. Measured on a real match, the old scan proposed
an aerial on 3 ticks out of 7,200.

Each mechanic is admissible only inside a hard ball-height band and only with a
minimum lead time. The bands make "wait for it to land" impossible for the
aerial search rather than merely unattractive. A small arbiter then picks
between the surviving typed candidates.

Structure follows Botimus Prime (Darxeal/BotimusPrime, maneuvers/strikes/*),
cross-checked against VirxERLU and Kamael.
"""

from __future__ import annotations

from .constants import BALL_RADIUS, SIDE_WALL_X
from .mechanics import (
    AERIAL_MAX_Z,
    AERIAL_MIN_Z,
    DODGE_MAX_Z,
    DOUBLE_JUMP_MAX_TIME,
    DOUBLE_JUMP_MAX_Z,
    DOUBLE_JUMP_MIN_Z,
    FAST_AERIAL_MAX_Z,
    FAST_AERIAL_MIN_Z,
    GROUND_MAX_Z,
    aerial_lead,
    double_jump_time_needed,
    jump_duration,
)
from .physics import ReachCurve, aerial_feasible, turn_penalty
from .vec import Vec3

# Roughly how far the contact point sits from the ball centre: ball radius plus
# the front of the car.
CONTACT_OFFSET = BALL_RADIUS + 60.0

# Only every Nth prediction slice is tested; the prediction is 120Hz, so a
# stride of 3 still gives 40ms resolution -- far finer than our decisions need.
SLICE_STRIDE = 3

# Kinds, in the order the arbiter prefers to consider them.
GROUND = "ground"
DODGE = "dodge"
DOUBLE_JUMP = "double_jump"
AERIAL = "aerial"
WALL = "wall"

# An aerial needs boost in hand, but the old affordability test was
# `dt < boost / 33.3` -- the cost of boosting for the ENTIRE flight at full
# thrust. Nothing aerials like that. `aerial_feasible` has already solved the
# actual thrust required and returned it as `boost_cost`, and its own
# feasibility condition guarantees that cost is affordable; the arbiter then
# ignored that number and applied a test 1.2-2.9x stricter, which is why the
# bot essentially never flew. Use the cost the solver computed.
# Measurably inert: sweeping this 0/10/20/25 changed the outcome by 5 ticks in
# 12,431 (0.04%), because the reserve gate below always binds first. Kept at a
# nominal floor rather than deleted so the two call sites still read as a
# deliberate "needs some boost" rather than looking like an oversight.
AERIAL_MIN_BOOST = 10.0
# Held back for the landing and the recovery afterwards.
# 12 -> 4, worth +20.4% aerial candidates when it was measured in the
# 2026-08-23 audit (notes/why-the-bots-never-fly.md). Held back then so two
# behavioural changes in one run stayed attributable; shipped now with the
# inert MIN_BOOST, because both are boost-gate changes and move together.
AERIAL_BOOST_RESERVE = 4.0

# How much earlier a dodge must be than a plain ground touch to be preferred.
DODGE_PREFERENCE = 0.1
# A double jump may be slightly later than the ground option and still win,
# because it meets the ball above the ground game where it is harder to defend.
DOUBLE_JUMP_GRACE = 0.2


class Intercept:
    """
    Where the ball will be when we can reach it, and by what means.

    `pos` is the BALL CENTRE, not a driving target. Callers needing somewhere to
    aim the car apply their own contact offset -- see `strike.contact_point`.
    `approach` is the offset point this solver tested against.

    That distinction matters: returning a pre-offset position here once caused
    the striker to offset a second time, parking the car ~300uu from a ball it
    had to be within ~150uu of to touch.
    """

    __slots__ = (
        "time", "dt", "pos", "approach", "ball_vel",
        "kind", "feasible", "boost_cost", "slice_index",
        "aerial_reject", "aerial_cost",
    )

    def __init__(
        self,
        time: float = 0.0,
        dt: float = 99.0,
        pos: Vec3 | None = None,
        ball_vel: Vec3 | None = None,
        kind: str = "none",
        feasible: bool = False,
        boost_cost: float = 0.0,
        slice_index: int = -1,
        approach: Vec3 | None = None,
    ):
        self.time = time
        self.dt = dt
        self.pos = pos or Vec3()
        self.approach = approach if approach is not None else self.pos
        self.ball_vel = ball_vel or Vec3()
        self.kind = kind
        self.feasible = feasible
        self.boost_cost = boost_cost
        self.slice_index = slice_index
        # Why no aerial was offered, and what the cheapest one would have cost.
        #
        # The aerial gates used to fail completely silently: an unaffordable
        # aerial returned feasible=False and was simply never recorded, so
        # nothing downstream could tell "too expensive" from "out of reach"
        # from "never searched". That is why the real cause took an audit to
        # find, and why `adecl` -- which fires only AFTER an aerial has already
        # been chosen -- was misread as evidence that aerials were fine.
        self.aerial_reject = ""
        self.aerial_cost = 0.0

    @property
    def is_air(self) -> bool:
        return self.kind in (AERIAL, DOUBLE_JUMP)

    def __repr__(self) -> str:
        return f"Intercept({self.kind}, dt={self.dt:.2f}s, {self.pos}, ok={self.feasible})"


def _on_wall(pos: Vec3) -> bool:
    return abs(pos.x) > SIDE_WALL_X - BALL_RADIUS - 40.0 and pos.z > 200.0


def _drivable(car, curve: ReachCurve, ground_target: Vec3, usable: float) -> bool:
    """Can we drive to this ground point with `usable` seconds available?"""
    if usable <= 0.0:
        return False
    lead = usable - turn_penalty(car, ground_target)
    if lead <= 0.0:
        return False
    return curve.distance_at(lead) >= car.pos.flat_dist(ground_target)


class _Facing:
    """Minimal orientation stand-in: aerial_feasible only reads `.forward`."""

    __slots__ = ("forward",)

    def __init__(self, forward: Vec3):
        self.forward = forward


class _LaunchState:
    """A virtual car at the point an aerial would actually leave the ground."""

    __slots__ = ("pos", "vel", "ori", "on_ground", "boost")

    def __init__(self, pos: Vec3, vel: Vec3, forward: Vec3, boost: float):
        self.pos = pos
        self.vel = vel
        self.ori = _Facing(forward)
        self.on_ground = True
        # The drive phase runs on a no-boost curve, so the full tank is still
        # in hand at the moment of launch.
        self.boost = boost


# Flight time as a fraction of the whole window, tried in order. 1.0 is the old
# behaviour -- fly the entire way from where the car stands -- so this search is
# a strict superset of what it replaces and can never find less.
DRIVE_FRACTIONS = (1.0, 0.75, 0.55, 0.40)
# Below this an "aerial" is really a jump; the double-jump band owns that.
MIN_FLIGHT_TIME = 0.35


def _aerial_after_drive(car, ground_curve: ReachCurve, target: Vec3, dt: float):
    """
    Price an aerial as drive-then-fly instead of as one long burn.

    GROUND, DODGE and DOUBLE_JUMP all subtract a drive segment before their
    mechanic fires. The aerial branch did not -- it called `aerial_feasible`
    straight from the car's current position, charging boost for the ENTIRE
    displacement as powered flight. Measured consequences:

      * median priced cost 47.4 boost, and nearly flat with distance (40.8
        under 500uu horizontal, only 58.2 beyond 4000uu), because the solver
        was buying the horizontal closing speed with thrust rather than wheels
      * 69.5% of every kinematically flyable aerial destroyed on cost
      * a car at 1200 uu/s angled 45, 90 or 150 degrees off the ball line was
        called infeasible at EVERY window from 0.5 to 4.0s, where drive-then-fly
        succeeds at all three

    The ground phase deliberately runs on a NO-BOOST curve. Boost spent driving
    is boost unavailable for the climb, and the whole point is to arrive under
    the ball with a full tank -- which is what a player actually does.

    Returns (feasible, boost_cost) for the cheapest split found.
    """
    ground_under = Vec3(target.x, target.y, 17.0)
    to_target = ground_under - car.pos
    flat_dist = to_target.flat_length()

    best = None
    for frac in DRIVE_FRACTIONS:
        flight = dt * frac
        if flight < MIN_FLIGHT_TIME:
            continue
        drive_t = dt - flight

        if drive_t <= 0.0:
            launch = car
        else:
            lead = drive_t - turn_penalty(car, ground_under)
            if lead <= 0.0:
                continue
            # How far under the ball we can get on wheels alone.
            reach = min(ground_curve.distance_at(lead), flat_dist)
            direction = to_target.flat()
            if direction.length_sq() < 1.0:
                direction = car.ori.forward.flat()
            direction = direction.normalized()
            speed = ground_curve.speed_at(lead)
            launch = _LaunchState(
                car.pos + direction * reach,
                direction * speed,
                direction,
                car.boost,
            )

        ok, cost = aerial_feasible(launch, target, flight)
        if ok and (best is None or cost < best):
            best = cost

    return (best is not None), (best or 0.0)


def find_intercept(
    car,
    prediction,
    now: float,
    curve: ReachCurve | None = None,
    aim_dir: Vec3 | None = None,
    allow_aerial: bool = True,
    max_time: float = 6.0,
    contested: bool = False,
    shoot_target: Vec3 | None = None,
) -> Intercept:
    """
    Best way for this car to meet the ball.

    Returns the BALL CENTRE in `Intercept.pos`, tagged with the mechanic. Set
    `contested` when an opponent is closing on the ball, which makes the arbiter
    prefer getting there sooner over getting there tidily.
    """
    if prediction is None or not prediction.slices:
        return Intercept()

    if curve is None:
        curve = ReachCurve(car.speed, car.boost)
    # Built lazily: only the aerial branch needs it, and most calls never
    # reach the aerial band.
    ground_curve = None

    slices = prediction.slices
    found: dict[str, Intercept] = {}
    nearest_miss = None
    # Cheapest aerial we saw but could not take, and why.
    aerial_reject = ""
    aerial_cost = 0.0

    for i in range(0, len(slices), SLICE_STRIDE):
        sl = slices[i]
        dt = sl.game_seconds - now
        if dt <= 0.0:
            continue
        if dt > max_time:
            break

        bpos = Vec3(sl.physics.location)
        z = bpos.z

        # Stand off from the ball centre on the side we would strike from.
        if aim_dir is not None:
            target = bpos - aim_dir.normalized() * CONTACT_OFFSET
        else:
            target = bpos - (bpos - car.pos).normalized() * CONTACT_OFFSET
        ground_target = Vec3(target.x, target.y, 17.0)
        bvel = Vec3(sl.physics.velocity)

        def record(kind: str, boost_cost: float = 0.0):
            found[kind] = Intercept(
                sl.game_seconds, dt, bpos, bvel, kind, True, boost_cost, i,
                approach=target,
            )

        # --- wall: drivable surface, but only if we can actually climb it ---
        if WALL not in found and _on_wall(bpos) and z < 600.0:
            if car.ori.up.x * (1.0 if bpos.x > 0 else -1.0) < -0.5:
                if _drivable(car, curve, Vec3(target.x, target.y, target.z), dt):
                    record(WALL, curve.boost_used_by(dt))
            continue

        # --- ground: ball settling on the floor -----------------------------
        if GROUND not in found and z <= GROUND_MAX_Z:
            if abs(bvel.z) < 300.0 or z <= BALL_RADIUS + 30.0:
                if _drivable(car, curve, ground_target, dt):
                    record(GROUND, curve.boost_used_by(dt))

        # --- dodge: low ball, jump into it ----------------------------------
        if DODGE not in found and z < DODGE_MAX_Z:
            need = jump_duration(z)
            if dt >= need and _drivable(car, curve, ground_target, dt - need):
                record(DODGE, curve.boost_used_by(max(0.0, dt - need)))

        # --- double jump: the mid band --------------------------------------
        if DOUBLE_JUMP not in found and DOUBLE_JUMP_MIN_Z <= z <= DOUBLE_JUMP_MAX_Z:
            need = double_jump_time_needed(z)
            # The curve extrapolates past what a double jump can actually do.
            if need <= DOUBLE_JUMP_MAX_TIME and dt >= need and _drivable(
                car, curve, ground_target, dt - need
            ):
                record(DOUBLE_JUMP, curve.boost_used_by(max(0.0, dt - need)))

        # --- aerial: high ball, committed to early ---------------------------
        if allow_aerial and AERIAL not in found and AERIAL_MIN_Z <= z <= FAST_AERIAL_MAX_Z:
            if dt >= aerial_lead(z):
                if ground_curve is None:
                    ground_curve = ReachCurve(car.speed, car.boost, allow_boost=False)
                ok, cost = _aerial_after_drive(car, ground_curve, target, dt)
                if ok:
                    record(AERIAL, cost)
                else:
                    if not aerial_reject or (cost and cost < aerial_cost):
                        aerial_reject = "unreachable"
                        aerial_cost = cost
                    if nearest_miss is None:
                        nearest_miss = (sl, dt, bpos, bvel, target, AERIAL)
            elif allow_aerial and AERIAL not in found and z > AERIAL_MIN_Z:
                if not aerial_reject:
                    aerial_reject = "no_lead"

        if nearest_miss is None and not found:
            nearest_miss = (sl, dt, bpos, bvel, target, GROUND)

        # Everything worth comparing is in hand.
        if len(found) >= 4:
            break

    # An aerial was found but the arbiter passed it over: record which clause.
    if AERIAL in found:
        a = found[AERIAL]
        if car.boost < AERIAL_MIN_BOOST:
            aerial_reject, aerial_cost = "min_boost", a.boost_cost
        elif a.boost_cost > car.boost - AERIAL_BOOST_RESERVE:
            aerial_reject, aerial_cost = "reserve", a.boost_cost
        else:
            ref = found.get(DODGE) or found.get(GROUND)
            if ref is not None and a.time >= ref.time:
                aerial_reject, aerial_cost = "slower_than_" + ref.kind, a.boost_cost

    chosen = _arbitrate(found, car, contested, shoot_target)
    if chosen is not None:
        if chosen.kind != AERIAL:
            chosen.aerial_reject = aerial_reject
            chosen.aerial_cost = aerial_cost
        return chosen

    # Nothing reachable: report the closest attempt so callers can still use the
    # position for shadowing rather than getting a null.
    if nearest_miss is not None:
        sl, dt, bpos, bvel, target, kind = nearest_miss
        miss = Intercept(sl.game_seconds, dt, bpos, bvel, kind, False, 0.0, -1,
                         approach=target)
        miss.aerial_reject = aerial_reject
        miss.aerial_cost = aerial_cost
        return miss
    return Intercept()


def _arbitrate(
    found: dict[str, Intercept], car, contested: bool,
    shoot_target: Vec3 | None = None,
) -> Intercept | None:
    """
    Choose between the typed candidates.

    Follows Botimus Prime's `direct_shot`: take the aerial when we can afford it
    and it genuinely gets there first, otherwise prefer meeting the ball in the
    air over waiting for it, otherwise dodge, otherwise drive.
    """
    ground = found.get(GROUND)
    dodge = found.get(DODGE)
    double_jump = found.get(DOUBLE_JUMP)
    aerial = found.get(AERIAL)
    wall = found.get(WALL)

    # --- aerial ----------------------------------------------------------
    # Requires real boost, must beat the grounded options, and the flight has
    # to fit inside the boost we are carrying.
    if aerial is not None and car.boost >= AERIAL_MIN_BOOST:
        reference = dodge or ground
        if reference is None or aerial.time < reference.time:
            if aerial.boost_cost <= car.boost - AERIAL_BOOST_RESERVE:
                return aerial

    # --- double jump -------------------------------------------------------
    if double_jump is not None:
        if ground is None or double_jump.time < ground.time + DOUBLE_JUMP_GRACE:
            return double_jump

    # --- dodge -------------------------------------------------------------
    #
    # Four independent reasons to jump into the ball rather than drive into it,
    # ORed, following Botimus Prime's `direct_shot`. Requiring the dodge to be
    # strictly *earlier* (the first clause alone) left 93.9% of touches as plain
    # ground contacts -- the bot nudged the ball along at walking pace, roughly
    # one touch per second per car, with no power in anything.
    #
    # The third clause is the important one and the least obvious: if the ball
    # and the car are travelling at similar velocities, driving into it transfers
    # almost no momentum, however fast the car is going. That is exactly the
    # "no power behind their shots" case, and it is most touches.
    if dodge is not None:
        near_goal = (
            shoot_target is not None
            and dodge.pos.flat_dist(shoot_target) < 4000.0
        )
        closing_speed = (ground.ball_vel - car.vel).length() if ground else 9e9
        if (
            ground is None
            or dodge.time < ground.time - DODGE_PREFERENCE
            or near_goal
            or closing_speed < 500.0
            or contested
        ):
            return dodge

    if ground is not None:
        return ground
    if wall is not None:
        return wall
    # Any air option is better than nothing at all.
    return aerial or double_jump or dodge


def time_to_ball(car, prediction, now: float, curve: ReachCurve | None = None) -> float:
    """
    Seconds until this car could reach the ball. Used for role assignment, so it
    stays cheap and returns a large number rather than inf when hopeless.
    """
    ic = find_intercept(car, prediction, now, curve)
    return ic.dt if ic.feasible else 15.0
