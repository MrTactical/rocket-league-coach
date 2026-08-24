"""
Hitting the ball: choosing where to send it, and executing the contact.

Aim selection is deliberately teammate-aware. A bot that shoots at goal from
every position is easy to defend and constantly steals its partner's setups;
a good partner recognises when the better ball is a pass, a cross, or simply
putting it somewhere safe.
"""

from __future__ import annotations

import math

from rlbot import flat

from ..control.aerial import Aerial, apply_orientation
from ..control.dodge import Dodge
from ..control.drive import arrive_at, drive_to, local_angle_to
from ..core.constants import (
    BACK_WALL_Y,
    BALL_RADIUS,
    GOAL_HALF_WIDTH,
    MAX_SPEED,
    MAX_SPEED_NO_BOOST,
    SIDE_WALL_X,
)
from ..core.constants import max_turn_radius
from ..core.intercept import AERIAL, DODGE, DOUBLE_JUMP, GROUND, WALL
from ..core.shots import shot_vector
from ..core.mechanics import (
    DODGE_WINDOW_MIN,
    HIT_OFFSET,
    MIN_USEFUL_SPEED,
    STRIKE_MIN_SPEED,
    STRIKE_SPEED_MARGIN,
    RUNWAY_PER_SECOND,
    TIME_SHIFT_FACTOR,
    arrive_shift,
    double_jump_time_needed,
    jump_duration,
)
from ..core.vec import Vec3, clamp

SHOT = "shot"
PASS = "pass"
CLEAR = "clear"

# How far from the ball centre we aim the car's contact point.
CONTACT_OFFSET = BALL_RADIUS + 55.0

# How much better a teammate's shooting position must be before we give them
# the ball. Small enough that passing is a live option, large enough that we do
# not hand away a shot we could take ourselves.
PASS_ADVANTAGE = 0.06

# How far back along the aim line to approach from when we are badly lined up.
#
# A car cannot pivot on the spot. Driving straight at a contact point that sits
# off to one side demands a turn radius the car does not have, so it circles the
# ball indefinitely -- which is exactly what it was doing. Aiming at a point
# further BACK along the line the ball must travel converts that impossible turn
# into an arc the car can actually drive, and has the side effect of lining the
# car up with the shot before it arrives.
MAX_SETBACK = 1400.0
SETBACK_PER_RAD = 900.0


def _keeper(state):
    """The opponent best placed to defend their own net."""
    best, best_d = None, float("inf")
    for opp in state.opponents:
        if opp.is_demolished:
            continue
        d = opp.pos.dist(state.enemy_goal)
        if d < best_d:
            best, best_d = opp, d
    return best


def goal_target(state, from_pos: Vec3, car_pos: Vec3 | None = None) -> Vec3:
    """
    Where to send the ball so it goes between the posts.

    Uses the post-clamped shot vector rather than picking a point in the goal
    mouth by hand. The difference matters: aiming at a *point* means the car is
    forever trying to line up on one exact spot, whereas clamping the car's
    natural approach line into the cone between the two posts means that when
    the line already points into the mouth we simply drive straight through the
    ball, and only snap to a post when we genuinely have to. That is what makes
    shots accurate without a separate aiming controller.

    The keeper still biases the aim, but only within the cone -- it can no
    longer send the ball wide of the frame.
    """
    ball = from_pos
    direction, scorable = shot_vector(state, car_pos or state.me.pos, ball)

    # Nudge away from whoever is covering, but stay inside the posts: rotate
    # the shot vector a little rather than moving the aim point bodily.
    keeper = _keeper(state)
    if keeper is not None and scorable:
        side = -1.0 if keeper.pos.x >= 0.0 else 1.0
        direction = direction.rotate_2d(side * 0.06).normalized()

    return ball + direction * 2500.0


def shot_quality(state, from_pos: Vec3) -> float:
    """
    How good a shooting position this is, in [0,1].

    Angle to the goal mouth and distance, with a penalty for anyone standing in
    the way. Used to compare our own position against a teammate's, which is
    what turns "I always shoot" into "I pass when you are better placed".
    """
    goal = state.enemy_goal
    to_goal = (goal - from_pos).flat()
    dist = to_goal.length()
    if dist < 1.0:
        return 0.0

    # Straight on is worth far more than a tight angle.
    angle_off = abs(from_pos.x) / SIDE_WALL_X
    q = (1.0 - clamp(angle_off, 0.0, 1.0)) * 0.6

    # Closer is better, but only down to a sensible range.
    q += clamp(1.0 - dist / 6000.0, 0.0, 1.0) * 0.4

    # Somebody in the shooting lane ruins it.
    for opp in state.opponents:
        if opp.is_demolished:
            continue
        to_opp = (opp.pos - from_pos).flat()
        if to_opp.length() < 1.0 or to_opp.length() > dist:
            continue
        if to_opp.normalized().dot(to_goal.normalized()) > 0.93:
            q *= 0.45
            break

    return clamp(q, 0.0, 1.0)


def pass_target(state, lead_time: float = 1.0) -> Vec3 | None:
    """
    Where to put the ball for a teammate, if any of them is worth passing to.

    Considers EVERY teammate, not just the first one. With three a side the old
    version looked only at `state.ally` and ignored the rest, which was part of
    why passing was 5% of touches and each bot effectively played alone.
    """
    if not state.teammates or state.ball is None:
        return None

    ball_depth = state.ball.pos.y * state.goal_sign
    best, best_q = None, 0.0

    for mate in state.teammates:
        if mate.is_demolished:
            continue
        lead = mate.pos + mate.vel * lead_time

        # Only pass forward -- never back into our own half.
        if lead.y * state.goal_sign > ball_depth:
            continue

        # Do not pass into heavy pressure. The radius is deliberately modest:
        # at 1100uu almost every teammate on a 3v3 pitch counts as marked, and
        # passing collapsed to 2% of touches.
        opp = state.nearest_opponent_to(lead)
        if opp is not None and opp.pos.flat_dist(lead) < 700.0:
            continue

        q = shot_quality(state, lead)
        if q > best_q:
            best_q, best = q, lead

    if best is None:
        return None
    best.z = 150.0
    return best


def clear_target(state) -> Vec3:
    """
    Somewhere safe. Up the wall and toward the corner on whichever side the
    ball already is, which is the shortest route out of danger.
    """
    ball = state.ball.pos if state.ball else Vec3()
    side = 1.0 if ball.x >= 0 else -1.0
    return Vec3(side * (SIDE_WALL_X - 500.0), -state.goal_sign * 1500.0, 300.0)


def choose_aim(state, model, intercept, threat: float) -> tuple[Vec3, str]:
    """
    Decide where this touch should send the ball.

    Order of preference: get out of danger, then find the partner if they are
    better placed, then shoot.
    """
    ball_at = intercept.pos
    my_depth = ball_at.y * state.goal_sign

    # Dangerous ball in our third: clear it, no heroics.
    if threat > 0.55 or my_depth > BACK_WALL_Y * 0.55:
        return clear_target(state), CLEAR

    # Pass when a teammate is simply in a better place to score than we are.
    #
    # The old test was three ANDs -- teammate ahead of the ball, teammate
    # unmarked, AND our own angle poor -- which almost never coincided: 5% of
    # touches were passes and each bot played as though alone. Comparing shot
    # quality directly is both simpler and closer to how the decision is
    # actually made.
    p = pass_target(state)
    if p is not None:
        mine = shot_quality(state, ball_at)
        theirs = shot_quality(state, p)
        if theirs > mine + PASS_ADVANTAGE:
            return p, PASS

    return goal_target(state, ball_at, state.me.pos), SHOT


def legal_actions(state, intercept, threat: float) -> list[str]:
    """
    Which actions are worth considering here at all.

    Experience should choose between genuinely available options, not be
    allowed to learn that shooting at goal from our own six-yard box is fine
    because it happened to work once.
    """
    ball_at = intercept.pos
    depth = ball_at.y * state.goal_sign
    options = [CLEAR]

    # Shooting only makes sense from somewhere you could plausibly score.
    if depth < BACK_WALL_Y * 0.45 and threat < 0.7:
        options.append(SHOT)

    if pass_target(state) is not None:
        options.append(PASS)

    return options


def aim_for(state, action: str, intercept) -> Vec3:
    """The aim point for a given action, or a clear if it is unavailable."""
    if action == PASS:
        p = pass_target(state)
        if p is not None:
            return p
        return clear_target(state)
    if action == SHOT:
        return goal_target(state, intercept.pos, state.me.pos)
    return clear_target(state)


def contact_point(intercept_pos: Vec3, aim: Vec3, offset_scale: float = 1.0) -> Vec3:
    """
    Where the car must be to send the ball toward `aim`.

    Directly opposite the aim direction, offset by the ball radius plus a bit
    of car. `offset_scale` is the calibrator's learned correction -- if we keep
    whiffing it closes the standoff up, and successful contacts relax it back.
    """
    direction = (aim - intercept_pos).normalized()
    if direction.length_sq() < 0.5:
        direction = Vec3(0.0, 1.0, 0.0)
    return intercept_pos - direction * (CONTACT_OFFSET * offset_scale)


def strike_difficulty(state, intercept) -> float:
    """
    How hard this touch is, in [0,1]. Feeds the whiff model so easy rolling
    shots are rarely missed and awkward aerials often are.
    """
    d = 0.0
    if intercept.kind == AERIAL:
        d += 0.5
    elif intercept.kind == DOUBLE_JUMP:
        d += 0.3
    elif intercept.kind == DODGE:
        d += 0.22
    elif intercept.kind == WALL:
        d += 0.35

    d += clamp(intercept.pos.z / 1400.0, 0.0, 0.25)
    d += clamp(intercept.ball_vel.length() / 4000.0, 0.0, 0.2)

    # Very short reaction windows are harder.
    if intercept.dt < 0.6:
        d += 0.2

    opp = state.nearest_opponent_to(intercept.pos)
    if opp is not None and opp.pos.dist(intercept.pos) < 900.0:
        d += 0.15

    return clamp(d, 0.0, 1.0)


class Strike:
    """
    Executes one attempt on the ball.

    Holds any sub-manoeuvre (aerial or dodge) so it survives across ticks, and
    carries a stable key so the humanizer commits to one aim error and one
    whiff decision per attempt rather than resampling every frame.
    """

    def __init__(self, intercept, aim: Vec3, kind: str):
        self.intercept = intercept
        self.aim = aim
        self.kind = kind
        self.key = f"{kind}:{intercept.slice_index}:{round(intercept.time, 1)}"
        self.maneuver = None
        self.dodged = False

    def refresh(self, intercept, aim: Vec3, kind: str):
        """
        Update the target without discarding an in-flight manoeuvre.

        The key is rebuilt only on a real kind change. It used to be left
        untouched entirely, so once a held aerial finished the humanizer went
        on keying its aim error and whiff decision off a stale `aerial:...`
        string for whatever ground strike came next.
        """
        if kind != self.kind:
            self.key = f"{kind}:{intercept.slice_index}:{round(intercept.time, 1)}"
        self.intercept = intercept
        self.aim = aim
        self.kind = kind

    def step(self, state, humanizer, calibrator=None, debug=None,
             contesting: bool = False) -> flat.ControllerState:
        car = state.me
        ic = self.intercept

        # Continue any manoeuvre already underway.
        #
        # An in-flight Aerial gets a real viability check. `Aerial.is_viable`
        # has existed since the class was written and had zero callers, so the
        # only way an aerial ever ended was the manoeuvre timing out -- or, far
        # more often, the Strike being rebuilt underneath it.
        if self.maneuver is not None:
            if isinstance(self.maneuver, Aerial) and not self.maneuver.is_viable(state):
                self.maneuver = None
            else:
                out = self.maneuver.step(state)
                if out is not None:
                    if debug is not None:
                        debug.mechanic = type(self.maneuver).__name__.lower()
                    return out
                self.maneuver = None

        # Learned corrections for our own systematic errors first, then this
        # attempt's deliberate human aim error on top of them.
        cal_aim, cal_timing, cal_offset = (0.0, 0.0, 1.0)
        if calibrator is not None:
            cal_aim, cal_timing, cal_offset = calibrator.apply()

        aim_err = humanizer.aim_offset(self.key, state)
        raw_aim = (self.aim - ic.pos).flat()
        if raw_aim.length_sq() < 1.0:
            raw_aim = (state.enemy_goal - ic.pos).flat()
        intended = raw_aim.normalized()
        aim_dir = raw_aim.rotate_2d(aim_err + cal_aim).normalized()
        adjusted_aim = ic.pos + aim_dir * 1000.0

        target = contact_point(ic.pos, adjusted_aim, cal_offset)
        contact_time = ic.time + cal_timing

        # Register what this attempt is trying to do. The *intended* direction
        # is recorded, not the noised one, so the calibrator measures error
        # against our real goal; the humanizer's noise is zero-mean and averages
        # out rather than being learned away.
        if calibrator is not None:
            calibrator.plan(self.key, intended, contact_time, ic.pos)

        # --- aerial ---------------------------------------------------------
        if ic.kind == AERIAL:
            difficulty = strike_difficulty(state, ic)
            if humanizer.attempt_aerial(difficulty):
                self.maneuver = Aerial(target, contact_time)
                out = self.maneuver.step(state)
                if out is not None:
                    return out
            # Declined the aerial: fall back to positioning underneath it.
            if debug is not None:
                debug.aerial_declined = True
            return drive_to(car, Vec3(target.x, target.y, 17.0), MAX_SPEED * 0.7)

        ground_target = Vec3(target.x, target.y, 17.0)
        dist = car.pos.flat_dist(ground_target)
        ball_dist = car.pos.flat_dist(ic.pos)

        # --- Arrive: aim back down the shot line, and arrive EARLY -----------
        #
        # Botimus Prime's Arrive, and the mechanism that turns hovering into
        # striking. Driving straight at the contact point asks for a turn the
        # car often cannot make (so it orbits) and arrives with no speed (so the
        # touch is a nudge). Instead aim at a point pulled back along the line
        # the ball must travel, and demand arrival there EARLIER than contact.
        # Having to cover more ground in less time makes the speed controller
        # ask for more speed, so the car is still accelerating on contact --
        # which is where the power in a touch comes from.
        #
        # The shift decays to zero as the car closes, and is dropped entirely
        # once it would fall inside the car's own turning circle, because a
        # target inside the turn circle is exactly what causes orbiting.
        jump_time = 0.0
        if ic.kind == DODGE:
            jump_time = jump_duration(ic.pos.z)
        elif ic.kind == DOUBLE_JUMP:
            jump_time = double_jump_time_needed(ic.pos.z)
        additional = jump_time * RUNWAY_PER_SECOND

        turn_r = max_turn_radius(clamp(car.speed, 500.0, MAX_SPEED))
        shift = arrive_shift(dist, car.speed, additional, turn_r)
        if contesting:
            # In a 50/50 there is no time to line up: the setback and the
            # arrival timing both exist to make a tidy approach, and tidy loses
            # the ball. Go straight at it, flat out.
            shift = 0.0
        shifted = ground_target - aim_dir * shift
        shifted = Vec3(shifted.x, shifted.y, 17.0)

        time_shift = shift / clamp(car.speed, 500.0, MAX_SPEED) * TIME_SHIFT_FACTOR
        time_left = (contact_time - time_shift) - state.time
        shifted_dist = car.pos.flat_dist(shifted)
        if time_left > 0.02:
            target_speed = clamp(shifted_dist / time_left, 0.0, MAX_SPEED)
        else:
            target_speed = MAX_SPEED

        # Floor the approach at something that can actually move the ball.
        # Timing the arrival perfectly at 400 uu/s behind a ball rolling at 450
        # produces a nudge, not a shot -- the transfer depends on CLOSING speed.
        # Arriving early is acceptable; arriving feeble is not.
        strike_floor = clamp(
            ic.ball_vel.flat_length() + STRIKE_SPEED_MARGIN,
            STRIKE_MIN_SPEED,
            MAX_SPEED,
        )
        # Power and boost are in direct tension. Demanding 1250+ on every
        # approach burns boost on routine touches and then there is none left
        # when a ball goes up -- measured at boost median 29 with 21% of ticks
        # empty, against the ~50 an aerial costs. Below the no-boost cap the
        # throttle alone gets us there, so a car that is not already flush caps
        # its ambition there and keeps what it has for the air.
        if contesting and car.boost > 20.0:
            # Worth spending everything on a 50/50 -- but only if there is
            # something to spend. Forcing max speed on an empty car just pins
            # the throttle and drains what little is left; boost fell to a
            # median of 20 when this was unconditional.
            strike_floor = MAX_SPEED
        elif car.boost < 45.0:
            strike_floor = min(strike_floor, MAX_SPEED_NO_BOOST)
        target_speed = max(target_speed, strike_floor)

        # Whiff: nudge the approach off just enough to mistime the touch.
        difficulty = strike_difficulty(state, ic)
        if humanizer.should_whiff(self.key, state, difficulty):
            c = drive_to(car, shifted, target_speed, min_turn_speed=900.0)
            return humanizer.apply_whiff(c, state)

        c = drive_to(car, shifted, target_speed, allow_slide=True, min_turn_speed=900.0)

        # Never creep. Botimus zeroes the throttle rather than inching forward:
        # a car crawling at the ball cannot strike it and cannot react.
        if target_speed < MIN_USEFUL_SPEED:
            c.throttle = 0.0
            c.boost = False

        angle = abs(local_angle_to(car, shifted))

        # --- jump into it ----------------------------------------------------
        #
        # Three clauses ORed the way the prior art does it, not ANDed tight.
        # The previous gate needed dt < 0.45 AND within 500uu AND on the ground
        # simultaneously, which almost never coincided: 13% of intercepts were
        # tagged "jump" and virtually none of them jumped. The speed-match band
        # is deliberately loose (1000 uu/s) and there is a low-speed escape so a
        # nearly stationary car jumps regardless of heading.
        if ic.kind in (DODGE, DOUBLE_JUMP) and car.on_ground and car.can_dodge:
            vel_dir = car.vel.flat()
            aligned = False
            if vel_dir.length() > 1.0 and dist > 1.0:
                to_t = (shifted - car.pos).flat().normalized()
                aligned = vel_dir.normalized().dot(to_t) > 0.75
            if car.speed < 500.0:
                aligned = True
            # The window must open early enough for the flip to land on the
            # ball: the Dodge takes ~0.16s to reach the flip, and for a floor
            # ball jump_duration is only ~0.05s, so the raw window was 0.18s and
            # the car reached the ball before it ever opened. Zero dodges fired
            # in a full match.
            window = max(jump_time + 0.13, DODGE_WINDOW_MIN)
            # The speed-match clause compares the SPEED WE ASKED FOR against
            # our actual speed, and the strike floor deliberately asks for far
            # more than we are doing -- so this silently closed the gate and
            # the Dodge manoeuvre fired zero times across every match. What it
            # is really guarding against is dodging while still accelerating
            # hard from a standstill, so compare against a sane ceiling instead
            # of the floor we just invented.
            speed_ok = car.speed > 500.0 or dist < 400.0
            if (
                (contact_time - state.time) < window
                and speed_ok
                and aligned
            ):
                self.maneuver = Dodge(target=ic.pos)
                out = self.maneuver.step(state)
                if out is not None:
                    if debug is not None:
                        debug.mechanic = "dodge"
                    return out

        # --- power dodge on a ground strike ----------------------------------
        if (
            not self.dodged
            and self.kind in (SHOT, CLEAR)
            and car.on_ground
            and car.can_dodge
            and ic.kind in (GROUND, DODGE)
            and shift < 90.0
            and 110.0 < ball_dist < 700.0
            and angle < 0.65
            and car.speed > 600.0
            and ic.dt < 0.85
        ):
            self.dodged = True
            self.maneuver = Dodge(target=ic.pos)
            out = self.maneuver.step(state)
            if out is not None:
                if debug is not None:
                    debug.mechanic = "power_dodge"
                return out

        return c
