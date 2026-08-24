"""
Role assignment and the positions each role wants to occupy.

Three roles, and the whole point is that the choice is made *relative to the
human*, not in isolation:

  ATTACK   -- we are taking this ball.
  SUPPORT  -- the human has it; sit where their touch can find us.
  DEFEND   -- last man; protect the net and shadow.

Two things matter more than the raw decision. First, hysteresis: a role that
flickers every tick produces a car that twitches instead of committing.
Second, deference -- when in doubt, we give the ball to the human, because two
players contesting the same ball is worse than either taking it alone.
"""

from __future__ import annotations

import math

from ..core.constants import BACK_WALL_Y, GOAL_HALF_WIDTH, SIDE_WALL_X
from ..core.vec import Vec3, clamp

ATTACK = "attack"
SUPPORT = "support"
DEFEND = "defend"

# Minimum time to hold a role before switching, and the extra advantage
# required to abandon a committed attack.
ROLE_MIN_HOLD = 0.35
ATTACK_STICKINESS = 0.45


class RoleState:
    """Carries role commitment across ticks."""

    def __init__(self):
        self.role = SUPPORT
        self.since = 0.0
        self.last_eval = 0.0

    def set(self, role: str, now: float):
        if role != self.role:
            self.role = role
            self.since = now

    def held_for(self, now: float) -> float:
        return now - self.since


def _depth(state, car) -> float:
    """How deep in our own half a car is: +BACK_WALL_Y at our net, -at theirs."""
    return car.pos.y * state.goal_sign


def threat_level(state, prediction) -> float:
    """
    How dangerous the ball currently is to our net, in [0, 1].

    Combines how close the ball is to our goal with how fast it is heading
    there -- a slow ball on our doorstep and a rocket from midfield are both
    threats, for different reasons.
    """
    if state.ball is None:
        return 0.0
    ball = state.ball.pos
    vel = state.ball.vel

    dist = ball.dist(state.own_goal)
    proximity = clamp(1.0 - dist / 7000.0, 0.0, 1.0)

    to_goal = (state.own_goal - ball)
    closing = vel.dot(to_goal.normalized()) if to_goal.length_sq() > 1.0 else 0.0
    closing_norm = clamp(closing / 2000.0, 0.0, 1.0)

    # A ball predicted to actually enter our net is maximal threat.
    on_target = 0.0
    if prediction is not None and prediction.slices:
        for i in range(0, min(len(prediction.slices), 360), 6):
            p = prediction.slices[i].physics.location
            if abs(p.y) > BACK_WALL_Y - 30.0 and p.y * state.goal_sign > 0:
                if abs(p.x) < GOAL_HALF_WIDTH and p.z < 700.0:
                    on_target = 1.0
                    break

    return clamp(max(on_target, 0.55 * proximity + 0.45 * closing_norm * proximity), 0.0, 1.0)


def assign_role(
    state,
    model,
    role_state: RoleState,
    my_time: float,
    ally_time: float,
    prediction=None,
) -> str:
    """Pick this tick's role, respecting commitment and deference."""
    now = state.time

    # No teammate: we do everything.
    if not state.has_ally:
        role_state.set(ATTACK, now)
        return ATTACK

    ally = state.ally
    threat = threat_level(state, prediction)

    my_depth = _depth(state, state.me)
    ally_depth = _depth(state, ally)
    i_am_last = my_depth > ally_depth + 150.0

    # How much faster we must be before taking a ball off the human.
    margin = model.defer_bias
    if role_state.role == ATTACK and role_state.held_for(now) > ROLE_MIN_HOLD:
        # Already committed: don't bail out for a marginal difference.
        margin -= ATTACK_STICKINESS

    advantage = ally_time - my_time  # positive means we are quicker

    # --- hard overrides ---------------------------------------------------

    # Serious threat and we are the last line: defend, no argument.
    if threat > 0.6 and i_am_last and ally_time > my_time - 0.8:
        role_state.set(DEFEND, now)
        return DEFEND

    # Human is demolished or has no boost and is miles away: we take over.
    if ally.is_demolished or (ally_time > my_time + 1.5):
        if threat > 0.75 and i_am_last:
            role_state.set(DEFEND, now)
            return DEFEND
        role_state.set(ATTACK, now)
        return ATTACK

    # --- normal flow ------------------------------------------------------

    if role_state.held_for(now) < ROLE_MIN_HOLD:
        return role_state.role

    if advantage > margin:
        want = ATTACK
    elif i_am_last or threat > 0.4:
        want = DEFEND
    else:
        want = SUPPORT

    # A cover-heavy partner profile nudges ambiguous calls toward defence.
    if want == ATTACK and model.cover_bias > 0.7 and i_am_last and threat > 0.25:
        want = DEFEND

    role_state.set(want, now)
    return want


# --- role positions -------------------------------------------------------


def support_position(state, model, prediction=None, ordinal: int = 1) -> Vec3:
    """
    Where to sit while someone else attacks.

    `ordinal` is our place in the queue for the ball: 1 is second man, 2 is
    third man, and so on. This parameter is the whole point of the function.
    Without it every non-attacking teammate computes the same spot from the
    same ball position and they drive around side by side -- which is exactly
    what "no rotation" looks like from the outside.

    Second man sits close enough to convert a loose ball. Third man sits much
    deeper and wider, covering the counter rather than the follow-up.
    """
    ball = state.ball.pos if state.ball else Vec3()

    # Goal-side of the ball.
    back_dir = (state.own_goal - ball).flat().normalized()
    if back_dir.length_sq() < 0.1:
        back_dir = Vec3(0.0, state.goal_sign, 0.0)

    # Depth grows sharply with ordinal: 2nd man supports, 3rd man covers.
    distance = 1700.0 + 1900.0 * max(0, ordinal - 1)
    pos = ball + back_dir * distance

    # Lateral separation. Alternate sides by ordinal so two supporters never
    # occupy the same channel, and widen the deeper players.
    side = model.preferred_side if ordinal % 2 else -model.preferred_side
    ally_x = state.ally.pos.x if state.has_ally else 0.0
    if ordinal == 1 and abs(ally_x) > 800.0:
        # Second man mirrors whoever is on the ball.
        side = -1.0 if ally_x >= 0.0 else 1.0
    offset = 1000.0 + 350.0 * max(0, ordinal - 1)
    pos.x = clamp(pos.x + side * offset, -SIDE_WALL_X + 400.0, SIDE_WALL_X - 400.0)

    # Never drift ahead of the ball.
    if pos.y * state.goal_sign < ball.y * state.goal_sign:
        pos.y = ball.y + back_dir.y * 400.0

    pos.y = clamp(pos.y, -BACK_WALL_Y + 300.0, BACK_WALL_Y - 300.0)
    pos.z = 17.0
    return pos


def defend_position(state, prediction=None, ordinal: int = 0) -> Vec3:
    """
    Shadow position: between the ball and our net, biased to the near post,
    dropping deeper as the ball gets closer.

    `ordinal` separates multiple defenders -- the deeper one takes the far
    post rather than stacking on top of the near-post player.
    """
    ball = state.ball.pos if state.ball else Vec3()
    goal = state.own_goal

    to_goal = (goal - ball).flat()
    dist_to_goal = to_goal.length()
    if dist_to_goal < 1.0:
        return Vec3(0.0, goal.y + state.goal_sign * -300.0, 17.0)

    direction = to_goal / dist_to_goal

    # Sit closer to our own net when the ball is near, further up when it is
    # far -- classic shadow depth. Extra defenders sit deeper still.
    standoff = clamp(dist_to_goal * 0.45, 900.0, 3200.0)
    standoff *= max(0.35, 1.0 - 0.4 * max(0, ordinal - 1))
    pos = ball + direction * standoff

    # Near-post bias: cover the side the ball is on. A second defender takes
    # the opposite post instead of doubling up.
    post_side = 1.0 if ball.x >= 0.0 else -1.0
    if ordinal >= 2:
        post_side = -post_side
    pos.x = clamp(pos.x + post_side * (280.0 + 420.0 * max(0, ordinal - 1)),
                  -SIDE_WALL_X + 350.0, SIDE_WALL_X - 350.0)

    # Never sit inside our own goal.
    limit = BACK_WALL_Y - 380.0
    if abs(pos.y) > limit:
        pos.y = math.copysign(limit, pos.y)

    pos.z = 17.0
    return pos


def net_position(state) -> Vec3:
    """Last-ditch goal-line position, used when the ball is about to arrive."""
    ball = state.ball.pos if state.ball else Vec3()
    x = clamp(ball.x * 0.35, -GOAL_HALF_WIDTH + 120.0, GOAL_HALF_WIDTH - 120.0)
    return Vec3(x, state.own_goal.y - state.goal_sign * 250.0, 17.0)
