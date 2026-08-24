"""
Contesting the ball: challenge, shadow, or take it.

The gap this fills: the bot used to drive at its own intercept regardless of
whether an opponent would get there first, so on any ball it was losing it
arrived late and did nothing useful -- hovering near the ball rather than
either committing to it or defending against it.

The fix is to solve the intercept a second time with the OPPONENT as the actor,
form `ball_advantage = their_time - my_time`, and branch on it:

    advantage > 0            we get there first -- strike
    -CONTEST < advantage < 0 a genuine 50/50 -- challenge at full speed
    advantage < -CONTEST     we are losing it -- shadow, goal-side

The principle from the prior art (Botimus, ReliefBot, Kamael, Wildfire) is that
nobody ever hovers beside the ball. You are either behind it on the shot line
or driving through it.

Geometry follows ReliefBot's ChallengeStep; the contest threshold is Kamael's
0.5s; the "already on the ball, contest anyway" override and the boost-scaled
threat term are Botimus's shadow gate.

Cost: the opponent solve is NOT run every tick. Prior art re-decides at about
2Hz and so do we -- the result is cached and only refreshed when stale.
"""

from __future__ import annotations

from ..core.constants import BACK_WALL_Y, SIDE_WALL_X
from ..core.intercept import find_intercept
from ..core.physics import ReachCurve
from ..core.vec import Vec3, clamp

STRIKE = "strike"
CHALLENGE = "challenge"
SHADOW = "shadow"

# Within this much of the opponent it is a real 50/50: go, at speed. Kamael's
# contestedTimeLimit.
CONTEST_WINDOW = 0.5

# How far goal-side of the opponent's contact to stand, and how much further
# per extra second we are losing by. ReliefBot's defensiveNodeDistance.
SHADOW_MIN_DISTANCE = 900.0
SHADOW_PER_SECOND = 1000.0
SHADOW_MAX_DISTANCE = 4000.0

# Re-solve the opponent intercept at roughly 2Hz rather than every tick.
OPPONENT_REFRESH = 0.45

# Dead-band on the shadow point, which is what stops the car jittering between
# shadow positions. VirxEB uses 320uu with a 250uu pop.
SHADOW_DEADBAND = 320.0

# An opponent basically on top of the ball must be contested regardless of the
# timing arithmetic -- letting them have a free touch is worse than losing a
# 50/50. Botimus's `> 300` clause.
ALREADY_THERE = 300.0


class ContestState:
    """Caches the opponent solve and holds the current stance."""

    def __init__(self):
        self.stance = STRIKE
        self.advantage = 0.0
        self.their_time = 99.0
        self.their_contact: Vec3 | None = None
        self._solved_at = -99.0
        self._threat_car = None

    def refresh(self, state, prediction, my_time: float):
        """Re-solve the opponent's intercept if the cached one is stale."""
        if state.time - self._solved_at < OPPONENT_REFRESH:
            return
        self._solved_at = state.time

        best_time, best_ic, best_car = 99.0, None, None
        for opp in state.opponents:
            if opp.is_demolished:
                continue
            ic = find_intercept(
                opp, prediction, state.time,
                ReachCurve(opp.speed, opp.boost),
                max_time=4.0,
            )
            if ic.feasible and ic.dt < best_time:
                best_time, best_ic, best_car = ic.dt, ic, opp

        if best_ic is None:
            self.their_time = 99.0
            self.their_contact = None
            self._threat_car = None
        else:
            self.their_time = best_time
            self.their_contact = best_ic.pos
            self._threat_car = best_car

        self.advantage = self.their_time - my_time

    def decide(self, state, my_time: float, is_last_man: bool) -> str:
        """
        Strike, challenge, or shadow.

        Deliberately biased toward contesting: conceding a free touch is worse
        than losing a 50/50, so an opponent already on the ball is challenged
        whatever the arithmetic says.
        """
        if self.their_contact is None or self._threat_car is None:
            self.stance = STRIKE
            return self.stance

        advantage = self.their_time - my_time
        self.advantage = advantage

        # They are effectively on the ball already: contest, always.
        if self._threat_car.pos.flat_dist(self.their_contact) < ALREADY_THERE:
            self.stance = CHALLENGE
            return self.stance

        if advantage > 0.0:
            self.stance = STRIKE
        elif advantage > -CONTEST_WINDOW:
            self.stance = CHALLENGE
        else:
            # Losing it clearly. The last man shadows; anyone else may still
            # press if they have the boost to recover afterwards.
            self.stance = SHADOW if (is_last_man or state.me.boost < 30.0) else CHALLENGE

        return self.stance

    # --- geometry ---------------------------------------------------------

    def shadow_point(self, state) -> Vec3:
        """
        Where to stand while shadowing: on the line from the opponent's contact
        to our own goal, receding further the more we are losing by.
        """
        contact = self.their_contact or (state.ball.pos if state.ball else Vec3())
        to_goal = (state.own_goal - contact).flat()
        if to_goal.length_sq() < 1.0:
            to_goal = Vec3(0.0, state.goal_sign, 0.0)
        direction = to_goal.normalized()

        distance = clamp(
            max(-self.advantage * SHADOW_PER_SECOND, SHADOW_MIN_DISTANCE),
            SHADOW_MIN_DISTANCE,
            SHADOW_MAX_DISTANCE,
        )
        pos = contact + direction * distance
        pos.x = clamp(pos.x, -SIDE_WALL_X + 300.0, SIDE_WALL_X - 300.0)
        pos.y = clamp(pos.y, -BACK_WALL_Y + 250.0, BACK_WALL_Y - 250.0)
        pos.z = 17.0
        return pos

    def challenge_point(self, state) -> Vec3:
        """
        Where to drive for a 50/50: goal-side of their contact but close enough
        to meet it, so we arrive through the ball rather than beside it.
        """
        contact = self.their_contact or (state.ball.pos if state.ball else Vec3())
        to_goal = (state.own_goal - contact).flat()
        if to_goal.length_sq() < 1.0:
            to_goal = Vec3(0.0, state.goal_sign, 0.0)
        pos = contact + to_goal.normalized() * SHADOW_MIN_DISTANCE * 0.35
        pos.x = clamp(pos.x, -SIDE_WALL_X + 250.0, SIDE_WALL_X - 250.0)
        pos.z = 17.0
        return pos

    def shadow_speed(self, state, target: Vec3) -> float:
        """
        Speed that puts us AT the shadow point when the opponent touches the
        ball, rather than ambling there and arriving after the fact.

        Wildfire's kinematic solve: given distance s, current speed u and the
        time t until their contact, solve s = ut + 0.5*a*t^2 and turn the
        required acceleration back into a speed to ask for.
        """
        t = clamp(self.their_time, 0.2, 4.0)
        s = state.me.pos.flat_dist(target)
        u = state.me.speed
        # Average speed needed over the window, with a floor so the car keeps
        # moving and can react.
        return clamp(s / t, 550.0, 2300.0) if t > 0.05 else 1400.0

    def summary(self) -> str:
        return f"{self.stance} adv={self.advantage:+.2f}s theirs={self.their_time:.2f}s"
