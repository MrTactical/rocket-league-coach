"""
Kickoffs.

The decision is not "can I win this kickoff" but "should I take it, given what
my partner is about to do". Two players going for the same kickoff loses the
game on the spot; two players both cheating back hands the opponent a free
shot. So the go/no-go call leans on the learned kickoff_commit trait: if the
human reliably takes them, we set up behind instead.
"""

from __future__ import annotations

import math

from ..core.vec import Vec3, clamp

# The five kickoff spawns, mirrored per team. Values are for the -Y (blue)
# side; multiply y by the team's goal sign to get the other side.
SPAWNS = {
    "corner_right": Vec3(-2048.0, -2560.0, 17.0),
    "corner_left": Vec3(2048.0, -2560.0, 17.0),
    "back_right": Vec3(-256.0, -3840.0, 17.0),
    "back_left": Vec3(256.0, -3840.0, 17.0),
    "back_center": Vec3(0.0, -4608.0, 17.0),
}


def classify_spawn(car, goal_sign: float) -> str:
    """Which of the five kickoff positions this car is standing on."""
    x, y = car.pos.x, car.pos.y * goal_sign
    if abs(x) > 1500.0:
        return "corner_left" if x > 0 else "corner_right"
    if abs(x) > 100.0:
        return "back_left" if x > 0 else "back_right"
    return "back_center"


def kickoff_distance(car) -> float:
    """Straight-line distance to the ball at centre field."""
    return car.pos.flat_length()


def should_take_kickoff(state, model) -> bool:
    """
    Decide whether we go for the ball or set up behind.

    Compares against EVERY teammate, not just one. The old version looked only
    at `state.ally`, so on a three-a-side kickoff the third car was invisible
    and two bots could each conclude they were the taker -- both then raced the
    ball and collided in the middle. Measured: two of three cars per team
    reported "kickoff/go" on every kickoff.

    Distance settles it when the gap is clear. When two spawns are mirrored and
    genuinely equidistant, the learned trait decides whether to defer to a
    human, and a stable index tiebreak settles the rest so the choice cannot
    oscillate mid-run or disagree between cars.
    """
    mates = [c for c in state.teammates if not c.is_demolished]
    if not mates:
        return True

    me = state.me
    my_d = kickoff_distance(me)

    closest_d = my_d
    contenders = [me.index]
    for mate in mates:
        d = kickoff_distance(mate)
        if d < closest_d - 250.0:
            # Someone is clearly closer: not our kickoff.
            return False
        if abs(d - my_d) <= 250.0:
            contenders.append(mate.index)
            closest_d = min(closest_d, d)

    # Nobody is meaningfully closer. If a human is one of the contenders, the
    # learned trait decides -- a human who expects to take kickoffs will take
    # them whatever we do.
    human = next((c for c in mates if c.is_human
                  and abs(kickoff_distance(c) - my_d) <= 250.0), None)
    if human is not None:
        if model.take_kickoff_bias > 0.55:
            return True
        if model.take_kickoff_bias < 0.45:
            return False

    # Stable tiebreak: the lowest index among the equidistant contenders goes.
    # Every car computes the same answer, so exactly one of them takes it.
    return me.index == min(contenders)


def cheat_position(state, model) -> Vec3:
    """
    Where to sit when the human is taking the kickoff.

    Slightly back and to the opposite side, close enough to punish a 50/50
    that pops loose but not so close that we get tangled up with them.
    """
    sign = state.goal_sign
    side = model.preferred_side
    ally_x = state.ally.pos.x if state.has_ally else 0.0
    if abs(ally_x) > 400.0:
        side = -1.0 if ally_x > 0 else 1.0

    return Vec3(side * 1100.0, sign * 2600.0, 17.0)


def back_boost_target(state) -> Vec3:
    """Corner big pad on our side, for a no-go kickoff that collects instead."""
    sign = state.goal_sign
    side = 1.0 if state.me.pos.x >= 0 else -1.0
    return Vec3(side * 3072.0, sign * 4096.0, 73.0)


class KickoffPlan:
    """
    Runs a kickoff from start to first touch.

    Holds its own manoeuvre state so a speedflip started at t=0.2s is not
    re-decided at t=0.3s.
    """

    def __init__(self, take: bool, spawn: str, use_speedflip: bool):
        self.take = take
        self.spawn = spawn
        self.use_speedflip = use_speedflip
        self.maneuver = None
        self.started = False
        self.start_time = 0.0
        self.flip_fired = False

    def aim_point(self, state) -> Vec3:
        """
        Where to drive on a kickoff we are taking.

        From the corner spawns we angle slightly off-centre so the 50/50 pops
        toward our own side rather than straight back at our net.
        """
        ball = state.ball.pos if state.ball else Vec3(0.0, 0.0, 93.0)
        if self.spawn in ("corner_left", "corner_right"):
            offset = -1.0 if self.spawn == "corner_left" else 1.0
            return Vec3(ball.x + offset * 55.0, ball.y, ball.z)
        return Vec3(ball.x, ball.y, ball.z)
