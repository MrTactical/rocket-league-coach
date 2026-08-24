"""
Decision tests.

Roles are deliberately sticky and gated behind a reaction time, so a single
tick tells you almost nothing. Each case runs a fresh brain for a stretch of
simulated ticks and asserts on the role the bot *settles* into.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlbot import flat  # noqa: E402

from bot.brain.decide import Brain  # noqa: E402
from bot.brain.humanize import Humanizer, get_profile  # noqa: E402
from bot.brain.teammate import TeammateModel  # noqa: E402
from bot.core.game import GameState  # noqa: E402
from tests.harness import (  # noqa: E402
    FIELD_INFO,
    ball,
    car,
    make_prediction,
    packet,
)


def simulate(cars, the_ball, ticks=180, phase=flat.MatchPhase.Active, rank="diamond",
             seed=11, model=None, start_time=10.0):
    """
    Run the brain for `ticks` frames.

    Cars are advanced by their own velocity and the ball follows its predicted
    path. This is not a physics engine -- it exists to let role commitment,
    reaction gating and hysteresis play out over time.
    """
    model = model or TeammateModel("_test")
    hz = Humanizer(get_profile(rank), seed=seed)
    brain = Brain(model, hz)

    base = packet(cars, the_ball, time=start_time, phase=phase)
    full_pred = make_prediction(base.balls[0], start_time, horizon=8.0)

    roles, actions = [], []
    dt = 1.0 / 120.0
    for i in range(ticks):
        t = start_time + i * dt

        # Ball follows its own prediction.
        sl = full_pred.slices[min(i, len(full_pred.slices) - 1)]
        b = flat.BallInfo(physics=sl.physics, shape=flat.SphereShape(92.75))

        # Cars drift along their velocity vectors.
        moved = []
        for c in cars:
            p = c.physics
            moved.append(
                flat.PlayerInfo(
                    physics=flat.Physics(
                        location=flat.Vector3(
                            p.location.x + p.velocity.x * i * dt,
                            p.location.y + p.velocity.y * i * dt,
                            p.location.z,
                        ),
                        rotation=p.rotation,
                        velocity=p.velocity,
                        angular_velocity=p.angular_velocity,
                    ),
                    hitbox=c.hitbox,
                    air_state=c.air_state,
                    dodge_timeout=c.dodge_timeout,
                    demolished_timeout=c.demolished_timeout,
                    is_bot=c.is_bot,
                    name=c.name,
                    team=c.team,
                    boost=c.boost,
                )
            )

        pkt = packet(moved, b, time=t, phase=phase)
        st = GameState(pkt, 0, moved[0].team, FIELD_INFO, t - dt)
        pred = make_prediction(b, t, horizon=6.0)

        brain.decide(st, pred)
        roles.append(brain.debug.role)
        actions.append(brain.debug.action)

    return roles, actions, brain, model


def settled(seq, tail=60):
    """Most common value over the final stretch."""
    window = seq[-tail:]
    return max(set(window), key=window.count)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("bot closer than human -> attacks")
def _():
    roles, acts, *_ = simulate(
        [car(x=0, y=-1200, yaw=math.pi / 2, boost=60, name="Ally"),
         car(x=1800, y=-3800, yaw=math.pi / 2, boost=40, name="Joe", is_bot=False),
         car(x=0, y=3000, yaw=-math.pi / 2, team=1, name="Opp")],
        ball(x=0, y=0))
    return settled(roles), "attack"


@case("human closer -> defers, does not contest")
def _():
    roles, acts, *_ = simulate(
        [car(x=-2200, y=-3600, yaw=math.pi / 2, boost=60, name="Ally"),
         car(x=0, y=-500, yaw=math.pi / 2, boost=60, name="Joe", is_bot=False),
         car(x=0, y=3000, yaw=-math.pi / 2, team=1, name="Opp")],
        ball(x=0, y=0))
    return settled(roles), ("support", "defend")


@case("shot heading at our net, bot is last -> defends")
def _():
    roles, acts, *_ = simulate(
        [car(x=0, y=-3600, yaw=math.pi / 2, boost=40, name="Ally"),
         car(x=1200, y=2000, yaw=math.pi / 2, boost=30, name="Joe", is_bot=False),
         car(x=0, y=700, yaw=-math.pi / 2, team=1, name="Opp")],
        ball(x=0, y=-1200, vy=-1800))
    return settled(roles), "defend"


@case("human upfield, ball loose in our half -> bot covers")
def _():
    roles, acts, *_ = simulate(
        [car(x=-1500, y=-2500, yaw=math.pi / 2, boost=50, name="Ally"),
         car(x=500, y=3500, yaw=math.pi / 2, boost=20, name="Joe", is_bot=False),
         car(x=800, y=1500, yaw=-math.pi / 2, team=1, name="Opp")],
        ball(x=300, y=-1800, vy=-300))
    return settled(roles), ("attack", "defend")


@case("1v1, no teammate -> always attacks")
def _():
    roles, acts, *_ = simulate(
        [car(x=0, y=-3000, yaw=math.pi / 2, boost=50, name="Ally"),
         car(x=0, y=3000, yaw=-math.pi / 2, team=1, name="Opp")],
        ball(x=0, y=0))
    return settled(roles), "attack"


@case("kickoff produces a decision and moves")
def _():
    roles, acts, brain, _ = simulate(
        [car(x=-2048, y=-2560, yaw=math.pi / 4, boost=34, name="Ally"),
         car(x=2048, y=-2560, yaw=math.pi * 0.75, boost=34, name="Joe", is_bot=False),
         car(x=-2048, y=2560, yaw=-math.pi / 4, team=1, name="O1"),
         car(x=2048, y=2560, yaw=-math.pi * 0.75, team=1, name="O2")],
        ball(x=0, y=0, z=93), phase=flat.MatchPhase.Kickoff, ticks=90)
    act = settled(acts)
    return act.split("/")[0], "kickoff"


@case("upside down in the air -> recovers")
def _():
    roles, acts, *_ = simulate(
        [car(x=0, y=-1000, z=900, vx=600, vz=-300, roll=math.pi,
             on_ground=False, boost=40, name="Ally"),
         car(x=1500, y=-2500, name="Joe", is_bot=False)],
        ball(x=0, y=0), ticks=30)
    return ("recovery" in acts[:20]), True


def main():
    print(f"{'case':52s} {'got':12s} {'want':22s} result")
    print("-" * 100)
    failures = 0
    for name, fn in CASES:
        got, want = fn()
        ok = (got in want) if isinstance(want, tuple) else (got == want)
        if not ok:
            failures += 1
        print(f"{name:52s} {str(got):12s} {str(want):22s} {'PASS' if ok else 'FAIL'}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
