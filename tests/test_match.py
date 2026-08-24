"""
End-to-end simulated match.

Runs the full stack -- brain, teammate model, telemetry, coaching -- against a
scripted human who ball-chases and runs their boost dry, so the coach has
something real to find. Verifies the whole pipeline produces a report rather
than checking any single number.

The car physics here are crude (move toward a target, bleed boost). The point
is to exercise the plumbing over thousands of ticks, not to reproduce Rocket
League.
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
from bot.core.vec import Vec3  # noqa: E402
from bot.learn.coach import analyse, format_report  # noqa: E402
from bot.learn.telemetry import MatchRecorder  # noqa: E402
from tests.harness import FIELD_INFO, ball, car, make_prediction, packet  # noqa: E402

DT = 1.0 / 30.0
MINUTES = 3.0


class Actor:
    """A minimal kinematic car."""

    def __init__(self, x, y, team, name, is_bot, boost=50.0):
        self.pos = Vec3(x, y, 17.0)
        self.vel = Vec3()
        self.yaw = math.pi / 2 if team == 0 else -math.pi / 2
        self.team = team
        self.name = name
        self.is_bot = is_bot
        self.boost = boost

    def drive_to(self, target: Vec3, speed=1400.0, use_boost=True):
        d = (target - self.pos).flat()
        n = d.length()
        if n < 1.0:
            self.vel = Vec3()
            return
        direction = d / n
        if use_boost and self.boost > 0:
            speed = min(speed * 1.5, 2300.0)
            self.boost = max(0.0, self.boost - 33.3 * DT)
        self.vel = direction * min(speed, n / DT)
        self.pos = self.pos + self.vel * DT
        self.pos.z = 17.0
        self.yaw = math.atan2(direction.y, direction.x)

    def to_flat(self) -> flat.PlayerInfo:
        return car(
            x=self.pos.x, y=self.pos.y, z=self.pos.z,
            vx=self.vel.x, vy=self.vel.y, vz=0.0,
            yaw=self.yaw, boost=self.boost, team=self.team,
            name=self.name, is_bot=self.is_bot,
        )


def run_match(rank="diamond", verbose=True):
    model = TeammateModel("SimPartner")
    # Start from a neutral profile so the run's learning is visible.
    model.traits.chase_rate = 0.5
    model.traits.rotation_discipline = 0.5

    hz = Humanizer(get_profile(rank), seed=3)
    brain = Brain(model, hz)
    rec = MatchRecorder("SimPartner", rank, enabled=True)

    me = Actor(0, -3000, 0, "Ally", True, boost=60)
    ally = Actor(1200, -2500, 0, "SimPartner", False, boost=45)
    opp1 = Actor(-800, 2800, 1, "Opp1", True, boost=50)
    opp2 = Actor(800, 3400, 1, "Opp2", True, boost=50)

    bpos = Vec3(0, 0, 93)
    bvel = Vec3(250, 600, 0)

    ticks = int(MINUTES * 60 / DT)
    score = [0, 0]
    t0 = 10.0

    for i in range(ticks):
        t = t0 + i * DT

        # --- ball: bounce around the pitch ------------------------------
        bpos = bpos + bvel * DT
        if abs(bpos.x) > 4000:
            bvel.x *= -1
            bpos.x = math.copysign(4000, bpos.x)
        if abs(bpos.y) > 5000:
            # Count it as a goal if it is near the middle, then reset.
            if abs(bpos.x) < 892:
                if bpos.y > 0:
                    score[0] += 1
                else:
                    score[1] += 1
                bpos = Vec3(0, 0, 93)
                bvel = Vec3(300 * (1 if i % 2 else -1), 500 * (1 if i % 3 else -1), 0)
            else:
                bvel.y *= -1
                bpos.y = math.copysign(5000, bpos.y)
        bvel = bvel * (1.0 - 0.15 * DT)
        if bvel.flat_length() < 250:
            bvel = Vec3(400 * math.cos(i * 0.05), 400 * math.sin(i * 0.07), 0)

        # --- the scripted human: chases the ball, ignores boost ----------
        # Aim at a point just short of the ball rather than the ball centre.
        # Driving at the centre makes the actor converge to zero distance and
        # sit still on top of it, which drags its measured pace to zero and is
        # not what a chasing player actually looks like.
        chase_dir = (bpos - ally.pos).flat()
        standoff = chase_dir.normalized() * 120.0 if chase_dir.length() > 200.0 else Vec3()
        ally.drive_to(bpos - standoff, 1500, use_boost=ally.boost > 0)
        if ally.pos.flat_dist(bpos) < 200:
            # Direction to push the ball. If the actor is sitting exactly on
            # the ball the separation vector is degenerate and normalising it
            # yields zero, which stops the ball dead and freezes the whole
            # simulation -- so fall back to the direction of travel.
            push = (bpos - ally.pos).flat()
            if push.length() < 60.0:
                push = ally.vel.flat()
            if push.length() < 60.0:
                push = Vec3(math.cos(i * 0.11), math.sin(i * 0.11), 0.0)
            bvel = push.normalized() * 1400
            bvel.z = 0

        # Opponents shadow their half.
        opp1.drive_to(Vec3(bpos.x * 0.5, max(bpos.y, 500), 17), 1100, use_boost=False)
        opp2.drive_to(Vec3(bpos.x * 0.5 + 900, 2500, 17), 900, use_boost=False)

        # --- build the packet and run the brain --------------------------
        cars = [me.to_flat(), ally.to_flat(), opp1.to_flat(), opp2.to_flat()]
        b = ball(x=bpos.x, y=bpos.y, z=bpos.z, vx=bvel.x, vy=bvel.y, vz=bvel.z)
        pkt = packet(cars, b, time=t, score=tuple(score))
        st = GameState(pkt, 0, 0, FIELD_INFO, t - DT)
        pred = make_prediction(b, t, horizon=4.0)

        brain.decide(st, pred)
        d = brain.debug

        # Move the bot roughly toward whatever it decided on.
        if d.target is not None:
            me.drive_to(d.target, 1500, use_boost=me.boost > 20)
        elif d.intercept is not None:
            me.drive_to(d.intercept, 1600, use_boost=me.boost > 20)
        me.boost = min(100.0, me.boost + 6.0 * DT)  # stand-in for pad pickups

        rec.sample(st, d, d.my_time, d.ally_time)

        # Fire the events the real bot detects from the packet.
        if ally.pos.flat_dist(bpos) < 220 and i % 8 == 0:
            model.on_ally_touch(st, bpos)
            rec.event("ally_touch", st, z=round(bpos.z))
        if i > 0 and i % 900 == 0:
            model.on_kickoff(st, ally_went=True)

    model.tally.goals_for = score[0]
    model.tally.goals_against = score[1]

    insights = analyse(model, tuple(score))
    report = format_report(model, insights, tuple(score), rank)
    summary_path = rec.close(model, tuple(score))

    if verbose:
        print(report)
        print(f"\ntelemetry : {rec.trace_path.name}")
        print(f"events    : {rec.events_path.name}")
        print(f"summary   : {summary_path.name if summary_path else 'none'}")

    return model, insights, rec


if __name__ == "__main__":
    model, insights, rec = run_match()
    ok = True
    if not rec.trace_path.exists():
        print("FAIL: no telemetry trace written")
        ok = False
    if model.traits.samples < 1000:
        print(f"FAIL: model saw only {model.traits.samples} samples")
        ok = False
    if not insights:
        print("FAIL: coach produced no insights")
        ok = False
    print("\nOK" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
