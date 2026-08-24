"""
Does recovery work near real geometry?

Free-space search said the orientation controller and the dodge parameters are
both already near optimal, yet live telemetry says the cars spend most of their
airborne time inverted. Those two findings can only be reconciled by something
free space does not contain. This measures the obvious candidate: walls.

`predict_landing` says so itself -- "ignoring walls" -- and `Recovery` always
levels the roof to world up. Both are correct on open floor and wrong anywhere
near the curve, the wall or the corner.

    python tools/wallcheck.py

Reports, per zone, how well the SHIPPED recovery controller lands. Changes
nothing.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.control.recovery import Recovery  # noqa: E402
from bot.core.vec import Mat3, Vec3  # noqa: E402
from tools.simrs import (  # noqa: E402
    RSCar, make_arena, nearest_surface, place, run, surface_alignment, to_rs_vec,
)

TICK = 1.0 / 120.0
EPISODE = 600  # 5 seconds
SETTLE_GRACE = 90  # 0.75s -- long enough to right a tumble from any attitude


def tumbled(rng) -> tuple[Mat3, Vec3]:
    """A random attitude and spin, of the sort a car has after a challenge."""
    yaw = rng.uniform(-math.pi, math.pi)
    pitch = rng.uniform(-math.pi, math.pi)
    roll = rng.uniform(-math.pi, math.pi)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    f = Vec3(cp * cy, cp * sy, sp)
    u = Vec3(
        -cr * cy * sp - sr * sy,
        -cr * sy * sp + sr * cy,
        cr * cp,
    )
    f = f.normalized()
    u = (u - f * u.dot(f)).normalized()
    spin = Vec3(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))
    return Mat3(f, u.cross(f), u), spin


ZONES = [
    # name,            x,     y,    z,   vx,    vy,   vz
    ("open floor",      0.0,   0.0, 700.0,   300.0,  600.0, 250.0),
    ("near side wall", 3300.0, 0.0, 700.0,   700.0,  200.0, 250.0),
    ("at side wall",   3900.0, 0.0, 900.0,   500.0,  100.0, 200.0),
    ("corner",         3300.0, 4200.0, 800.0, 500.0,  500.0, 200.0),
    ("back wall",         0.0, 4900.0, 900.0, 100.0,  600.0, 200.0),
    ("high, midfield",    0.0,   0.0, 1400.0,   0.0,  400.0, 300.0),
]

TRIALS = 24


def park_ball(arena):
    """The ball spawns at the origin and cars land on top of it."""
    b = arena.ball.get_state()
    b.pos = to_rs_vec(Vec3(-3600.0, -4600.0, 93.15))
    b.vel = to_rs_vec(Vec3(0.0, 0.0, -1.0))
    arena.ball.set_state(b)


def episode(arena, car, rsc, start, ori, spin, rng):
    park_ball(arena)
    place(car, start[0], start[1], ori, spin, boost=60.0)
    rsc.sync()
    rec = Recovery(allow_wavedash=True)

    stats = {
        "landings": 0,        # air -> ground transitions
        "bad_landings": 0,    # ... where the roof was turned away from the surface hit
        "first_align": None,  # alignment at the FIRST touchdown
        "settle": None,
        "air_ticks": 0,
        "ground_ticks": 0,
    }
    was_air = [not rsc.on_ground]

    def observe(i, c):
        # Alignment against the *nearest* surface is confounded in the air: a
        # car at x=3900, z=700 is nearer the wall than the floor, yet landing
        # on the floor may be exactly right. Only at the moment of contact is
        # the relevant surface unambiguous, so that is where we judge.
        wheels = sum(1 for w in c.wheels_with_contact if w)
        align = surface_alignment(c)
        if c.on_ground:
            stats["ground_ticks"] += 1
            if was_air[0]:
                stats["landings"] += 1
                if stats["first_align"] is None:
                    stats["first_align"] = align
                if align < 0.5:
                    stats["bad_landings"] += 1
            if wheels == 4 and align > 0.85 and stats["settle"] is None:
                stats["settle"] = i
        else:
            stats["air_ticks"] += 1
        was_air[0] = not c.on_ground

    run(car, rsc, arena, rec.step, EPISODE, observe=observe)
    return stats


def main() -> int:
    arena = make_arena()
    car = arena.add_car(__import__("RocketSim").Team.BLUE)
    rsc = RSCar(car)

    print("shipped Recovery controller, %d trials per zone, %.1fs episodes"
          % (TRIALS, EPISODE * TICK))
    print()
    print("%-16s %8s %8s %11s %10s %9s"
          % ("zone", "settled", "settle_s", "1st_align", "bad_land%", "touches"))
    print("-" * 70)

    rows = []
    for name, x, y, z, vx, vy, vz in ZONES:
        rng = random.Random(hash(name) & 0xFFFF)
        settles, first, badpct, bounces, never = [], [], [], [], 0
        for _ in range(TRIALS):
            ori, spin = tumbled(rng)
            s = episode(
                arena, car, rsc,
                (Vec3(x, y, z), Vec3(vx, vy, vz)),
                ori, spin, rng,
            )
            if s["settle"] is None:
                never += 1
            else:
                settles.append(s["settle"] * TICK)
            if s["first_align"] is not None:
                first.append(s["first_align"])
            if s["landings"]:
                badpct.append(100.0 * s["bad_landings"] / s["landings"])
                bounces.append(s["landings"])

        settled_pct = 100.0 * (TRIALS - never) / TRIALS
        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        row = (
            name,
            settled_pct,
            avg(settles),
            avg(first),
            avg(badpct),
            avg(bounces),
        )
        rows.append(row)
        print("%-16s %7.0f%% %8.2f %11.2f %9.0f%% %9.1f" % row)

    print("-" * 70)
    print()
    print("1st_align is the roof-vs-surface dot product at the FIRST touchdown:")
    print("1.0 is wheels-down, 0 is on its side, -1 is roof-first. bad_land% is")
    print("the share of all touchdowns below 0.5. touches counts air->ground")
    print("transitions, so above 1 means it bounced rather than stuck.")
    print()
    base = next(r for r in rows if r[0] == "open floor")
    print()
    print("open floor is the control. A zone that settles far less often, or")
    print("stays misaligned with the surface it is touching, is a geometry bug")
    print("the free-space optimiser could not have found.")
    print()
    for r in rows[1:]:
        d = r[1] - base[1]
        if d < -10.0:
            print("  %-16s settles %.0f points less often than open floor" % (r[0], -d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
