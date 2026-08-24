"""
Offline parameter search against the simulated physics core.

    python tools/optimise.py orient        # tune the orientation PD
    python tools/optimise.py orient --apply
    python tools/optimise.py --list

This is the "self-learning" that is actually achievable on one machine. The bot
has around thirty hand-set constants, and hand-tuning them by watching matches
is slow and unreliable -- several fixes this session were aimed at the wrong
cause until a number said otherwise. Here the constants are searched against a
measurable objective over hundreds of simulated scenarios, which takes seconds.

What this is NOT: reinforcement learning over raw controls. That needs billions
of timesteps and distributed hardware; a five minute match yields about 36,000
samples. Searching a low-dimensional parameter space is the version of the same
idea that fits in the budget available.

Results are printed with the baseline alongside, and only written to source when
`--apply` is passed -- a search that improves a simulated objective has not yet
proved anything about the real game.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlbot import flat  # noqa: E402

import bot.control.aerial as aerial_mod  # noqa: E402
from bot.core.vec import Mat3, Vec3  # noqa: E402
from tools.simcore import SimCar, upright  # noqa: E402

# --- scenarios ------------------------------------------------------------
#
# A spread of attitudes and spins a car realistically ends a flip or a bump in.
# Fixed, not random, so two runs of the search are comparable.
ATTITUDES = [
    ("inverted", (0.0, 0.0, math.pi), (0.0, 0.0, 0.0)),
    ("inverted + roll spin", (0.0, 0.0, math.pi), (3.0, 0.0, 0.0)),
    ("on left side", (0.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0)),
    ("on right side", (0.0, 0.0, -math.pi / 2), (0.0, 0.0, 0.0)),
    ("nose down", (-math.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("nose up", (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("tumbling", (0.8, 1.2, 2.4), (2.0, -1.5, 1.0)),
    ("post-flip pitch", (1.4, 0.3, 0.2), (0.0, -4.0, 0.0)),
    ("post-flip diagonal", (0.9, 0.6, 2.2), (1.5, -2.5, 0.5)),
    ("yaw spin upright", (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
    ("slight roll", (0.0, 0.0, 0.6), (0.0, 0.0, 0.0)),
    ("inverted + yaw", (0.0, 1.0, math.pi), (0.0, 0.0, 2.0)),
]


def make_car(rot, spin, height=400.0):
    return SimCar(
        pos=Vec3(0.0, 0.0, height),
        vel=Vec3(600.0, 0.0, 0.0),
        ori=Mat3.from_rotator(flat.Rotator(rot[0], rot[1], rot[2])),
        ang_vel=Vec3(*spin),
        boost=50.0,
        on_ground=False,
    )


def orientation_cost(p_gain: float, omega_gain: float, limit: int = 300) -> tuple[float, int]:
    """
    Mean ticks to settle upright across the attitude set.

    Unsettled cases are charged the full limit, so a gain pair that is quick on
    easy attitudes but never recovers a tumble scores badly rather than being
    flattered by an average over successes only.
    """
    old_p, old_w = aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN
    aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN = p_gain, omega_gain
    try:
        total, settled = 0, 0
        for _name, rot, spin in ATTITUDES:
            car = make_car(rot, spin)
            took = None
            for i in range(limit):
                c = flat.ControllerState()
                c.pitch, c.yaw, c.roll = aerial_mod.orient_controls(
                    car, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)
                )
                car.step(c)
                if upright(car) > 0.95 and car.ang_vel.length() < 0.8:
                    took = i
                    break
            if took is None:
                total += limit
            else:
                total += took
                settled += 1
        return total / len(ATTITUDES), settled
    finally:
        aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN = old_p, old_w


def search_orientation(coarse: int = 9, refine: int = 7):
    """Grid search, then a local refinement around the best cell."""
    base_cost, base_settled = orientation_cost(aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN)
    print(f"baseline  ORIENT_P={aerial_mod.ORIENT_P:.2f} OMEGA_GAIN={aerial_mod.OMEGA_GAIN:.2f}"
          f"  ->  {base_cost:6.1f} ticks ({base_cost / 120:.2f}s), settled {base_settled}/{len(ATTITUDES)}")

    p_range = [1.0 + i * (14.0 - 1.0) / (coarse - 1) for i in range(coarse)]
    w_range = [0.1 + i * (1.6 - 0.1) / (coarse - 1) for i in range(coarse)]

    best = (base_cost, base_settled, aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN)
    print("\ncoarse grid...")
    for p, w in itertools.product(p_range, w_range):
        cost, settled = orientation_cost(p, w)
        # More attitudes recovered always beats a faster average.
        if (settled, -cost) > (best[1], -best[0]):
            best = (cost, settled, p, w)
    print(f"  best so far  P={best[2]:.2f} W={best[3]:.2f}  {best[0]:.1f} ticks, settled {best[1]}")

    print("refining...")
    span_p, span_w = 1.6, 0.22
    for _ in range(3):
        cp, cw = best[2], best[3]
        for p in [cp + (i / (refine - 1) - 0.5) * 2 * span_p for i in range(refine)]:
            for w in [cw + (i / (refine - 1) - 0.5) * 2 * span_w for i in range(refine)]:
                if p <= 0.2 or w <= 0.02:
                    continue
                cost, settled = orientation_cost(p, w)
                if (settled, -cost) > (best[1], -best[0]):
                    best = (cost, settled, p, w)
        span_p *= 0.5
        span_w *= 0.5

    cost, settled, p, w = best
    print(f"\nbest      ORIENT_P={p:.2f} OMEGA_GAIN={w:.2f}"
          f"  ->  {cost:6.1f} ticks ({cost / 120:.2f}s), settled {settled}/{len(ATTITUDES)}")
    if base_cost > 0:
        print(f"improvement: {100 * (base_cost - cost) / base_cost:+.1f}% settle time,"
              f" {settled - base_settled:+d} attitudes recovered")

    print("\nper-attitude, baseline -> tuned:")
    for name, rot, spin in ATTITUDES:
        a = _single(rot, spin, aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN)
        b = _single(rot, spin, p, w)
        fmt = lambda v: "never" if v is None else f"{v / 120:.2f}s"  # noqa: E731
        print(f"  {name:22s} {fmt(a):>7s} -> {fmt(b):>7s}")

    return p, w


def _single(rot, spin, p_gain, omega_gain, limit=300):
    old_p, old_w = aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN
    aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN = p_gain, omega_gain
    try:
        car = make_car(rot, spin)
        for i in range(limit):
            c = flat.ControllerState()
            c.pitch, c.yaw, c.roll = aerial_mod.orient_controls(
                car, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)
            )
            car.step(c)
            if upright(car) > 0.95 and car.ang_vel.length() < 0.8:
                return i
        return None
    finally:
        aerial_mod.ORIENT_P, aerial_mod.OMEGA_GAIN = old_p, old_w


def apply_orientation(p: float, w: float):
    path = ROOT / "bot" / "control" / "aerial.py"
    s = path.read_text(encoding="utf-8")
    s = s.replace(f"ORIENT_P = {aerial_mod.ORIENT_P}", f"ORIENT_P = {p:.2f}")
    s = s.replace(f"OMEGA_GAIN = {aerial_mod.OMEGA_GAIN}", f"OMEGA_GAIN = {w:.2f}")
    path.write_text(s, encoding="utf-8")
    print(f"\nwritten to {path.relative_to(ROOT)}: ORIENT_P={p:.2f} OMEGA_GAIN={w:.2f}")



# --- dodge landing --------------------------------------------------------
#
# The orientation search showed the PD rights any attitude in ~0.66s, so the
# 62% of airborne time spent inverted is NOT slow righting -- a car flipped at
# 99uu lands long before any controller could recover it. What matters is the
# attitude the Dodge HANDS BACK, and how long it holds on trying to fix it.


def dodge_landing_cost(recover_time: float, max_time: float, upright_gate: float):
    """
    Run the real Dodge manoeuvre from a driving car and score the landing.

    Cost is the fraction of post-dodge ticks spent inverted, plus a penalty for
    still being inverted when control is handed back -- that is the state that
    produces a car sliding along on its roof.
    """
    import bot.control.dodge as dodge_mod

    old = (dodge_mod.Dodge.RECOVER_TIME, dodge_mod.Dodge.MAX_TIME, dodge_mod.Dodge.UPRIGHT)
    dodge_mod.Dodge.RECOVER_TIME = recover_time
    dodge_mod.Dodge.MAX_TIME = max_time
    dodge_mod.Dodge.UPRIGHT = upright_gate
    try:
        inverted = 0
        total = 0
        bad_handbacks = 0
        cases = 0
        for speed in (600.0, 1200.0, 1800.0):
            for ang in (0.0, 0.6, 1.4, 2.4, -0.9):
                cases += 1
                car = SimCar(
                    pos=Vec3(0.0, 0.0, 17.0),
                    vel=Vec3(speed, 0.0, 0.0),
                    boost=40.0,
                    on_ground=True,
                )
                target = car.pos + Vec3(math.cos(ang), math.sin(ang), 0.0) * 400.0
                dodge = dodge_mod.Dodge(target=target)

                class _S:
                    pass

                st = _S()
                st.me = car
                st.time = 0.0
                handback = None
                for i in range(200):
                    st.time = i / 120.0
                    out = dodge.step(st)
                    if out is None:
                        handback = i
                        break
                    car.step(out)
                if handback is None:
                    handback = 200
                if upright(car) < 0.7:
                    bad_handbacks += 1
                # Watch the 1.5s after control returns.
                for i in range(180):
                    total += 1
                    if upright(car) < 0.5:
                        inverted += 1
                    car.step(flat.ControllerState(throttle=1.0))
        return (inverted / max(1, total)) + 0.5 * (bad_handbacks / max(1, cases)), bad_handbacks, cases
    finally:
        (
            dodge_mod.Dodge.RECOVER_TIME,
            dodge_mod.Dodge.MAX_TIME,
            dodge_mod.Dodge.UPRIGHT,
        ) = old


def search_dodge():
    import bot.control.dodge as dodge_mod

    base = (dodge_mod.Dodge.RECOVER_TIME, dodge_mod.Dodge.MAX_TIME, dodge_mod.Dodge.UPRIGHT)
    bc, bbad, ncase = dodge_landing_cost(*base)
    print(f"baseline  RECOVER_TIME={base[0]:.2f} MAX_TIME={base[1]:.2f} UPRIGHT={base[2]:.2f}")
    print(f"          cost {bc:.4f}   handed back inverted {bbad}/{ncase}")

    best = (bc, bbad) + base
    print("\nsearching...")
    for rt in (0.55, 0.70, 0.85, 1.00):
        for mt in (1.0, 1.4, 1.8, 2.2):
            if mt <= rt:
                continue
            for ug in (0.70, 0.80, 0.90):
                cost, bad, _ = dodge_landing_cost(rt, mt, ug)
                if (-bad, -cost) > (-best[1], -best[0]):
                    best = (cost, bad, rt, mt, ug)

    cost, bad, rt, mt, ug = best
    print(f"\nbest      RECOVER_TIME={rt:.2f} MAX_TIME={mt:.2f} UPRIGHT={ug:.2f}")
    print(f"          cost {cost:.4f}   handed back inverted {bad}/{ncase}   (was {bbad}/{ncase})")
    if bc > 0:
        print(f"improvement: {100 * (bc - cost) / bc:+.1f}% cost, {bbad - bad:+d} fewer bad handbacks")
    return rt, mt, ug


def apply_dodge(rt: float, mt: float, ug: float):
    import bot.control.dodge as dodge_mod

    path = ROOT / "bot" / "control" / "dodge.py"
    s = path.read_text(encoding="utf-8")
    s = s.replace(f"RECOVER_TIME = {dodge_mod.Dodge.RECOVER_TIME}", f"RECOVER_TIME = {rt:.2f}")
    s = s.replace(f"MAX_TIME = {dodge_mod.Dodge.MAX_TIME}", f"MAX_TIME = {mt:.2f}")
    s = s.replace(f"UPRIGHT = {dodge_mod.Dodge.UPRIGHT}", f"UPRIGHT = {ug:.2f}")
    path.write_text(s, encoding="utf-8")
    print(f"\nwritten to {path.relative_to(ROOT)}")


TARGETS = {
    "orient": (
        "orientation PD gains (ORIENT_P, OMEGA_GAIN) -- used by every airborne "
        "manoeuvre: aerials, dodges, half-flips, wavedashes and recovery",
        search_orientation,
        apply_orientation,
    ),
    "dodge": (
        "dodge landing (RECOVER_TIME, MAX_TIME, UPRIGHT) -- how long a flip "
        "holds control while righting the car, which decides whether it lands "
        "on its wheels or slides on its roof",
        search_dodge,
        apply_dodge,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune bot constants against simulated physics.")
    ap.add_argument("target", nargs="?", help="what to tune (see --list)")
    ap.add_argument("--list", action="store_true", help="show available targets")
    ap.add_argument("--apply", action="store_true", help="write the result into the source")
    args = ap.parse_args()

    if args.list or not args.target:
        print("targets:\n")
        for name, (desc, _, _) in TARGETS.items():
            print(f"  {name:10s} {desc}")
        print("\nNothing is written unless --apply is passed.")
        return 0

    if args.target not in TARGETS:
        print(f"unknown target {args.target!r}; try --list")
        return 1

    desc, search, apply_fn = TARGETS[args.target]
    print(f"tuning: {desc}\n")
    result = search()
    if args.apply:
        apply_fn(*result)
    else:
        print("\n(dry run -- pass --apply to write it, then measure in a real match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
