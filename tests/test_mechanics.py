"""
Per-mechanic intercept selection.

Guards the architectural fix. The bot previously ran ONE earliest-touch search
across all techniques, which cannot ever choose an aerial: a lofted ball
descends into ground range, and the descending low slice is always earlier than
any aerial slice. Measured on a real match, that solver proposed an aerial on 3
ticks out of 7,200.

The fix is one banded search per mechanic, so "wait for it to land" is
inadmissible to the aerial search rather than merely unattractive. These tests
assert the bands hold and the arbiter behaves.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.core.intercept import (  # noqa: E402
    AERIAL,
    DODGE,
    DOUBLE_JUMP,
    GROUND,
    find_intercept,
)
from bot.core.mechanics import (  # noqa: E402
    aerial_lead,
    arrive_shift,
    double_jump_time_needed,
    jump_duration,
)
from bot.core.physics import ReachCurve  # noqa: E402
from tests.harness import ball, car, scenario  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def solve(car_y=-1200, boost=60, z=93, vz=0):
    st, pred = scenario(
        me=car(x=0, y=car_y, yaw=math.pi / 2, boost=boost),
        the_ball=ball(x=0, y=0, z=z, vz=vz),
    )
    return find_intercept(st.me, pred, st.time, ReachCurve(st.me.speed, st.me.boost))


def test_rolling_ball_is_a_ground_level_mechanic():
    """
    A ball on the floor must be met on the floor -- but either by driving into
    it or by dodging into it. Which of the two is a power question, not a
    height question, and is covered by the closing-speed test below.
    """
    ic = solve(z=93)
    assert ic.kind in (GROUND, DODGE), f"rolling ball gave {ic.kind}"
    assert ic.feasible


def test_low_closing_speed_prefers_a_dodge():
    """
    The clause that fixes "no power behind their shots".

    If the ball and the car are moving at similar velocities, driving into the
    ball transfers almost nothing however fast the car is going, so the strike
    has to be a dodge. Omitting this left 93.9% of touches as plain ground
    contacts and the bots nudged the ball around at walking pace.
    """
    # Car and ball both travelling +y at a similar rate: driving into it is
    # pointless, so the solver must pick the dodge.
    st, pred = scenario(
        me=car(x=0, y=-1200, yaw=math.pi / 2, vy=900, boost=60),
        the_ball=ball(x=0, y=0, vy=800),
    )
    ic = find_intercept(
        st.me, pred, st.time, ReachCurve(st.me.speed, st.me.boost),
        shoot_target=st.enemy_goal,
    )
    assert ic.kind == DODGE, (
        f"car and ball moving together at similar speed should be a dodge, got {ic.kind}"
    )


def test_high_ball_within_reach_is_an_aerial():
    """The whole point. This was impossible before the rewrite."""
    ic = solve(car_y=-900, boost=100, z=900, vz=500)
    assert ic.kind == AERIAL, f"expected an aerial for a lofted ball, got {ic.kind}"
    assert ic.pos.z > 500.0, f"aerial contact at only z={ic.pos.z:.0f}"


def test_mid_height_ball_uses_a_double_jump():
    ic = solve(car_y=-900, boost=60, z=600, vz=300)
    assert ic.kind in (DOUBLE_JUMP, AERIAL), f"expected air play, got {ic.kind}"


def test_no_aerial_without_boost():
    """An aerial needs boost in hand; without it the bot must stay grounded."""
    ic = solve(car_y=-900, boost=10, z=900, vz=500)
    assert ic.kind != AERIAL, "attempted an aerial on 10 boost"


def test_no_aerial_when_too_far_to_get_there():
    ic = solve(car_y=-4500, boost=100, z=700, vz=0)
    assert ic.kind != AERIAL, (
        f"claimed an aerial from 4500uu away on a ball already falling ({ic.kind})"
    )


def test_bands_do_not_admit_absurd_choices():
    """A ball on the floor must never be tagged as an air mechanic."""
    for vz in (0, -200, -600):
        ic = solve(z=93, vz=vz)
        assert ic.kind in (GROUND, DODGE), f"floor ball tagged {ic.kind}"


def test_jump_duration_is_monotonic_and_bounded():
    prev = -1.0
    for z in range(92, 1400, 50):
        d = jump_duration(z)
        assert d >= prev, f"jump duration not monotonic at z={z}"
        assert 0.0 < d <= 1.6, f"jump duration {d:.2f} out of range at z={z}"
        prev = d


def test_double_jump_curve_is_sane_in_band():
    """
    The fitted curve keeps rising past what the mechanic can do -- 1.66s at
    z=550, against a 1.25s dodge window. Inside the usable range it must be
    sensible, and beyond it the guard must reject rather than the curve lie.
    """
    from bot.core.mechanics import DOUBLE_JUMP_MAX_TIME

    for z in (250, 350, 450):
        t = double_jump_time_needed(z)
        assert 0.1 < t <= DOUBLE_JUMP_MAX_TIME, (
            f"double jump time {t:.2f}s at z={z} is implausible"
        )
    assert double_jump_time_needed(550) > DOUBLE_JUMP_MAX_TIME, (
        "expected the curve to exceed the window at the top of the band, so the "
        "guard is what rejects it"
    )


def test_aerial_lead_grows_with_height():
    leads = [aerial_lead(z) for z in (500, 650, 800, 1200, 1800)]
    assert leads == sorted(leads), f"aerial lead not increasing: {leads}"
    assert 0.7 < leads[0] < 1.0 and 2.0 < leads[-1] < 2.6


def test_arrive_shift_collapses_inside_the_turn_circle():
    """
    The guard that stops the shift itself causing orbiting: when the shift
    would land inside the car's own turning circle, it must go to zero.
    """
    turn_r = 640.0
    far = arrive_shift(3000.0, 1400.0, 0.0, turn_r)
    near = arrive_shift(400.0, 1400.0, 0.0, turn_r)
    assert far > 0.0, "expected a shift at range"
    assert near == 0.0, f"shift {near:.0f} should collapse to 0 when close"


def test_arrive_shift_decays_with_distance():
    turn_r = 300.0
    vals = [arrive_shift(d, 1400.0, 0.0, turn_r) for d in (3000, 2000, 1200, 700)]
    nonzero = [v for v in vals if v > 0]
    assert nonzero == sorted(nonzero, reverse=True), f"shift not decaying: {vals}"


def test_grounded_car_can_dodge():
    """
    The single most damaging bug found in this bot.

    `can_dodge` was `dodge_timeout > 0`, but RLBot documents dodge_timeout as
    "-1 while on ground". Every dodge in the codebase was guarded by
    `car.on_ground and car.can_dodge` -- a contradiction that could never be
    true. Across a full match the Dodge manoeuvre executed on 0 of 7,300 ticks
    while 89.5% of intercepts were asking for one: no flips, no half-flips, and
    no power in any touch.

    A grounded car can always dodge. It jumps, then dodges.
    """
    st, _ = scenario(me=car(x=0, y=0, yaw=0, boost=50))
    assert st.me.on_ground, "harness should give a grounded car"
    assert st.me.can_dodge, (
        "a grounded car must be able to dodge -- this is the guard that silently "
        "disabled every flip in the bot"
    )
    # And the combination the strike code actually tests must be satisfiable.
    assert st.me.on_ground and st.me.can_dodge


def test_demolished_car_cannot_dodge():
    st, _ = scenario(me=car(x=0, y=0, demolished=True))
    assert not st.me.can_dodge, "a demolished car must not be able to dodge"


def test_airborne_dodge_respects_the_window():
    open_st, _ = scenario(me=car(x=0, y=0, z=500, on_ground=False, can_dodge=True))
    shut_st, _ = scenario(me=car(x=0, y=0, z=500, on_ground=False, can_dodge=False))
    assert open_st.me.can_dodge, "airborne with the window open should allow a dodge"
    assert not shut_st.me.can_dodge, (
        "airborne with the window expired must NOT allow a dodge -- the ground "
        "exemption must not leak into the air case"
    )


def main() -> int:
    for name, fn in [
        ("rolling ball -> ground-level", test_rolling_ball_is_a_ground_level_mechanic),
        ("low closing speed -> dodge", test_low_closing_speed_prefers_a_dodge),
        ("lofted ball -> AERIAL", test_high_ball_within_reach_is_an_aerial),
        ("mid ball -> air play", test_mid_height_ball_uses_a_double_jump),
        ("no aerial without boost", test_no_aerial_without_boost),
        ("no aerial from too far", test_no_aerial_when_too_far_to_get_there),
        ("floor ball never air-tagged", test_bands_do_not_admit_absurd_choices),
        ("jump duration monotonic", test_jump_duration_is_monotonic_and_bounded),
        ("double jump curve sane", test_double_jump_curve_is_sane_in_band),
        ("aerial lead grows with height", test_aerial_lead_grows_with_height),
        ("shift collapses in turn circle", test_arrive_shift_collapses_inside_the_turn_circle),
        ("shift decays with distance", test_arrive_shift_decays_with_distance),
        ("grounded car CAN dodge", test_grounded_car_can_dodge),
        ("demolished car cannot dodge", test_demolished_car_cannot_dodge),
        ("airborne dodge window respected", test_airborne_dodge_respects_the_window),
    ]:
        check(name, fn)

    print(f"{'test':38s} result")
    print("-" * 74)
    fails = 0
    for name, ok, err in RESULTS:
        print(f"{name:38s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            fails += 1
    print("-" * 74)
    print(f"{len(RESULTS) - fails}/{len(RESULTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
