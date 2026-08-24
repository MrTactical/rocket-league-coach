"""
Multi-teammate rotation.

The failure this guards against: with three players a side, every non-attacking
teammate computed its support position from the same ball position and got the
same answer, so they drove around side by side. From the outside that looks
exactly like "no rotation" -- and no unit test caught it, because with two
players there is only ever one supporter and the bug is invisible.

The invariant is simple and worth stating plainly: **no two teammates should
want to be in the same place.**
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.brain.roles import defend_position, support_position  # noqa: E402
from bot.brain.teammate import TeammateModel  # noqa: E402
from bot.core.game import GameState  # noqa: E402
from tests.harness import FIELD_INFO, ball, car, packet  # noqa: E402

# Two cars closer than this are effectively in the same place.
MIN_SEPARATION = 900.0

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def three_v_three(ball_x=0.0, ball_y=500.0):
    cars = [
        car(x=-800, y=-1500, yaw=math.pi / 2, name="Ally1", team=0),
        car(x=200, y=-2600, yaw=math.pi / 2, name="Ally2", team=0),
        car(x=1400, y=-3600, yaw=math.pi / 2, name="Ally3", team=0),
        car(x=-500, y=2000, yaw=-math.pi / 2, name="Opp1", team=1),
        car(x=500, y=2800, yaw=-math.pi / 2, name="Opp2", team=1),
        car(x=0, y=3600, yaw=-math.pi / 2, name="Opp3", team=1),
    ]
    pkt = packet(cars, ball(x=ball_x, y=ball_y), time=10.0)
    return GameState(pkt, 0, 0, FIELD_INFO, 10.0 - 1 / 120)


def test_support_positions_separate_by_ordinal():
    st = three_v_three()
    model = TeammateModel("_rot")
    p1 = support_position(st, model, None, ordinal=1)
    p2 = support_position(st, model, None, ordinal=2)
    d = p1.flat_dist(p2)
    assert d > MIN_SEPARATION, (
        f"second and third man want to be {d:.0f}uu apart -- they will drive "
        f"side by side. p1={p1} p2={p2}"
    )


def test_third_man_is_deeper_than_second():
    st = three_v_three()
    model = TeammateModel("_rot")
    p1 = support_position(st, model, None, ordinal=1)
    p2 = support_position(st, model, None, ordinal=2)
    # goal_sign is -1 for team 0, so deeper means a more negative y.
    depth1 = p1.y * st.goal_sign
    depth2 = p2.y * st.goal_sign
    assert depth2 > depth1, (
        f"third man (depth {depth2:.0f}) should sit deeper than second "
        f"({depth1:.0f})"
    )


def test_defenders_take_opposite_posts():
    st = three_v_three(ball_x=1200.0, ball_y=-2000.0)
    d1 = defend_position(st, None, ordinal=1)
    d2 = defend_position(st, None, ordinal=2)
    sep = d1.flat_dist(d2)
    assert sep > 500.0, f"two defenders only {sep:.0f}uu apart; they will stack"
    assert d1.x * d2.x < 0 or abs(d1.x - d2.x) > 700.0, (
        f"defenders should split the posts, got x={d1.x:.0f} and {d2.x:.0f}"
    )


def test_all_three_slots_are_mutually_separated():
    """The full set: attacker aside, no two support/defend slots collide."""
    st = three_v_three()
    model = TeammateModel("_rot")
    spots = [
        ("support#2", support_position(st, model, None, ordinal=1)),
        ("support#3", support_position(st, model, None, ordinal=2)),
        ("defend#2", defend_position(st, None, ordinal=1)),
        ("defend#3", defend_position(st, None, ordinal=2)),
    ]
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            (n1, a), (n2, b) = spots[i], spots[j]
            # Only slots of the same kind must separate; a supporter and a
            # defender legitimately occupy different depths anyway.
            if n1.split("#")[0] == n2.split("#")[0]:
                d = a.flat_dist(b)
                assert d > MIN_SEPARATION, f"{n1} and {n2} only {d:.0f}uu apart"


def test_positions_stay_in_bounds():
    from bot.core.constants import BACK_WALL_Y, SIDE_WALL_X

    model = TeammateModel("_rot")
    for bx in (-3500, 0, 3500):
        for by in (-4500, 0, 4500):
            st = three_v_three(bx, by)
            for ordinal in (1, 2, 3):
                for name, p in (
                    ("support", support_position(st, model, None, ordinal)),
                    ("defend", defend_position(st, None, ordinal)),
                ):
                    assert abs(p.x) < SIDE_WALL_X, f"{name}#{ordinal} x={p.x:.0f} out of bounds"
                    assert abs(p.y) < BACK_WALL_Y, f"{name}#{ordinal} y={p.y:.0f} out of bounds"


def main() -> int:
    for name, fn in [
        ("support slots separate by ordinal", test_support_positions_separate_by_ordinal),
        ("third man sits deeper than second", test_third_man_is_deeper_than_second),
        ("defenders split the posts", test_defenders_take_opposite_posts),
        ("no two same-kind slots collide", test_all_three_slots_are_mutually_separated),
        ("all positions stay in bounds", test_positions_stay_in_bounds),
    ]:
        check(name, fn)

    print(f"{'test':40s} result")
    print("-" * 76)
    fails = 0
    for name, ok, err in RESULTS:
        print(f"{name:40s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            fails += 1
    print("-" * 76)
    print(f"{len(RESULTS) - fails}/{len(RESULTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
