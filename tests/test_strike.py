"""
Strike geometry.

The bot's first real in-game failure was "drives up to the ball but never hits
it". The cause was purely geometric and entirely invisible to the decision
tests: `find_intercept` returned a position already backed off 151uu from the
ball, and `Strike` then applied its own 146uu contact offset on top, so the car
aimed at a point ~300uu away from a ball it must be within ~150uu of to touch.
It drove there correctly and stopped, doing exactly what it was told.

Nothing in the role tests could catch that -- the role was right, the driving
was right, only the target was wrong. These tests assert the geometric
invariants directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.brain.strike import CONTACT_OFFSET, contact_point, goal_target  # noqa: E402
from bot.core.constants import BALL_RADIUS  # noqa: E402
from bot.core.intercept import find_intercept  # noqa: E402
from bot.core.physics import ReachCurve  # noqa: E402
from bot.core.vec import Vec3  # noqa: E402
from tests.harness import ball, car, scenario  # noqa: E402

# A car's nose reaches roughly 60uu past its origin, so the origin has to be
# within about ball radius + 60 for contact. Allow a little slack.
MAX_TOUCH_DISTANCE = BALL_RADIUS + 75.0

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def test_intercept_reports_ball_centre():
    """Intercept.pos must be the ball itself, not a pre-offset drive target."""
    for bx, by in ((0, 0), (1500, -800), (-2200, 1900)):
        st, pred = scenario(
            me=car(x=0, y=-3000, yaw=math.pi / 2, boost=60),
            the_ball=ball(x=bx, y=by),
        )
        ic = find_intercept(st.me, pred, st.time, ReachCurve(st.me.speed, st.me.boost))
        assert ic.feasible, f"ball at ({bx},{by}) should be reachable"
        # The ball moves a little over the intercept window, so compare against
        # the prediction slice rather than the spawn point.
        err = abs(ic.pos.z - BALL_RADIUS)
        assert err < 25.0, (
            f"Intercept.pos.z is {ic.pos.z:.0f}, expected ~{BALL_RADIUS} "
            f"(a ball centre). A pre-offset point would sit lower."
        )


def test_contact_point_is_within_touching_range():
    """The point we drive at must be close enough to actually strike."""
    for bx, by in ((0, 0), (2000, 1000), (-1800, -2400)):
        st, pred = scenario(
            me=car(x=0, y=-3000, yaw=math.pi / 2, boost=60),
            the_ball=ball(x=bx, y=by),
        )
        ic = find_intercept(st.me, pred, st.time, ReachCurve(st.me.speed, st.me.boost))
        aim = goal_target(st, ic.pos)
        cp = contact_point(ic.pos, aim)
        d = ic.pos.flat_dist(cp)
        assert d <= MAX_TOUCH_DISTANCE, (
            f"contact point is {d:.0f}uu from the ball, too far to touch "
            f"(limit {MAX_TOUCH_DISTANCE:.0f}). Double offset regression?"
        )


def test_contact_point_is_on_the_correct_side():
    """
    The car must approach from the side opposite the aim, so the ball is
    driven toward the target rather than away from it.
    """
    ball_pos = Vec3(0.0, 0.0, BALL_RADIUS)
    aim = Vec3(0.0, 5120.0, 300.0)  # shooting at +Y
    cp = contact_point(ball_pos, aim)
    assert cp.y < ball_pos.y, (
        f"contact point y={cp.y:.0f} should be behind the ball (y<0) when "
        f"shooting toward +Y"
    )
    # And the offset should be the configured distance.
    assert abs(ball_pos.dist(cp) - CONTACT_OFFSET) < 1.0


def test_offset_direction_follows_the_aim():
    """Aiming the other way must flip which side we strike from."""
    b = Vec3(0.0, 0.0, BALL_RADIUS)
    left = contact_point(b, Vec3(-4000.0, 0.0, 100.0))
    right = contact_point(b, Vec3(4000.0, 0.0, 100.0))
    assert left.x > 0 > right.x, (
        f"expected opposite approach sides, got left={left.x:.0f} right={right.x:.0f}"
    )


def test_approach_field_still_offset():
    """
    `Intercept.approach` keeps the internal reachability point, which SHOULD
    be offset -- it is what the solver tested against.
    """
    st, pred = scenario(
        me=car(x=0, y=-3000, yaw=math.pi / 2, boost=60), the_ball=ball(x=0, y=0)
    )
    ic = find_intercept(st.me, pred, st.time, ReachCurve(st.me.speed, st.me.boost))
    d = ic.pos.dist(ic.approach)
    assert d > 50.0, f"approach should be offset from the ball, got {d:.0f}uu"


def main() -> int:
    for name, fn in [
        ("intercept reports ball centre", test_intercept_reports_ball_centre),
        ("contact point within touching range", test_contact_point_is_within_touching_range),
        ("contact point on correct side", test_contact_point_is_on_the_correct_side),
        ("offset direction follows aim", test_offset_direction_follows_the_aim),
        ("approach field remains offset", test_approach_field_still_offset),
    ]:
        check(name, fn)

    print(f"{'test':42s} result")
    print("-" * 76)
    failures = 0
    for name, ok, err in RESULTS:
        print(f"{name:42s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            failures += 1
    print("-" * 76)
    print(f"{len(RESULTS) - failures}/{len(RESULTS)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
