"""
Challenge / shadow ladder.

The gap this closes: the bot drove at its own intercept regardless of whether
an opponent would get there first, so on any ball it was losing it arrived late
and hovered beside the ball doing nothing.

The invariant from the prior art is blunt and worth stating: **nobody ever
hovers next to the ball**. You are either striking it, contesting it at speed,
or standing goal-side of it. There is no fourth option.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.brain.challenge import (  # noqa: E402
    CHALLENGE,
    CONTEST_WINDOW,
    SHADOW,
    SHADOW_MIN_DISTANCE,
    STRIKE,
    ContestState,
)
from bot.core.vec import Vec3  # noqa: E402
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


def contest(their_time, contact=None, opp_pos=None):
    """A ContestState primed with a known opponent solve."""
    st, _ = scenario(
        me=car(x=0, y=-2000, yaw=math.pi / 2, boost=60),
        opponents=[car(x=opp_pos[0], y=opp_pos[1], team=1) if opp_pos
                   else car(x=0, y=2000, team=1)],
        the_ball=ball(x=0, y=0),
    )
    c = ContestState()
    c.their_time = their_time
    c.their_contact = contact or Vec3(0, 0, 93)
    c._threat_car = st.opponents[0]
    return st, c


def test_winning_the_ball_strikes():
    st, c = contest(their_time=2.0)
    assert c.decide(st, my_time=1.0, is_last_man=False) == STRIKE


def test_narrow_loss_is_a_challenge():
    """Inside the contest window it is a genuine 50/50 -- go."""
    st, c = contest(their_time=1.0)
    stance = c.decide(st, my_time=1.0 + CONTEST_WINDOW * 0.5, is_last_man=True)
    assert stance == CHALLENGE, f"expected a challenge inside the window, got {stance}"


def test_clear_loss_as_last_man_shadows():
    st, c = contest(their_time=1.0)
    stance = c.decide(st, my_time=1.0 + CONTEST_WINDOW * 3, is_last_man=True)
    assert stance == SHADOW, f"last man losing clearly must shadow, got {stance}"


def test_clear_loss_with_cover_may_still_press():
    """Not the last man and with boost: pressing is legitimate."""
    st, c = contest(their_time=1.0)
    st.me.boost = 80.0
    stance = c.decide(st, my_time=1.0 + CONTEST_WINDOW * 3, is_last_man=False)
    assert stance == CHALLENGE, f"expected a press with cover behind, got {stance}"


def test_opponent_already_on_the_ball_is_always_contested():
    """
    Conceding a free touch is worse than losing a 50/50, so an opponent
    effectively on the ball is challenged whatever the arithmetic says.
    """
    st, c = contest(their_time=0.1, contact=Vec3(0, 2050, 93), opp_pos=(0, 2000))
    stance = c.decide(st, my_time=5.0, is_last_man=True)
    assert stance == CHALLENGE, (
        f"an opponent already on the ball must be contested, got {stance}"
    )


def test_shadow_point_is_goal_side():
    st, c = contest(their_time=1.0)
    c.decide(st, my_time=3.0, is_last_man=True)
    p = c.shadow_point(st)
    # goal_sign is -1 for team 0, so goal-side means a more negative y.
    assert p.y * st.goal_sign > c.their_contact.y * st.goal_sign, (
        f"shadow point {p} is not between the contact and our goal"
    )
    d = p.flat_dist(c.their_contact)
    assert d >= SHADOW_MIN_DISTANCE - 1, f"shadow only {d:.0f}uu off the contact"


def test_shadow_recedes_as_we_lose_by_more():
    st, c = contest(their_time=1.0)
    c.decide(st, my_time=1.4, is_last_man=True)
    near = c.shadow_point(st).flat_dist(c.their_contact)
    c.decide(st, my_time=3.5, is_last_man=True)
    far = c.shadow_point(st).flat_dist(c.their_contact)
    assert far > near, f"expected to drop deeper when losing by more: {near:.0f} -> {far:.0f}"


def test_shadow_point_stays_in_the_arena():
    from bot.core.constants import BACK_WALL_Y, SIDE_WALL_X

    for cx, cy in ((-4000, -4800), (4000, 4800), (0, 5000)):
        st, c = contest(their_time=1.0, contact=Vec3(cx, cy, 93))
        c.decide(st, my_time=4.0, is_last_man=True)
        p = c.shadow_point(st)
        assert abs(p.x) < SIDE_WALL_X, f"shadow x={p.x:.0f} outside the arena"
        assert abs(p.y) < BACK_WALL_Y, f"shadow y={p.y:.0f} outside the arena"


def test_shadow_speed_arrives_on_time_not_eventually():
    """
    Shadowing must not be a stroll. The speed asked for has to put us at the
    point by the time the opponent touches the ball.
    """
    st, c = contest(their_time=1.5)
    c.decide(st, my_time=4.0, is_last_man=True)
    p = c.shadow_point(st)
    v = c.shadow_speed(st, p)
    dist = st.me.pos.flat_dist(p)
    assert v >= 550.0, f"shadow speed {v:.0f} is a crawl"
    # Should be in the right ballpark for covering the distance in the window.
    assert v >= min(2300.0, dist / 1.5) - 1.0, (
        f"speed {v:.0f} will not cover {dist:.0f}uu in 1.5s"
    )


def test_no_opponent_means_just_strike():
    st, _ = scenario(me=car(x=0, y=-2000, yaw=math.pi / 2, boost=60), the_ball=ball())
    c = ContestState()
    assert c.decide(st, my_time=1.0, is_last_man=True) == STRIKE


def main() -> int:
    for name, fn in [
        ("winning the ball -> strike", test_winning_the_ball_strikes),
        ("narrow loss -> challenge", test_narrow_loss_is_a_challenge),
        ("clear loss, last man -> shadow", test_clear_loss_as_last_man_shadows),
        ("clear loss with cover -> press", test_clear_loss_with_cover_may_still_press),
        ("opponent on the ball -> contest", test_opponent_already_on_the_ball_is_always_contested),
        ("shadow point is goal-side", test_shadow_point_is_goal_side),
        ("shadow recedes when losing more", test_shadow_recedes_as_we_lose_by_more),
        ("shadow stays in the arena", test_shadow_point_stays_in_the_arena),
        ("shadow arrives on time", test_shadow_speed_arrives_on_time_not_eventually),
        ("no opponent -> strike", test_no_opponent_means_just_strike),
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
