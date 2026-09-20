"""
Drill geometry checks.

The exposed-recovery drill only teaches the right thing if it actually builds
the state the coach measured: blue beaten and upfield, orange on the ball, in
blue's half. Getting a sign wrong would silently train the opposite habit,
and that is not visible from watching one reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.trainer import _exposed_placement, build_drills  # noqa: E402

OWN_NET_Y = -5120.0          # blue defends -y


def test_exposed_drill_exists_and_is_weighted():
    names = [d.name for d in build_drills()]
    assert names.count("exposed_recover") == 3, \
        "the measured state should come up more often than a novelty"


def test_ball_starts_in_blue_half_moving_home():
    d = next(x for x in build_drills() if x.name == "exposed_recover")
    assert d.ball_pos[1] < 0, "ball must be in blue's half"
    assert d.ball_vel[1] < 0, "ball must be travelling toward blue's net"


def test_blue_first_man_starts_beaten():
    d = next(x for x in build_drills() if x.name == "exposed_recover")
    bx, by = d.ball_pos[0], d.ball_pos[1]
    x, y = _exposed_placement(0, 0, d.ball_pos)
    assert y > by, "trainee must start AHEAD of the ball, not goal-side"
    assert abs(y - OWN_NET_Y) > abs(by - OWN_NET_Y), \
        "trainee must be further from their own net than the ball is"


def test_blue_has_someone_goal_side():
    """Beaten, not abandoned -- otherwise it is a 1v3, not a recovery drill."""
    d = next(x for x in build_drills() if x.name == "exposed_recover")
    by = d.ball_pos[1]
    behind = [s for s in (1, 2) if _exposed_placement(0, s, d.ball_pos)[1] < by]
    assert behind, "at least one blue car should be goal-side of the ball"


def test_orange_is_on_the_ball():
    d = next(x for x in build_drills() if x.name == "exposed_recover")
    bx, by = d.ball_pos[0], d.ball_pos[1]
    x, y = _exposed_placement(1, 0, d.ball_pos)
    assert ((x - bx) ** 2 + (y - by) ** 2) ** 0.5 < 900.0, \
        "the opponent must start on the ball -- they have possession"


def test_placements_are_inside_the_field():
    d = next(x for x in build_drills() if x.name == "exposed_recover")
    for team in (0, 1):
        for slot in range(3):
            x, y = _exposed_placement(team, slot, d.ball_pos)
            assert -4096 < x < 4096, "car spawned outside the side wall"
            assert -5120 < y < 5120, "car spawned outside the back wall"


def demo():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
    print("%d/%d passed" % (n, n))


if __name__ == "__main__":
    demo()
