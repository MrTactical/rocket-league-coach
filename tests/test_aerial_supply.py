"""
Aerial supply: the search must offer aerials the car can actually fly.

Guards the two defects found by the commitment audit (2026-08-23):

  * The aerial branch priced the WHOLE displacement as powered flight, because
    unlike GROUND/DODGE/DOUBLE_JUMP it never subtracted a drive segment. Median
    priced cost was 47.4 boost and barely distance-sensitive, which destroyed
    69.5% of every kinematically flyable aerial.

  * The resulting envelope had a hard blind spot off-axis: a car at 1200 uu/s
    angled away from the ball line was called infeasible at EVERY window from
    0.5s to 4.0s, where a drive-then-fly model succeeds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.core.intercept import DRIVE_FRACTIONS, _aerial_after_drive  # noqa: E402
from bot.core.physics import ReachCurve, aerial_feasible  # noqa: E402
from bot.core.vec import Mat3, Vec3  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, "%s: %s" % (type(e).__name__, e)))


class FakeCar:
    def __init__(self, pos, vel, boost):
        self.pos, self.vel, self.boost = pos, vel, boost
        f = vel.normalized() if vel.length() > 1.0 else Vec3(1.0, 0.0, 0.0)
        up = Vec3(0.0, 0.0, 1.0)
        self.ori = Mat3(f, up.cross(f), up)
        self.on_ground = True

    @property
    def speed(self):
        return self.vel.length()


def solve(car, target, dt):
    curve = ReachCurve(car.speed, car.boost, allow_boost=False)
    return _aerial_after_drive(car, curve, target, dt)


def envelope():
    """A realistic sweep of car states and aerial targets."""
    for dist in (500, 1000, 2000, 3000, 4000):
        for z in (500, 700, 900, 1200, 1500, 1800):
            for dt in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                for boost in (25, 40, 60, 80, 100):
                    for spd in (0, 700, 1400):
                        for ang in (0, 45, 90, 150):
                            a = math.radians(ang)
                            yield (
                                FakeCar(Vec3(0.0, 0.0, 17.0),
                                        Vec3(math.cos(a), math.sin(a), 0.0) * spd,
                                        float(boost)),
                                Vec3(float(dist), 0.0, float(z)),
                                dt,
                            )


def test_drive_then_fly_is_a_strict_superset():
    """
    THE load-bearing invariant. DRIVE_FRACTIONS includes 1.0, which is "fly the
    whole way from here" -- exactly the old model. So anything the old test
    accepted the new one must still accept. If this ever fails, the change has
    started losing aerials rather than only adding them.
    """
    assert 1.0 in DRIVE_FRACTIONS, "the fly-the-whole-way case was removed"
    lost = []
    for car, target, dt in envelope():
        if aerial_feasible(car, target, dt)[0] and not solve(car, target, dt)[0]:
            lost.append((car.vel, target, dt))
    assert not lost, "%d cases the old model accepted are now rejected: %r" % (
        len(lost), lost[:3])


def test_drive_then_fly_finds_more():
    old = new = tot = 0
    for car, target, dt in envelope():
        tot += 1
        old += 1 if aerial_feasible(car, target, dt)[0] else 0
        new += 1 if solve(car, target, dt)[0] else 0
    assert new > old * 1.3, (
        "expected a substantial gain, got %d -> %d over %d cells" % (old, new, tot))


def test_offaxis_blind_spot_is_gone():
    """
    The audit's headline case: ball at z=900, 2000uu away, 60 boost, car at
    1200 uu/s angled off the ball line. Previously infeasible at every window.
    """
    target = Vec3(2000.0, 0.0, 900.0)
    windows = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    for ang in (45, 150):
        a = math.radians(ang)
        vel = Vec3(math.cos(a), math.sin(a), 0.0) * 1200.0
        ok = any(solve(FakeCar(Vec3(0.0, 0.0, 17.0), vel, 60.0), target, dt)[0]
                 for dt in windows)
        assert ok, "still infeasible at every window, %d degrees off axis" % ang


def test_cost_falls_with_a_drive_phase():
    """
    The mechanism, not just the outcome: a far target should get cheaper once
    the wheels do the horizontal work.
    """
    car = FakeCar(Vec3(0.0, 0.0, 17.0), Vec3(1200.0, 0.0, 0.0), 100.0)
    target = Vec3(3000.0, 0.0, 900.0)
    dt = 3.0
    old_ok, old_cost = aerial_feasible(car, target, dt)
    new_ok, new_cost = solve(car, target, dt)
    assert new_ok, "a 3s window to a 3000uu/900z ball should be flyable"
    if old_ok:
        assert new_cost <= old_cost + 1e-6, (
            "drive-then-fly should not cost more: %.1f -> %.1f" % (old_cost, new_cost))


def test_no_boost_ground_phase_keeps_the_tank():
    """
    The ground phase must not spend boost -- the whole point is arriving under
    the ball with a full tank. A no-boost curve caps at 1410 uu/s.
    """
    curve = ReachCurve(0.0, 100.0, allow_boost=False)
    assert curve.speed_at(4.0) <= 1415.0, (
        "the ground phase is boosting: reached %.0f uu/s" % curve.speed_at(4.0))


def main() -> int:
    for name, fn in [
        ("drive-then-fly is a superset", test_drive_then_fly_is_a_strict_superset),
        ("drive-then-fly finds more", test_drive_then_fly_finds_more),
        ("off-axis blind spot is gone", test_offaxis_blind_spot_is_gone),
        ("a drive phase lowers the cost", test_cost_falls_with_a_drive_phase),
        ("ground phase keeps the tank", test_no_boost_ground_phase_keeps_the_tank),
    ]:
        check(name, fn)

    print("%-40s result" % "test")
    print("-" * 76)
    fails = 0
    for name, ok, err in RESULTS:
        print("%-40s %s" % (name, "PASS" if ok else "FAIL  " + err))
        if not ok:
            fails += 1
    print("-" * 76)
    print("%d/%d passed" % (len(RESULTS) - fails, len(RESULTS)))
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
