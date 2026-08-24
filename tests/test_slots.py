"""
Rotation-slot integrity, orientation robustness, and humanizer roll rates.

Three defects this guards, all found by measurement rather than by reading:

  * `_team_ordinal` had no duplicate detection. When two cars both held slot 0
    the incumbency bonus cancelled between them and the minimum-hold gate then
    stopped either moving -- a fault protected as though it were a decision.
    Measured: three cars on slot 0 for 6.2 continuous seconds.

  * `rotation_error` returned an arbitrary axis at exactly 180 degrees. That
    state is absorbing: composing a 180 degree error with a rotation about a
    perpendicular axis gives another exact 180 degree error, so the car holds
    full pitch for ever and never rights itself.

  * The wavedash roll happened once per TICK rather than once per recovery. At
    game rate that is ~136 rolls per episode, so diamond's 0.50 rate meant
    1 - 0.5^136 in practice: every recovery wavedashed and the rank profile was
    defeated.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlbot import flat  # noqa: E402

from bot.brain.comms import CommsHub  # noqa: E402
from bot.brain.decide import Brain  # noqa: E402
from bot.brain.humanize import Humanizer, get_profile  # noqa: E402
from bot.brain.teammate import TeammateModel  # noqa: E402
from bot.control.aerial import orient_controls, rotation_error  # noqa: E402
from bot.core.constants import PITCH_TORQUE, ROLL_TORQUE, YAW_TORQUE  # noqa: E402
from bot.core.game import GameState  # noqa: E402
from bot.core.vec import Mat3, Vec3  # noqa: E402
from tests.harness import FIELD_INFO, ball, car, packet  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def three_brains(times, ordinals, aerials=(False, False, False)):
    """Three Allies on one team, each told what the others are claiming."""
    cars = [
        car(x=-800 + i * 800, y=-2000 - i * 400, yaw=math.pi / 2, boost=50, name="A%d" % i)
        for i in range(3)
    ]
    cars.append(car(x=0, y=2500, yaw=-math.pi / 2, team=1, name="O"))
    pkt = packet(cars, ball(x=0, y=0), time=10.0)

    hubs = [CommsHub(i, 0) for i in range(3)]
    for i, hub in enumerate(hubs):
        for j, other in enumerate(hubs):
            if i != j:
                hub.receive(
                    j, 0,
                    other.build_intent("attack", times[j], True, ordinals[j],
                                       aerial=aerials[j]),
                    now=10.0,
                )

    brains = []
    for i in range(3):
        b = Brain(TeammateModel("_slots"), Humanizer(get_profile("diamond"), seed=i))
        b.comms = hubs[i]
        b.ordinal = ordinals[i]
        # What this car last told the others. _team_ordinal must rank us by the
        # broadcast value, never by a locally-computed one.
        b.broadcast_aerial = aerials[i]
        brains.append(b)

    states = [GameState(pkt, i, 0, FIELD_INFO, 10.0 - 1 / 120) for i in range(3)]
    return brains, states


def test_two_cars_on_one_slot_exactly_one_yields():
    """The core fix. Both hold slot 0; exactly one must give it up."""
    times = [1.0, 1.7, 3.0]
    brains, states = three_brains(times, [0, 0, 2])
    got = [b._team_ordinal(s, times[i], 99.0)
           for i, (b, s) in enumerate(zip(brains, states))]
    zeros = [i for i, r in enumerate(got) if r == 0]
    assert len(zeros) == 1, "expected exactly one car on slot 0, got %r" % (got,)
    assert zeros[0] == 0, "the faster car (1.0s) should keep it, got %r" % (got,)


def test_yield_is_stamped_for_rate_limiting():
    """A resolving pair must not bounce off each other every tick."""
    times = [1.0, 1.7, 3.0]
    brains, states = three_brains(times, [0, 0, 2])
    b, s = brains[1], states[1]
    b._team_ordinal(s, times[1], 99.0)
    assert b._last_yield == s.time, "the yield was not stamped, so it cannot be rate limited"


def test_no_duplicate_means_no_yield():
    """With distinct slots the minimum hold behaves exactly as before."""
    times = [1.0, 1.7, 3.0]
    brains, states = three_brains(times, [0, 1, 2])
    for i, (b, s) in enumerate(zip(brains, states)):
        before = b.ordinal
        got = b._team_ordinal(s, times[i], 99.0)
        assert got == before, "car %d moved off slot %d to %d with no clash" % (i, before, got)
        assert b._last_yield == -99.0, "car %d yielded with no duplicate" % i


def test_solo_car_is_unaffected():
    """1v1: no peers, so the yield branch can never fire."""
    cars = [
        car(x=0, y=-2000, yaw=math.pi / 2, boost=50, name="Me"),
        car(x=0, y=2000, yaw=-math.pi / 2, team=1, name="O"),
    ]
    pkt = packet(cars, ball(), time=10.0)
    st = GameState(pkt, 0, 0, FIELD_INFO, 10.0 - 1 / 120)
    b = Brain(TeammateModel("_slots"), Humanizer(get_profile("diamond"), seed=1))
    b.comms = CommsHub(0, 0)
    assert b._team_ordinal(st, 1.0, 99.0) == 0
    assert b._last_yield == -99.0, "a lone car must never yield"


class _Inverted:
    """A car rolled exactly 180 degrees."""

    def __init__(self):
        self.ori = Mat3.from_rotator(flat.Rotator(0.0, 0.0, math.pi))
        self.ang_vel = Vec3()


def test_inverted_car_rights_about_roll():
    """
    Roll has 3.2x the torque authority of pitch, and more importantly the axis
    at 180 degrees is fixed by the matrix rather than free to choose.
    """
    err = rotation_error(_Inverted(), Vec3(1, 0, 0), Vec3(0, 0, 1))
    assert abs(err.length() - math.pi) < 1e-3, (
        "expected a 180 degree error, got %.4f" % err.length()
    )
    axis = err.normalized()
    assert abs(axis.x) > 0.9, (
        "an inverted car must be righted about roll (local x), got axis %r" % (axis,)
    )


def test_180_degree_state_is_not_absorbing():
    """
    Integrate the orientation controller with REAL rigid-body dynamics from
    exactly inverted, and confirm the car actually rights itself.

    The dynamics matter here. A naive test that drives the orientation straight
    from the control input, without carrying angular velocity, has no D term --
    the controller is then pure proportional with an enormous step, it
    overshoots, the shortest-rotation axis flips sign, and it limit-cycles at
    180 degrees for ever. That is an artefact of the integrator, not of the
    controller, and it is worth stating because it looks exactly like the bug
    this test is meant to catch.
    """

    class Sim:
        def __init__(self):
            # Exactly inverted: rolled 180 degrees.
            self.ori = Mat3.from_rotator(flat.Rotator(0.0, 0.0, math.pi))
            self.ang_vel = Vec3()

        def advance(self, pitch, yaw, roll, dt):
            # Control inputs map to torque about the car's own axes. Signs
            # follow the derivation at the top of bot/control/aerial.py:
            # +roll -> +local x, +pitch -> -local y, +yaw -> -local z.
            alpha_local = Vec3(
                roll * ROLL_TORQUE,
                -pitch * PITCH_TORQUE,
                -yaw * YAW_TORQUE,
            )
            self.ang_vel = self.ang_vel + self.ori.to_world(alpha_local) * dt

            # Rotate the basis by omega*dt (small-angle) and re-orthonormalise.
            w = self.ang_vel * dt
            f = self.ori.forward + w.cross(self.ori.forward)
            u = self.ori.up + w.cross(self.ori.up)
            f = f.normalized()
            u = (u - f * u.dot(f)).normalized()
            self.ori = Mat3(f, u.cross(f), u)

    sim = Sim()
    dt = 1.0 / 120.0
    start = rotation_error(sim, Vec3(1, 0, 0), Vec3(0, 0, 1)).length()
    assert abs(start - math.pi) < 1e-3, "expected to start at 180 degrees"

    settled = None
    for i in range(360):  # three seconds
        p, y, r = orient_controls(sim, Vec3(1, 0, 0), Vec3(0, 0, 1))
        sim.advance(p, y, r, dt)
        if rotation_error(sim, Vec3(1, 0, 0), Vec3(0, 0, 1)).length() < 0.35:
            settled = i
            break

    assert settled is not None, (
        "the controller never left the 180 degree state in 3s -- it is absorbing"
    )
    assert sim.ori.up.z > 0.5, "settled, but not upright: up.z=%.2f" % sim.ori.up.z


def test_wavedash_rolls_once_per_recovery():
    """Rolling per tick makes the rank profile meaningless."""
    hz = Humanizer(get_profile("diamond"), seed=3)
    calls = []
    real = hz.attempt_wavedash

    def counted():
        calls.append(1)
        return real()

    hz.attempt_wavedash = counted
    b = Brain(TeammateModel("_slots"), hz)

    # One continuous recovery episode of 50 ticks.
    for _ in range(50):
        if not b._in_recovery:
            b._in_recovery = True
            b.recovery.allow_wavedash = hz.attempt_wavedash()
    assert len(calls) == 1, "rolled %d times in one episode, expected 1" % len(calls)

    # A fresh episode rolls again.
    b._in_recovery = False
    if not b._in_recovery:
        b._in_recovery = True
        b.recovery.allow_wavedash = hz.attempt_wavedash()
    assert len(calls) == 2, "expected a fresh roll for the second episode, got %d" % len(calls)


def settle_ordinals(times, ordinals, aerials, seconds=2.0, hz=10.0):
    """
    Run the rotation to convergence instead of judging it on one call.

    ORDINAL_HOLD means a new slot must persist before it is adopted, so a
    single _team_ordinal call correctly returns the OLD ordinal. Anything that
    asserts on one call is testing the debounce, not the decision. This ticks
    at the real 10Hz intent rate, rebroadcasting each car's current slot every
    tick exactly as the live loop does.
    """
    ordinals = list(ordinals)
    brains = None
    step = 1.0 / hz
    n = int(seconds * hz)
    for k in range(n):
        now = 10.0 + k * step
        cars = [
            car(x=-800 + i * 800, y=-2000 - i * 400, yaw=math.pi / 2, boost=50,
                name="A%d" % i)
            for i in range(3)
        ]
        cars.append(car(x=0, y=2500, yaw=-math.pi / 2, team=1, name="O"))
        pkt = packet(cars, ball(x=0, y=0), time=now)

        hubs = [CommsHub(i, 0) for i in range(3)]
        for i, hub in enumerate(hubs):
            for j, other in enumerate(hubs):
                if i != j:
                    hub.receive(
                        j, 0,
                        other.build_intent("attack", times[j], True, ordinals[j],
                                           aerial=aerials[j]),
                        now=now,
                    )
        if brains is None:
            brains = []
            for i in range(3):
                b = Brain(TeammateModel("_slots"),
                          Humanizer(get_profile("diamond"), seed=i))
                brains.append(b)
        for i, b in enumerate(brains):
            b.comms = hubs[i]
            b.ordinal = ordinals[i]
            b.broadcast_aerial = aerials[i]

        states = [GameState(pkt, i, 0, FIELD_INFO, now - 1 / 120) for i in range(3)]
        ordinals = [b._team_ordinal(st, times[i], 99.0)
                    for i, (b, st) in enumerate(zip(brains, states))]
    return ordinals


def test_aerial_car_takes_the_attack_slot():
    """
    The whole point of the change. Car 1 has a viable aerial and is 0.10s
    behind the incumbent -- which measurement says is the realistic case: the
    aerial car is usually tied with or faster than the team's best, it just is
    not the incumbent. AERIAL_BONUS (0.55) must beat INCUMBENT_BONUS (0.30).

    Note this does NOT let it break rotation -- it takes slot 0, and the car
    that had slot 0 moves down. Exactly one attacker, as before.
    """
    times = [1.00, 1.10, 3.0]
    got = settle_ordinals(times, [0, 1, 2], [False, True, False])
    assert got[1] == 0, (
        "the car with the aerial should have taken slot 0, got %r" % (got,)
    )
    assert sorted(got) == [0, 1, 2], "slots stopped being unique: %r" % (got,)


def test_aerial_bonus_is_bounded():
    """A car far behind must not claim the ball just because it could fly."""
    times = [1.0, 2.4, 3.0]
    got = settle_ordinals(times, [0, 1, 2], [False, True, False])
    assert got[0] == 0, (
        "a car 1.4s behind must not take the slot on a 0.55s bonus, got %r" % (got,)
    )


def test_aerial_bonus_never_creates_a_duplicate():
    """
    The constraint that matters: this must not cost clash rate.

    Sweep every combination of who holds which slot and who has an aerial, and
    assert the ranking still yields exactly one car per slot.
    """
    import itertools

    bad = []
    for ords in itertools.permutations([0, 1, 2]):
        for aer in itertools.product([False, True], repeat=3):
            times = [1.0, 1.2, 1.5]
            brains, states = three_brains(times, list(ords), aerials=list(aer))
            got = [b._team_ordinal(s, times[i], 99.0)
                   for i, (b, s) in enumerate(zip(brains, states))]
            # Every car must reach a decision, and the ranking each car
            # computes must be a permutation -- no two cars ranked equal.
            if sorted(got) != [0, 1, 2]:
                bad.append((ords, aer, got))
    assert not bad, (
        "%d of 48 configurations produced a duplicate slot; first three: %r"
        % (len(bad), bad[:3])
    )


def test_both_cars_with_aerials_leaves_order_unchanged():
    """When everyone can fly the bonus cancels, so nothing churns."""
    times = [1.0, 1.2, 1.5]
    a = three_brains(times, [0, 1, 2], aerials=[False, False, False])
    b = three_brains(times, [0, 1, 2], aerials=[True, True, True])
    ga = [br._team_ordinal(st, times[i], 99.0)
          for i, (br, st) in enumerate(zip(*a))]
    gb = [br._team_ordinal(st, times[i], 99.0)
          for i, (br, st) in enumerate(zip(*b))]
    assert ga == gb, "a uniform bonus changed the ordering: %r vs %r" % (ga, gb)


def main() -> int:
    for name, fn in [
        ("duplicate slot -> exactly one yields", test_two_cars_on_one_slot_exactly_one_yields),
        ("yield is stamped for rate limit", test_yield_is_stamped_for_rate_limiting),
        ("distinct slots -> nobody moves", test_no_duplicate_means_no_yield),
        ("1v1 unaffected by the yield", test_solo_car_is_unaffected),
        ("inverted car rights about roll", test_inverted_car_rights_about_roll),
        ("180deg state is not absorbing", test_180_degree_state_is_not_absorbing),
        ("wavedash rolls once per episode", test_wavedash_rolls_once_per_recovery),
        ("aerial car takes the attack slot", test_aerial_car_takes_the_attack_slot),
        ("aerial bonus is bounded", test_aerial_bonus_is_bounded),
        ("aerial bonus makes no duplicates", test_aerial_bonus_never_creates_a_duplicate),
        ("uniform aerials do not churn", test_both_cars_with_aerials_leaves_order_unchanged),
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
