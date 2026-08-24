"""
Kickoff deconfliction.

Two cars taking the same kickoff loses the game on the spot, and two cars both
hanging back hands the opponent a free shot. Exactly one must go.

The bug this guards: `should_take_kickoff` compared against `state.ally`, a
single teammate, so on a three-a-side kickoff the third car was invisible and
two bots each concluded they were the taker. Measured in a real match, two of
three cars per team reported "kickoff/go" on every single kickoff.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlbot import flat  # noqa: E402

from bot.brain.kickoff import should_take_kickoff  # noqa: E402
from bot.brain.teammate import TeammateModel  # noqa: E402
from bot.core.game import GameState  # noqa: E402
from tests.harness import FIELD_INFO, ball, car, packet  # noqa: E402

# Real kickoff spawns for the -Y side.
CORNER_L = (2048, -2560)
CORNER_R = (-2048, -2560)
BACK_L = (256, -3840)
BACK_R = (-256, -3840)
BACK_C = (0, -4608)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def takers(spawns, human_index=None, bias=0.5):
    """How many of our cars decide to take the kickoff."""
    model = TeammateModel("_k")
    model.traits.kickoff_commit = 1.0 - bias
    cars = [
        car(x=x, y=y, yaw=math.pi / 2, boost=34, name=f"A{i}",
            is_bot=(human_index != i))
        for i, (x, y) in enumerate(spawns)
    ]
    # Mirror opponents so the packet is realistic.
    cars += [car(x=-x, y=-y, yaw=-math.pi / 2, team=1, name=f"O{i}")
             for i, (x, y) in enumerate(spawns)]
    pkt = packet(cars, ball(x=0, y=0, z=93), time=10.0, phase=flat.MatchPhase.Kickoff)

    out = []
    for i in range(len(spawns)):
        st = GameState(pkt, i, 0, FIELD_INFO, 10.0 - 1 / 120)
        if should_take_kickoff(st, model):
            out.append(i)
    return out


def test_three_a_side_has_exactly_one_taker():
    t = takers([CORNER_R, CORNER_L, BACK_C])
    assert len(t) == 1, f"expected exactly one taker, got {len(t)}: {t}"


def test_two_a_side_has_exactly_one_taker():
    t = takers([CORNER_R, CORNER_L])
    assert len(t) == 1, f"expected exactly one taker, got {len(t)}: {t}"


def test_nearest_car_takes_it_when_the_gap_is_clear():
    # Back-centre is much further from the ball than a corner.
    t = takers([CORNER_R, BACK_C])
    assert t == [0], f"the closer car should take it, got {t}"


def test_mirrored_spawns_resolve_deterministically():
    """Two equidistant cars must not both go, and must not both defer."""
    for _ in range(5):
        t = takers([CORNER_R, CORNER_L])
        assert len(t) == 1, f"mirrored spawns gave {len(t)} takers"


def test_alone_always_takes_it():
    t = takers([CORNER_R])
    assert t == [0], "a lone car must take its own kickoff"


def test_full_three_a_side_every_spawn_combination():
    """Whatever three spawns the game hands out, exactly one car goes."""
    combos = [
        [CORNER_R, CORNER_L, BACK_C],
        [CORNER_R, BACK_L, BACK_C],
        [BACK_L, BACK_R, BACK_C],
        [CORNER_L, BACK_R, BACK_C],
        [CORNER_R, CORNER_L, BACK_L],
    ]
    for spawns in combos:
        t = takers(spawns)
        assert len(t) == 1, f"spawns {spawns} gave {len(t)} takers: {t}"


def test_human_teammate_is_deferred_to_when_they_usually_go():
    """A human who always takes kickoffs should be left to it."""
    # bias high -> take_kickoff_bias low -> we defer.
    t = takers([CORNER_R, CORNER_L], human_index=1, bias=0.9)
    assert 0 not in t or len(t) == 1, f"got {t}"


def main() -> int:
    for name, fn in [
        ("3v3: exactly one taker", test_three_a_side_has_exactly_one_taker),
        ("2v2: exactly one taker", test_two_a_side_has_exactly_one_taker),
        ("nearest car takes it", test_nearest_car_takes_it_when_the_gap_is_clear),
        ("mirrored spawns resolve", test_mirrored_spawns_resolve_deterministically),
        ("alone always takes it", test_alone_always_takes_it),
        ("every spawn combination", test_full_three_a_side_every_spawn_combination),
        ("defers to a human taker", test_human_teammate_is_deferred_to_when_they_usually_go),
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
