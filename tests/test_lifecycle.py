"""
Bot lifecycle tests.

The brain was covered from the start, but the `Bot` subclass wrapping it was
not -- and that is where the first real in-game failure lived. `initialize()`
read `PlayerConfiguration.name`, which does not exist: a Human entry carries
only a team. RLBot catches exceptions from `initialize()`, logs them and exits
the process, so the symptom was simply "the bot never joins the match" with
nothing obviously wrong in any unit test.

These tests drive the real `Ally` class through its real lifecycle against a
real parsed match config, with the socket never connected. Anything that
throws in initialize() or on the first ticks fails here instead of silently
in-game.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlbot import flat  # noqa: E402
from rlbot.config import load_match_config  # noqa: E402

from bot.ally import Ally  # noqa: E402
from tests.harness import FIELD_INFO, ball, car, make_prediction, packet  # noqa: E402

HUMAN_NAME = "MrTactical"


def build_bot(match_file="2v2-vs-allstar.toml", index=1, team=0) -> Ally:
    """
    An Ally wired up as RLBot would wire it, minus the socket.

    Constructing Bot only creates a SocketRelay; nothing connects until run(),
    so we can exercise the whole tick path offline.
    """
    bot = Ally()
    # Keep side effects out of the test run.
    bot.cfg = dict(bot.cfg)
    bot.cfg.update(rendering=False, callouts=False, telemetry=False, learning=False)

    bot.match_config = load_match_config(ROOT / "matches" / match_file)
    bot.field_info = FIELD_INFO
    bot.index = index
    bot.team = team
    bot.name = "Ally"
    return bot


def make_packet(t=10.0, human_name=HUMAN_NAME):
    """Human at index 0, Ally at index 1, two opponents."""
    cars = [
        car(x=900, y=-2400, yaw=math.pi / 2, boost=48, name=human_name, is_bot=False, team=0),
        car(x=-600, y=-3400, yaw=math.pi / 2, boost=62, name="Ally", is_bot=True, team=0),
        car(x=-400, y=2600, yaw=-math.pi / 2, boost=50, name="Opp1", is_bot=True, team=1),
        car(x=700, y=3300, yaw=-math.pi / 2, boost=50, name="Opp2", is_bot=True, team=1),
    ]
    return packet(cars, ball(x=100, y=-300, vx=200, vy=-450), time=t)


RESULTS = []


def check(name: str, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except Exception as e:
        import traceback

        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


def test_initialize_does_not_throw():
    """The original bug: any exception here and RLBot exits the process."""
    bot = build_bot()
    bot.initialize()
    assert bot.brain is not None, "brain not built"
    assert bot.model is not None, "model not built"


def test_first_tick_returns_controls():
    bot = build_bot()
    bot.initialize()
    pkt = make_packet()
    bot.ball_prediction = make_prediction(pkt.balls[0], 10.0)
    out = bot.get_output(pkt)
    assert isinstance(out, flat.ControllerState), f"got {type(out)}"


def test_partner_name_adopted_from_packet():
    """The human's name only exists in the packet, never in the config."""
    bot = build_bot()
    bot.initialize()
    assert bot.model.name == "partner", f"expected placeholder, got {bot.model.name}"
    pkt = make_packet()
    bot.ball_prediction = make_prediction(pkt.balls[0], 10.0)
    bot.get_output(pkt)
    assert bot.model.name == HUMAN_NAME, f"partner not adopted: {bot.model.name}"


def test_many_ticks_stable():
    """Run a few seconds of ticks; nothing should throw and time must advance."""
    bot = build_bot()
    bot.initialize()
    dt = 1.0 / 120.0
    for i in range(240):
        t = 10.0 + i * dt
        pkt = make_packet(t)
        bot.ball_prediction = make_prediction(pkt.balls[0], t)
        out = bot.get_output(pkt)
        assert -1.0 <= out.steer <= 1.0, f"steer out of range at tick {i}: {out.steer}"
        assert -1.0 <= out.throttle <= 1.0, f"throttle out of range at tick {i}"
    assert bot.model.traits.samples > 100, f"only {bot.model.traits.samples} samples"


def test_survives_kickoff_and_inactive_phases():
    bot = build_bot()
    bot.initialize()
    for phase in (
        flat.MatchPhase.Inactive,
        flat.MatchPhase.Countdown,
        flat.MatchPhase.Kickoff,
        flat.MatchPhase.Active,
        flat.MatchPhase.GoalScored,
        flat.MatchPhase.Replay,
        flat.MatchPhase.Paused,
    ):
        cars = [
            car(x=0, y=-2400, yaw=math.pi / 2, name=HUMAN_NAME, is_bot=False, team=0),
            car(x=0, y=-3400, yaw=math.pi / 2, name="Ally", is_bot=True, team=0),
        ]
        pkt = packet(cars, ball(), time=20.0, phase=phase)
        bot.ball_prediction = make_prediction(pkt.balls[0], 20.0)
        out = bot.get_output(pkt)
        assert isinstance(out, flat.ControllerState), f"phase {phase} returned {type(out)}"


def test_handles_missing_teammate():
    """1v1: state.ally is None. Nothing may assume a partner exists."""
    bot = build_bot(match_file="1v1-vs-ally.toml", index=0, team=1)
    bot.initialize()
    cars = [
        car(x=0, y=-2400, yaw=math.pi / 2, name="Ally", is_bot=True, team=1),
        car(x=0, y=2400, yaw=-math.pi / 2, name=HUMAN_NAME, is_bot=False, team=0),
    ]
    pkt = packet(cars, ball(), time=10.0)
    bot.ball_prediction = make_prediction(pkt.balls[0], 10.0)
    out = bot.get_output(pkt)
    assert isinstance(out, flat.ControllerState)


def test_handles_demolished_and_empty_ball():
    bot = build_bot()
    bot.initialize()
    cars = [
        car(x=0, y=-2400, name=HUMAN_NAME, is_bot=False, team=0),
        car(x=0, y=-3400, name="Ally", is_bot=True, team=0, demolished=True),
    ]
    pkt = packet(cars, ball(), time=10.0)
    bot.ball_prediction = flat.BallPrediction([])  # no prediction yet
    out = bot.get_output(pkt)
    assert isinstance(out, flat.ControllerState)


def main() -> int:
    tests = [
        ("initialize() does not throw", test_initialize_does_not_throw),
        ("first tick returns controls", test_first_tick_returns_controls),
        ("partner name adopted from packet", test_partner_name_adopted_from_packet),
        ("240 ticks stable, inputs in range", test_many_ticks_stable),
        ("survives every match phase", test_survives_kickoff_and_inactive_phases),
        ("handles missing teammate (1v1)", test_handles_missing_teammate),
        ("handles demolition + no prediction", test_handles_demolished_and_empty_ball),
    ]
    for name, fn in tests:
        check(name, fn)

    print()
    print(f"{'test':44s} result")
    print("-" * 78)
    failures = 0
    for name, ok, err in RESULTS:
        print(f"{name:44s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            failures += 1
    print("-" * 78)
    print(f"{len(RESULTS) - failures}/{len(RESULTS)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
