"""
Peer coordination protocol tests.

The property that matters: with several Allies in a match, exactly one of them
takes any given ball. Two bots both deferring leaves the ball to nobody, which
is worse than both committing -- so the tie-breaking is tested as carefully as
the ordinary case.

Inbound messages are also treated as untrusted. A peer that crashes mid-write,
or some other bot's traffic that happens to land in our handler, must not be
able to take a match down.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.brain.comms import CommsHub  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def test_intent_round_trip():
    a = CommsHub(my_index=0, my_team=0)
    b = CommsHub(my_index=1, my_team=0)
    msg = a.build_intent("attack", 1.25, True)
    b.receive(0, 0, msg, now=10.0)
    peers = b.teammates(10.0)
    assert len(peers) == 1, f"expected 1 peer, got {len(peers)}"
    p = peers[0]
    assert p.role == "attack" and abs(p.time_to_ball - 1.25) < 1e-6 and p.claiming


def test_better_placed_peer_wins():
    me = CommsHub(my_index=2, my_team=0)
    faster = CommsHub(my_index=1, my_team=0)
    me.receive(1, 0, faster.build_intent("attack", 0.8, True), now=10.0)
    peer = me.better_placed_peer(10.0, my_time=1.6)
    assert peer is not None and peer.index == 1, "should defer to the faster peer"


def test_we_keep_the_ball_when_we_are_faster():
    me = CommsHub(my_index=2, my_team=0)
    slower = CommsHub(my_index=1, my_team=0)
    me.receive(1, 0, slower.build_intent("attack", 2.4, True), now=10.0)
    assert me.better_placed_peer(10.0, my_time=0.9) is None, "should not defer"


def test_exact_tie_resolves_to_one_taker():
    """
    Two bots with identical estimates. Exactly one must defer -- if both do,
    nobody takes the ball.
    """
    lo = CommsHub(my_index=1, my_team=0)
    hi = CommsHub(my_index=3, my_team=0)
    t = 1.5
    lo.receive(3, 0, hi.build_intent("attack", t, True), now=10.0)
    hi.receive(1, 0, lo.build_intent("attack", t, True), now=10.0)

    lo_defers = lo.better_placed_peer(10.0, my_time=t) is not None
    hi_defers = hi.better_placed_peer(10.0, my_time=t) is not None
    assert lo_defers != hi_defers, (
        f"exactly one must defer; lo_defers={lo_defers} hi_defers={hi_defers}"
    )
    assert hi_defers and not lo_defers, "the lower index should take it"


def test_opposing_team_intent_ignored():
    me = CommsHub(my_index=0, my_team=0)
    enemy = CommsHub(my_index=4, my_team=1)
    me.receive(4, 1, enemy.build_intent("attack", 0.2, True), now=10.0)
    assert me.teammates(10.0) == [], "opposition intent must not count as a teammate"
    assert me.better_placed_peer(10.0, my_time=5.0) is None


def test_stale_peers_expire():
    me = CommsHub(my_index=0, my_team=0)
    other = CommsHub(my_index=1, my_team=0)
    me.receive(1, 0, other.build_intent("attack", 0.5, True), now=10.0)
    assert len(me.teammates(10.1)) == 1
    assert me.teammates(20.0) == [], "a peer gone quiet must expire, not linger"


def test_malformed_messages_are_survived():
    me = CommsHub(my_index=0, my_team=0)
    for junk in (
        b"", b"not json", b"{", b"[]", b"null",
        b'{"m":"other","v":1,"t":"intent"}',            # someone else's bot
        b'{"m":"ally","v":999,"t":"intent"}',           # future protocol
        b'{"m":"ally","v":1,"t":"intent","tt":"NaN-ish"}',
        b'{"m":"ally","v":1,"t":"calib","n":"lots"}',
        "\udcff".encode("utf-8", "surrogateescape"),    # invalid utf-8
    ):
        me.receive(1, 0, junk, now=10.0)  # must not raise
    assert me.teammates(10.0) == [], "no junk should have registered as a peer"


def test_calibration_pool_round_trip():
    from bot.learn.calibrate import StrikeCalibrator

    a = CommsHub(my_index=0, my_team=0)
    b = CommsHub(my_index=1, my_team=1)  # cross-team pooling is intended
    cal = StrikeCalibrator(persist=False)
    cal.c.aim_bias = -0.12
    cal.c.contacts = 30

    b.receive(0, 0, a.build_calib(cal.c), now=10.0)
    pending = b.drain_calibration()
    assert len(pending) == 1, f"expected 1 report, got {len(pending)}"
    assert abs(pending[0]["aim"] + 0.12) < 1e-3
    assert b.drain_calibration() == [], "draining twice must not repeat reports"


def test_pool_is_bounded():
    """A chatty peer must not grow the buffer without limit."""
    from bot.learn.calibrate import StrikeCalibrator

    hub = CommsHub(my_index=0, my_team=0)
    other = CommsHub(my_index=1, my_team=0)
    cal = StrikeCalibrator(persist=False)
    cal.c.contacts = 5
    for _ in range(200):
        hub.receive(1, 0, other.build_calib(cal.c), now=10.0)
    assert len(hub.calib_pool) <= 16, f"pool grew to {len(hub.calib_pool)}"


def main() -> int:
    for name, fn in [
        ("intent round trip", test_intent_round_trip),
        ("defers to a faster peer", test_better_placed_peer_wins),
        ("keeps the ball when faster", test_we_keep_the_ball_when_we_are_faster),
        ("exact tie -> exactly one taker", test_exact_tie_resolves_to_one_taker),
        ("ignores opposition intent", test_opposing_team_intent_ignored),
        ("stale peers expire", test_stale_peers_expire),
        ("survives malformed messages", test_malformed_messages_are_survived),
        ("calibration pool round trip", test_calibration_pool_round_trip),
        ("calibration pool is bounded", test_pool_is_bounded),
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
