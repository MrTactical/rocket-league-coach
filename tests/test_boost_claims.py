"""
Boost pad claim tests.

Telemetry showed teammates converging on the same pad on 7% of ticks. Distance
based courtesy could not fix that on its own: two bots at similar range each
conclude they are the closer one, and both go.

The invariant here is the same shape as the ball-claim one -- for any pad,
exactly one teammate should end up taking it. Both deferring is as bad as both
going, because then nobody collects and both arrive at the next fight empty.
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


def link(*hubs):
    """Every hub hears every other hub's current intent."""
    msgs = {}
    for h in hubs:
        msgs[h.my_index] = h
    return msgs


def test_claim_round_trip():
    a = CommsHub(0, 0)
    b = CommsHub(1, 0)
    b.receive(0, 0, a.build_intent("support", 3.0, False, 1, boost_pad=14,
                                   boost_eta=0.9), now=10.0)
    claims = b.pad_claims(10.0)
    assert 14 in claims, f"claim not registered: {claims}"
    assert claims[14][0] == 0 and abs(claims[14][1] - 0.9) < 1e-6


def test_closer_teammate_takes_the_pad():
    me = CommsHub(2, 0)
    fast = CommsHub(1, 0)
    me.receive(1, 0, fast.build_intent("support", 3.0, False, 1, boost_pad=7,
                                       boost_eta=0.5), now=10.0)
    assert me.pad_is_taken(10.0, 7, my_eta=1.4), "should yield to the closer teammate"


def test_we_keep_the_pad_when_closer():
    me = CommsHub(2, 0)
    slow = CommsHub(1, 0)
    me.receive(1, 0, slow.build_intent("support", 3.0, False, 1, boost_pad=7,
                                       boost_eta=2.2), now=10.0)
    assert not me.pad_is_taken(10.0, 7, my_eta=0.6), "should not yield when closer"


def test_exact_tie_yields_exactly_one():
    lo, hi = CommsHub(1, 0), CommsHub(3, 0)
    eta = 1.0
    lo.receive(3, 0, hi.build_intent("support", 3.0, False, 1, boost_pad=5,
                                     boost_eta=eta), now=10.0)
    hi.receive(1, 0, lo.build_intent("support", 3.0, False, 1, boost_pad=5,
                                     boost_eta=eta), now=10.0)
    lo_yields = lo.pad_is_taken(10.0, 5, eta)
    hi_yields = hi.pad_is_taken(10.0, 5, eta)
    assert lo_yields != hi_yields, (
        f"exactly one must yield on a tie; lo={lo_yields} hi={hi_yields}"
    )
    assert hi_yields and not lo_yields, "the lower car index should take it"


def test_unclaimed_pads_are_free():
    me = CommsHub(0, 0)
    other = CommsHub(1, 0)
    me.receive(1, 0, other.build_intent("support", 3.0, False, 1, boost_pad=14,
                                        boost_eta=0.5), now=10.0)
    assert not me.pad_is_taken(10.0, 99, my_eta=5.0), "unrelated pad reported taken"


def test_not_going_for_boost_claims_nothing():
    me = CommsHub(0, 0)
    other = CommsHub(1, 0)
    me.receive(1, 0, other.build_intent("attack", 1.0, True, 0), now=10.0)
    assert me.pad_claims(10.0) == {}, "a bot not detouring must claim no pad"


def test_claims_expire_with_the_peer():
    me = CommsHub(0, 0)
    other = CommsHub(1, 0)
    me.receive(1, 0, other.build_intent("support", 3.0, False, 1, boost_pad=3,
                                        boost_eta=0.4), now=10.0)
    assert me.pad_claims(10.1), "claim should be live immediately"
    assert me.pad_claims(20.0) == {}, "a stale peer must not hold a pad forever"


def test_strongest_claim_wins_when_two_peers_want_one_pad():
    me = CommsHub(0, 0)
    p1, p2 = CommsHub(1, 0), CommsHub(2, 0)
    me.receive(1, 0, p1.build_intent("support", 3.0, False, 1, boost_pad=9,
                                     boost_eta=1.8), now=10.0)
    me.receive(2, 0, p2.build_intent("support", 3.0, False, 2, boost_pad=9,
                                     boost_eta=0.7), now=10.0)
    holder, eta = me.pad_claims(10.0)[9]
    assert holder == 2 and abs(eta - 0.7) < 1e-6, f"wrong holder: {holder}, {eta}"


def test_opposition_claims_ignored():
    me = CommsHub(0, 0)
    enemy = CommsHub(4, 1)
    me.receive(4, 1, enemy.build_intent("support", 3.0, False, 1, boost_pad=14,
                                        boost_eta=0.1), now=10.0)
    assert me.pad_claims(10.0) == {}, "we do not yield pads to the opposition"


def main() -> int:
    for name, fn in [
        ("claim round trip", test_claim_round_trip),
        ("yields to a closer teammate", test_closer_teammate_takes_the_pad),
        ("keeps the pad when closer", test_we_keep_the_pad_when_closer),
        ("exact tie -> exactly one yields", test_exact_tie_yields_exactly_one),
        ("unclaimed pads stay free", test_unclaimed_pads_are_free),
        ("no detour claims no pad", test_not_going_for_boost_claims_nothing),
        ("claims expire with the peer", test_claims_expire_with_the_peer),
        ("strongest of two claims wins", test_strongest_claim_wins_when_two_peers_want_one_pad),
        ("opposition claims ignored", test_opposition_claims_ignored),
    ]:
        check(name, fn)

    print(f"{'test':42s} result")
    print("-" * 76)
    fails = 0
    for name, ok, err in RESULTS:
        print(f"{name:42s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            fails += 1
    print("-" * 76)
    print(f"{len(RESULTS) - fails}/{len(RESULTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
