"""
Shared-experience tests.

The claim: a bot should end up preferring the action that has actually worked
in a given situation, should learn that from *other bots'* attempts as well as
its own, and should not become confident about a bucket nobody has really
explored.

That last one is the subtle risk. Every Ally merges every other Ally's table,
and those tables already contain merged copies of each other -- so a naive
merge lets four samples echo around six bots until they look like forty.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.learn.experience import (  # noqa: E402
    MIN_SAMPLES,
    ExperienceTable,
    situation_key,
)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def test_prefers_the_action_that_worked():
    t = ExperienceTable(persist=False, seed=4)
    key = "att/C/gnd/free"
    for _ in range(20):
        t.record(key, "shot", 0.8)
        t.record(key, "clear", -0.3)
    # Exploration is random, so check the tendency over many draws.
    picks = [t.choose(key, ["shot", "clear"], "clear")[0] for _ in range(200)]
    shot_rate = picks.count("shot") / len(picks)
    assert shot_rate > 0.75, f"only chose the better action {shot_rate:.0%} of the time"


def test_falls_back_to_default_when_unseen():
    t = ExperienceTable(persist=False, seed=4)
    action, why = t.choose("mid/L/air/press", ["shot", "clear"], "clear")
    # With no data it either explores or takes the default; both are fine, but
    # it must not claim to have learned something.
    assert "learned" not in why, f"claimed knowledge with no data: {why}"


def test_needs_minimum_samples_before_trusting():
    t = ExperienceTable(persist=False, seed=1)
    key = "def/R/gnd/press"
    for _ in range(MIN_SAMPLES - 1):
        t.record(key, "shot", 1.0)
    assert t.value(key, "shot") is None, "trusted a bucket below the sample floor"
    t.record(key, "shot", 1.0)
    assert t.value(key, "shot") is not None, "still untrusted at the sample floor"


def test_learns_from_a_peer_without_own_attempts():
    """The whole point: learn from another bot's mistake."""
    peer = ExperienceTable(persist=False, seed=2)
    key = "att/L/gnd/free"
    for _ in range(30):
        peer.record(key, "shot", -0.7)   # peer discovers this is bad
        peer.record(key, "pass", 0.6)

    me = ExperienceTable(persist=False, seed=3)
    assert me.value(key, "shot") is None, "should start with no opinion"
    me.merge(peer.export())

    assert me.peer_samples > 0, "merge recorded nothing"
    v_shot = me.value(key, "shot")
    v_pass = me.value(key, "pass")
    assert v_shot is not None and v_pass is not None, "peer data not usable"
    assert v_pass > v_shot, (
        f"did not inherit the peer's finding: pass={v_pass:+.2f} shot={v_shot:+.2f}"
    )


def test_merge_discounts_echoed_counts():
    """
    Peers merge our data back to us. The incoming count must be discounted or
    a handful of samples inflates into false confidence.
    """
    a = ExperienceTable(persist=False, seed=1)
    key = "mid/C/gnd/free"
    for _ in range(10):
        a.record(key, "shot", 0.5)
    before = a.samples(key)

    b = ExperienceTable(persist=False, seed=2)
    b.merge(a.export())
    # Echo it back and forth a few times.
    for _ in range(3):
        a.merge(b.export())
        b.merge(a.export())

    growth = a.samples(key) / before
    assert growth < 12.0, (
        f"sample count grew {growth:.1f}x through echo alone -- discounting is "
        "not holding"
    )


def test_exploration_decays_as_a_bucket_fills():
    t = ExperienceTable(persist=False, seed=5)
    key = "att/C/gnd/free"
    empty = t.explore_rate(key)
    for _ in range(60):
        t.record(key, "shot", 0.3)
    full = t.explore_rate(key)
    assert full < empty, f"exploration did not decay: {empty:.2f} -> {full:.2f}"
    assert full >= 0.0499, "exploration must never reach zero entirely"


def test_situation_keys_are_coarse_but_distinct():
    class FakeBall:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeState:
        goal_sign = -1.0

    st = FakeState()
    keys = {
        situation_key(st, FakeBall(0, -4000, 90), 0.1),
        situation_key(st, FakeBall(0, 4000, 90), 0.1),
        situation_key(st, FakeBall(-3000, 0, 90), 0.1),
        situation_key(st, FakeBall(0, 0, 900), 0.1),
        situation_key(st, FakeBall(0, 0, 90), 0.9),
    }
    assert len(keys) == 5, f"expected 5 distinct buckets, got {len(keys)}: {keys}"
    # And nearby situations should share a bucket.
    a = situation_key(st, FakeBall(100, 200, 90), 0.1)
    b = situation_key(st, FakeBall(150, 260, 95), 0.15)
    assert a == b, f"over-fine bucketing: {a} vs {b}"


def main() -> int:
    for name, fn in [
        ("prefers the action that worked", test_prefers_the_action_that_worked),
        ("no false confidence when unseen", test_falls_back_to_default_when_unseen),
        ("respects the sample floor", test_needs_minimum_samples_before_trusting),
        ("learns from a peer's attempts", test_learns_from_a_peer_without_own_attempts),
        ("merge discounts echoed counts", test_merge_discounts_echoed_counts),
        ("exploration decays with data", test_exploration_decays_as_a_bucket_fills),
        ("situation buckets sane", test_situation_keys_are_coarse_but_distinct),
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
