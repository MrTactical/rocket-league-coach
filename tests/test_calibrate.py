"""
Online self-calibration tests.

The claim being tested is narrow and checkable: given touches that are
systematically off in a consistent direction, the calibrator should converge on
a correction that cancels the bias -- and given touches that are merely noisy,
it should NOT chase the noise.

That second property is the one that matters most. The humanizer deliberately
injects random aim error to keep the bot human; if calibration learned that
away, the bot would quietly become a machine again.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.core.vec import Vec3  # noqa: E402
from bot.learn.calibrate import StrikeCalibrator  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


class FakeTouch:
    def __init__(self, t):
        self.game_seconds = t
        self.location = Vec3()


class FakeCar:
    def __init__(self):
        self.latest_touch = None


class FakeBall:
    def __init__(self, vel):
        self.vel = vel


class FakeState:
    def __init__(self, time, ball_vel, touch_time=None):
        self.time = time
        self.me = FakeCar()
        if touch_time is not None:
            self.me.latest_touch = FakeTouch(touch_time)
        self.ball = FakeBall(ball_vel)


def run_attempts(cal, n, bias_rad, noise_fn=None, seed=1):
    """Feed n attempts whose outcome is off by `bias_rad` from the aim."""
    import random

    rng = random.Random(seed)
    t = 0.0
    for i in range(n):
        t += 3.0
        aim = Vec3(1.0, 0.0, 0.0)
        cal.plan(f"shot:{i}", aim, t + 1.0, Vec3(0, 0, 92))
        # The correction the bot would have applied this attempt.
        applied, _, _ = cal.apply()
        err = bias_rad + (noise_fn(rng) if noise_fn else 0.0)
        # Outgoing direction = intended, rotated by the applied correction and
        # then by the real-world error.
        out = aim.rotate_2d(applied + err) * 1500.0
        cal.observe(FakeState(t + 1.0, out, touch_time=t + 1.0))
    return cal


def test_learns_systematic_bias():
    """A consistent 0.2 rad error should be largely cancelled."""
    cal = StrikeCalibrator(persist=False)
    run_attempts(cal, 40, bias_rad=0.20)
    assert cal.c.contacts == 40, f"expected 40 contacts, got {cal.c.contacts}"
    # The learned correction should oppose the bias.
    assert cal.c.aim_bias < -0.10, (
        f"expected a negative correction opposing +0.20 bias, got {cal.c.aim_bias:+.3f}"
    )
    # And the residual error should be small.
    applied, _, _ = cal.apply()
    residual = abs(0.20 + applied)
    assert residual < 0.08, f"residual error {residual:.3f} rad still too large"


def test_does_not_chase_zero_mean_noise():
    """Random error must average out, not be learned away."""
    cal = StrikeCalibrator(persist=False)
    run_attempts(cal, 60, bias_rad=0.0, noise_fn=lambda r: r.gauss(0.0, 0.15))
    assert abs(cal.c.aim_bias) < 0.06, (
        f"calibrator chased noise: aim_bias drifted to {cal.c.aim_bias:+.3f} "
        "with zero true bias. This would erode the humanizer's intended error."
    )


def test_learns_timing_bias():
    cal = StrikeCalibrator(persist=False)
    t = 0.0
    for i in range(30):
        t += 3.0
        planned = t + 1.0
        cal.plan(f"shot:{i}", Vec3(1, 0, 0), planned, Vec3(0, 0, 92))
        # Always arrive 0.15s late.
        cal.observe(FakeState(planned + 0.15, Vec3(1500, 0, 0), touch_time=planned + 0.15))
    assert cal.c.timing_bias < -0.05, (
        f"expected negative timing correction for arriving late, got {cal.c.timing_bias:+.3f}"
    )


def test_misses_tighten_the_contact_offset():
    """Repeated whiffs should close the standoff up."""
    cal = StrikeCalibrator(persist=False)
    t = 0.0
    for i in range(6):
        t += 5.0
        cal.plan(f"shot:{i}", Vec3(1, 0, 0), t + 1.0, Vec3(0, 0, 92))
        # Never touch; run the clock past the grace period.
        cal.observe(FakeState(t + 2.0, Vec3(0, 0, 0)))
    assert cal.c.misses == 6, f"expected 6 misses, got {cal.c.misses}"
    assert cal.c.offset_scale < 0.95, (
        f"offset should have tightened after whiffs, got {cal.c.offset_scale:.2f}"
    )


def test_corrections_are_bounded():
    """A pathological bias must not produce a runaway correction."""
    cal = StrikeCalibrator(persist=False)
    run_attempts(cal, 80, bias_rad=3.0)  # absurd
    assert abs(cal.c.aim_bias) <= 0.401, f"aim_bias escaped bounds: {cal.c.aim_bias}"
    assert 0.6 <= cal.c.offset_scale <= 1.5, f"offset escaped bounds: {cal.c.offset_scale}"


def test_confidence_ramps_in():
    """Corrections apply gradually, not from the first sample."""
    cal = StrikeCalibrator(persist=False)
    assert cal.confidence() == 0.0
    run_attempts(cal, 3, bias_rad=0.2)
    partial = cal.confidence()
    assert 0.0 < partial < 1.0, f"expected partial confidence, got {partial}"
    run_attempts(cal, 10, bias_rad=0.2)
    assert cal.confidence() == 1.0


def main() -> int:
    for name, fn in [
        ("learns a systematic aim bias", test_learns_systematic_bias),
        ("ignores zero-mean noise", test_does_not_chase_zero_mean_noise),
        ("learns a timing bias", test_learns_timing_bias),
        ("whiffs tighten contact offset", test_misses_tighten_the_contact_offset),
        ("corrections stay bounded", test_corrections_are_bounded),
        ("confidence ramps in", test_confidence_ramps_in),
    ]:
        check(name, fn)

    print(f"{'test':38s} result")
    print("-" * 72)
    fails = 0
    for name, ok, err in RESULTS:
        print(f"{name:38s} {'PASS' if ok else 'FAIL  ' + err}")
        if not ok:
            fails += 1
    print("-" * 72)
    print(f"{len(RESULTS) - fails}/{len(RESULTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
