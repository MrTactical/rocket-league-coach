"""
Online self-calibration.

The bot corrects its own systematic errors during play. After every attempt on
the ball it compares what it intended with what happened, and nudges a small
set of correction terms:

  aim_bias      the ball keeps leaving at an angle to where we aimed
  timing_bias   we keep arriving early or late relative to the plan
  offset_scale  our contact point is consistently too far from the ball
  power_scale   our touches are consistently too soft or too heavy

This is not reinforcement learning, and it is deliberately not trying to be.
RL over raw controls needs millions of samples and would learn nothing useful
inside a match. Estimating four scalar corrections from observed outcomes
converges in tens of touches, which is a single game.

One subtlety worth stating, because it looks like a bug otherwise. The
humanizer deliberately injects random aim error to keep the bot human. That
error is zero-mean, so averaging over attempts cancels it and leaves only the
systematic component -- which is exactly what we want to correct. Calibration
therefore sharpens the bot's *model* without removing its intended
imperfection. If it ever started cancelling the humanizer, that would show up
as aim_bias drifting to track the injected noise rather than settling.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.vec import Vec3, clamp

CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration.json"

# Learning rates. Deliberately small: these corrections sit underneath every
# touch, so a term that chases the last sample makes the bot worse, not better.
AIM_RATE = 0.18
TIMING_RATE = 0.15
OFFSET_RATE = 0.10
POWER_RATE = 0.10

# Bounds. A correction that runs away is worse than no correction, and any
# error large enough to exceed these is a bug to fix rather than trim.
AIM_LIMIT = 0.40          # radians (~23 degrees)
TIMING_LIMIT = 0.35       # seconds
OFFSET_RANGE = (0.65, 1.45)
POWER_RANGE = (0.70, 1.40)

# How long after the planned contact we keep waiting before calling it a miss.
MISS_GRACE = 0.45

# Contacts needed before corrections are trusted enough to apply at full
# strength; before this they are scaled in gradually.
WARMUP_CONTACTS = 6


@dataclass
class Corrections:
    aim_bias: float = 0.0
    timing_bias: float = 0.0
    offset_scale: float = 1.0
    power_scale: float = 1.0

    attempts: int = 0
    contacts: int = 0
    misses: int = 0

    # Evidence gathered by other Ally instances and folded in via comms.
    # Tracked separately from our own contacts so it can be weighted
    # differently and so pooled runs are distinguishable in the logs.
    peer_samples: int = 0


class StrikeCalibrator:
    """
    Tracks one attempt at a time and folds its outcome into the corrections.

    Usage per tick:
        cal.plan(key, aim_dir, contact_time, ball_pos)   # when a strike starts
        cal.observe(state)                               # every tick
        aim, timing, offset = cal.apply()                # when building controls
    """

    def __init__(self, persist: bool = True):
        self.c = Corrections()
        self.persist = persist

        # Current intent.
        self._key: str | None = None
        self._aim_dir: Vec3 | None = None
        self._planned_time: float = 0.0
        self._planned_ball: Vec3 | None = None
        self._planned_speed: float = 0.0

        # Touch detection.
        self._last_touch_time: float = -1.0
        self._awaiting: bool = False
        self._recent: list[str] = []

        self.load()

    # --- persistence ------------------------------------------------------

    def load(self) -> bool:
        if not self.persist:
            return False
        try:
            if CALIBRATION_PATH.exists():
                data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
                known = set(Corrections.__dataclass_fields__)
                self.c = Corrections(**{k: v for k, v in data.items() if k in known})
                return True
        except Exception:
            pass
        return False

    def save(self):
        if not self.persist:
            return
        try:
            CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION_PATH.write_text(json.dumps(asdict(self.c), indent=2), encoding="utf-8")
        except Exception:
            pass

    # --- registering intent ----------------------------------------------

    def plan(self, key: str, aim_dir: Vec3, contact_time: float, ball_pos: Vec3, speed: float = 0.0):
        """Record what this attempt is trying to do. Called when a strike begins."""
        if key == self._key:
            return
        # A new attempt while still waiting on the last one means the previous
        # attempt was abandoned rather than missed; do not penalise it.
        self._key = key
        self._aim_dir = aim_dir.flat().normalized()
        self._planned_time = contact_time
        self._planned_ball = ball_pos.copy()
        self._planned_speed = speed
        self._awaiting = True
        self.c.attempts += 1

    # --- observing outcomes ----------------------------------------------

    def observe(self, state):
        """Call every tick. Detects the touch, or gives up and records a miss."""
        if not self._awaiting or state.ball is None:
            return

        touch = state.me.latest_touch
        if touch is not None and touch.game_seconds > self._last_touch_time + 1e-3:
            # Only count it if it is plausibly the touch we planned, not a
            # stale one from before this attempt began.
            if touch.game_seconds >= self._planned_time - 1.5:
                self._last_touch_time = touch.game_seconds
                self._on_contact(state, touch)
                return
            self._last_touch_time = touch.game_seconds

        if state.time > self._planned_time + MISS_GRACE:
            self._on_miss()

    def _on_contact(self, state, touch):
        self._awaiting = False
        self.c.contacts += 1
        self._recent.append("hit")
        del self._recent[:-20]

        # --- aim error ----------------------------------------------------
        # Where the ball actually left, versus where we meant to send it.
        outgoing = state.ball.vel.flat()
        if outgoing.length() > 250.0 and self._aim_dir is not None:
            actual = outgoing.normalized()
            intended = self._aim_dir
            # Signed angle from intended to actual, in the ground plane.
            cross = intended.x * actual.y - intended.y * actual.x
            dot = clamp(intended.dot(actual), -1.0, 1.0)
            err = math.atan2(cross, dot)
            # Only learn from touches that roughly went where we asked. A ball
            # sent 120 degrees off was a mishit or a deflection, not a bias.
            if abs(err) < 1.2:
                self.c.aim_bias = clamp(
                    self.c.aim_bias - err * AIM_RATE, -AIM_LIMIT, AIM_LIMIT
                )

        # --- timing error --------------------------------------------------
        t_err = touch.game_seconds - self._planned_time
        if abs(t_err) < 1.0:
            self.c.timing_bias = clamp(
                self.c.timing_bias - t_err * TIMING_RATE, -TIMING_LIMIT, TIMING_LIMIT
            )

        # --- power ----------------------------------------------------------
        if self._planned_speed > 100.0:
            got = state.ball.vel.flat_length()
            if got > 50.0:
                ratio = self._planned_speed / got
                if 0.3 < ratio < 3.0:
                    self.c.power_scale = clamp(
                        self.c.power_scale + (ratio - 1.0) * POWER_RATE,
                        *POWER_RANGE,
                    )

        # Contact made: our standoff is working, relax it back toward nominal.
        self.c.offset_scale = clamp(
            self.c.offset_scale + (1.0 - self.c.offset_scale) * 0.05, *OFFSET_RANGE
        )

    def _on_miss(self):
        self._awaiting = False
        self.c.misses += 1
        self._recent.append("miss")
        del self._recent[:-20]

        # Whiffed: the most common cause is standing off too far, so close the
        # contact point up a little. Bounded, and undone by successful touches.
        self.c.offset_scale = clamp(
            self.c.offset_scale - OFFSET_RATE, *OFFSET_RANGE
        )

    # --- pooling across Ally instances ------------------------------------

    def absorb(self, reports: list[dict]):
        """
        Fold in calibration from other Ally instances.

        Every Ally runs the same controllers, so their corrections estimate the
        same underlying quantity and pooling is legitimate -- six cars gather
        six times the evidence.

        The weighting is deliberately timid. Peers are also absorbing *our*
        reports, so aggressive averaging would let two bots echo a value back
        and forth until they agreed on something neither had evidence for. The
        pull is capped well below 1 and scaled by how much more evidence the
        peer has than we do, so a peer with 50 touches moves a bot with 2, and
        barely moves a bot with 200.
        """
        for r in reports:
            n = int(r.get("n", 0))
            if n < 2:
                continue

            mine = max(self.c.contacts, 1)
            # Relative evidence, hard-capped so no single report dominates.
            w = min(0.20, 0.35 * n / (n + mine))

            self.c.aim_bias = clamp(
                self.c.aim_bias + (float(r.get("aim", 0.0)) - self.c.aim_bias) * w,
                -AIM_LIMIT, AIM_LIMIT,
            )
            self.c.timing_bias = clamp(
                self.c.timing_bias + (float(r.get("time", 0.0)) - self.c.timing_bias) * w,
                -TIMING_LIMIT, TIMING_LIMIT,
            )
            self.c.offset_scale = clamp(
                self.c.offset_scale + (float(r.get("off", 1.0)) - self.c.offset_scale) * w,
                *OFFSET_RANGE,
            )
            self.c.peer_samples += n

    # --- applying ---------------------------------------------------------

    def confidence(self) -> float:
        """
        How much of the correction to apply, ramping in over early touches.

        Pooled peer evidence counts, but at a discount: it is second-hand and
        may already include an echo of our own reports.
        """
        effective = self.c.contacts + 0.4 * self.c.peer_samples
        return clamp(effective / WARMUP_CONTACTS, 0.0, 1.0)

    def apply(self) -> tuple[float, float, float]:
        """Returns (aim_bias, timing_bias, offset_scale) scaled by confidence."""
        k = self.confidence()
        return (
            self.c.aim_bias * k,
            self.c.timing_bias * k,
            1.0 + (self.c.offset_scale - 1.0) * k,
        )

    # --- reporting --------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.c.contacts + self.c.misses
        return self.c.contacts / total if total else 0.0

    def recent_hit_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(1 for r in self._recent if r == "hit") / len(self._recent)

    def summary(self) -> str:
        pooled = f" +{self.c.peer_samples} pooled" if self.c.peer_samples else ""
        return (
            f"aim{self.c.aim_bias:+.3f}rad time{self.c.timing_bias:+.3f}s "
            f"off x{self.c.offset_scale:.2f} pow x{self.c.power_scale:.2f} | "
            f"{self.c.contacts} hit / {self.c.misses} miss{pooled} "
            f"({100 * self.hit_rate:.0f}%, recent {100 * self.recent_hit_rate():.0f}%) "
            f"conf {self.confidence():.2f}"
        )
