"""
Human realism layer.

A bot with perfect reactions and perfect touches is a poor practice partner:
you end up training against something that does not exist, and habits you
build against it do not transfer. This module degrades the bot's play in the
specific ways human play is degraded, calibrated to a target rank.

What gets modelled:

  * Reaction time  -- decisions act on a slightly stale view of the world.
  * Aim error      -- touches land a few degrees off intent, with the error
                      persisting across a whole strike rather than resampled
                      per tick (jitter would average out to perfect aim).
  * Whiffs         -- occasional outright misses on hard touches.
  * Consistency    -- mechanics degrade under pressure and at speed.
  * Input noise    -- small steering imprecision, like a real thumbstick.
  * Attention      -- brief lapses where the bot commits late.

The rank presets are drawn from how play actually differs by rank: reaction
time compresses, aerials become reliable, and consistency rises.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..core.vec import Vec3, clamp


@dataclass
class SkillProfile:
    """How good, and how flawed, the bot should be."""

    name: str = "diamond"

    # Seconds of delay between the world changing and the bot acting on it.
    reaction_time: float = 0.22
    reaction_jitter: float = 0.06

    # Standard deviation of aim error on a strike, in radians.
    aim_error: float = 0.055

    # Probability of a meaningfully bad touch on a difficult ball.
    whiff_chance: float = 0.07

    # Probability of attempting an aerial that is available.
    aerial_confidence: float = 0.55

    # Steering noise amplitude.
    input_noise: float = 0.035

    # Chance per second of a brief attention lapse.
    lapse_rate: float = 0.035
    lapse_duration: float = 0.35

    # How much worse things get under pressure (opponent close, high speed).
    pressure_sensitivity: float = 0.5

    # Speedflip reliability on kickoff.
    speedflip_skill: float = 0.6

    # How often recovery includes a wavedash rather than a plain landing.
    wavedash_rate: float = 0.5


RANKS: dict[str, SkillProfile] = {
    "bronze": SkillProfile(
        "bronze", reaction_time=0.42, reaction_jitter=0.12, aim_error=0.19,
        whiff_chance=0.30, aerial_confidence=0.03, input_noise=0.11,
        lapse_rate=0.13, pressure_sensitivity=0.95, speedflip_skill=0.0,
        wavedash_rate=0.0,
    ),
    "silver": SkillProfile(
        "silver", reaction_time=0.36, reaction_jitter=0.10, aim_error=0.15,
        whiff_chance=0.24, aerial_confidence=0.07, input_noise=0.09,
        lapse_rate=0.10, pressure_sensitivity=0.85, speedflip_skill=0.0,
        wavedash_rate=0.05,
    ),
    "gold": SkillProfile(
        "gold", reaction_time=0.31, reaction_jitter=0.09, aim_error=0.12,
        whiff_chance=0.18, aerial_confidence=0.15, input_noise=0.075,
        lapse_rate=0.08, pressure_sensitivity=0.75, speedflip_skill=0.1,
        wavedash_rate=0.12,
    ),
    "platinum": SkillProfile(
        "platinum", reaction_time=0.26, reaction_jitter=0.07, aim_error=0.09,
        whiff_chance=0.13, aerial_confidence=0.30, input_noise=0.055,
        lapse_rate=0.06, pressure_sensitivity=0.65, speedflip_skill=0.25,
        wavedash_rate=0.28,
    ),
    "diamond": SkillProfile(
        "diamond", reaction_time=0.22, reaction_jitter=0.06, aim_error=0.055,
        whiff_chance=0.07, aerial_confidence=0.55, input_noise=0.035,
        lapse_rate=0.035, pressure_sensitivity=0.50, speedflip_skill=0.6,
        wavedash_rate=0.50,
    ),
    "champion": SkillProfile(
        "champion", reaction_time=0.185, reaction_jitter=0.045, aim_error=0.038,
        whiff_chance=0.045, aerial_confidence=0.75, input_noise=0.025,
        lapse_rate=0.022, pressure_sensitivity=0.38, speedflip_skill=0.82,
        wavedash_rate=0.68,
    ),
    "grand_champion": SkillProfile(
        "grand_champion", reaction_time=0.16, reaction_jitter=0.035, aim_error=0.026,
        whiff_chance=0.028, aerial_confidence=0.88, input_noise=0.018,
        lapse_rate=0.014, pressure_sensitivity=0.28, speedflip_skill=0.93,
        wavedash_rate=0.80,
    ),
    "ssl": SkillProfile(
        # Even at the top there is no such thing as clean. SSLs miss, misread
        # and get beaten to balls; they just do it far less often.
        "ssl", reaction_time=0.14, reaction_jitter=0.028, aim_error=0.018,
        whiff_chance=0.018, aerial_confidence=0.95, input_noise=0.013,
        lapse_rate=0.009, pressure_sensitivity=0.20, speedflip_skill=0.97,
        wavedash_rate=0.88,
    ),
}


class Humanizer:
    """
    Applies the skill profile to the bot's perception and output.

    Errors are *committed*: an aim error is sampled once when a strike begins
    and held for its duration. Resampling every tick would average out to
    perfect aim, which is exactly the artificial precision we're removing.
    """

    def __init__(self, profile: SkillProfile, seed: int | None = None):
        self.profile = profile
        self.rng = random.Random(seed)

        self._reaction_deadline = 0.0
        self._pending_role: str | None = None

        self._aim_error: float = 0.0
        self._aim_key: str = ""

        self._whiff_key: str = ""
        self._whiff_active: bool = False

        self._lapse_until: float = 0.0
        self._last_lapse_roll: float = 0.0

        self._noise_phase = self.rng.random() * 10.0

    # --- perception -------------------------------------------------------

    def reaction_delay(self) -> float:
        p = self.profile
        return max(0.0, self.rng.gauss(p.reaction_time, p.reaction_jitter))

    def gate_decision(self, now: float, new_role: str, current_role: str) -> str:
        """
        Delay acting on a role change by a reaction time.

        Real players do not switch intent the instant the situation changes;
        they notice, then react. This is where most of the "feels human"
        comes from.

        Yielding the ball is exempt. Gating a *downgrade* out of attack means
        that during every handover the bot giving up the ball still reads as
        attacking while the one taking it also does -- so two teammates appear
        committed at once for a reaction time, every single time possession
        changes hands. Backing off is also the safe direction to get wrong, and
        realistically it is the faster judgement: noticing a teammate is better
        placed takes less deliberation than deciding to commit yourself.
        """
        if new_role == current_role:
            self._pending_role = None
            return current_role

        if current_role == "attack" and new_role != "attack":
            self._pending_role = None
            return new_role

        if self._pending_role != new_role:
            self._pending_role = new_role
            self._reaction_deadline = now + self.reaction_delay()
            return current_role

        if now >= self._reaction_deadline:
            self._pending_role = None
            return new_role
        return current_role

    def is_lapsed(self, now: float, dt: float) -> bool:
        """Brief attention lapses: the bot hesitates, like a distracted player."""
        if now < self._lapse_until:
            return True
        if now - self._last_lapse_roll > 0.25:
            self._last_lapse_roll = now
            if self.rng.random() < self.profile.lapse_rate * 0.25:
                self._lapse_until = now + self.profile.lapse_duration
                return True
        return False

    # --- execution error --------------------------------------------------

    def pressure(self, state) -> float:
        """0..1 estimate of how pressured this touch is."""
        if state.ball is None:
            return 0.0
        opp = state.nearest_opponent_to(state.ball.pos)
        p = 0.0
        if opp is not None:
            d = opp.pos.dist(state.ball.pos)
            p += clamp(1.0 - d / 2500.0, 0.0, 1.0) * 0.6
        p += clamp(state.me.speed / 2300.0, 0.0, 1.0) * 0.2
        if state.ball.pos.z > 300.0:
            p += 0.2
        return clamp(p, 0.0, 1.0)

    def aim_offset(self, key: str, state) -> float:
        """
        Angular aim error for the current strike, in radians.

        `key` identifies the strike (e.g. the intercept slice), so the same
        error is held for its duration and only resampled on a new attempt.
        """
        if key != self._aim_key:
            self._aim_key = key
            scale = 1.0 + self.pressure(state) * self.profile.pressure_sensitivity
            self._aim_error = self.rng.gauss(0.0, self.profile.aim_error * scale)
        return self._aim_error

    def should_whiff(self, key: str, state, difficulty: float) -> bool:
        """
        Decide once per strike whether this one gets badly mistimed.

        `difficulty` in [0,1] scales the base whiff chance -- easy rolling
        shots are rarely missed, awkward aerials often are.
        """
        if key != self._whiff_key:
            self._whiff_key = key
            chance = self.profile.whiff_chance * (0.35 + 1.3 * difficulty)
            chance *= 1.0 + self.pressure(state) * self.profile.pressure_sensitivity
            self._whiff_active = self.rng.random() < clamp(chance, 0.0, 0.6)
        return self._whiff_active

    def attempt_aerial(self, difficulty: float) -> bool:
        """Whether to commit to an available aerial at all."""
        conf = self.profile.aerial_confidence * (1.0 - 0.55 * clamp(difficulty, 0.0, 1.0))
        return self.rng.random() < clamp(conf, 0.0, 1.0)

    def attempt_speedflip(self) -> bool:
        return self.rng.random() < self.profile.speedflip_skill

    def attempt_wavedash(self) -> bool:
        return self.rng.random() < self.profile.wavedash_rate

    # --- output ----------------------------------------------------------

    def add_input_noise(self, controls, now: float):
        """
        Small continuous steering imprecision.

        Uses smooth low-frequency noise rather than white noise: a real thumb
        drifts, it does not vibrate.
        """
        n = self.profile.input_noise
        if n <= 0.0:
            return controls
        drift = math.sin(now * 2.3 + self._noise_phase) * 0.6 + math.sin(now * 5.7 + self._noise_phase * 2) * 0.4
        controls.steer = clamp(controls.steer + drift * n, -1.0, 1.0)
        return controls

    def apply_whiff(self, controls, state):
        """
        Turn a committed strike into a realistic miss: slightly wrong angle and
        slightly wrong timing, rather than an obvious swerve away.
        """
        controls.steer = clamp(controls.steer + self.rng.uniform(-0.55, 0.55), -1.0, 1.0)
        if self.rng.random() < 0.4:
            controls.throttle = clamp(controls.throttle * 0.55, -1.0, 1.0)
            controls.boost = False
        return controls


def rotate_aim(direction: Vec3, angle: float) -> Vec3:
    """Rotate an aim direction in the ground plane by `angle` radians."""
    return direction.rotate_2d(angle)


def get_profile(name: str) -> SkillProfile:
    return RANKS.get(name.lower().strip(), RANKS["diamond"])
