"""
An online model of the human teammate.

This is what separates "a bot that plays Rocket League" from "a bot that plays
Rocket League *with you*". Every tick it watches what the human actually does
and maintains a handful of traits; every decision downstream is biased by
them. The traits persist to disk between matches, so match two starts from
what match one learned.

All traits are exponential moving averages in [0, 1] unless noted. EMAs are
used rather than running means so the model tracks *current* form -- if you
start rotating better, it notices within a match rather than being anchored to
last week.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.constants import BACK_WALL_Y, MAX_SPEED, SIDE_WALL_X
from ..core.vec import Vec3, clamp

# EMA smoothing per *sample*. Traits sampled every tick use SLOW; traits
# sampled only on rare events (touches, kickoffs) use FAST, since they get far
# fewer updates and would otherwise never move.
ALPHA_SLOW = 0.002
ALPHA_FAST = 0.08

PROFILE_DIR = Path(__file__).resolve().parents[2] / "data" / "profiles"


def _ema(old: float, new: float, alpha: float) -> float:
    return old + (new - old) * alpha


@dataclass
class Traits:
    """The learned picture of how this human plays."""

    # How often they pursue the ball when they are NOT the better-placed
    # player. 0 = perfectly disciplined, 1 = pure ball-chaser.
    chase_rate: float = 0.5

    # Fraction of the match spent glued to the ball (within ~1600uu).
    #
    # This exists because chase_rate alone misses the worst case. A player who
    # never leaves the ball is always the nearest player, so they are never
    # "worse placed" and never trip the chase test -- the most extreme
    # ball-chaser scores zero. Occupancy catches what pursuit cannot.
    ball_glue: float = 0.5

    # After committing to a touch in the attacking half, do they rotate back?
    # 1 = always retreats and gives space, 0 = stays glued to the ball.
    rotation_discipline: float = 0.5

    # Fraction of kickoffs they go for.
    kickoff_commit: float = 0.5

    # Typical boost level, normalised to [0,1]. Low means they run dry a lot.
    boost_economy: float = 0.35

    # Fraction of their touches that happen above the ground game.
    aerial_rate: float = 0.15

    # Mean lateral position, [-1,1]. Negative = favours the -X side.
    side_bias: float = 0.0

    # Mean (their time-to-ball - our time-to-ball). Negative means they are
    # generally getting there first. In seconds, not normalised.
    relative_speed: float = 0.0

    # How often both of us end up committed to the same ball at once.
    double_commit_rate: float = 0.2

    # Mean depth in our own half when we are defending, [0,1] where 1 is deep
    # in our own net. Tells us whether they help defend or float upfield.
    defensive_depth: float = 0.4

    # Mean speed they carry, normalised. Slow players need more cover.
    pace: float = 0.5

    # --- bookkeeping ---
    samples: int = 0
    touches: int = 0
    kickoffs: int = 0
    matches: int = 0
    seconds_observed: float = 0.0
    last_updated: float = 0.0


@dataclass
class MatchTally:
    """Per-match counters, reset each match and fed to the coach afterwards."""

    goals_for: int = 0
    goals_against: int = 0
    my_touches: int = 0
    ally_touches: int = 0
    ally_shots: int = 0
    ally_saves: int = 0
    ally_goals: int = 0
    ally_demos: int = 0
    ally_demoed: int = 0
    double_commits: int = 0
    ally_low_boost_time: float = 0.0
    ally_no_boost_time: float = 0.0
    ally_time_in_own_third: float = 0.0
    ally_time_in_attacking_third: float = 0.0
    conceded_while_ally_upfield: int = 0
    ally_first_to_ball: int = 0
    my_first_to_ball: int = 0
    kickoffs_taken_by_ally: int = 0
    kickoffs_total: int = 0
    ally_aerial_touches: int = 0
    ally_backpost_saves: int = 0
    duration: float = 0.0
    notes: list[str] = field(default_factory=list)


class TeammateModel:
    """
    Watches the human and exposes both raw traits and the derived knobs the
    strategy layer actually consumes.
    """

    def __init__(self, name: str = "partner"):
        self.name = name
        self.traits = Traits()
        self.tally = MatchTally()

        # Transient per-tick tracking state.
        self._last_ally_touch_time = -99.0
        self._pending_rotation_check: tuple[float, float] | None = None
        self._last_kickoff_time = -99.0
        self._kickoff_recorded = False
        self._prev_time = 0.0
        self._contest_cooldown = 0.0

        self.load()

    # --- persistence ------------------------------------------------------

    @property
    def path(self) -> Path:
        safe = "".join(ch for ch in self.name if ch.isalnum() or ch in "-_") or "partner"
        return PROFILE_DIR / f"{safe}.json"

    def load(self) -> bool:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                known = {f for f in Traits.__dataclass_fields__}
                self.traits = Traits(**{k: v for k, v in data.items() if k in known})
                return True
        except Exception:
            # A corrupt profile should never stop a match starting.
            pass
        return False

    def bind(self, name: str) -> bool:
        """
        Attach to a named partner once we discover who they are.

        The human's name is not in the match configuration -- a Human player
        entry carries only a team, no name -- so it can only be read from the
        packet at runtime. This lets the bot start under a placeholder and
        adopt the real profile on the first tick it sees them.

        Returns True if a stored profile was loaded.
        """
        if not name or name == self.name:
            return False
        # Only rebind before we have accumulated meaningful observations,
        # so a mid-match name change cannot discard a match's learning.
        if self.traits.samples > 600:
            return False
        self.name = name
        self.traits = Traits()
        return self.load()

    def save(self):
        try:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            self.traits.last_updated = time.time()
            self.path.write_text(json.dumps(asdict(self.traits), indent=2), encoding="utf-8")
        except Exception:
            pass

    # --- per-tick observation --------------------------------------------

    def update(self, state, my_time: float, ally_time: float):
        """Called every tick with both players' estimated times to the ball."""
        ally = state.ally
        if ally is None or state.ball is None or not state.is_active:
            return

        dt = max(0.0, state.time - self._prev_time) if self._prev_time else 0.0
        self._prev_time = state.time
        if dt > 1.0:  # scene change / respawn; skip the discontinuity
            dt = 0.0

        t = self.traits
        t.samples += 1
        t.seconds_observed += dt
        self.tally.duration += dt

        # --- cheap positional traits --------------------------------------
        t.boost_economy = _ema(t.boost_economy, ally.boost / 100.0, ALPHA_SLOW)
        t.side_bias = _ema(t.side_bias, clamp(ally.pos.x / SIDE_WALL_X, -1.0, 1.0), ALPHA_SLOW)
        t.pace = _ema(t.pace, clamp(ally.speed / MAX_SPEED, 0.0, 1.0), ALPHA_SLOW)

        # Depth in our own half, 1 = in our net.
        depth = clamp((ally.pos.y * state.goal_sign + BACK_WALL_Y) / (2 * BACK_WALL_Y), 0.0, 1.0)
        t.defensive_depth = _ema(t.defensive_depth, depth, ALPHA_SLOW)

        # --- boost starvation (for coaching) ------------------------------
        if ally.boost < 20.0:
            self.tally.ally_low_boost_time += dt
        if ally.boost < 5.0:
            self.tally.ally_no_boost_time += dt

        # --- field occupancy ----------------------------------------------
        own_third = ally.pos.y * state.goal_sign > BACK_WALL_Y / 3.0
        att_third = ally.pos.y * state.goal_sign < -BACK_WALL_Y / 3.0
        if own_third:
            self.tally.ally_time_in_own_third += dt
        elif att_third:
            self.tally.ally_time_in_attacking_third += dt

        # --- chase / double-commit ----------------------------------------
        self._contest_cooldown = max(0.0, self._contest_cooldown - dt)
        ball = state.ball.pos
        to_ball = (ball - ally.pos)
        closing = ally.vel.dot(to_ball.normalized()) if to_ball.length_sq() > 1.0 else 0.0

        # Occupancy: are they sitting on top of the ball? Sampled every tick,
        # regardless of whether the ball is contestable, because the point is
        # to measure how much of the match they spend near it at all.
        t.ball_glue = _ema(t.ball_glue, 1.0 if to_ball.flat_length() < 1600.0 else 0.0, ALPHA_SLOW)

        # Only sample when the ball is genuinely contestable by both of us.
        if my_time < 4.0 and ally_time < 4.0:
            t.relative_speed = _ema(t.relative_speed, ally_time - my_time, ALPHA_SLOW)

            # They are chasing if they are clearly the worse-placed player yet
            # still driving hard at the ball.
            worse_placed = ally_time > my_time + 0.45
            committed = closing > 700.0
            self._observe_chase(worse_placed and committed)

            # Both of us bearing down on the same ball, close in.
            if (
                closing > 600.0
                and abs(ally_time - my_time) < 0.5
                and ally.pos.dist(ball) < 2200.0
                and state.me.vel.dot((ball - state.me.pos).normalized()) > 600.0
            ):
                if self._contest_cooldown <= 0.0:
                    self.tally.double_commits += 1
                    self._contest_cooldown = 2.0
                self._observe_double_commit(True)
            else:
                self._observe_double_commit(False)

        # --- rotation discipline, checked 1.5s after an attacking touch ----
        if self._pending_rotation_check is not None:
            due, y_at_touch = self._pending_rotation_check
            if state.time >= due:
                retreated = (ally.pos.y - y_at_touch) * state.goal_sign > 250.0
                t.rotation_discipline = _ema(t.rotation_discipline, 1.0 if retreated else 0.0, ALPHA_FAST)
                self._pending_rotation_check = None

    def _observe_chase(self, chasing: bool):
        self.traits.chase_rate = _ema(self.traits.chase_rate, 1.0 if chasing else 0.0, ALPHA_SLOW * 3)

    def _observe_double_commit(self, doubling: bool):
        self.traits.double_commit_rate = _ema(
            self.traits.double_commit_rate, 1.0 if doubling else 0.0, ALPHA_SLOW * 3
        )

    # --- discrete events --------------------------------------------------

    def on_ally_touch(self, state, touch_pos: Vec3):
        """Called when the human contacts the ball."""
        t = self.traits
        t.touches += 1
        self.tally.ally_touches += 1
        self._last_ally_touch_time = state.time

        if touch_pos.z > 300.0:
            self.tally.ally_aerial_touches += 1
            t.aerial_rate = _ema(t.aerial_rate, 1.0, ALPHA_FAST)
        else:
            t.aerial_rate = _ema(t.aerial_rate, 0.0, ALPHA_FAST)

        # If the touch happened in the attacking half, check shortly whether
        # they rotated out of the way afterwards.
        if touch_pos.y * state.goal_sign < 0.0 and state.ally is not None:
            self._pending_rotation_check = (state.time + 1.5, state.ally.pos.y)

    def on_kickoff(self, state, ally_went: bool):
        t = self.traits
        t.kickoffs += 1
        self.tally.kickoffs_total += 1
        if ally_went:
            self.tally.kickoffs_taken_by_ally += 1
        t.kickoff_commit = _ema(t.kickoff_commit, 1.0 if ally_went else 0.0, ALPHA_FAST)

    def on_goal_conceded(self, state):
        self.tally.goals_against += 1
        ally = state.ally
        if ally is not None and ally.pos.y * state.goal_sign < -BACK_WALL_Y / 3.0:
            self.tally.conceded_while_ally_upfield += 1

    def on_goal_scored(self, state):
        self.tally.goals_for += 1

    def note_accolades(self, ally):
        """Fold in the game's own per-tick event list."""
        for a in ally.accolades:
            if a == "Shot":
                self.tally.ally_shots += 1
            elif a in ("Save", "EpicSave"):
                self.tally.ally_saves += 1
            elif a == "Goal":
                self.tally.ally_goals += 1
            elif a == "Demolition":
                self.tally.ally_demos += 1

    # --- derived strategy knobs ------------------------------------------
    # These are what the decision layer actually reads.

    @property
    def chase_index(self) -> float:
        """
        Overall ball-chasing tendency in [0,1], from both available signals.

        Pursuit (`chase_rate`) catches the player who contests balls they
        should leave. Occupancy (`ball_glue`) catches the player who simply
        never leaves the ball, who by construction scores zero on pursuit.
        Taking the stronger of the two means either pattern is detected.

        Occupancy is rescaled first: being near the ball roughly half the time
        is normal in 2v2, so 0.5 maps to 0 and 1.0 maps to 1.
        """
        pursuit = clamp(self.traits.chase_rate, 0.0, 1.0)
        occupancy = clamp((self.traits.ball_glue - 0.5) * 2.0, 0.0, 1.0)
        return max(pursuit, occupancy)

    @property
    def defer_bias(self) -> float:
        """
        How much extra time-advantage we demand before taking a ball the human
        could also take. A heavy ball-chaser gets more room, because fighting
        them for it just loses possession for both of us.

        Returns seconds, roughly 0.05 to 0.6.
        """
        return 0.05 + 0.55 * self.chase_index

    @property
    def cover_bias(self) -> float:
        """
        How strongly we should sit back and cover. High when the human chases
        hard, rotates poorly, or floats upfield while we defend.
        """
        poor_rotation = 1.0 - clamp(self.traits.rotation_discipline, 0.0, 1.0)
        upfield = 1.0 - clamp(self.traits.defensive_depth, 0.0, 1.0)
        return clamp(0.45 * self.chase_index + 0.35 * poor_rotation + 0.20 * upfield, 0.0, 1.0)

    @property
    def take_kickoff_bias(self) -> float:
        """
        1 = we should take kickoffs, 0 = the human reliably takes them and we
        should set up behind instead.
        """
        return 1.0 - clamp(self.traits.kickoff_commit, 0.0, 1.0)

    @property
    def leave_boost_bias(self) -> float:
        """
        How much we should avoid taking big pads the human is heading for.
        High when they habitually run empty.
        """
        return clamp(1.0 - self.traits.boost_economy * 2.2, 0.0, 1.0)

    @property
    def preferred_side(self) -> float:
        """
        Which side WE should occupy: the opposite of the human's habit, so we
        naturally spread the field. Returns -1 or +1.
        """
        return -1.0 if self.traits.side_bias >= 0.0 else 1.0

    def summary(self) -> str:
        t = self.traits
        return (
            f"chase={t.chase_rate:.2f} glue={t.ball_glue:.2f} rot={t.rotation_discipline:.2f} "
            f"kickoff={t.kickoff_commit:.2f} boost={t.boost_economy:.2f} "
            f"aerial={t.aerial_rate:.2f} pace={t.pace:.2f} rel_speed={t.relative_speed:+.2f}s "
            f"-> chase_idx={self.chase_index:.2f} defer={self.defer_bias:.2f} cover={self.cover_bias:.2f}"
        )

    def end_match(self):
        self.traits.matches += 1
        self.save()
