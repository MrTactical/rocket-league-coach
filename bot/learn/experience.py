"""
Shared experience: learning from what other bots did, not just what you did.

Self-calibration fixes *mechanics* -- am I aiming where I think I am. This
fixes *choices* -- was shooting from that corner ever a good idea. It is a
contextual bandit: the situation is discretised into a bucket, each bucket
records how well each available action has worked out, and choices are biased
toward what has actually paid off.

The part that makes it interesting with six Allies in a match is that the
experience is pooled. When one bot shoots from the left corner and the ball
comes straight back for a counter, every other Ally learns that too, without
having to make the mistake itself. Six cars exploring in parallel fill the
table roughly six times faster than one, and every bot benefits from the whole
team's mistakes.

Outcomes are deliberately judged a couple of seconds after the touch rather
than instantly. Whether a shot was a good idea is not visible at the moment of
contact -- it is visible when you see where the ball ended up and who has it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..core.constants import BACK_WALL_Y, SIDE_WALL_X
from ..core.vec import clamp

EXPERIENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "experience.json"

# How long after a touch we wait before judging it.
JUDGE_DELAY = 2.5
# Give up on attributing an outcome after this.
JUDGE_TIMEOUT = 5.0

# Exploration: how often to try something other than the current best.
# Decays as a bucket fills, so early play explores and later play exploits.
MIN_EXPLORE = 0.05
EXPLORE_AT_ZERO = 0.6
EXPLORE_HALFLIFE = 12.0

# A bucket needs this many samples before its estimate is trusted at all.
MIN_SAMPLES = 4


def situation_key(state, ball_pos, pressure: float) -> str:
    """
    Discretise the situation into a bucket label.

    Coarse on purpose. Too fine a split and no bucket ever collects enough
    samples to say anything; these four dimensions give 3x3x2x2 = 36 buckets,
    which a few matches can actually fill.
    """
    depth = ball_pos.y * state.goal_sign
    if depth > BACK_WALL_Y * 0.33:
        third = "def"
    elif depth < -BACK_WALL_Y * 0.33:
        third = "att"
    else:
        third = "mid"

    x = ball_pos.x
    if x < -SIDE_WALL_X * 0.33:
        lane = "L"
    elif x > SIDE_WALL_X * 0.33:
        lane = "R"
    else:
        lane = "C"

    height = "air" if ball_pos.z > 300.0 else "gnd"
    press = "press" if pressure > 0.45 else "free"
    return f"{third}/{lane}/{height}/{press}"


@dataclass
class Stat:
    n: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


class ExperienceTable:
    """
    bucket -> action -> running mean outcome.

    Outcomes are in [-1, 1]: +1 is a goal for us, -1 a goal against, with
    possession changes in between.
    """

    def __init__(self, persist: bool = True, seed: int | None = None):
        self.table: dict[str, dict[str, Stat]] = {}
        self.persist = persist
        self.rng = random.Random(seed)
        self.own_samples = 0
        self.peer_samples = 0
        self.load()

    # --- recording --------------------------------------------------------

    def record(self, key: str, action: str, reward: float, own: bool = True, n: int = 1):
        bucket = self.table.setdefault(key, {})
        stat = bucket.setdefault(action, Stat())
        stat.n += n
        stat.total += reward * n
        if own:
            self.own_samples += n
        else:
            self.peer_samples += n

    # --- querying ---------------------------------------------------------

    def value(self, key: str, action: str) -> float | None:
        stat = self.table.get(key, {}).get(action)
        if stat is None or stat.n < MIN_SAMPLES:
            return None
        return stat.mean

    def samples(self, key: str) -> int:
        return sum(s.n for s in self.table.get(key, {}).values())

    def explore_rate(self, key: str) -> float:
        """Explore a lot in an unfamiliar bucket, little in a well-known one."""
        n = self.samples(key)
        decay = 0.5 ** (n / EXPLORE_HALFLIFE)
        return max(MIN_EXPLORE, EXPLORE_AT_ZERO * decay)

    def choose(self, key: str, options: list[str], default: str) -> tuple[str, str]:
        """
        Pick an action for this situation.

        Returns (action, reason) -- the reason is for the on-screen overlay so
        the choice is legible while watching.
        """
        if not options:
            return default, "no options"

        if self.rng.random() < self.explore_rate(key):
            pick = self.rng.choice(options)
            return pick, f"explore({self.samples(key)}n)"

        scored = [(self.value(key, a), a) for a in options]
        known = [(v, a) for v, a in scored if v is not None]
        if not known:
            return default, "unseen"

        best_v, best_a = max(known)
        # Only override the situational default if the evidence is meaningful.
        default_v = self.value(key, default)
        if default_v is not None and best_v - default_v < 0.08:
            return default, f"default({default_v:+.2f})"
        return best_a, f"learned({best_v:+.2f})"

    # --- pooling ----------------------------------------------------------

    def export(self, limit: int = 12) -> list[dict]:
        """
        Most-informative buckets, for broadcasting to peers.

        Bounded so a message stays small; the busiest buckets carry the most
        useful information anyway.
        """
        rows = []
        for key, actions in self.table.items():
            for action, stat in actions.items():
                if stat.n >= 2:
                    rows.append({"k": key, "a": action, "n": stat.n,
                                 "v": round(stat.mean, 3)})
        rows.sort(key=lambda r: -r["n"])
        return rows[:limit]

    def merge(self, rows: list[dict]):
        """
        Fold in a peer's experience.

        Peers are also merging ours, so their counts partly reflect our own
        samples coming back. Discounting the incoming count stops that echo
        inflating confidence in a bucket nobody has really explored.
        """
        for r in rows:
            try:
                key = str(r["k"])
                action = str(r["a"])
                n = max(1, int(r["n"]) // 2)  # discount for echo
                v = float(r["v"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-1.5 <= v <= 1.5):
                continue
            self.record(key, action, v, own=False, n=n)

    # --- persistence ------------------------------------------------------

    def load(self) -> bool:
        if not self.persist:
            return False
        try:
            if EXPERIENCE_PATH.exists():
                raw = json.loads(EXPERIENCE_PATH.read_text(encoding="utf-8"))
                for key, actions in raw.get("table", {}).items():
                    self.table[key] = {
                        a: Stat(int(d["n"]), float(d["total"]))
                        for a, d in actions.items()
                    }
                self.own_samples = int(raw.get("own_samples", 0))
                self.peer_samples = int(raw.get("peer_samples", 0))
                return True
        except Exception:
            pass
        return False

    def save(self):
        if not self.persist:
            return
        try:
            EXPERIENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "own_samples": self.own_samples,
                "peer_samples": self.peer_samples,
                "table": {
                    key: {a: {"n": s.n, "total": round(s.total, 4)}
                          for a, s in actions.items()}
                    for key, actions in self.table.items()
                },
            }
            EXPERIENCE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except Exception:
            pass

    def summary(self) -> str:
        buckets = len(self.table)
        total = sum(s.n for a in self.table.values() for s in a.values())
        best = ""
        ranked = []
        for key, actions in self.table.items():
            for action, stat in actions.items():
                if stat.n >= MIN_SAMPLES:
                    ranked.append((stat.mean, stat.n, key, action))
        if ranked:
            ranked.sort()
            worst_v, worst_n, worst_k, worst_a = ranked[0]
            best_v, best_n, best_k, best_a = ranked[-1]
            best = (f" | best {best_k}:{best_a}={best_v:+.2f}({best_n}) "
                    f"worst {worst_k}:{worst_a}={worst_v:+.2f}({worst_n})")
        return (f"{buckets} buckets, {total} samples "
                f"({self.own_samples} own / {self.peer_samples} pooled){best}")


class OutcomeWatcher:
    """
    Attributes a reward to a decision, a couple of seconds after the fact.

    Holds at most one pending judgement -- if a new touch happens first the
    previous one is judged early on what has happened so far, which is more
    honest than discarding it.
    """

    def __init__(self, table: ExperienceTable):
        self.table = table
        self._pending: tuple[str, str, float, float] | None = None  # key, action, at, ball_depth
        self.last_reward: float | None = None
        self.judged = 0

    def note_decision(self, state, key: str, action: str):
        if self._pending is not None and self._pending[0] == key and self._pending[1] == action:
            return
        if self._pending is not None:
            self._judge(state, early=True)
        depth = state.ball.pos.y * state.goal_sign if state.ball else 0.0
        self._pending = (key, action, state.time, depth)

    def update(self, state, scored_for: bool = False, scored_against: bool = False):
        if self._pending is None:
            return
        key, action, at, _ = self._pending

        # A goal settles it immediately, in either direction.
        if scored_for or scored_against:
            self._finish(key, action, 1.0 if scored_for else -1.0)
            return

        elapsed = state.time - at
        if elapsed >= JUDGE_DELAY:
            self._judge(state)
        elif elapsed > JUDGE_TIMEOUT:
            self._pending = None

    def _judge(self, state, early: bool = False):
        if self._pending is None:
            return
        key, action, at, start_depth = self._pending
        self._finish(key, action, self._reward(state, start_depth) * (0.6 if early else 1.0))

    def _reward(self, state, start_depth: float) -> float:
        """
        How well did that turn out?

        Territory plus possession: did the ball move away from our goal, and
        does anyone on our side have a claim on it now.
        """
        if state.ball is None:
            return 0.0
        depth = state.ball.pos.y * state.goal_sign
        # Negative depth is toward their goal, so a drop in depth is progress.
        territory = clamp((start_depth - depth) / (2 * BACK_WALL_Y), -0.6, 0.6)

        ball = state.ball.pos
        mine = min((c.pos.dist(ball) for c in [state.me] + list(state.teammates)
                    if not c.is_demolished), default=9e9)
        theirs = min((c.pos.dist(ball) for c in state.opponents
                      if not c.is_demolished), default=9e9)
        if theirs < 9e8 and mine < 9e8:
            possession = clamp((theirs - mine) / 3000.0, -0.4, 0.4)
        else:
            possession = 0.0

        return clamp(territory + possession, -1.0, 1.0)

    def _finish(self, key: str, action: str, reward: float):
        self.table.record(key, action, reward, own=True)
        self.last_reward = reward
        self.judged += 1
        self._pending = None
