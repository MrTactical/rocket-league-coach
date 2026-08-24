"""
Match recording.

Two streams are written per match: a downsampled positional trace, and a list
of discrete events. The trace is what makes questions like "where were you
when we conceded?" answerable after the fact; the events are what the coaching
report is built from.

Sampling is at 10Hz rather than the full 120Hz tick rate. At 120Hz a five
minute match is 36,000 rows of almost entirely redundant data; at 10Hz it is
3,000 rows and loses nothing that matters at the timescale of a rotation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

TELEMETRY_DIR = Path(__file__).resolve().parents[2] / "data" / "telemetry"

SAMPLE_HZ = 10.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_HZ
FLUSH_EVERY = 200  # rows buffered before touching disk


class MatchRecorder:
    """Buffers samples and events, flushing periodically."""

    def __init__(
        self,
        ally_name: str = "partner",
        rank: str = "diamond",
        enabled: bool = True,
        slot: int | None = None,
    ):
        self.enabled = enabled
        self.ally_name = ally_name
        self.rank = rank
        self.started = datetime.now()
        self.stamp = self.started.strftime("%Y%m%d-%H%M%S")

        # With several Allies in one match every instance starts inside the
        # same second, so a timestamp alone collides and all of them append to
        # one interleaved file. Tagging the car index keeps the streams
        # separate and makes per-bot analysis possible.
        self.slot = slot
        if slot is not None:
            self.stamp = f"{self.stamp}-b{slot}"

        self._samples: list[dict] = []
        self._events: list[dict] = []
        self._last_sample = 0.0
        self._pending = 0
        self._match_start_time: float | None = None
        self._closed = False

        # Set by the bot each tick before sample() so coordination state
        # lands in the trace alongside everything else.
        self.ordinal = 0
        self.peer_count = 0

        if self.enabled:
            try:
                TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.enabled = False

    # --- paths ------------------------------------------------------------

    @property
    def trace_path(self) -> Path:
        return TELEMETRY_DIR / f"{self.stamp}-trace.jsonl"

    @property
    def events_path(self) -> Path:
        return TELEMETRY_DIR / f"{self.stamp}-events.jsonl"

    @property
    def summary_path(self) -> Path:
        return TELEMETRY_DIR / f"{self.stamp}-summary.json"

    # --- recording --------------------------------------------------------

    def sample(self, state, debug, my_time: float, ally_time: float):
        if not self.enabled or not state.is_active:
            return
        if self._match_start_time is None:
            self._match_start_time = state.time
        if state.time - self._last_sample < SAMPLE_INTERVAL:
            return
        self._last_sample = state.time

        me = state.me
        ally = state.ally
        ball = state.ball

        row = {
            "t": round(state.time - (self._match_start_time or 0.0), 2),
            "role": debug.role,
            "act": debug.action,
            "threat": round(debug.threat, 3),
            "mt": round(my_time, 2),
            "at": round(ally_time, 2),
            "me": [round(me.pos.x), round(me.pos.y), round(me.pos.z), round(me.boost)],
            # Attitude. Without these, "airborne" can only be guessed from
            # height -- and most high-z ticks are driving up a wall with the
            # wheels planted, so every recovery measurement taken that way
            # was wrong. up.z tells upright from inverted, |w| tells settled
            # from tumbling, and gnd is ground truth.
            "ug": round(me.ori.up.z, 3),
            "w": round(me.ang_vel.length(), 2),
            "gnd": 1 if me.on_ground else 0,
            "ball": [round(ball.pos.x), round(ball.pos.y), round(ball.pos.z)] if ball else None,
            # Coordination state. Without these the analyser cannot tell a
            # broken ordinal from silent comms -- both look identical from
            # positions alone.
            "ord": self.ordinal,
            "peers": self.peer_count,
            "note": debug.note or "",
            "k": getattr(debug, "kind", ""),
            # CAREFUL: `adecl` fires only AFTER the arbiter has already chosen
            # an aerial -- it is the humanizer declining to fly one, not a
            # rejection by the search. A low count here says nothing about
            # whether aerials are being found. Reading it as though it did cost
            # a long detour once. `arej` below is the field that answers that.
            "adecl": 1 if getattr(debug, "aerial_declined", False) else 0,
            # Why no aerial was offered this tick: "" (none needed / taken),
            # "no_lead", "unreachable", "min_boost", "reserve", or
            # "slower_than_<kind>". `acost` is what the cheapest one would
            # have cost in boost.
            "arej": getattr(debug, "aerial_reject", "") or "",
            "acost": round(float(getattr(debug, "aerial_cost", 0.0) or 0.0), 1),
            "mech": getattr(debug, "mechanic", ""),
        }
        if ally is not None:
            row["ally"] = [round(ally.pos.x), round(ally.pos.y), round(ally.pos.z), round(ally.boost)]
            row["ally_v"] = round(ally.speed)

        self._samples.append(row)
        self._pending += 1
        if self._pending >= FLUSH_EVERY:
            self.flush()

    def event(self, kind: str, state, **extra):
        if not self.enabled:
            return
        row = {
            "t": round(state.time - (self._match_start_time or state.time), 2),
            "kind": kind,
            "score": [state.score_us, state.score_them],
        }
        me, ally, ball = state.me, state.ally, state.ball
        row["me"] = [round(me.pos.x), round(me.pos.y), round(me.boost)]
        if ally is not None:
            row["ally"] = [round(ally.pos.x), round(ally.pos.y), round(ally.boost)]
        if ball is not None:
            row["ball"] = [round(ball.pos.x), round(ball.pos.y), round(ball.pos.z)]
        row.update(extra)
        self._events.append(row)

    # --- output -----------------------------------------------------------

    def flush(self):
        if not self.enabled:
            return
        try:
            if self._samples:
                with self.trace_path.open("a", encoding="utf-8") as f:
                    for row in self._samples:
                        f.write(json.dumps(row) + "\n")
                self._samples.clear()
            if self._events:
                with self.events_path.open("a", encoding="utf-8") as f:
                    for row in self._events:
                        f.write(json.dumps(row) + "\n")
                self._events.clear()
            self._pending = 0
        except Exception:
            # Telemetry must never take the match down with it.
            self.enabled = False

    def close(self, model, final_score=(0, 0)):
        """Flush everything and write the match summary."""
        if not self.enabled or self._closed:
            return None
        self._closed = True
        self.flush()
        try:
            summary = {
                "stamp": self.stamp,
                "started": self.started.isoformat(timespec="seconds"),
                "ended": datetime.now().isoformat(timespec="seconds"),
                "ally_name": self.ally_name,
                "bot_rank": self.rank,
                "score": {"us": final_score[0], "them": final_score[1]},
                "tally": asdict(model.tally),
                "traits": asdict(model.traits),
                "derived": {
                    "defer_bias": round(model.defer_bias, 3),
                    "cover_bias": round(model.cover_bias, 3),
                    "take_kickoff_bias": round(model.take_kickoff_bias, 3),
                    "leave_boost_bias": round(model.leave_boost_bias, 3),
                },
            }
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return self.summary_path
        except Exception:
            return None
