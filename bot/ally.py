"""
Ally -- an adaptive Rocket League teammate.

Entry point that RLBot launches. Everything interesting lives in bot/brain and
bot/learn; this file is the wiring: build the game state, run the brain, detect
events, record them, and draw what the bot is thinking.

Runs in offline matches only -- RLBot connects to a local match through the
RLBotServer socket, which is exactly why it cannot be used in online play.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlbot import flat  # noqa: E402
from rlbot.managers import Bot  # noqa: E402

from bot.brain.comms import CommsHub  # noqa: E402
from bot.brain.decide import Brain  # noqa: E402
from bot.brain.humanize import (Humanizer, get_profile,  # noqa: E402
                                profile_from_file)
from bot.brain.teammate import TeammateModel  # noqa: E402
from bot.core.game import GameState  # noqa: E402
from bot.core.vec import Vec3  # noqa: E402
from bot.learn.coach import LiveCoach, analyse, format_report  # noqa: E402
from bot.learn.telemetry import MatchRecorder  # noqa: E402

CONFIG_PATH = ROOT / "config" / "ally.toml"

DEFAULTS = {
    "rank": "diamond",
    "seed": 0,
    "callouts": True,
    "telemetry": True,
    "rendering": True,
    "learning": True,
    "profile_name": "",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("rb") as f:
                data = tomllib.load(f)
            cfg.update(data.get("ally", {}))
    except Exception as e:  # a broken config should not stop the match
        print(f"[ally] could not read config ({e}); using defaults")
    # Environment overrides are handy for quick experiments.
    if os.environ.get("ALLY_RANK"):
        cfg["rank"] = os.environ["ALLY_RANK"]
    if os.environ.get("ALLY_PROFILE"):
        cfg["profile_file"] = os.environ["ALLY_PROFILE"]
    return cfg


class Ally(Bot):
    def __init__(self):
        super().__init__("joe/ally")
        self.cfg = load_config()

        self.model: TeammateModel | None = None
        self.brain: Brain | None = None
        self.coach: LiveCoach | None = None
        self.recorder: MatchRecorder | None = None
        self.comms: CommsHub | None = None

        self._prev_time = 0.0
        self._prev_score = (0, 0)
        self._ally_touch_at = -1.0
        self._my_touch_at = -1.0
        self._reported = False
        self._last_phase = None

    # --- lifecycle --------------------------------------------------------

    def initialize(self):
        cfg = self.cfg
        # A measured profile beats a rank label: "diamond" is an average of
        # everyone at that rank and nobody plays like the average. Generate
        # one with coach/profile_export.py and point `profile_file` at it.
        prof_file = cfg.get("profile_file")
        if prof_file:
            # Resolve against BOTH the bot directory and the repo root. The
            # config lives in bot/, so "profile-me.json" is natural there --
            # but the generator prints "bot/profile-me.json", which is natural
            # from the repo root, and joining that onto bot/ gives bot/bot/.
            # Accept either rather than making the user think about it.
            path = Path(prof_file)
            if not path.is_absolute():
                here = Path(__file__).resolve().parent
                for cand in (here / path, here.parent / path):
                    if cand.is_file():
                        path = cand
                        break
                else:
                    path = here / path      # let the loader log the miss
            profile = profile_from_file(path)
            rank = profile.name
            # Through the BOT's logger, not a bare module one -- RLBot
            # configures this and it actually reaches the console. The load
            # failure was visible only because warnings pass a default root
            # logger; the success was silent, which made "did the measured
            # profile actually load" unanswerable from the match output.
            m = getattr(profile, "measured", {}) or {}
            self.logger.info(
                "skill: MEASURED %r from %s matches (band %s, aerial %.2f, "
                "whiff %.3f, noise %.3f)",
                profile.name, m.get("matches", "?"), rank,
                profile.aerial_confidence, profile.whiff_chance,
                profile.input_noise)
        else:
            rank = str(cfg.get("rank", "diamond"))
            profile = get_profile(rank)
            self.logger.info("skill: preset %r (no measured profile set)", rank)

        # Name the profile after the human, so different people playing on
        # this machine get their own learned model.
        #
        # Their name cannot be read here: a Human entry in the match config
        # carries only a team, with no name field, and touching one throws.
        # It is only available at runtime from the packet, so unless the user
        # pinned a profile in config we start on a placeholder and adopt the
        # real one on the first tick (see _bind_partner).
        ally_name = str(cfg.get("profile_name") or "").strip()
        self._profile_pinned = bool(ally_name)
        self._partner_bound = self._profile_pinned

        self.model = TeammateModel(ally_name or "partner")
        loaded = self.model.path.exists()

        seed = int(cfg.get("seed", 0)) or None
        humanizer = Humanizer(profile, seed=seed)
        self.brain = Brain(self.model, humanizer, cfg)
        self.coach = LiveCoach(enabled=bool(cfg.get("callouts", True)))
        # Peer coordination with any other Ally instances in the match.
        self.comms = CommsHub(self.index, self.team)
        self.brain.comms = self.comms
        self.recorder = MatchRecorder(
            ally_name or "partner",
            rank,
            enabled=bool(cfg.get("telemetry", True)),
            slot=self.index,
        )

        self.logger.info(
            "Ally ready | rank=%s | partner=%s | reaction=%.0fms",
            rank,
            ally_name if ally_name else "(detect from packet)",
            profile.reaction_time * 1000,
        )
        if loaded:
            self.logger.info("Recalled: %s", self.model.summary())

    def retire(self):
        self._finish_match()

    # --- main tick --------------------------------------------------------

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        if self.brain is None or not packet.players:
            return flat.ControllerState()

        state = GameState(packet, self.index, self.team, self.field_info, self._prev_time)
        self._prev_time = state.time

        if not self._partner_bound:
            self._bind_partner(state)

        self._track_events(state)

        controls = self.brain.decide(state, self.ball_prediction)
        debug = self.brain.debug

        if self.recorder is not None:
            self.recorder.ordinal = self.brain.ordinal
            self.recorder.peer_count = (
                len(self.comms.teammates(state.time)) if self.comms else 0
            )
            self.recorder.sample(state, debug, debug.my_time, debug.ally_time)

        if self.coach is not None and state.has_ally:
            line = self.coach.check(state, debug, self.model, debug.my_time, debug.ally_time)
            if line:
                self.send_match_comm(b"", line, team_only=True)

        self._broadcast(state, debug)

        if self.cfg.get("rendering", True):
            self._render(state, debug)

        # Match over: write the report once.
        if packet.match_info.match_phase == flat.MatchPhase.Ended:
            self._finish_match()

        return controls

    def _bind_partner(self, state: GameState):
        """
        Adopt the human's profile once the packet tells us who they are.

        Runs on the first tick a teammate is visible. Harmless if there is no
        teammate (1v1 or free play) -- it simply keeps the placeholder.
        """
        ally = state.ally
        if ally is None or not ally.name:
            return
        self._partner_bound = True
        loaded = self.model.bind(ally.name)
        if self.recorder is not None:
            self.recorder.ally_name = ally.name
        self.logger.info(
            "Partner is '%s' (%s)",
            ally.name,
            "profile loaded" if loaded else "new profile",
        )
        if loaded:
            self.logger.info("Recalled: %s", self.model.summary())

    def handle_match_comm(self, index, team, content, display, team_only):
        """Inbound peer traffic. Treated as untrusted: never allowed to throw."""
        if self.comms is None or not content:
            return
        try:
            self.comms.receive(index, team, content, self._prev_time)
        except Exception:
            pass

    def _broadcast(self, state, debug):
        """Tell other Allies what we are doing, and share calibration."""
        if self.comms is None or self.brain is None:
            return
        now = state.time
        try:
            # A wrecked car must not publish an intent at all. During a dead
            # ball `decide` returns before my_time is solved, so it would send
            # tt = 0.0 -- "I am on the ball right now" -- and out-rank every
            # live teammate for the whole respawn.
            if not state.me.is_demolished and self.comms.should_send_intent(now):
                if debug.my_time > 0.0:
                    # Remember what we told them, so our own ranking uses the
                    # same number they are ranking us by.
                    self.brain.broadcast_time = debug.my_time
                tt = (
                    self.brain.broadcast_time
                    if self.brain.broadcast_time < 90.0
                    else debug.my_time
                )
                claiming = debug.role == "attack" and tt < 3.0
                # Remember this too, for the same reason as broadcast_time: the
                # aerial bonus must be applied to this car by every bot using
                # the same value, including by this car itself.
                self.brain.broadcast_aerial = self.brain.aerial_claim
                self.send_match_comm(
                    self.comms.build_intent(
                        debug.role, tt, claiming, self.brain.ordinal,
                        self.brain.boost_pad, self.brain.boost_eta,
                        aerial=self.brain.aerial_claim,
                    ),
                    None,
                    team_only=True,
                )
            cal = self.brain.calibrator
            if cal.c.contacts >= 2 and self.comms.should_send_calib(now):
                # Not team_only: every Ally has identical mechanics, so this is
                # shared engineering data, not tactical information.
                self.send_match_comm(self.comms.build_calib(cal.c), None, team_only=False)
            pending = self.comms.drain_calibration()
            if pending:
                cal.absorb(pending)

            exp = self.brain.experience
            if exp.own_samples >= 3 and self.comms.should_send_experience(now):
                rows = exp.export()
                if rows:
                    self.send_match_comm(
                        self.comms.build_experience(rows), None, team_only=False
                    )
            incoming = self.comms.drain_experience()
            if incoming:
                exp.merge(incoming)
        except Exception:
            pass

    # --- events -----------------------------------------------------------

    def _track_events(self, state: GameState):
        """Detect goals and touches by watching for changes between ticks."""
        model, rec = self.model, self.recorder
        if model is None:
            return

        # --- goals ---------------------------------------------------------
        score = (state.score_us, state.score_them)
        if score != self._prev_score:
            if score[0] > self._prev_score[0]:
                model.on_goal_scored(state)
                if self.brain is not None:
                    self.brain.outcomes.update(state, scored_for=True)
                if rec:
                    rec.event("goal_for", state)
            if score[1] > self._prev_score[1]:
                model.on_goal_conceded(state)
                if self.brain is not None:
                    self.brain.outcomes.update(state, scored_against=True)
                if rec:
                    rec.event("goal_against", state)
            self._prev_score = score

        # --- kickoff boundary ----------------------------------------------
        phase = state.phase
        if phase != self._last_phase:
            if phase == flat.MatchPhase.Kickoff and rec:
                rec.event("kickoff", state)
            self._last_phase = phase

        # --- touches --------------------------------------------------------
        ally = state.ally
        if ally is not None:
            model.note_accolades(ally)
            touch = ally.latest_touch
            if touch is not None and touch.game_seconds > self._ally_touch_at + 1e-3:
                self._ally_touch_at = touch.game_seconds
                pos = Vec3(touch.location)
                model.on_ally_touch(state, pos)
                if rec:
                    rec.event("ally_touch", state, z=round(pos.z))

        my_touch = state.me.latest_touch
        if my_touch is not None and my_touch.game_seconds > self._my_touch_at + 1e-3:
            self._my_touch_at = my_touch.game_seconds
            model.tally.my_touches += 1
            if rec:
                rec.event("my_touch", state, z=round(my_touch.location.z))

    # --- reporting --------------------------------------------------------

    def _finish_match(self):
        if self._reported or self.model is None:
            return
        self._reported = True

        try:
            if self.cfg.get("learning", True):
                self.model.end_match()
                if self.brain is not None:
                    self.brain.calibrator.save()
                    self.brain.experience.save()

            insights = analyse(self.model, self._prev_score)
            report = format_report(
                self.model, insights, self._prev_score, str(self.cfg.get("rank", ""))
            )
            print("\n" + report, flush=True)

            path = None
            if self.recorder is not None:
                path = self.recorder.close(self.model, self._prev_score)

            # Save the human-readable report next to the telemetry.
            if path is not None:
                report_path = path.with_name(path.name.replace("-summary.json", "-report.txt"))
                report_path.write_text(report, encoding="utf-8")
                self.logger.info("Report written to %s", report_path)
        except Exception as e:
            self.logger.error("Failed to write match report: %s", e)

    # --- rendering --------------------------------------------------------

    def _render(self, state: GameState, debug):
        r = self.renderer
        try:
            r.begin_rendering("ally")

            colour = {
                "attack": r.red,
                "support": r.yellow,
                "defend": r.cyan,
                "kickoff": r.lime,
            }.get(debug.role, r.white)

            lines = [
                f"{debug.role.upper()}  {debug.action}",
                f"threat {debug.threat:.2f}   me {debug.my_time:.1f}s  you {debug.ally_time:.1f}s",
            ]
            if debug.note:
                lines.append(debug.note)
            r.draw_string_3d(
                "\n".join(lines),
                flat.CarAnchor(self.index),
                0.6,
                colour,
                r.transparent,
                flat.TextHAlign.Center,
            )

            if debug.target is not None:
                t = flat.Vector3(debug.target.x, debug.target.y, max(debug.target.z, 20.0))
                r.draw_line_3d(flat.CarAnchor(self.index), t, colour)

            if debug.intercept is not None:
                ic = debug.intercept
                r.draw_line_3d(
                    flat.Vector3(ic.x, ic.y, 20.0),
                    flat.Vector3(ic.x, ic.y, max(ic.z, 120.0)),
                    r.orange,
                )

            r.end_rendering()
        except Exception:
            # Never let a rendering problem interrupt play.
            try:
                if r.is_rendering():
                    r.end_rendering()
            except Exception:
                pass


if __name__ == "__main__":
    Ally().run()
