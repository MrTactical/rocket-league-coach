"""
Training director.

An RLBot script that turns a self-play match into a drill machine. It does not
control a car -- it manipulates the world so the Allies in the match see far
more useful situations per minute than ordinary play would ever produce.

Why this matters more than adding cars. Self-calibration learns from touches,
and ordinary play is mostly *not* touching the ball: driving, rotating,
waiting for a kickoff. Measured from a real match, the bot touched the ball
roughly 2-30 times a minute depending on circumstances. This rig resets the
ball into a fresh, deliberately chosen scenario the moment the previous one
stops being interesting, so nearly every second is a touch opportunity.

Three multipliers stack:

  cars       six Allies pooling observations           ~6x
  drills     no dead time between useful situations    ~3-5x
  speed      game_speed above 1.0                      ~2-4x

They compose, so an hour of this is worth a great many hours of normal play.

Set `ALLY_TRAIN_SPEED` to change the game speed multiplier (default 2.0).
Above about 4x the physics stay correct but bots running Python can start
missing ticks, which teaches them the wrong lesson -- so it is capped.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlbot import flat  # noqa: E402
from rlbot.managers import Script  # noqa: E402

MAX_SPEED_MULT = 4.0
DEFAULT_SPEED_MULT = 2.0

# A drill is abandoned when it stops being useful.
DRILL_MAX_SECONDS = 9.0
BALL_IDLE_SPEED = 220.0
BALL_IDLE_SECONDS = 1.6

BOOST_ON_RESET = 60.0


def _v(x, y, z):
    return flat.Vector3Partial(x, y, z)


class Drill:
    """
    One scenario: where the ball starts and roughly where the cars go.

    `placement` overrides the default scatter when a drill is about a
    SPECIFIC shape rather than a contested ball. It takes (team, index within
    that team, ball position) and returns (x, y) for that car, or None to fall
    back to the scatter.
    """

    def __init__(self, name, ball_pos, ball_vel, spread=2200.0, height_bias=0.0,
                 placement=None, note=""):
        self.name = name
        self.ball_pos = ball_pos
        self.ball_vel = ball_vel
        self.spread = spread
        self.height_bias = height_bias
        self.placement = placement
        self.note = note


def _exposed_placement(team, slot, ball):
    """
    The state the coach measured as costing the most goals.

    Across 30 Champion matches a frame where you are ahead of the ball runs
    2.78x the base chance of conceding within six seconds -- but ONLY while
    the opponent has it in your half. During your own attack the same position
    is 0.41x, safer than average. So this drill builds the expensive half
    exactly: blue is defending, the ball is in blue's half with orange on it,
    and blue's first man is stranded upfield past it.

    Blue slot 0 is the one being trained: it starts beaten and has to recover.
    """
    bx, by = ball[0], ball[1]
    if team == 0:                       # defending, own net at -5120
        if slot == 0:
            return (bx + 700.0, by + 2100.0)      # upfield, ahead of the ball
        if slot == 1:
            return (bx - 1500.0, by - 900.0)      # covering, but wide
        return (bx + 400.0, by - 2600.0)          # last man, deep
    # attacking side: one on the ball, the rest supporting behind it
    if slot == 0:
        return (bx + 260.0, by + 520.0)
    return (bx - 900.0 + slot * 600.0, by + 1800.0)


def build_drills() -> list[Drill]:
    """
    A spread of situations that exercise different parts of the striker.

    Deliberately weighted toward things the bot is bad at rather than an even
    sample of match play: rolling balls are easy and already well calibrated,
    while bouncing and airborne balls are where the timing error lives.
    """
    d = []
    # Rolling balls across the middle -- baseline ground striking.
    for sign in (1, -1):
        d.append(Drill("roll_cross", (sign * 1800, 0, 93), (-sign * 900, 350, 0)))
    # Balls running toward a goal: clears and saves.
    d.append(Drill("attack_run", (0, -1200, 93), (150, 1500, 0)))
    d.append(Drill("defend_run", (0, 1200, 93), (-150, -1500, 0)))
    # Bouncing balls -- timing is hardest here.
    d.append(Drill("bounce_mid", (600, 0, 900), (-300, 400, -200)))
    d.append(Drill("bounce_deep", (-1400, 1800, 1100), (200, -600, -150)))
    # High balls: aerial opportunities.
    d.append(Drill("lofted", (0, 500, 1500), (250, -300, 0), height_bias=1.0))
    # Corner and wall play.
    d.append(Drill("corner", (3300, 3600, 300), (-400, -500, 0)))
    d.append(Drill("wall", (3900, -800, 900), (-200, 300, 0)))
    # Loose ball scramble in front of goal.
    d.append(Drill("scramble", (300, -3200, 200), (-200, -300, 0), spread=1500.0))
    # The measured one: beaten, ball in your half, opponent on it. Weighted
    # three times because it is the state the numbers say costs the goals,
    # and because recovering from it is a habit rather than a touch.
    for _ in range(3):
        d.append(Drill("exposed_recover", (400, -1900, 93), (-120, -900, 0),
                       placement=_exposed_placement,
                       note="you are ahead of the ball while they have it in "
                            "your half -- 2.78x concede risk; turn and go"))
    return d


class Trainer(Script):
    def __init__(self):
        super().__init__("joe/ally-trainer")
        self.drills = build_drills()
        self.rng = random.Random()
        self.speed = self._speed_from_env()

        self._drill_started = 0.0
        self._idle_since: float | None = None
        self._current: Drill | None = None
        self._resets = 0
        self._last_score = None
        self._speed_applied = False
        self._next_index = 0

    @staticmethod
    def _speed_from_env() -> float:
        raw = os.environ.get("ALLY_TRAIN_SPEED", "")
        try:
            v = float(raw) if raw else DEFAULT_SPEED_MULT
        except ValueError:
            v = DEFAULT_SPEED_MULT
        return max(0.5, min(v, MAX_SPEED_MULT))

    def initialize(self):
        self.logger.info(
            "Trainer ready | %d drills | game speed x%.1f",
            len(self.drills), self.speed,
        )

    def handle_packet(self, packet: flat.GamePacket):
        info = packet.match_info
        if info.match_phase not in (flat.MatchPhase.Active, flat.MatchPhase.Kickoff):
            return
        if not packet.balls or not packet.players:
            return

        now = info.seconds_elapsed

        # Apply the speed multiplier once the match is actually live.
        if not self._speed_applied:
            self._speed_applied = True
            self.set_game_state(match_info=flat.DesiredMatchInfo(game_speed=self.speed))
            self.logger.info("Game speed set to x%.1f", self.speed)

        if self._current is None:
            self._reset(packet, now)
            return

        ball = packet.balls[0].physics
        speed = math.sqrt(
            ball.velocity.x ** 2 + ball.velocity.y ** 2 + ball.velocity.z ** 2
        )

        # Ball has gone quiet: nothing more to learn from this one.
        if speed < BALL_IDLE_SPEED and ball.location.z < 200.0:
            if self._idle_since is None:
                self._idle_since = now
            elif now - self._idle_since > BALL_IDLE_SECONDS:
                self._reset(packet, now)
                return
        else:
            self._idle_since = None

        # Drill has run long enough.
        if now - self._drill_started > DRILL_MAX_SECONDS:
            self._reset(packet, now)
            return

        # A goal ends the scenario immediately -- do not wait for the kickoff.
        score = tuple(t.score for t in packet.teams)
        if self._last_score is not None and score != self._last_score:
            self._last_score = score
            self._reset(packet, now)
            return
        self._last_score = score

    # --- scenario setup ---------------------------------------------------

    def _reset(self, packet: flat.GamePacket, now: float):
        # Cycle rather than sample randomly, so every drill gets equal
        # attention instead of the RNG under-sampling the rare ones.
        drill = self.drills[self._next_index % len(self.drills)]
        self._next_index += 1
        self._current = drill
        self._drill_started = now
        self._idle_since = None
        self._resets += 1

        bx, by, bz = drill.ball_pos
        vx, vy, vz = drill.ball_vel
        # Jitter so the bots learn the situation, not one memorised position.
        j = self.rng.uniform
        ball = flat.DesiredBallState(
            flat.DesiredPhysics(
                location=_v(bx + j(-250, 250), by + j(-250, 250), max(93.0, bz + j(-80, 80))),
                velocity=_v(vx + j(-150, 150), vy + j(-150, 150), vz),
                angular_velocity=_v(j(-2, 2), j(-2, 2), j(-2, 2)),
            )
        )

        cars: dict[int, flat.DesiredCarState] = {}
        for i, player in enumerate(packet.players):
            # Spread each team behind the ball on its own side, so the drill
            # starts as a contest rather than a free hit.
            sign = -1.0 if player.team == 0 else 1.0
            spot = None
            if drill.placement is not None:
                slot = sum(1 for q in packet.players[:i] if q.team == player.team)
                spot = drill.placement(player.team, slot, drill.ball_pos)
            if spot is not None:
                x = spot[0] + j(-200, 200)
                y = spot[1] + j(-200, 200)
            else:
                lane = (i // 2) - 1
                x = lane * 1400.0 + j(-350, 350)
                y = by + sign * (drill.spread + j(-400, 400))
            x = max(-3900.0, min(3900.0, x))
            y = max(-5000.0, min(5000.0, y))
            yaw = math.atan2(by - y, bx - x)

            cars[i] = flat.DesiredCarState(
                physics=flat.DesiredPhysics(
                    location=_v(x, y, 17.0),
                    rotation=flat.RotatorPartial(pitch=0.0, yaw=yaw, roll=0.0),
                    velocity=_v(0.0, 0.0, 0.0),
                    angular_velocity=_v(0.0, 0.0, 0.0),
                ),
                boost_amount=BOOST_ON_RESET,
            )

        self.set_game_state(balls={0: ball}, cars=cars)

        if self._resets % 20 == 0:
            self.logger.info(
                "%d drills run (latest: %s) at x%.1f speed",
                self._resets, drill.name, self.speed,
            )

    def retire(self):
        self.logger.info("Trainer finished after %d drills", self._resets)


if __name__ == "__main__":
    Trainer().run()
