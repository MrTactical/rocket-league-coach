"""
A friendlier view over the raw RLBot packet.

Everything downstream reads `GameState` rather than `flat.GamePacket`, so the
strategy code deals in Vec3/Mat3 and named roles instead of raw indices.
"""

from __future__ import annotations

from rlbot import flat

from .constants import BALL_RADIUS, own_goal_sign
from .vec import Mat3, Vec3


class Car:
    """One player's car, with the derived quantities we need every tick."""

    __slots__ = (
        "index", "name", "team", "is_human", "boost",
        "pos", "vel", "ang_vel", "ori", "rot",
        "speed", "forward_speed", "on_ground", "air_state",
        "can_dodge", "is_demolished", "is_supersonic",
        "hitbox", "latest_touch", "accolades", "has_flip_reset",
        "dodge_elapsed", "jumped",
    )

    def __init__(self, info: flat.PlayerInfo, index: int):
        p = info.physics
        self.index = index
        self.name = info.name
        self.team = info.team
        self.is_human = not info.is_bot
        self.boost = info.boost

        self.pos = Vec3(p.location)
        self.vel = Vec3(p.velocity)
        self.ang_vel = Vec3(p.angular_velocity)
        self.rot = p.rotation
        self.ori = Mat3.from_rotator(p.rotation)

        self.speed = self.vel.length()
        self.forward_speed = self.vel.dot(self.ori.forward)

        self.air_state = info.air_state
        self.on_ground = info.air_state == flat.AirState.OnGround
        self.is_demolished = info.demolished_timeout > 0.0
        self.is_supersonic = info.is_supersonic

        # Whether a dodge is available to us right now.
        #
        # NOT simply `dodge_timeout > 0`. RLBot documents dodge_timeout as
        # "-1 while on ground", so that test is false for every grounded car --
        # which made `car.on_ground and car.can_dodge`, the guard on every
        # dodge in this bot, a contradiction that could never be true. The
        # result was zero flips, zero half-flips and no power in any touch:
        # measured across a full match, the Dodge manoeuvre executed on 0 of
        # 7,300 ticks while 89.5% of intercepts were asking for one.
        #
        # A car on the ground can always dodge: it jumps, then dodges. The
        # timeout only describes the remaining window once already airborne.
        self.can_dodge = not self.is_demolished and (
            info.dodge_timeout > 0.0 or self.on_ground
        )
        self.jumped = info.has_jumped
        self.dodge_elapsed = info.dodge_elapsed

        # Airborne with every jump still unused == flip reset held.
        self.has_flip_reset = (
            info.air_state == flat.AirState.InAir
            and not info.has_jumped
            and not info.has_double_jumped
            and not info.has_dodged
        )

        self.hitbox = info.hitbox
        self.latest_touch = info.latest_touch
        self.accolades = info.accolades

    def to_ball_axis(self, ball_pos: Vec3) -> Vec3:
        return ball_pos - self.pos

    def facing_toward(self, target: Vec3) -> float:
        """Cosine of the angle between car forward and the direction to target."""
        return self.ori.forward.dot((target - self.pos).normalized())


class Ball:
    __slots__ = ("pos", "vel", "ang_vel", "radius")

    def __init__(self, info: flat.BallInfo):
        p = info.physics
        self.pos = Vec3(p.location)
        self.vel = Vec3(p.velocity)
        self.ang_vel = Vec3(p.angular_velocity)
        self.radius = BALL_RADIUS


class BoostPad:
    __slots__ = ("pos", "is_big", "is_active", "timer", "index")

    def __init__(self, index: int, static: flat.BoostPad, state: flat.BoostPadState):
        self.index = index
        self.pos = Vec3(static.location)
        self.is_big = static.is_full_boost
        self.is_active = state.is_active
        self.timer = state.timer


class GameState:
    """
    Per-tick snapshot. Constructed fresh each frame; the objects are small
    enough that reuse would cost more in complexity than it saves in GC.
    """

    __slots__ = (
        "packet", "time", "dt", "phase", "is_kickoff", "is_active",
        "me", "ally", "teammates", "opponents", "all_cars",
        "ball", "boost_pads", "big_pads",
        "team", "own_goal", "enemy_goal", "goal_sign",
        "score_us", "score_them",
    )

    def __init__(
        self,
        packet: flat.GamePacket,
        my_index: int,
        my_team: int,
        field_info: flat.FieldInfo,
        prev_time: float,
    ):
        self.packet = packet
        self.time = packet.match_info.seconds_elapsed
        self.dt = max(1e-4, self.time - prev_time) if prev_time else 1.0 / 120.0

        self.phase = packet.match_info.match_phase
        self.is_kickoff = self.phase == flat.MatchPhase.Kickoff
        self.is_active = self.phase in (flat.MatchPhase.Active, flat.MatchPhase.Kickoff)

        self.team = my_team
        self.goal_sign = own_goal_sign(my_team)

        cars = [Car(p, i) for i, p in enumerate(packet.players)]
        self.all_cars = cars
        self.me = cars[my_index]

        self.teammates = [c for c in cars if c.team == my_team and c.index != my_index]
        self.opponents = [c for c in cars if c.team != my_team]

        # Our human partner: the non-bot teammate. Falls back to any teammate so
        # the bot still behaves sanely in bot-only test matches.
        human = next((c for c in self.teammates if c.is_human), None)
        self.ally = human or (self.teammates[0] if self.teammates else None)

        self.ball = Ball(packet.balls[0]) if packet.balls else None

        pads = []
        static_pads = field_info.boost_pads
        for i, state in enumerate(packet.boost_pads):
            if i < len(static_pads):
                pads.append(BoostPad(i, static_pads[i], state))
        self.boost_pads = pads
        self.big_pads = [p for p in pads if p.is_big]

        # Own goal is the one we defend; enemy goal is what we shoot at.
        from .constants import BACK_WALL_Y, GOAL_HEIGHT

        self.own_goal = Vec3(0.0, self.goal_sign * BACK_WALL_Y, GOAL_HEIGHT * 0.5)
        self.enemy_goal = Vec3(0.0, -self.goal_sign * BACK_WALL_Y, GOAL_HEIGHT * 0.5)

        teams = packet.teams
        self.score_us = teams[my_team].score if my_team < len(teams) else 0
        self.score_them = teams[1 - my_team].score if (1 - my_team) < len(teams) else 0

    # --- convenience ------------------------------------------------------

    @property
    def has_ally(self) -> bool:
        return self.ally is not None and not self.ally.is_demolished

    def ball_in_own_half(self) -> bool:
        return self.ball is not None and self.ball.pos.y * self.goal_sign > 0.0

    def dist_to_own_goal(self, p: Vec3) -> float:
        return p.dist(self.own_goal)

    def dist_to_enemy_goal(self, p: Vec3) -> float:
        return p.dist(self.enemy_goal)

    def nearest_opponent_to(self, p: Vec3) -> Car | None:
        alive = [c for c in self.opponents if not c.is_demolished]
        return min(alive, key=lambda c: c.pos.dist_sq(p)) if alive else None
