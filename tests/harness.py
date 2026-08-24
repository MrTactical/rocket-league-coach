"""
Headless test harness.

Builds synthetic GamePackets, FieldInfos and BallPredictions so the bot's
decision-making can be exercised without launching Rocket League. The ball
prediction here is a simple gravity+bounce integrator, which is accurate
enough to validate intercepts and role choices.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlbot import flat  # noqa: E402

from bot.core.constants import BALL_RADIUS, GRAVITY  # noqa: E402
from bot.core.game import GameState  # noqa: E402

# Standard soccar boost pad layout: the six big pads plus a representative
# spread of small ones. Enough for boost-routing logic to be meaningful.
BIG_PADS = [
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-3584.0, 0.0, 73.0),
    (3584.0, 0.0, 73.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
]
SMALL_PADS = [
    (0.0, -4240.0, 70.0), (-1792.0, -4184.0, 70.0), (1792.0, -4184.0, 70.0),
    (-940.0, -3308.0, 70.0), (940.0, -3308.0, 70.0), (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0), (3584.0, -2484.0, 70.0),
    (-1788.0, -1024.0, 70.0), (1788.0, -1024.0, 70.0), (0.0, -1024.0, 70.0),
    (-2048.0, 1024.0, 70.0), (2048.0, 1024.0, 70.0), (0.0, 1024.0, 70.0),
    (-1788.0, 2484.0, 70.0), (1788.0, 2484.0, 70.0),
    (-940.0, 3308.0, 70.0), (940.0, 3308.0, 70.0), (0.0, 2816.0, 70.0),
    (-1792.0, 4184.0, 70.0), (1792.0, 4184.0, 70.0), (0.0, 4240.0, 70.0),
]


def make_field_info() -> flat.FieldInfo:
    pads = [flat.BoostPad(flat.Vector3(*p), True) for p in BIG_PADS]
    pads += [flat.BoostPad(flat.Vector3(*p), False) for p in SMALL_PADS]
    goals = [
        flat.GoalInfo(0, flat.Vector3(0, -5120, 321), flat.Vector3(0, 1, 0), 1786.0, 642.775),
        flat.GoalInfo(1, flat.Vector3(0, 5120, 321), flat.Vector3(0, -1, 0), 1786.0, 642.775),
    ]
    return flat.FieldInfo(boost_pads=pads, goals=goals, tiles=[])


FIELD_INFO = make_field_info()


def car(
    x=0.0, y=0.0, z=17.0, vx=0.0, vy=0.0, vz=0.0,
    yaw=0.0, pitch=0.0, roll=0.0,
    boost=34.0, team=0, name="car", is_bot=True,
    on_ground=True, demolished=False, can_dodge=True,
) -> flat.PlayerInfo:
    return flat.PlayerInfo(
        physics=flat.Physics(
            location=flat.Vector3(x, y, z),
            rotation=flat.Rotator(pitch, yaw, roll),
            velocity=flat.Vector3(vx, vy, vz),
            angular_velocity=flat.Vector3(0, 0, 0),
        ),
        hitbox=flat.BoxShape(118.0, 84.2, 36.2),
        hitbox_offset=flat.Vector3(13.9, 0.0, 20.8),
        air_state=flat.AirState.OnGround if on_ground else flat.AirState.InAir,
        dodge_timeout=1.2 if can_dodge else -1.0,
        demolished_timeout=3.0 if demolished else -1.0,
        is_bot=is_bot,
        name=name,
        team=team,
        boost=boost,
    )


def ball(x=0.0, y=0.0, z=BALL_RADIUS, vx=0.0, vy=0.0, vz=0.0) -> flat.BallInfo:
    return flat.BallInfo(
        physics=flat.Physics(
            location=flat.Vector3(x, y, z),
            velocity=flat.Vector3(vx, vy, vz),
            angular_velocity=flat.Vector3(0, 0, 0),
        ),
        shape=flat.SphereShape(BALL_RADIUS),
    )


def packet(
    cars: list[flat.PlayerInfo],
    the_ball: flat.BallInfo | None = None,
    time: float = 10.0,
    phase=flat.MatchPhase.Active,
    score=(0, 0),
) -> flat.GamePacket:
    if the_ball is None:
        the_ball = ball()
    pad_states = [flat.BoostPadState(True, 0.0) for _ in FIELD_INFO.boost_pads]
    return flat.GamePacket(
        players=cars,
        boost_pads=pad_states,
        balls=[the_ball],
        match_info=flat.MatchInfo(
            seconds_elapsed=time,
            game_time_remaining=300.0,
            match_phase=phase,
            world_gravity_z=GRAVITY,
            game_speed=1.0,
            frame_num=int(time * 120),
        ),
        teams=[flat.TeamInfo(0, score[0]), flat.TeamInfo(1, score[1])],
    )


def make_prediction(b: flat.BallInfo, now: float, horizon: float = 6.0) -> flat.BallPrediction:
    """
    Gravity + floor-bounce integration of the ball. Not the game's exact model
    (no spin, no wall geometry beyond the floor) but close enough that
    intercept timing tests are meaningful.
    """
    dt = 1.0 / 120.0
    px, py, pz = b.physics.location.x, b.physics.location.y, b.physics.location.z
    vx, vy, vz = b.physics.velocity.x, b.physics.velocity.y, b.physics.velocity.z
    slices = []
    t = now
    # Ball air drag in Rocket League is about 0.0305 per second.
    air = 1.0 - 0.0305 * dt
    for _ in range(int(horizon / dt)):
        vz += GRAVITY * dt
        vx *= air
        vy *= air
        vz *= air
        px += vx * dt
        py += vy * dt
        pz += vz * dt
        if pz < BALL_RADIUS:
            pz = BALL_RADIUS
            if vz < -60.0:
                # A real bounce: restitution plus surface friction.
                vz = -vz * 0.6
                vx *= 0.97
                vy *= 0.97
            else:
                # Resting/rolling. Without this branch a ball sitting on the
                # floor "bounces" every tick and has its horizontal speed
                # multiplied by 0.97 at 120Hz, which stops it almost instantly.
                vz = 0.0
                # Once rolling, a Rocket League ball loses very little speed --
                # a firmly struck ball rolls the length of the pitch. Keep this
                # small; an over-damped roll makes shots look harmless and
                # suppresses the threat signal defenders key off.
                roll = 1.0 - 0.05 * dt
                vx *= roll
                vy *= roll
        t += dt
        slices.append(
            flat.PredictionSlice(
                t,
                flat.Physics(
                    location=flat.Vector3(px, py, pz),
                    velocity=flat.Vector3(vx, vy, vz),
                    angular_velocity=flat.Vector3(0, 0, 0),
                ),
            )
        )
    return flat.BallPrediction(slices)


def state_from(pkt: flat.GamePacket, my_index: int = 0, prev_time: float = 0.0) -> GameState:
    team = pkt.players[my_index].team
    return GameState(pkt, my_index, team, FIELD_INFO, prev_time)


def scenario(
    me: flat.PlayerInfo,
    ally: flat.PlayerInfo | None = None,
    opponents: list[flat.PlayerInfo] | None = None,
    the_ball: flat.BallInfo | None = None,
    time: float = 10.0,
    phase=flat.MatchPhase.Active,
):
    """Assemble a full scenario; returns (state, prediction)."""
    cars = [me]
    if ally is not None:
        cars.append(ally)
    if opponents:
        cars.extend(opponents)
    pkt = packet(cars, the_ball, time=time, phase=phase)
    st = state_from(pkt, 0, time - 1.0 / 120.0)
    pred = make_prediction(pkt.balls[0], time)
    return st, pred
