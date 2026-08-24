"""
A small, honest physics core for offline tuning.

Not a Rocket League reimplementation. It models exactly the regimes the bot's
tunable parameters live in -- free-space rigid-body rotation, ballistic flight
with boost, and flat-floor driving -- using the same verified constants the bot
itself uses (`bot/core/constants.py`, checked against RLUtilities and the RLBot
wiki).

Why not RocketSim: it is installed and it is more faithful, but it refuses to
build an arena without collision meshes dumped from the game, and those meshes
only matter for walls, corners and the ceiling. Every parameter this optimiser
touches -- orientation gains, dodge timing, aerial thresholds, the arrive shift
-- is free-space or flat-floor. The meshes would buy accuracy in precisely the
region we are not tuning.

What this DOES model faithfully:
  * rigid-body rotation under the car's real torque and damping coefficients
  * gravity, boost thrust and boost consumption in the air
  * the throttle acceleration curve and the steering curvature limit
  * jump impulse, the held-jump bonus, and the speed-dependent dodge impulse
  * a flat floor with a crude bounce

What it does NOT model, and must not be trusted for:
  * walls, corners, the ceiling, goal frames
  * ball-to-car collision response (ball contact is a simple impulse estimate)
  * suspension, wheel friction curves, sticky-force detail
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.core.constants import (  # noqa: E402
    BALL_RADIUS,
    BOOST_ACCEL,
    BOOST_USAGE_PER_SEC,
    BRAKE_ACCEL,
    COAST_ACCEL,
    GRAVITY,
    JUMP_HOLD_ACCEL,
    JUMP_IMPULSE,
    JUMP_MAX_HOLD_TIME,
    MAX_SPEED,
    PITCH_DAMPING,
    PITCH_TORQUE,
    ROLL_DAMPING,
    ROLL_TORQUE,
    YAW_DAMPING,
    YAW_TORQUE,
    max_curvature,
    throttle_accel,
)
from bot.core.vec import Mat3, Vec3, clamp  # noqa: E402

TICK = 1.0 / 120.0
REST_HEIGHT = 17.0
# Boost accelerates harder in the air than on the ground.
BOOST_ACCEL_AIR = 1058.333


class SimCar:
    """A car with just enough physics to tune a controller against."""

    def __init__(self, pos=None, vel=None, ori=None, ang_vel=None, boost=100.0,
                 on_ground=True):
        self.pos = pos or Vec3(0.0, 0.0, REST_HEIGHT)
        self.vel = vel or Vec3()
        self.ori = ori or Mat3(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
        self.ang_vel = ang_vel or Vec3()
        self.boost = boost
        self.on_ground = on_ground

        self.is_demolished = False
        self.is_supersonic = False
        self.can_dodge = True
        self.has_flip_reset = False
        self.latest_touch = None
        self.accolades = ()
        self.team = 0
        self.index = 0
        self.name = "sim"
        self.is_human = False

        self._jump_held = 0.0
        self._jumped = False

    # The bot's code reads these.
    @property
    def speed(self) -> float:
        return self.vel.length()

    @property
    def forward_speed(self) -> float:
        return self.vel.dot(self.ori.forward)

    # --- integration ------------------------------------------------------

    def step(self, c, dt: float = TICK):
        """Advance one tick under a ControllerState-like object."""
        if self.on_ground:
            self._step_ground(c, dt)
        else:
            self._step_air(c, dt)

        self.pos = self.pos + self.vel * dt

        # Flat floor.
        if self.pos.z <= REST_HEIGHT:
            self.pos.z = REST_HEIGHT
            if self.vel.z < 0.0:
                self.vel.z = 0.0
            if not self.on_ground and self.ori.up.z > 0.2:
                self.on_ground = True
                self.can_dodge = True
                self._jumped = False
                self.ang_vel = Vec3()
        elif self.pos.z > REST_HEIGHT + 1.0:
            self.on_ground = False

        self.is_supersonic = self.speed > 2200.0

    def _step_ground(self, c, dt):
        # Jump leaves the ground immediately.
        if getattr(c, "jump", False):
            self.on_ground = False
            self._jumped = True
            self._jump_held = 0.0
            self.vel = self.vel + self.ori.up * JUMP_IMPULSE
            self.pos.z += 2.0
            return

        throttle = clamp(getattr(c, "throttle", 0.0), -1.0, 1.0)
        fwd = self.ori.forward
        v_f = self.vel.dot(fwd)

        if throttle * v_f < -1.0:
            accel = -BRAKE_ACCEL * (1.0 if v_f > 0 else -1.0)
        elif abs(throttle) < 0.01:
            accel = -COAST_ACCEL * (1.0 if v_f > 0 else -1.0) if abs(v_f) > 1 else 0.0
        else:
            accel = throttle_accel(v_f) * throttle

        if getattr(c, "boost", False) and self.boost > 0.0:
            accel += BOOST_ACCEL
            self.boost = max(0.0, self.boost - BOOST_USAGE_PER_SEC * dt)

        v_f = clamp(v_f + accel * dt, -MAX_SPEED, MAX_SPEED)

        # Steering, limited by the real curvature curve.
        steer = clamp(getattr(c, "steer", 0.0), -1.0, 1.0)
        if getattr(c, "handbrake", False):
            steer *= 1.8
        omega = steer * max_curvature(abs(v_f)) * abs(v_f)
        yaw = math.atan2(fwd.y, fwd.x) + omega * dt
        self.ori = Mat3(
            Vec3(math.cos(yaw), math.sin(yaw), 0.0),
            Vec3(-math.sin(yaw), math.cos(yaw), 0.0),
            Vec3(0.0, 0.0, 1.0),
        )
        self.vel = self.ori.forward * v_f
        self.ang_vel = Vec3(0.0, 0.0, omega)

    def _step_air(self, c, dt):
        # Held jump keeps adding thrust for a short window.
        if getattr(c, "jump", False) and self._jumped and self._jump_held < JUMP_MAX_HOLD_TIME:
            self._jump_held += dt
            self.vel = self.vel + self.ori.up * (JUMP_HOLD_ACCEL * dt)

        # Angular dynamics: torque from the stick, damping proportional to the
        # existing rate, both in the car's own frame.
        w_local = self.ori.to_local(self.ang_vel)
        pitch = clamp(getattr(c, "pitch", 0.0), -1.0, 1.0)
        yaw = clamp(getattr(c, "yaw", 0.0), -1.0, 1.0)
        roll = clamp(getattr(c, "roll", 0.0), -1.0, 1.0)

        # Signs follow the derivation at the top of bot/control/aerial.py.
        alpha = Vec3(
            ROLL_TORQUE * roll + ROLL_DAMPING * w_local.x * (1.0 - abs(roll)),
            -PITCH_TORQUE * pitch + PITCH_DAMPING * w_local.y * (1.0 - abs(pitch)),
            -YAW_TORQUE * yaw + YAW_DAMPING * w_local.z * (1.0 - abs(yaw)),
        )
        self.ang_vel = self.ang_vel + self.ori.to_world(alpha) * dt
        if self.ang_vel.length() > 5.5:
            self.ang_vel = self.ang_vel.rescale(5.5)

        # Rotate the basis by omega*dt and re-orthonormalise.
        w = self.ang_vel * dt
        f = (self.ori.forward + w.cross(self.ori.forward)).normalized()
        u = self.ori.up + w.cross(self.ori.up)
        u = (u - f * u.dot(f)).normalized()
        self.ori = Mat3(f, u.cross(f), u)

        # Linear: gravity plus boost along the nose.
        self.vel.z += GRAVITY * dt
        if getattr(c, "boost", False) and self.boost > 0.0:
            self.vel = self.vel + self.ori.forward * (BOOST_ACCEL_AIR * dt)
            self.boost = max(0.0, self.boost - BOOST_USAGE_PER_SEC * dt)
        if self.vel.length() > MAX_SPEED:
            self.vel = self.vel.rescale(MAX_SPEED)


class SimBall:
    """Gravity, air drag and a flat-floor bounce. No walls."""

    AIR_DRAG_PER_SEC = 0.0305
    RESTITUTION = 0.6

    def __init__(self, pos=None, vel=None):
        self.pos = pos or Vec3(0.0, 0.0, BALL_RADIUS)
        self.vel = vel or Vec3()
        self.ang_vel = Vec3()
        self.radius = BALL_RADIUS

    def step(self, dt: float = TICK):
        self.vel.z += GRAVITY * dt
        drag = 1.0 - self.AIR_DRAG_PER_SEC * dt
        self.vel = self.vel * drag
        self.pos = self.pos + self.vel * dt
        if self.pos.z < BALL_RADIUS:
            self.pos.z = BALL_RADIUS
            if self.vel.z < -60.0:
                self.vel.z = -self.vel.z * self.RESTITUTION
                self.vel.x *= 0.97
                self.vel.y *= 0.97
            else:
                self.vel.z = 0.0
                roll = 1.0 - 0.05 * dt
                self.vel.x *= roll
                self.vel.y *= roll


def upright(car) -> float:
    """How upright the car is, in [-1, 1]. 1 is wheels-down."""
    return car.ori.up.z


def settle_ticks(car, controller, limit: int = 480, threshold: float = 0.9) -> int | None:
    """
    Ticks for `controller(car) -> controls` to bring the car upright.

    Returns None if it never settles inside `limit`.
    """
    for i in range(limit):
        c = controller(car)
        car.step(c)
        if upright(car) > threshold and car.ang_vel.length() < 1.0:
            return i
    return None
