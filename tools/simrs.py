"""
RocketSim-backed harness: the bot's real controllers against real arena geometry.

`tools/simcore.py` models free space and a flat floor. That was the honest
choice while no collision meshes existed, and it is still the right tool for
tuning orientation gains and dodge timing -- both of which it showed are
already near optimal.

It cannot explain the remaining defect. Telemetry says the cars spend most of
their non-grounded time on or near walls and corners, inverted. Free space has
no walls, so a free-space search will report every wall bug as "no problem
found". This module exists for exactly that gap.

The adapter is deliberately thin: it converts a RocketSim car into the
attribute surface `bot/` already reads, so the controllers under test are the
shipped ones, not reimplementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import RocketSim as rs  # noqa: E402

from bot.core.vec import Mat3, Vec3  # noqa: E402

TICK = 1.0 / 120.0


def to_vec3(v) -> Vec3:
    return Vec3(v.x, v.y, v.z)


def to_rs_vec(v) -> rs.Vec:
    return rs.Vec(v.x, v.y, v.z)


class RSCar:
    """A RocketSim car wearing the interface the bot's controllers expect."""

    def __init__(self, car, team: int = 0, index: int = 0, name: str = "rs"):
        self._car = car
        self.team = team
        self.index = index
        self.name = name
        self.is_human = False
        self.latest_touch = None
        self.accolades = ()
        self.has_flip_reset = False
        self.sync()

    def sync(self) -> None:
        s = self._car.get_state()
        self.pos = to_vec3(s.pos)
        self.vel = to_vec3(s.vel)
        self.ang_vel = to_vec3(s.ang_vel)
        m = s.rot_mat
        # RocketSim gives (forward, right, up); our Mat3 is (forward, left, up).
        self.ori = Mat3(to_vec3(m.forward), to_vec3(m.right) * -1.0, to_vec3(m.up))
        self.on_ground = bool(s.is_on_ground)
        self.boost = float(s.boost)
        self.is_demolished = bool(s.is_demoed)
        self.is_supersonic = bool(s.is_supersonic)
        # The bot's own rule: a dodge is available on the ground or inside the
        # post-jump window. RocketSim tracks this directly.
        self.can_dodge = bool(s.has_flip_or_jump)

        # Only RocketSim can supply these, and they are the point of this file.
        self.has_world_contact = bool(s.has_world_contact)
        self.contact_normal = to_vec3(s.world_contact_normal)
        self.wheels_with_contact = s.wheels_with_contact
        self.is_flipping = bool(s.is_flipping)

    @property
    def speed(self) -> float:
        return self.vel.length()

    @property
    def forward_speed(self) -> float:
        return self.vel.dot(self.ori.forward)


class RSState:
    """The slice of GameState the controllers actually read: `me` and `time`."""

    def __init__(self, me: RSCar, time: float = 0.0):
        self.me = me
        self.time = time
        self.dt = TICK
        self.teammates: list = []
        self.opponents: list = []
        self.ally = None


def to_rs_controls(c) -> rs.CarControls:
    out = rs.CarControls()
    out.throttle = float(getattr(c, "throttle", 0.0) or 0.0)
    out.steer = float(getattr(c, "steer", 0.0) or 0.0)
    out.pitch = float(getattr(c, "pitch", 0.0) or 0.0)
    out.yaw = float(getattr(c, "yaw", 0.0) or 0.0)
    out.roll = float(getattr(c, "roll", 0.0) or 0.0)
    out.jump = bool(getattr(c, "jump", False))
    out.boost = bool(getattr(c, "boost", False))
    out.handbrake = bool(getattr(c, "handbrake", False))
    return out


def make_arena():
    return rs.Arena(rs.GameMode.SOCCAR)


def place(car, pos: Vec3, vel: Vec3, ori: Mat3 | None = None,
          ang_vel: Vec3 | None = None, boost: float = 100.0) -> None:
    """Teleport a RocketSim car. Note the wake nudge caveat for the ball."""
    s = car.get_state()
    s.pos = to_rs_vec(pos)
    s.vel = to_rs_vec(vel)
    s.ang_vel = to_rs_vec(ang_vel or Vec3())
    if ori is not None:
        s.rot_mat = rs.RotMat(
            to_rs_vec(ori.forward), to_rs_vec(ori.left * -1.0), to_rs_vec(ori.up)
        )
    s.boost = boost
    s.has_jumped = False
    s.has_double_jumped = False
    s.has_flipped = False
    car.set_state(s)


def run(car, state_obj: RSCar, arena, controller, ticks: int, t0: float = 0.0,
        observe=None):
    """
    Drive `car` with `controller(state) -> ControllerState` for `ticks` ticks.

    `observe(tick, RSCar)` is called after every step, which is where the
    surface-contact measurements come from.
    """
    st = RSState(state_obj, time=t0)
    for i in range(ticks):
        state_obj.sync()
        st.time = t0 + i * TICK
        st.me = state_obj
        out = controller(st)
        car.set_controls(to_rs_controls(out))
        arena.step(1)
        state_obj.sync()
        if observe is not None:
            observe(i, state_obj)
    return state_obj


# --- arena geometry ------------------------------------------------------
#
# RocketSim reports `world_contact_normal` only for CHASSIS contact, and
# `wheels_with_contact` is a 4-tuple of bools rather than a count. Neither
# gives "what surface are the wheels on", which is the question that matters
# for recovery. Soccar geometry is simple enough to answer analytically.

WALL_X = 4096.0
WALL_Y = 5120.0
CEILING = 2044.0
# The four angled corner walls meet the flat walls along |x| + |y| = CORNER.
CORNER = 8064.0
ROOT2 = 1.4142135623730951


def nearest_surface(pos) -> tuple[Vec3, float]:
    """
    The normal of the closest arena surface, and the distance to it.

    Normals point INTO the arena, so a car correctly on a surface has
    `ori.up.dot(normal)` near 1 whether that surface is the floor, a wall,
    a corner or the ceiling.
    """
    best_n = Vec3(0.0, 0.0, 1.0)
    best_d = pos.z
    for n, d in (
        (Vec3(0.0, 0.0, -1.0), CEILING - pos.z),
        (Vec3(-1.0, 0.0, 0.0), WALL_X - pos.x),
        (Vec3(1.0, 0.0, 0.0), WALL_X + pos.x),
        (Vec3(0.0, -1.0, 0.0), WALL_Y - pos.y),
        (Vec3(0.0, 1.0, 0.0), WALL_Y + pos.y),
    ):
        if d < best_d:
            best_n, best_d = n, d
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            d = (CORNER - (sx * pos.x + sy * pos.y)) / ROOT2
            if d < best_d:
                best_n, best_d = Vec3(-sx / ROOT2, -sy / ROOT2, 0.0), d
    return best_n, best_d


def surface_alignment(car) -> float:
    """How well the roof matches the surface the car is nearest, in [-1, 1]."""
    n, _ = nearest_surface(car.pos)
    return car.ori.up.dot(n)
