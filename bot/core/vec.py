"""
Small, allocation-cheap 3D math for the tick loop.

Deliberately plain Python rather than numpy: these are 3-element vectors
evaluated a few hundred times per tick, and numpy's per-array overhead is far
worse than tuple-sized float math at this scale.
"""

from __future__ import annotations

import math

__all__ = ["Vec3", "Mat3", "clamp", "lerp", "sign", "angle_diff"]


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


def angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a-b, wrapped to [-pi, pi]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        # Accept a flat.Vector3 / any object exposing .x/.y/.z as the sole arg.
        if hasattr(x, "x"):
            self.x = float(x.x)
            self.y = float(x.y)
            self.z = float(getattr(x, "z", 0.0))
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    # --- construction helpers ---------------------------------------------

    def copy(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    # --- operators --------------------------------------------------------

    def __add__(self, o: Vec3) -> Vec3:
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: Vec3) -> Vec3:
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> Vec3:
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f"Vec3({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"

    # --- products ---------------------------------------------------------

    def dot(self, o: Vec3) -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: Vec3) -> Vec3:
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    # --- magnitude --------------------------------------------------------

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def dist(self, o: Vec3) -> float:
        dx = self.x - o.x
        dy = self.y - o.y
        dz = self.z - o.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def dist_sq(self, o: Vec3) -> float:
        dx = self.x - o.x
        dy = self.y - o.y
        dz = self.z - o.z
        return dx * dx + dy * dy + dz * dz

    def normalized(self) -> Vec3:
        n = self.length()
        if n < 1e-9:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / n, self.y / n, self.z / n)

    def rescale(self, new_len: float) -> Vec3:
        """Same direction, given magnitude."""
        return self.normalized() * new_len

    def capped(self, max_len: float) -> Vec3:
        n = self.length()
        if n > max_len and n > 1e-9:
            s = max_len / n
            return Vec3(self.x * s, self.y * s, self.z * s)
        return self.copy()

    # --- 2D helpers (ground plane) ----------------------------------------

    def flat(self) -> Vec3:
        """Projection onto the ground plane."""
        return Vec3(self.x, self.y, 0.0)

    def flat_length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def flat_dist(self, o: Vec3) -> float:
        dx = self.x - o.x
        dy = self.y - o.y
        return math.sqrt(dx * dx + dy * dy)

    def to_yaw(self) -> float:
        """Heading of this vector in the ground plane."""
        return math.atan2(self.y, self.x)

    def angle_to(self, o: Vec3) -> float:
        """Unsigned angle between two vectors, in radians."""
        d = self.normalized().dot(o.normalized())
        return math.acos(clamp(d, -1.0, 1.0))

    def rotate_2d(self, angle: float) -> Vec3:
        c, s = math.cos(angle), math.sin(angle)
        return Vec3(self.x * c - self.y * s, self.x * s + self.y * c, self.z)


ZERO = Vec3(0.0, 0.0, 0.0)


class Mat3:
    """
    Car orientation as three orthonormal basis vectors in world space.

    Uses Rocket League's convention: `left` is the car's +Y axis. Built from a
    (pitch, yaw, roll) Rotator with the standard RLBot rotation matrix.
    """

    __slots__ = ("forward", "left", "up")

    def __init__(self, forward: Vec3, left: Vec3, up: Vec3):
        self.forward = forward
        self.left = left
        self.up = up

    @classmethod
    def from_rotator(cls, rot) -> Mat3:
        pitch = float(rot.pitch)
        yaw = float(rot.yaw)
        roll = float(rot.roll)

        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        forward = Vec3(cp * cy, cp * sy, sp)
        left = Vec3(
            cy * sp * sr - cr * sy,
            sy * sp * sr + cr * cy,
            -cp * sr,
        )
        up = Vec3(
            -cr * cy * sp - sr * sy,
            -cr * sy * sp + sr * cy,
            cp * cr,
        )
        return cls(forward, left, up)

    def to_local(self, world: Vec3) -> Vec3:
        """World-space vector -> car-relative (forward, left, up) components."""
        return Vec3(world.dot(self.forward), world.dot(self.left), world.dot(self.up))

    def to_world(self, local: Vec3) -> Vec3:
        """Car-relative vector -> world space."""
        return self.forward * local.x + self.left * local.y + self.up * local.z
