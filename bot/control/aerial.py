"""
Air control: orientation and powered aerials.

Sign derivation (worth reading before touching the mapping below).
`Mat3.left` is the axis pointing +Y at yaw=0, and `left = up x forward`.

  * Positive `pitch` input raises the nose. Nose-up tilts forward toward up,
    which needs angular velocity w with w x forward = up. Taking w = -left:
    (-y) x x = z = up. So positive pitch => angular velocity about -left, i.e.
    a NEGATIVE local-y component.  Cross-check: a front flip is pitch = -1,
    and a front flip pitches the nose down. Consistent.
  * Positive `yaw` turns the nose right. Right is -left, and (-up) x forward
    = -left. So positive yaw => NEGATIVE local-z component.
  * Positive `roll` drops the right wing, tilting up toward -left, and
    forward x up = -left. So positive roll => POSITIVE local-x component.

Hence: roll = +dx, pitch = -dy, yaw = -dz.
"""

from __future__ import annotations

import math

from rlbot import flat

from ..core.constants import BOOST_ACCEL, GRAVITY
from ..core.vec import Mat3, Vec3, clamp

# Proportional gain converting orientation error into a target angular
# velocity, and the gain converting angular-velocity error into stick input.
ORIENT_P = 6.0
OMEGA_GAIN = 0.55


def rotation_error(car, target_forward: Vec3, target_up: Vec3 | None = None) -> Vec3:
    """
    Axis-angle orientation error, expressed in the car's own frame.

    Returns a vector whose direction is the rotation axis and whose magnitude
    is the angle in radians. Uses a full axis-angle solve rather than the usual
    small-angle shortcut, because the shortcut degenerates to zero error when
    the car is pointing exactly backwards.
    """
    f = target_forward.normalized()
    up_hint = target_up if target_up is not None else Vec3(0.0, 0.0, 1.0)

    # Gram-Schmidt the up hint against forward; fall back if they're parallel.
    u = up_hint - f * up_hint.dot(f)
    if u.length_sq() < 1e-6:
        alt = Vec3(0.0, 0.0, 1.0) if abs(f.z) < 0.9 else Vec3(1.0, 0.0, 0.0)
        u = alt - f * alt.dot(f)
    u = u.normalized()
    l = u.cross(f)

    # Target basis expressed in car-local coordinates.
    tf = car.ori.to_local(f)
    tl = car.ori.to_local(l)
    tu = car.ori.to_local(u)

    # Axis-angle of that local rotation matrix.
    w = Vec3(tl.z - tu.y, tu.x - tf.z, tf.y - tl.x)
    sin_a = w.length() * 0.5
    cos_a = (tf.x + tl.y + tu.z - 1.0) * 0.5
    angle = math.atan2(sin_a, cos_a)

    if sin_a < 1e-6:
        if cos_a > 0.0:
            return Vec3(0.0, 0.0, 0.0)  # already aligned

        # Exactly 180 degrees out. The axis is NOT free here, despite being
        # undefined in the cross-product form: it is fixed by the matrix, and
        # returning an arbitrary one is actively harmful. Composing a 180
        # degree error with a rotation about a perpendicular axis gives another
        # exact 180 degree error, so a wrong guess makes the state absorbing --
        # the car holds full pitch for ever and never rights itself.
        #
        # M has columns tf, tl, tu, so at 180 degrees M + I is symmetric and
        # equals 2*n*n^T. Take its largest-diagonal column and normalise.
        # For an inverted car this yields roll, which also has 3.2x the torque
        # authority of pitch.
        dx, dy, dz = tf.x + 1.0, tl.y + 1.0, tu.z + 1.0
        if dx >= dy and dx >= dz:
            axis = Vec3(dx, tf.y, tf.z)
        elif dy >= dz:
            axis = Vec3(tl.x, dy, tl.z)
        else:
            axis = Vec3(tu.x, tu.y, dz)
        if axis.length_sq() < 1e-12:
            return Vec3(math.pi, 0.0, 0.0)  # degenerate: roll has most authority
        return axis.normalized() * math.pi

    return w.normalized() * angle


def orient_controls(
    car, target_forward: Vec3, target_up: Vec3 | None = None
) -> tuple[float, float, float]:
    """PD stick inputs (pitch, yaw, roll) that rotate the car into alignment."""
    err = rotation_error(car, target_forward, target_up)
    desired_omega = err * ORIENT_P
    omega_local = car.ori.to_local(car.ang_vel)
    d = desired_omega - omega_local

    roll = clamp(d.x * OMEGA_GAIN, -1.0, 1.0)
    pitch = clamp(-d.y * OMEGA_GAIN, -1.0, 1.0)
    yaw = clamp(-d.z * OMEGA_GAIN, -1.0, 1.0)
    return pitch, yaw, roll


def apply_orientation(c: flat.ControllerState, car, forward: Vec3, up: Vec3 | None = None):
    c.pitch, c.yaw, c.roll = orient_controls(car, forward, up)


class Aerial:
    """
    A powered aerial to a point at a time.

    Stateful across ticks: it holds the jump at the start to gain height, then
    points the nose at the thrust vector and burns boost. `step` returns None
    once the manoeuvre is spent, and callers should re-plan.
    """

    def __init__(self, target: Vec3, arrival_time: float, double_jump: bool = True):
        self.target = target
        self.arrival_time = arrival_time
        self.double_jump = double_jump
        self.started = False
        self.start_time = 0.0
        self.jump_released = False
        self.double_jumped = False

    def step(self, state) -> flat.ControllerState | None:
        car = state.me
        now = state.time
        c = flat.ControllerState()

        if not self.started:
            self.started = True
            self.start_time = now

        time_left = self.arrival_time - now
        if time_left <= -0.1:
            return None

        elapsed = now - self.start_time
        delta = self.target - car.pos

        # Thrust needed to close the gap in the time remaining, with gravity
        # already accounted for.
        if time_left > 0.02:
            grav = Vec3(0.0, 0.0, GRAVITY)
            needed = (delta - car.vel * time_left - grav * (0.5 * time_left * time_left))
            needed = needed * (2.0 / (time_left * time_left))
        else:
            needed = delta

        # --- launch phase: hold jump briefly for height ---------------------
        if car.on_ground and elapsed < 0.20:
            c.jump = True
            c.pitch, c.yaw, c.roll = 0.0, 0.0, 0.0
            c.throttle = 1.0
            return c

        if elapsed < 0.25:
            c.jump = False
            self.jump_released = True

        # Second jump adds vertical impulse early in the aerial.
        if (
            self.double_jump
            and not self.double_jumped
            and self.jump_released
            and 0.25 <= elapsed < 0.35
            and not car.on_ground
        ):
            c.jump = True
            self.double_jumped = True

        # --- point at the thrust vector and burn ---------------------------
        mag = needed.length()
        aim = needed if mag > 1e-3 else car.vel
        apply_orientation(c, car, aim, Vec3(0.0, 0.0, 1.0))

        # Boost once ROUGHLY pointed, not once perfectly pointed.
        #
        # Requiring 0.85 alignment deadlocked the entire manoeuvre. Leaving the
        # ground the nose is horizontal while the thrust vector points steeply
        # up, so alignment starts near 0.2: the car will not boost until it is
        # aligned, but without boost it never rises, and falling back to the
        # floor is the only thing it ever achieves. Measured across 20 real
        # aerial episodes -- median boost spent 0, median height gained 0uu.
        #
        # 0.55 is loose enough to break the deadlock and still tight enough
        # that we are not thrusting sideways. Once the window gets short we
        # burn regardless: a late aerial that is slightly off-axis still
        # reaches the ball, whereas a perfectly aimed one that never fired
        # does not.
        alignment = car.ori.forward.dot(aim.normalized()) if mag > 1e-3 else 0.0
        urgent = time_left < 0.6
        c.boost = (
            car.boost > 0
            and mag > BOOST_ACCEL * 0.10
            and (alignment > 0.55 or (urgent and alignment > 0.2))
        )
        c.throttle = 1.0  # keeps the nose stable and prevents auto-flip

        return c

    # Abort only when the remaining thrust genuinely exceeds what boost can
    # deliver. ReliefBot brackets its launch decision on this ratio; in flight
    # we only need the upper bound.
    ABORT_ACCEL_RATIO = 0.95

    def is_viable(self, state) -> bool:
        """
        Can this aerial still be completed from where the car is NOW?

        Deliberately not `aerial_feasible`. That function answers a different
        question -- "could I START this aerial from here?" -- and it credits the
        jump only for a car still on the ground. Calling it mid-flight makes it
        strictly harsher than the test that authorised the takeoff, so every
        aerial aborted on its first airborne tick: one episode per match,
        0 boost spent.

        Here we simply ask whether the thrust still required is within what
        boost can produce.
        """
        car = state.me
        T = self.arrival_time - state.time
        if T <= 0.02:
            return False
        if car.boost <= 0:
            return False

        delta = self.target - car.pos
        grav = Vec3(0.0, 0.0, GRAVITY)
        needed = (delta - car.vel * T - grav * (0.5 * T * T)) * (2.0 / (T * T))
        return needed.length() / BOOST_ACCEL < self.ABORT_ACCEL_RATIO
