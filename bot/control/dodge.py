"""
Jump-based mechanics: flips, half-flips, speedflips and wavedashes.

These are timed sequences rather than closed-loop controllers, so each is a
small state machine driven by elapsed time. `step` returns None when the
manoeuvre is finished and the caller should go back to normal driving.

Convention reminder: a front flip is pitch = -1 (nose down).
"""

from __future__ import annotations

import math

from rlbot import flat

from ..core.vec import Vec3, clamp
from .aerial import apply_orientation

__all__ = ["Dodge", "HalfFlip", "SpeedFlip", "Wavedash", "flip_toward"]


def flip_toward(car, target: Vec3) -> tuple[float, float]:
    """
    Stick (pitch, yaw) for a dodge aimed at `target`, speed-compensated.

    A dodge fires along the stick direction in the car's own frame, so the
    world-space heading is converted to local space first. Local +x is forward,
    which is pitch -1; local +y is along `left`.

    The compensation matters. The impulse a dodge adds is speed-dependent --
    500 * (1 + 0.9 * speed/2300) uu/s -- and it is not applied evenly across
    the two axes, so at speed the car ends up flipping somewhere other than
    where it was aimed. RLUtilities' `dodge.cc` divides the requested direction
    through by the axis gains before normalising, so the RESULT matches what
    was asked for rather than the request doing.
    """
    local = car.ori.to_local((target - car.pos).normalized())
    x, y = local.x, local.y

    s = min(abs(car.forward_speed) / 2300.0, 1.0)
    if x < 0.0:
        # Backward component: the gains differ per axis, so undo them.
        x /= (16.0 / 15.0) * (1.0 + 1.5 * s)
        y /= 1.0 + 0.9 * s

    n = math.hypot(x, y)
    if n < 1e-6:
        return -1.0, 0.0
    return -x / n, y / n


class Dodge:
    """
    A standard dodge: jump, brief pause, then jump again with the stick pushed
    in the dodge direction.

    `direction` is a world-space vector; None means straight forward.
    """

    JUMP_TIME = 0.10       # how long the first jump is held
    PAUSE_TIME = 0.06      # gap before the second input
    DODGE_TIME = 0.20      # how long the stick is held into the dodge
    RECOVER_TIME = 0.70    # earliest we will hand back control
    MAX_TIME = 1.40        # hard ceiling, however the car is oriented

    # Hand back only once the wheels are pointing down enough to drive.
    UPRIGHT = 0.80

    def __init__(self, direction: Vec3 | None = None, target: Vec3 | None = None):
        self.direction = direction
        self.target = target
        self.start = None

    def step(self, state) -> flat.ControllerState | None:
        car = state.me
        if self.start is None:
            self.start = state.time
        t = state.time - self.start
        c = flat.ControllerState()
        c.throttle = 1.0

        if self.target is not None:
            pitch, yaw = flip_toward(car, self.target)
        elif self.direction is not None:
            pitch, yaw = flip_toward(car, car.pos + self.direction * 500.0)
        else:
            pitch, yaw = -1.0, 0.0

        if t < self.JUMP_TIME:
            c.jump = True
        elif t < self.JUMP_TIME + self.PAUSE_TIME:
            # Point the nose at the ball during the gap between the two jumps.
            # Botimus's AimDodge: the flip carries the car along its own axis,
            # so orienting first is where the extra power in a dodge strike
            # comes from -- an un-aimed flip mostly wastes the impulse.
            c.jump = False
            if self.target is not None:
                aim = (self.target - car.pos)
                if aim.length_sq() > 1.0:
                    apply_orientation(c, car, aim, Vec3(0.0, 0.0, 1.0))
        elif t < self.JUMP_TIME + self.PAUSE_TIME + self.DODGE_TIME:
            c.jump = True
            c.pitch = pitch
            c.yaw = yaw
        elif t < self.RECOVER_TIME:
            # Coming out of the flip, get the wheels back under us rather than
            # holding the stick and landing on the roof.
            c.jump = False
            if t < self.JUMP_TIME + self.PAUSE_TIME + self.DODGE_TIME + 0.12:
                c.pitch = pitch * 0.4
                c.yaw = yaw * 0.4
            else:
                forward = car.vel.flat()
                if forward.length_sq() < 1.0:
                    forward = car.ori.forward.flat()
                if forward.length_sq() < 1.0:
                    forward = Vec3(1.0, 0.0, 0.0)
                apply_orientation(c, car, forward, Vec3(0.0, 0.0, 1.0))
        elif t < self.MAX_TIME and not car.on_ground and car.ori.up.z < self.UPRIGHT:
            # Keep righting the car rather than handing it back mid-tumble.
            #
            # A flip imparts a lot of angular velocity and the fixed 0.70s
            # lifetime left only ~0.22s to absorb it, so the driver regularly
            # received a car that was still rotating -- and then fed it ground
            # steering while it was upside down. Measured: 69% of all airborne
            # time was spent inverted, in episodes with a median of 1.3s and a
            # tail out to 4.0s, at a median height of 99uu. That is the car
            # sliding along on its roof.
            #
            # The ceiling still applies: never hold control for ever, and give
            # up immediately once the wheels are back on the floor.
            forward = car.vel.flat()
            if forward.length_sq() < 1.0:
                forward = car.ori.forward.flat()
            if forward.length_sq() < 1.0:
                forward = Vec3(1.0, 0.0, 0.0)
            apply_orientation(c, car, forward, Vec3(0.0, 0.0, 1.0))
        else:
            return None

        return c


class HalfFlip:
    """
    Reverse direction quickly: dodge backwards, then cancel and roll upright
    while the car flips over. Faster than turning around when the target is
    behind us.
    """

    def __init__(self):
        self.start = None

    def step(self, state) -> flat.ControllerState | None:
        if self.start is None:
            self.start = state.time
        t = state.time - self.start
        c = flat.ControllerState()
        c.throttle = -1.0

        if t < 0.10:
            c.jump = True
        elif t < 0.16:
            c.jump = False
        elif t < 0.30:
            c.jump = True
            c.pitch = 1.0  # nose up == dodge backwards
        elif t < 0.55:
            c.jump = False
            c.pitch = 1.0
        elif t < 1.15:
            # Cancel the flip, then actively orient wheels-down.
            #
            # A fixed pitch/roll here is a guess: it rights the car only if it
            # happened to end the flip in the expected attitude, and otherwise
            # leaves it scrabbling on its roof. Driving the orientation
            # controller instead lands the car on its wheels from wherever the
            # flip actually left it, pointed where it is travelling.
            c.throttle = 1.0
            if t < 0.68:
                c.pitch = -1.0
            else:
                forward = state.me.vel.flat()
                if forward.length_sq() < 1.0:
                    forward = state.me.ori.forward.flat()
                if forward.length_sq() < 1.0:
                    forward = Vec3(1.0, 0.0, 0.0)
                apply_orientation(c, state.me, forward, Vec3(0.0, 0.0, 1.0))
        else:
            return None

        return c


class SpeedFlip:
    """
    Kickoff speedflip: a diagonal front flip with the handbrake tapped and the
    flip cancelled, which preserves nearly all forward speed. Worth roughly a
    car length of position at the first touch.

    `direction` is +1 to flip toward the car's right, -1 toward its left.
    """

    def __init__(self, direction: float = 1.0):
        self.dir = 1.0 if direction >= 0 else -1.0
        self.start = None

    def step(self, state) -> flat.ControllerState | None:
        car = state.me
        if self.start is None:
            self.start = state.time
        t = state.time - self.start
        c = flat.ControllerState()
        c.throttle = 1.0
        c.boost = True

        if t < 0.08:
            c.jump = True
            c.handbrake = True
        elif t < 0.14:
            c.jump = False
            c.handbrake = True
        elif t < 0.30:
            # Diagonal dodge: forward plus a sideways component.
            c.jump = True
            c.pitch = -1.0
            c.yaw = self.dir * 0.85
        elif t < 0.50:
            # Cancel: pull back to kill the flip's rotation before it lands.
            c.pitch = 1.0
            c.yaw = -self.dir * 0.3
        elif t < 0.90:
            aim = car.vel if car.vel.length() > 100.0 else car.ori.forward
            apply_orientation(c, car, aim, Vec3(0.0, 0.0, 1.0))
        else:
            return None

        return c


class Wavedash:
    """
    Land out of a jump with a forward dodge, converting the landing into speed.
    Used on recoveries when we are low and already moving forward.
    """

    def __init__(self, direction: Vec3 | None = None):
        self.direction = direction
        self.start = None
        self.dodged = False

    def step(self, state) -> flat.ControllerState | None:
        car = state.me
        if self.start is None:
            self.start = state.time
        t = state.time - self.start
        c = flat.ControllerState()
        c.throttle = 1.0

        if t > 1.2:
            return None
        if car.on_ground and t > 0.15:
            return None

        aim = self.direction or car.vel.flat()
        if aim.length_sq() < 1.0:
            aim = car.ori.forward.flat()

        # Stay slightly nose-down so the dodge lands flat and converts to speed.
        apply_orientation(c, car, aim, Vec3(0.0, 0.0, 1.0))
        c.pitch = clamp(c.pitch - 0.35, -1.0, 1.0)

        # Fire the dodge just before touching down.
        if not self.dodged and car.pos.z < 45.0 and car.vel.z < 0.0 and car.can_dodge:
            c.jump = True
            c.pitch = -1.0
            self.dodged = True

        return c
