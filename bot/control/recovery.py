"""
Recovery: getting the wheels back under the car after being airborne.

Points the car along its own velocity and levels the roof upward, so it lands
driving in the direction it is already moving rather than skidding sideways.
Converts the landing into a wavedash when that is cheap and useful.
"""

from __future__ import annotations

from rlbot import flat

from ..core.physics import predict_landing
from ..core.vec import Vec3
from .aerial import apply_orientation
from .dodge import Wavedash


class Recovery:
    """Stateful, so a wavedash begun on the way down survives across ticks."""

    def __init__(self, allow_wavedash: bool = True):
        self.allow_wavedash = allow_wavedash
        self.wavedash: Wavedash | None = None

    def step(self, state, aim: Vec3 | None = None) -> flat.ControllerState:
        car = state.me

        if self.wavedash is not None:
            out = self.wavedash.step(state)
            if out is not None:
                return out
            self.wavedash = None

        c = flat.ControllerState()
        c.throttle = 1.0

        _, t_land = predict_landing(car)

        # Face where we are going: velocity if we have meaningful speed,
        # otherwise toward whatever we were aiming for.
        forward = car.vel.flat()
        if forward.length_sq() < 250_000.0:  # under ~500 uu/s
            forward = (aim - car.pos).flat() if aim is not None else car.ori.forward.flat()
        if forward.length_sq() < 1.0:
            forward = Vec3(1.0, 0.0, 0.0)

        apply_orientation(c, car, forward, Vec3(0.0, 0.0, 1.0))

        # Low, upright and moving: turn the landing into speed.
        if (
            self.allow_wavedash
            and car.can_dodge
            and t_land < 0.35
            and car.ori.up.z > 0.75
            and car.vel.flat_length() > 500.0
        ):
            self.wavedash = Wavedash(forward)
            out = self.wavedash.step(state)
            if out is not None:
                return out

        return c

    def reset(self):
        self.wavedash = None


def needs_recovery(car) -> bool:
    """True when the car is airborne and not meaningfully upright."""
    if car.on_ground:
        return False
    return car.ori.up.z < 0.85
