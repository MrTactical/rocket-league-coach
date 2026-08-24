"""
Rocket League physics and field constants.

All values are in Rocket League's native "unreal units" (uu). These are the
community-verified constants from RLBot/RLUtilities; do not tweak them to
change bot behaviour -- they describe the game, not our strategy. Tunables
live in config/ally.toml instead.
"""

import math

# --- Field geometry -------------------------------------------------------

SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CEILING_Z = 2044.0

# The 45-degree corner walls satisfy |x| + |y| == CORNER_WALL_DIAG
CORNER_WALL_DIAG = 8064.0

GOAL_HEIGHT = 642.775
GOAL_HALF_WIDTH = 892.755
# Depth the goal extends behind the back wall.
GOAL_DEPTH = 880.0

BALL_RADIUS = 91.25

# Centre of each team's own goal mouth. Index by team (0 = blue, 1 = orange).
GOAL_CENTER = (
    (0.0, -BACK_WALL_Y, GOAL_HEIGHT / 2),
    (0.0, BACK_WALL_Y, GOAL_HEIGHT / 2),
)


def goal_center(team: int) -> tuple[float, float, float]:
    """Centre of `team`'s own goal."""
    return GOAL_CENTER[team]


def own_goal_sign(team: int) -> float:
    """-1 if this team defends the -Y goal, +1 if it defends +Y."""
    return -1.0 if team == 0 else 1.0


# --- Car physics ----------------------------------------------------------

MAX_SPEED = 2300.0
SUPERSONIC_THRESHOLD = 2200.0
MAX_SPEED_NO_BOOST = 1410.0

BOOST_ACCEL = 991.667  # uu/s^2 while boosting
BOOST_USAGE_PER_SEC = 33.3  # boost units consumed per second

BRAKE_ACCEL = 3500.0  # throttle opposing velocity
COAST_ACCEL = 525.0  # passive deceleration with no throttle

GRAVITY = -650.0

# Jumping
JUMP_IMPULSE = 291.667  # instantaneous dv on jump press
JUMP_HOLD_ACCEL = 1458.333  # extra accel while jump is held
JUMP_MAX_HOLD_TIME = 0.2
# Time after the first jump during which a second jump / dodge is available.
DODGE_WINDOW = 1.25
# Minimum airtime before a dodge is allowed to register cleanly.
DODGE_MIN_DELAY = 0.05

# Dodging: the flip adds a fixed horizontal impulse relative to car forward.
DODGE_IMPULSE = 500.0
DODGE_TORQUE_TIME = 0.65  # total duration of the dodge animation lock

# Angular control (torque coefficients / damping) -- used by the aerial PD
# controller. Values from RLUtilities' car model.
PITCH_TORQUE = 12.146
YAW_TORQUE = 9.11
ROLL_TORQUE = 38.34
PITCH_DAMPING = -2.798
YAW_DAMPING = -1.886
ROLL_DAMPING = -4.589

MAX_ANGULAR_SPEED = 5.5  # rad/s

# Default Octane-ish hitbox. The real hitbox comes from PlayerInfo.hitbox at
# runtime; this is only a fallback for offline math/tests.
OCTANE_HITBOX = (118.007, 84.2, 36.159)  # length, width, height
OCTANE_HITBOX_OFFSET = (13.88, 0.0, 20.755)


# --- Throttle acceleration curve -----------------------------------------
# Forward acceleration from throttle alone falls off to zero at 1410 uu/s.
_THROTTLE_CURVE = ((0.0, 1600.0), (1400.0, 160.0), (1410.0, 0.0))


def throttle_accel(speed: float) -> float:
    """Acceleration available from full throttle at the given forward speed."""
    speed = abs(speed)
    if speed >= MAX_SPEED_NO_BOOST:
        return 0.0
    for i in range(len(_THROTTLE_CURVE) - 1):
        s0, a0 = _THROTTLE_CURVE[i]
        s1, a1 = _THROTTLE_CURVE[i + 1]
        if speed <= s1:
            t = (speed - s0) / (s1 - s0)
            return a0 + t * (a1 - a0)
    return 0.0


# --- Steering curvature curve --------------------------------------------
# Maximum turn curvature (1/radius) the car can achieve at a given speed.
_CURVATURE_CURVE = (
    (0.0, 0.00690),
    (500.0, 0.00398),
    (1000.0, 0.00235),
    (1500.0, 0.001375),
    (1750.0, 0.00110),
    (2300.0, 0.00088),
)


def max_curvature(speed: float) -> float:
    """Tightest curvature (1/uu) the car can steer at this speed."""
    speed = min(max(abs(speed), 0.0), MAX_SPEED)
    for i in range(len(_CURVATURE_CURVE) - 1):
        s0, c0 = _CURVATURE_CURVE[i]
        s1, c1 = _CURVATURE_CURVE[i + 1]
        if speed <= s1:
            t = (speed - s0) / (s1 - s0)
            return c0 + t * (c1 - c0)
    return _CURVATURE_CURVE[-1][1]


def max_turn_radius(speed: float) -> float:
    """Tightest turn radius (uu) achievable at this speed."""
    c = max_curvature(speed)
    return 1.0 / c if c > 1e-9 else float("inf")


def speed_for_curvature(curvature: float) -> float:
    """
    Fastest speed at which the car can still hold the given curvature.
    Used to decide how much to slow down before a turn.
    """
    curvature = abs(curvature)
    if curvature >= _CURVATURE_CURVE[0][1]:
        return 0.0
    if curvature <= _CURVATURE_CURVE[-1][1]:
        return MAX_SPEED
    for i in range(len(_CURVATURE_CURVE) - 1):
        s0, c0 = _CURVATURE_CURVE[i]
        s1, c1 = _CURVATURE_CURVE[i + 1]
        # curvature decreases as speed increases
        if c1 <= curvature <= c0:
            t = (curvature - c0) / (c1 - c0)
            return s0 + t * (s1 - s0)
    return MAX_SPEED


# --- Boost pads -----------------------------------------------------------

BIG_BOOST_AMOUNT = 100.0
SMALL_BOOST_AMOUNT = 12.0
BIG_BOOST_RESPAWN = 10.0
SMALL_BOOST_RESPAWN = 4.0

# --- Misc -----------------------------------------------------------------

TICK_RATE = 120.0
TICK_DT = 1.0 / TICK_RATE

TWO_PI = 2.0 * math.pi
