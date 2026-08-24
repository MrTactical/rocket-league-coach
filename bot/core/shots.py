"""
Shot geometry: aiming at a goal mouth rather than at a point.

The idea, standard across the RLBot community (RLBot wiki, GoslingUtils
`post_correction`, VirxERLU-CLib `correct_for_posts`/`clamp2D`, Kamael
`is_shot_scorable`), is that a target is a PAIR of directions -- toward the
left post and toward the right post -- not a single centre point.

Two things fall out of that:

  * a cheap "is this even scorable from here" test, which prunes most of the
    ball prediction before any expensive reachability work; and
  * the exact direction the ball must leave in. When the car's natural line
    already points into the mouth we hit straight through; otherwise we snap
    to the nearer post. That is what makes shots accurate without needing a
    separate aiming controller bolted on afterwards.

Swapping the two post vectors turns the same code into an ANTI-target: clear
the ball anywhere except between them.
"""

from __future__ import annotations

from .constants import BALL_RADIUS, GOAL_HALF_WIDTH
from .vec import Vec3

# The posts are inflated by roughly a ball radius so a "scorable" shot has room
# to pass without clipping the woodwork. Community values sit between 95 and
# 135; 110 is GoslingUtils' number and is a reasonable middle.
POST_INFLATION = 110.0


def _rot_ccw(v: Vec3) -> Vec3:
    """v rotated +90 degrees in the ground plane: cross(v, -Z)."""
    return Vec3(-v.y, v.x, 0.0)


def _rot_cw(v: Vec3) -> Vec3:
    """v rotated -90 degrees in the ground plane: cross(v, +Z)."""
    return Vec3(v.y, -v.x, 0.0)


def goal_posts(state) -> tuple[Vec3, Vec3]:
    """The opponent's two posts, left then right as seen from our half."""
    y = state.enemy_goal.y
    return Vec3(-GOAL_HALF_WIDTH, y, 0.0), Vec3(GOAL_HALF_WIDTH, y, 0.0)


def correct_for_posts(ball: Vec3, left: Vec3, right: Vec3, radius: float = POST_INFLATION):
    """
    Pull the aiming posts inward by the ball's radius, and report whether a
    ball of that size can actually fit through the remaining gap from here.

    Returns (left_corrected, right_corrected, fits).
    """
    goal_line = right - left
    goal_perp = _rot_cw(goal_line)

    to_left = (left - ball).flat()
    to_right = (right - ball).flat()
    if to_left.length_sq() < 1.0 or to_right.length_sq() < 1.0:
        return left, right, False

    left_adj = left + _rot_ccw(to_left.normalized()) * radius
    right_adj = right + _rot_cw(to_right.normalized()) * radius

    left_c = left if (left_adj - left).dot(goal_perp) > 0.0 else left_adj
    right_c = right if (right_adj - right).dot(goal_perp) > 0.0 else right_adj

    span = right_c - left_c
    width = span.flat_length()
    if width < 1.0:
        return left_c, right_c, False

    new_line = span / width
    new_perp = _rot_cw(new_line)
    centre = left_c + new_line * (width * 0.5)
    ball_to_goal = (centre - ball).flat()
    if ball_to_goal.length_sq() < 1.0:
        return left_c, right_c, False
    ball_to_goal = ball_to_goal.normalized()

    fits = width * abs(new_perp.normalized().dot(ball_to_goal)) > 2.0 * radius
    return left_c, right_c, fits


def clamp2D(direction: Vec3, start: Vec3, end: Vec3) -> Vec3:
    """
    Clamp a ground-plane direction into the cone between `start` and `end`.

    Unchanged if it already points inside the cone, otherwise snapped to
    whichever bound it is nearer.

    WINDING MATTERS. The bounds must be given so that `start` is the clockwise
    edge and `end` the counter-clockwise one; passing them the other way round
    inverts the test and the function snaps everything that was already inside
    the cone out to a post. Callers should use `shot_vector`, which gets this
    right, rather than calling this directly.

    The `inside` test itself flips depending on whether the cone spans more or
    less than a half-plane, which is what the outer branch handles.
    """
    is_right = direction.dot(_rot_ccw(end)) < 0.0
    is_left = direction.dot(_rot_ccw(start)) > 0.0

    if end.dot(_rot_ccw(start)) > 0.0:
        inside = is_right and is_left
    else:
        inside = is_right or is_left

    if inside:
        return direction
    return end if start.dot(direction) < end.dot(direction) else start


def shot_vector(state, car_pos: Vec3, ball: Vec3) -> tuple[Vec3, bool]:
    """
    The direction the ball should leave in to go between the posts.

    Returns (direction, scorable). When the car's natural approach line already
    points into the mouth this is that line, so the car drives straight through
    the ball; otherwise it is the nearer post direction.
    """
    left, right = goal_posts(state)
    left_c, right_c, fits = correct_for_posts(ball, left, right)

    to_left = (left_c - ball).flat()
    to_right = (right_c - ball).flat()
    if to_left.length_sq() < 1.0 or to_right.length_sq() < 1.0:
        return (state.enemy_goal - ball).flat().normalized(), False

    natural = (ball - car_pos).flat()
    if natural.length_sq() < 1.0:
        natural = (state.enemy_goal - ball).flat()

    # Right post first: clamp2D wants the clockwise edge as `start`.
    clamped = clamp2D(natural.normalized(), to_right.normalized(), to_left.normalized())
    return clamped.normalized(), fits


def aim_point(state, car_pos: Vec3, ball: Vec3, reach: float = 2500.0) -> Vec3:
    """A point along the shot vector, for callers that want a target position."""
    direction, _ = shot_vector(state, car_pos, ball)
    return ball + direction * reach
