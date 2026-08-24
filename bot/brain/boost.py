"""
Boost economy.

Two rules do most of the work. First, only detour for boost when the detour is
actually cheap relative to where we were already going. Second -- and this is
the teammate-specific part -- do not take pads our partner is about to reach,
especially if their profile says they habitually run dry. Stealing your
teammate's corner boost is one of the most common ways to lose a game while
looking like you did nothing wrong.
"""

from __future__ import annotations

from ..core.constants import BACK_WALL_Y
from ..core.vec import Vec3, clamp

# Boost levels below which we start caring / start prioritising.
LOW_BOOST = 34.0
CRITICAL_BOOST = 12.0
FULL_ENOUGH = 80.0

# What a cover player should be carrying.
#
# Measured: when the ball went above 500uu the nearest bot was only 585uu away
# -- easily close enough to go up -- but was carrying a median of 34 boost,
# against the ~50 an aerial to that height costs. The team was not failing to
# find aerials, it was failing to afford them. Players who are not on the ball
# have time to collect, and should arrive at the next contest able to use it.
COVER_BOOST_TARGET = 70.0

# Keep this much in hand for challenges, saves and recoveries. Routine
# repositioning must not dip below it -- arriving somewhere half a second
# sooner is worth far less than having boost when a 50/50 appears.
BOOST_RESERVE = 15.0


def _pad_value(pad, state, model, destination: Vec3, my_boost: float) -> float:
    """
    Score a pad. Higher is better; returns -inf for pads we should ignore.

    The score is (boost gained) / (seconds of detour), with penalties for pads
    that belong to our partner or sit in dangerous places.
    """
    if not pad.is_active:
        # Respawning pads are still worth routing toward if the timer is short.
        if pad.timer > 1.2:
            return float("-inf")

    me = state.me
    amount = 100.0 if pad.is_big else 12.0
    # No credit for overfilling.
    gain = min(amount, 100.0 - my_boost)
    if gain < 4.0:
        return float("-inf")

    direct = me.pos.flat_dist(destination)
    via = me.pos.flat_dist(pad.pos) + pad.pos.flat_dist(destination)
    detour = via - direct
    if detour < 0.0:
        detour = 0.0

    # Convert to a rough time cost and keep the denominator sane.
    detour_time = detour / 1400.0 + 0.05

    score = gain / detour_time

    # --- teammate courtesy ------------------------------------------------
    # Every teammate, not just the primary one. The old version consulted
    # state.ally alone, so in a 3v3 the third car was invisible here and two
    # bots would happily converge on the same pad.
    if pad.is_big:
        my_dist = me.pos.flat_dist(pad.pos)
        for mate in state.teammates:
            if mate.is_demolished:
                continue
            if mate.pos.flat_dist(pad.pos) < my_dist and mate.boost < 60.0:
                # They are closer and need it. Scale by how badly our human
                # partner tends to run out; bot teammates coordinate through
                # explicit claims instead, so a mild nudge is enough.
                factor = model.leave_boost_bias if mate.is_human else 0.5
                score *= clamp(1.0 - 0.85 * factor, 0.05, 1.0)
                break

    # --- positional safety -------------------------------------------------
    # Pads deep in the opponent's corner are a trap when we are the last man.
    pad_depth = pad.pos.y * state.goal_sign
    if pad_depth < -BACK_WALL_Y * 0.55:
        score *= 0.35

    # Prefer pads that keep us goal-side when the ball is threatening.
    if state.ball is not None:
        ball_depth = state.ball.pos.y * state.goal_sign
        if ball_depth > 0.0 and pad_depth < ball_depth - 1500.0:
            score *= 0.4

    return score


def eta_to_pad(state, pad) -> float:
    """Rough seconds for us to reach this pad. Used as our claim strength."""
    speed = max(state.me.speed, 700.0)
    return state.me.pos.flat_dist(pad.pos) / speed


def find_boost(
    state,
    model,
    destination: Vec3,
    max_detour: float = 1600.0,
    require_big: bool = False,
    comms=None,
):
    """
    Best pad to pick up on the way to `destination`, or None.

    `max_detour` is in uu of extra travel; it is widened automatically when we
    are nearly empty, because at that point almost any detour is worth it.

    When `comms` is supplied, pads a teammate has already claimed with a better
    ETA are skipped outright. Distance-based courtesy alone is not enough --
    two bots at similar range both conclude they are closer and converge, which
    telemetry showed happening on 7% of ticks.
    """
    me = state.me
    if me.boost >= FULL_ENOUGH:
        return None

    if me.boost < CRITICAL_BOOST:
        max_detour *= 2.2
    elif me.boost < LOW_BOOST:
        max_detour *= 1.4

    direct = me.pos.flat_dist(destination)
    best = None
    best_score = 0.0

    for pad in state.boost_pads:
        if require_big and not pad.is_big:
            continue
        via = me.pos.flat_dist(pad.pos) + pad.pos.flat_dist(destination)
        if via - direct > max_detour:
            continue
        if comms is not None and comms.pad_is_taken(
            state.time, pad.index, eta_to_pad(state, pad)
        ):
            continue
        score = _pad_value(pad, state, model, destination, me.boost)
        if score > best_score:
            best_score = score
            best = pad

    return best


def wants_boost(state, role: str, ordinal: int = 0) -> bool:
    """
    Whether topping up is worth considering at all right now.

    Staggered by rotation slot. Raising the cover target to 70 made every cover
    player detour at the same moment -- two or more collecting simultaneously on
    39% of ticks, and pad contention back up to 11%. The deepest player has the
    most time and collects first; the second man only tops up once genuinely
    low, so the team is never all off the ball at once.
    """
    me = state.me
    if me.boost >= FULL_ENOUGH:
        return False
    if me.boost < CRITICAL_BOOST:
        return True
    # The player on the ball tolerates running low -- it is committed. Cover
    # players have time, and are the ones who will need boost for the next
    # challenge or aerial, so they bank up to a genuinely useful level rather
    # than merely topping out of the danger zone.
    from .roles import ATTACK

    if role == ATTACK:
        return me.boost < 18.0
    # Third man banks up to the full target; second man only tops up when it is
    # actually short, so the two do not go shopping together.
    target = COVER_BOOST_TARGET if ordinal >= 2 else LOW_BOOST
    return me.boost < target


def nearest_big_pad(state, from_pos: Vec3 | None = None, active_only: bool = True):
    origin = from_pos or state.me.pos
    best, best_d = None, float("inf")
    for pad in state.big_pads:
        if active_only and not pad.is_active:
            continue
        d = origin.flat_dist(pad.pos)
        if d < best_d:
            best, best_d = pad, d
    return best
