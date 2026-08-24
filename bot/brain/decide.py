"""
The decision policy: everything above joins up here.

Order matters. Reflexes that must never be overridden (respawning, recovery,
an in-flight manoeuvre) are handled first; then the situation is assessed;
then a role is chosen, passed through the human reaction-time gate, and turned
into controls. Output noise is applied last so it degrades the final input
rather than the intent.
"""

from __future__ import annotations

import math

from rlbot import flat

from ..control.dodge import HalfFlip, SpeedFlip
from ..control.drive import arrive_at, drive_to, local_angle_to, should_half_flip
from ..control.recovery import Recovery, needs_recovery
from ..core.constants import (
    BACK_WALL_Y,
    GOAL_HALF_WIDTH,
    MAX_SPEED,
    MAX_SPEED_NO_BOOST,
)
from ..core.intercept import AERIAL, find_intercept
from ..core.physics import ReachCurve
from ..core.vec import Vec3, clamp
from ..learn.calibrate import StrikeCalibrator
from ..learn.experience import ExperienceTable, OutcomeWatcher, situation_key
from .challenge import CHALLENGE, SHADOW, STRIKE, ContestState
from . import boost as boost_mod
from . import kickoff as kickoff_mod
from .roles import (
    ATTACK,
    DEFEND,
    SUPPORT,
    RoleState,
    assign_role,
    defend_position,
    net_position,
    support_position,
    threat_level,
)
from ..control.aerial import Aerial
from .strike import CLEAR, Strike, aim_for, choose_aim, clear_target, legal_actions

# Ordinal stickiness. Ordering teammates purely by time-to-ball is
# unstable -- similar estimates swap several times a second and the role
# swaps with them. Both constants exist to stop the car twitching.
ORDINAL_HOLD = 0.9        # a new ordinal must persist this long to count
INCUMBENT_BONUS = 0.30    # seconds of advantage the current holder keeps

# Seconds of advantage a car with a viable aerial gets in the rotation ranking.
#
# Measured cause of the bots never flying: 76.1% of every aerial the solver
# found was found by a car the ordinal had already forbidden from touching the
# ball, and `_do_support`/`_do_defend` then discarded it outright. Of 2,611
# aerial proposals only 302 ever reached the executor -- 88.4% thrown away.
#
# The fix deliberately does NOT let a non-first-man break rotation. That would
# reintroduce the pack behaviour the ordinal was built to kill. Instead the
# aerial competes for the ATTACK SLOT ITSELF, through the same ranking that
# already guarantees exactly one car holds slot 0. Nobody strikes who is not
# first man; we only change who first man is.
#
# Sized from measurement, not taste. Across 6,064 three-car team-instants the
# car proposing an aerial is usually ALREADY the fastest on the team -- the gap
# behind the team's best time-to-ball is 0.00s at p75 and only +0.12s at p90.
# It is not behind; it simply is not the incumbent. So the bonus has to beat
# INCUMBENT_BONUS, not a real deficit.
#
#   bonus   aerial proposals converted (of 164 not already on slot 0)
#   0.45    90.9%
#   0.55    95.1%
#   0.65    96.3%
#   0.75+   97.6%, flat thereafter
#
# 0.55 takes the knee. It lets a car within (0.55 - 0.30) = 0.25s of the
# team's fastest take the slot, which is tight enough that a genuinely
# out-of-position car still cannot, so clash rate is protected.
AERIAL_BONUS = 0.55
# Once claimed, hold the claim this long after the aerial stops being offered.
# A bonus that flickers on and off at tick rate would churn the ordering, which
# is the exact failure the minimum-hold and incumbency terms exist to prevent.
AERIAL_CLAIM_HOLD = 0.6
# One yield per broadcast interval (comms sends intent at 10Hz), so a pair
# resolving a duplicate slot cannot both bounce off each other repeatedly.
YIELD_COOLDOWN = 0.12

# Time constant for the threat low-pass filter used by role selection.
THREAT_SMOOTHING = 0.30
# Minimum time in a cover role before switching between support and
# defend, on top of the hysteresis band.
COVER_DWELL = 0.8

# How close to a positional target the car starts easing off, and the speed
# it will not drop below while holding that position.
ARRIVE_RADIUS = 260.0
ARRIVE_MIN_SPEED = 550.0

# Support/defend switching band for the cover players, against the SMOOTHED
# threat signal. Calibrated against 78,813 samples of real play: the smoothed
# threat has median 0.09 and p90 0.40, so a threshold in the 0.5s is above
# almost everything and produces a cover player who never defends.
COVER_ENTER = 0.26          # ~25% of ticks above, for second man
COVER_ORDINAL_STEP = 0.08   # third man commits to defending sooner
COVER_BAND = 0.10           # hysteresis width


class Debug:
    """Whatever the renderer wants to draw this tick."""

    def __init__(self):
        self.role = ""
        self.action = ""
        self.target: Vec3 | None = None
        self.intercept: Vec3 | None = None
        self.threat = 0.0
        self.my_time = 0.0
        self.ally_time = 0.0
        self.note = ""
        # Diagnostics: what the solver proposed, and whether the rank gate
        # turned down an aerial it offered.
        self.kind = ""
        self.aerial_declined = False
        self.mechanic = ""


def car_of(state):
    """Tiny accessor so the contest branches read as prose."""
    return state.me


class Brain:
    def __init__(self, model, humanizer, config=None):
        self.model = model
        self.hz = humanizer
        self.config = config or {}

        self.role_state = RoleState()
        # The role we are actually executing.
        #
        # This must NOT be read back out of role_state: assign_role() writes to
        # role_state internally for its own hysteresis, and the ordinal logic
        # then overrides the result. Feeding role_state.role to the reaction
        # gate therefore compares the new decision against a value that was
        # already overwritten this tick, so the gate "holds" by re-emitting
        # something nobody chose. That was the real cause of the role thrash.
        self.active_role = SUPPORT
        self.recovery = Recovery()
        self.strike: Strike | None = None
        self.maneuver = None
        self.kickoff_plan: kickoff_mod.KickoffPlan | None = None
        # Corrects our own systematic strike errors during play.
        self.calibrator = StrikeCalibrator()
        # Which choices actually pay off, pooled across every Ally.
        self.experience = ExperienceTable()
        self.outcomes = OutcomeWatcher(self.experience)
        # Set by the bot when other Ally instances are in the match.
        self.comms = None
        # Our place in the queue for the ball; drives which support or
        # defensive slot we occupy so teammates do not stack.
        self.ordinal = 0
        self._ordinal_pending = None
        self._ordinal_pending_since = 0.0
        self._last_yield = -99.0
        self._in_recovery = False
        # Latched state for the support/defend Schmitt trigger, plus the
        # smoothed threat signal it compares against.
        self._second_man_defending = False
        self._cover_switched_at = 0.0
        self._threat_smooth = 0.0
        # Whether we beat the opponent to this ball, and what to do if not.
        self.contest = ContestState()
        # Boost pad we are currently routing to, broadcast so teammates
        # can avoid it. -1 when we are not detouring.
        self.boost_pad = -1
        self.boost_eta = 99.0
        # The action committed for the current attempt, held until the
        # situation bucket changes.
        self._action_key = None
        self._action_choice = None
        self._action_why = ""
        # Half-flips are visually loud and easy to over-fire; one at a time.
        self._last_half_flip = -99.0
        # The time-to-ball we last told our teammates. Ranking ourselves by
        # this rather than by a freshly solved value removes the 10Hz
        # staleness asymmetry: peers are comparing against what we SAID, so
        # we must compare against the same thing or the orderings disagree
        # and two bots claim the same slot.
        self.broadcast_time = 99.0
        # What we last told the team about having an aerial, so our own ranking
        # uses the same value they are ranking us by -- same reasoning as
        # `broadcast_time`.
        self.broadcast_aerial = False
        self.aerial_claim = False
        self._aerial_seen_at = -99.0

        self.debug = Debug()
        self._prev_phase = None

    # --- main entry -------------------------------------------------------

    def decide(self, state, prediction) -> flat.ControllerState:
        d = self.debug
        # Clear every frame: stale debug values are worse than empty ones,
        # because they read as real readings while drawn on screen.
        d.note = ""
        d.action = ""
        d.target = None
        d.intercept = None
        d.threat = 0.0
        d.my_time = 0.0
        d.ally_time = 0.0
        # Role was never cleared here, so a wrecked car kept broadcasting a
        # stale "attack" and drew it on screen while dead.
        d.role = ""
        d.kind = ""
        d.aerial_declined = False
        d.mechanic = ""

        # --- non-playing states -------------------------------------------
        if state.me.is_demolished:
            self._reset_transient()
            d.action = "demolished"
            return flat.ControllerState()

        if not state.is_active or state.ball is None:
            self._reset_transient()
            d.action = "waiting"
            return flat.ControllerState(throttle=0.0)

        # --- perceive -------------------------------------------------------
        # Observation runs before any early return. Learning about our partner
        # must not stop just because we happen to be mid-half-flip or upside
        # down -- those are exactly the moments they are doing something
        # unsupervised, and skipping them biases the model.
        my_curve = ReachCurve(state.me.speed, state.me.boost)
        # An opponent bearing down changes the calculus: get there sooner
        # rather than tidily. Half a second of opponent lookahead, as the
        # prior art does it.
        contested = False
        if state.ball is not None:
            for opp in state.opponents:
                if opp.is_demolished:
                    continue
                ahead = opp.pos + opp.vel * 0.5
                if ahead.flat_dist(state.ball.pos) < 900.0:
                    contested = True
                    break
        my_ic = find_intercept(
            state.me, prediction, state.time, my_curve,
            contested=contested, shoot_target=state.enemy_goal,
        )
        my_time = my_ic.dt if my_ic.feasible else 12.0

        # An aerial is on offer. The arbiter has already checked we can afford
        # it, so reaching here means this car can actually fly to the ball.
        # Sticky, per AERIAL_CLAIM_HOLD.
        if my_ic.feasible and my_ic.kind == AERIAL:
            self._aerial_seen_at = state.time
        self.aerial_claim = (state.time - self._aerial_seen_at) < AERIAL_CLAIM_HOLD

        if state.has_ally:
            ally_curve = ReachCurve(state.ally.speed, state.ally.boost)
            ally_ic = find_intercept(state.ally, prediction, state.time, ally_curve)
            ally_time = ally_ic.dt if ally_ic.feasible else 12.0
        else:
            ally_time = 99.0

        self.model.update(state, my_time, ally_time)
        self.calibrator.observe(state)
        self.outcomes.update(state)
        # Solving the opponent's intercept is what tells us whether this
        # ball is ours at all. Cached at ~2Hz, as the prior art does.
        self.contest.refresh(state, prediction, my_time)

        threat = threat_level(state, prediction)

        # Smoothed threat for anything that picks a *role*.
        #
        # threat_level contains a binary on-target test that snaps between 0
        # and 1 in a single tick as the ball prediction shifts, so the raw
        # signal is far too jumpy to compare against a threshold: it produced
        # 202 support/defend flips in under two minutes. Reflexes (saves,
        # clears) still use the raw value, because there responsiveness is the
        # whole point.
        alpha = clamp(state.dt / THREAT_SMOOTHING, 0.0, 1.0)
        self._threat_smooth += (threat - self._threat_smooth) * alpha
        threat_s = self._threat_smooth

        d.threat = threat
        d.my_time = my_time
        d.ally_time = ally_time
        d.intercept = my_ic.pos if my_ic.feasible else None
        d.kind = my_ic.kind if my_ic.feasible else "none"
        # Why we are NOT flying, when we are not. Empty when an aerial was
        # taken or when the ball was never in the aerial band.
        d.aerial_reject = my_ic.aerial_reject
        d.aerial_cost = my_ic.aerial_cost

        # Rotation slot, computed BEFORE the kickoff early-return so the first
        # live tick already has distinct slots. Whatever duplicate existed when
        # the ball died used to be carried straight into open play and then held
        # for another ORDINAL_HOLD on top -- measured at three cars on slot 0 for
        # 6.2 seconds after one kickoff.
        #
        # Free during a kickoff: _do_kickoff forces role="kickoff" and reads no
        # ordinal. Skipped while recovering or mid-manoeuvre, where my_time comes
        # from a ReachCurve that models a car which is driving, so a tumbling car
        # would get an optimistic time and could wrongly claim slot 0.
        if self.maneuver is None and not needs_recovery(state.me):
            self.ordinal = self._team_ordinal(state, my_time, ally_time)

        # --- kickoff -------------------------------------------------------
        if state.is_kickoff:
            return self._do_kickoff(state, prediction)
        if self.kickoff_plan is not None:
            self.kickoff_plan = None

        # --- in-flight manoeuvre -------------------------------------------
        if self.maneuver is not None:
            out = self.maneuver.step(state)
            if out is not None:
                d.action = type(self.maneuver).__name__.lower()
                return out
            self.maneuver = None

        # --- recovery ------------------------------------------------------
        if needs_recovery(state.me):
            d.action = "recovery"
            aim = state.ball.pos if state.ball else None
            # Roll for the wavedash ONCE per recovery, not once per tick.
            #
            # decide() runs at game rate, so a single recovery was rerolling
            # this ~136 times. At diamond's 0.50 that is not "half of
            # recoveries", it is 1 - 0.5^136: the rank profile was completely
            # defeated and every recovery wavedashed. This restores the
            # intended imperfection rather than removing it.
            if not self._in_recovery:
                self._in_recovery = True
                self.recovery.allow_wavedash = self.hz.attempt_wavedash()
            return self.hz.add_input_noise(self.recovery.step(state, aim), state.time)
        self._in_recovery = False
        self.recovery.reset()

        # --- role, gated by reaction time ----------------------------------
        wanted = assign_role(state, self.model, self.role_state, my_time, ally_time, prediction)

        # Where we sit in the queue for the ball. With two or more teammates
        # this drives the role outright: without an ordinal every non-attacker
        # computes the same support spot and the team drives around in a pack.
        ordinal = self.ordinal

        if len(state.teammates) >= 2:
            if ordinal == 0:
                wanted = ATTACK
            else:
                # Second and third man share a role and differ only in depth,
                # which is expressed in the position rather than the role.
                # Making the distinction a role difference means every swap
                # between the two slots flips the role and the car twitches --
                # and those two are the closest in time-to-ball, so they swap
                # constantly.
                #
                # Schmitt trigger rather than a plain threshold, because threat
                # is continuous and noisy: a single comparison point makes the
                # role flicker every time it jitters across the line. The
                # deeper player commits to defending sooner.
                enter = COVER_ENTER - COVER_ORDINAL_STEP * min(ordinal - 1, 1)
                leave = enter - COVER_BAND
                want_defend = self._second_man_defending
                if self._second_man_defending:
                    want_defend = threat_s > leave
                else:
                    want_defend = threat_s > enter
                # Dwell time on top of the band: even a smoothed signal can
                # loiter on a threshold, and a cover player flicking between
                # roles is worse than one that is briefly in the wrong one.
                if want_defend != self._second_man_defending:
                    if state.time - self._cover_switched_at >= COVER_DWELL:
                        self._second_man_defending = want_defend
                        self._cover_switched_at = state.time
                wanted = DEFEND if self._second_man_defending else SUPPORT
            if threat_s > 0.75 and ordinal >= 1:
                wanted = DEFEND
            d.note = f"#{ordinal + 1}"
        elif self.comms is not None and wanted == ATTACK:
            # Two-player case: a peer Ally may already have called this ball.
            # Their estimate comes from their own car and beats our guess
            # about them, so it wins.
            peer = self.comms.better_placed_peer(state.time, my_time)
            if peer is not None:
                wanted = DEFEND if threat > 0.45 else SUPPORT
                d.note = f"peer {peer.index} has it"

        role = self.hz.gate_decision(state.time, wanted, self.active_role)
        self.active_role = role
        d.role = role

        # Attention lapse: hesitate briefly instead of acting.
        if self.hz.is_lapsed(state.time, state.dt) and threat < 0.7:
            d.action = "hesitate"
            c = drive_to(state.me, state.me.pos + state.me.ori.forward * 400.0, 600.0, allow_boost=False)
            return self.hz.add_input_noise(c, state.time)

        # --- act -------------------------------------------------------------
        if role == ATTACK:
            c = self._do_attack(state, prediction, my_ic, threat)
        elif role == DEFEND:
            c = self._do_defend(state, prediction, my_ic, threat)
        else:
            c = self._do_support(state, prediction, my_ic)

        return self.hz.add_input_noise(c, state.time)

    def _team_ordinal(self, state, my_time: float, ally_time: float) -> int:
        """
        Our place in the queue for the ball among teammates. 0 = first man.

        Peer Allies report their own time-to-ball over comms, which is better
        information than anything we could compute about them from outside
        their car. The human has no comms, so we use the estimate we already
        solved for them rather than paying for extra intercept solves.

        Two forms of stickiness, both necessary. Ordering purely by
        time-to-ball is unstable: as the ball moves, two cars with similar
        estimates swap places several times a second, the role swaps with
        them, and the car twitches instead of committing. Measured at 62-72
        role changes a minute before this was added.

          * Incumbency -- whoever currently holds a slot gets a bonus when
            comparing, so a 50ms difference does not hand the ball over.
          * Minimum hold -- a new ordinal must persist before it is adopted.
        """
        # Incumbency must be applied to whoever *actually* holds the slot, not
        # to whoever is doing the calculating. Discounting our own time while
        # comparing against peers' raw times makes every bot the incumbent, so
        # all three conclude they are first man -- measured at 52% of ticks
        # believing they were #1 when only 33% can be. Peers broadcast their
        # ordinal so everyone applies the same bonus to the same car and
        # arrives at the same ordering.
        my_bonus = INCUMBENT_BONUS if self.ordinal == 0 else 0.0
        # Use the broadcast value, not the live one, so we rank ourselves by
        # the same number the others are ranking us by.
        if self.broadcast_aerial:
            my_bonus += AERIAL_BONUS
        # Rank by the value the others have actually seen from us.
        mine = self.broadcast_time if self.broadcast_time < 90.0 else my_time
        entries = [(mine - my_bonus, state.me.index)]
        covered = {state.me.index}

        peers = self.comms.teammates(state.time) if self.comms is not None else []
        # A wrecked teammate reports a meaningless time-to-ball, so skip it here
        # and let the local `is_demolished` check below handle that car.
        demoed = {c.index for c in state.teammates if c.is_demolished}
        for p in peers:
            if p.index in covered or p.index in demoed:
                continue
            peer_bonus = INCUMBENT_BONUS if p.ordinal == 0 else 0.0
            if p.aerial:
                peer_bonus += AERIAL_BONUS
            entries.append((p.time_to_ball - peer_bonus, p.index))
            covered.add(p.index)

        primary = state.ally.index if state.ally is not None else None
        for c in state.teammates:
            if c.index in covered:
                continue
            if c.is_demolished:
                continue
            # Only the teammate we actually solved for gets a real estimate;
            # any other un-comm'd car sorts to the back rather than being
            # given a number we did not compute.
            entries.append((ally_time if c.index == primary else 99.0, c.index))
            covered.add(c.index)

        # Sort by time, then index so ties resolve identically on every car.
        entries.sort()
        raw = 0
        for i, (_, idx) in enumerate(entries):
            if idx == state.me.index:
                raw = i
                break

        # Two cars holding one slot is a FAULT, not a preference.
        #
        # The minimum hold below exists to damp churn between two valid
        # orderings. While a duplicate stands, the ordering is not valid: the
        # incumbency bonus cancels between the pair and the hold then stops
        # either of them moving, so a fault gets protected as though it were a
        # decision. Measured, three cars sat on slot 0 for 6.2 seconds. `raw`
        # puts exactly one of the pair back where it belongs, so exactly one
        # yields -- and the cooldown keeps that to one yield per broadcast.
        if (
            raw != self.ordinal
            and any(p.ordinal == self.ordinal for p in peers)
            and state.time - self._last_yield >= YIELD_COOLDOWN
        ):
            self._ordinal_pending = None
            self._last_yield = state.time
            return raw

        # Minimum hold: a different ordinal must persist before we act on it.
        if raw == self.ordinal:
            self._ordinal_pending = None
            return self.ordinal

        if self._ordinal_pending != raw:
            self._ordinal_pending = raw
            self._ordinal_pending_since = state.time
            return self.ordinal

        if state.time - self._ordinal_pending_since >= ORDINAL_HOLD:
            self._ordinal_pending = None
            return raw
        return self.ordinal

    # --- behaviours -------------------------------------------------------

    def _do_attack(self, state, prediction, my_ic, threat) -> flat.ControllerState:
        d = self.debug
        d.action = "attack"

        if not my_ic.feasible:
            # Cannot reach it: fall back to shadowing the play.
            target = defend_position(state, prediction, self.ordinal)
            d.target = target
            d.action = "attack/unreachable"
            return self._drive_with_boost(state, target, ATTACK, urgency=threat)

        # --- do we actually win this ball? ---------------------------------
        #
        # Driving at our own intercept regardless of the opponent is how the
        # bot ended up arriving late and hovering. Either we get there first
        # and strike, or it is a 50/50 and we commit at speed, or we are losing
        # it and we get goal-side instead. Never beside it.
        is_last = True
        for mate in state.teammates:
            if not mate.is_demolished and (
                mate.pos.y * state.goal_sign > state.me.pos.y * state.goal_sign
            ):
                is_last = False
                break
        stance = self.contest.decide(state, d.my_time, is_last)

        if stance == SHADOW:
            target = self.contest.shadow_point(state)
            d.target = target
            d.action = "attack/shadow"
            d.note = self.contest.summary()
            speed = self.contest.shadow_speed(state, target)
            return drive_to(car_of(state), target, speed,
                            allow_boost=state.me.boost > boost_mod.BOOST_RESERVE)

        # A 50/50 is won by arriving fast and committing, so the striker is
        # told to abandon its arrival timing and drive straight through.
        contesting = stance == CHALLENGE

        aim, kind = choose_aim(state, self.model, my_ic, threat)

        # The situational default is a decent prior, but which option actually
        # pays off from here is something the team has evidence about -- much
        # of it gathered by other Allies. Consult that before committing.
        key = situation_key(state, my_ic.pos, self.hz.pressure(state))

        # Commit to one action for the duration of an attempt.
        #
        # Consulting the experience table every tick re-rolls the choice --
        # including its exploration, which is 60% in an unfamiliar bucket -- so
        # the aim flipped on 5% of consecutive ticks. Every flip swings the
        # contact point to the OTHER SIDE of the ball, and the car visibly
        # wags its wheels trying to follow. Decide once, then hold, exactly as
        # the humanizer already commits its aim error and whiff per attempt.
        if self._action_key != key or self._action_choice is None:
            options = legal_actions(state, my_ic, threat)
            if kind not in options:
                options.append(kind)
            chosen, why = self.experience.choose(key, options, kind)
            self._action_key = key
            self._action_choice = chosen
            self._action_why = why

        if self._action_choice != kind:
            kind = self._action_choice
            aim = aim_for(state, kind, my_ic)
        why = self._action_why
        self.outcomes.note_decision(state, key, kind)

        d.action = f"attack/{kind}" if not contesting else f"contest/{kind}"
        d.note = self.contest.summary() if contesting else why
        d.target = aim

        # Never rebuild the Strike out from under a live aerial.
        #
        # Strike owns its manoeuvre, so replacing the Strike destroys the
        # Aerial mid-flight. The intercept kind flips constantly as a lofted
        # ball falls out of the aerial band, which is why real aerial episodes
        # lasted a median of 0.3s against a required lead of 0.8-2.5s and spent
        # a median of 0 boost. `is_viable` is now the thing that ends them.
        # Test the manoeuvre itself, not the key. Strike.key is built from the
        # ACTION (shot / pass / clear), never the mechanic, so a startswith on
        # "aerial" is dead code that silently protects nothing.
        holding_aerial = self.strike is not None and isinstance(
            self.strike.maneuver, Aerial
        )
        if self.strike is None or (
            self.strike.key.split(":")[0] != kind and not holding_aerial
        ):
            self.strike = Strike(my_ic, aim, kind)
        else:
            self.strike.refresh(my_ic, aim, kind)

        return self.strike.step(state, self.hz, self.calibrator, d, contesting)

    def _do_defend(self, state, prediction, my_ic, threat) -> flat.ControllerState:
        d = self.debug
        d.action = "defend"
        ball = state.ball.pos

        # Ball is on our doorstep and we can get to it: clear it.
        imminent = threat > 0.65 and my_ic.feasible and my_ic.dt < 1.6
        if imminent:
            d.action = "defend/save"
            aim = clear_target(state)
            holding_aerial = self.strike is not None and isinstance(
                self.strike.maneuver, Aerial
            )
            if self.strike is None or (
                self.strike.key.split(":")[0] != CLEAR and not holding_aerial
            ):
                self.strike = Strike(my_ic, aim, CLEAR)
            else:
                self.strike.refresh(my_ic, aim, CLEAR)
            # A save is a commitment by definition: never time the approach.
            return self.strike.step(state, self.hz, self.calibrator, d, True)

        self.strike = None

        # Very high threat and we cannot reach the ball: sit on the line.
        if threat > 0.8 and (not my_ic.feasible or my_ic.dt > 1.8):
            target = net_position(state)
            d.action = "defend/net"
            d.target = target
            return self._drive_with_boost(state, target, DEFEND, arrive=False, urgency=1.0)

        target = defend_position(state, prediction, self.ordinal)
        d.target = target
        return self._drive_with_boost(state, target, DEFEND, urgency=threat)

    def _do_support(self, state, prediction, my_ic) -> flat.ControllerState:
        d = self.debug
        d.action = "support"
        self.strike = None

        target = support_position(state, self.model, prediction, self.ordinal)
        d.target = target
        return self._drive_with_boost(state, target, SUPPORT)

    # --- shared movement --------------------------------------------------

    def _drive_with_boost(
        self, state, target: Vec3, role: str, arrive: bool = True, urgency: float = 0.0
    ) -> flat.ControllerState:
        """
        Drive to `target`, detouring for boost when that is worthwhile.

        Routine repositioning cruises at the no-boost cap rather than demanding
        max speed. Asking for 2300 everywhere means the throttle controller
        sees a large speed error on every tick of every rotation and holds
        boost flat out, which empties the tank and leaves nothing for the
        moments that actually decide goals. 1410 gets there nearly as quickly
        and arrives with boost still in hand.
        """
        car = state.me

        self.boost_pad = -1
        self.boost_eta = 99.0
        if boost_mod.wants_boost(state, role, self.ordinal):
            pad = boost_mod.find_boost(state, self.model, target, comms=self.comms)
            if pad is not None:
                self.debug.note = "boost"
                self.boost_pad = pad.index
                self.boost_eta = boost_mod.eta_to_pad(state, pad)
                # Aim through the pad rather than braking onto it.
                target = Vec3(pad.pos.x, pad.pos.y, 17.0)

        # Half-flip rather than turning around when the target is behind us.
        if (
            should_half_flip(car, target)
            and car.can_dodge
            and car.speed > 300.0
            and state.time - self._last_half_flip > 2.5
        ):
            self._last_half_flip = state.time
            self.maneuver = HalfFlip()
            out = self.maneuver.step(state)
            if out is not None:
                return out

        dist = car.pos.flat_dist(target)

        if dist < ARRIVE_RADIUS:
            # Ease in, but keep real pace.
            #
            # The old floor of 180 uu/s let a cover player creep around its
            # holding spot: measured at 26% of support time and 34% of defend
            # time under 400 uu/s, with speed actually FALLING the closer a car
            # got to the ball. From outside that reads as the bot being afraid
            # to touch it. Real players hold a position by circling at pace,
            # because a slow car cannot react to anything and gets bumped off
            # the ball by anyone arriving with speed.
            speed = clamp(dist * 3.0, ARRIVE_MIN_SPEED, 1100.0)
            allow_boost = False
        elif urgency > 0.35 or dist > 2000.0:
            # Genuinely need to be somewhere: spend boost.
            speed = MAX_SPEED
            allow_boost = car.boost > boost_mod.BOOST_RESERVE
        else:
            speed = MAX_SPEED_NO_BOOST
            allow_boost = False

        return drive_to(car, target, speed, allow_boost=allow_boost)

    # --- kickoff ----------------------------------------------------------

    def _do_kickoff(self, state, prediction) -> flat.ControllerState:
        d = self.debug
        d.role = "kickoff"

        if self.kickoff_plan is None:
            take = kickoff_mod.should_take_kickoff(state, self.model)
            spawn = kickoff_mod.classify_spawn(state.me, state.goal_sign)
            use_flip = take and self.hz.attempt_speedflip() and spawn != "back_center"
            self.kickoff_plan = kickoff_mod.KickoffPlan(take, spawn, use_flip)
            self.kickoff_plan.start_time = state.time
            # Record what our partner did, for the model.
            if state.has_ally:
                ally_going = kickoff_mod.kickoff_distance(state.ally) <= kickoff_mod.kickoff_distance(state.me) + 250.0
                self.model.on_kickoff(state, ally_going)

        plan = self.kickoff_plan
        car = state.me

        if not plan.take:
            d.action = "kickoff/cheat"
            target = kickoff_mod.cheat_position(state, self.model)
            if car.boost < 50.0:
                pad = boost_mod.nearest_big_pad(state)
                if pad is not None and pad.pos.flat_dist(car.pos) < 2600.0:
                    target = pad.pos
            d.target = target
            return drive_to(car, Vec3(target.x, target.y, 17.0), MAX_SPEED)

        d.action = "kickoff/go"
        aim = plan.aim_point(state)
        d.target = aim

        # Continue an in-flight speedflip.
        if plan.maneuver is not None:
            out = plan.maneuver.step(state)
            if out is not None:
                return out
            plan.maneuver = None

        elapsed = state.time - plan.start_time
        dist = car.pos.flat_dist(aim)

        # Fire the speedflip early in the run, once up to speed AND pointed
        # at the ball. The flip steers blind for ~0.9s, so starting it while
        # still turning throws the car off the line entirely -- measured
        # closest approach on kickoffs was 178-722uu against the ~150uu needed
        # to touch the ball, i.e. it missed every time.
        aligned = abs(local_angle_to(car, Vec3(aim.x, aim.y, 17.0))) < 0.25
        if (
            plan.use_speedflip
            and not plan.flip_fired
            and 0.25 < elapsed < 0.9
            and car.speed > 700.0
            and dist > 1900.0
            and aligned
            and car.on_ground
        ):
            plan.flip_fired = True
            direction = 1.0 if car.pos.x <= 0 else -1.0
            plan.maneuver = SpeedFlip(direction)
            out = plan.maneuver.step(state)
            if out is not None:
                return out

        # Never brake on a kickoff. The default turn cap slows the car whenever
        # the angle to the ball is wide, which from the corner spawns is the
        # whole approach -- the same mistake that was turning strikes into
        # nudges. A kickoff is a straight race; hold speed and steer.
        c = drive_to(car, Vec3(aim.x, aim.y, 17.0), MAX_SPEED,
                     allow_slide=False, min_turn_speed=1600.0)
        c.boost = car.speed < 2200.0
        return c

    # --- misc -------------------------------------------------------------

    def _reset_transient(self):
        self.strike = None
        self.maneuver = None
        # A kickoff or manoeuvre between two recoveries must not merge them
        # into one episode, or the wavedash roll is skipped for the second.
        self._in_recovery = False
        self._action_key = None
        self._action_choice = None
        self.recovery.reset()
