"""
Boost economy.

Reading boost out of a replay is not as simple as averaging car["boost"],
because TAGame.CarComponent_Boost_TA:ReplicatedBoost is only sent when
something *changes the state* of the component -- the frame a burst starts,
the frame it ends, and the frame a pad is collected. In between, the replay
repeats the stale pre-burst number. Averaging it raw reports a player who
never spends anything:

    t=99.366  boost 85.1  boosting True     <- value sent, burst begins
    t=99.4 .. 100.5       boosting True     <- 85.1 repeated for 1.2 s
    t=100.565 boost 45.5  boosting False    <- value sent, burst ends

So the curve is reconstructed instead. While `boosting` is set the tank
drains at a fixed rate, and every fresh replicated value snaps the estimate
back to the truth. Fitting that rate against 210 bursts in the sample replay
gives a median of 33.0-33.2 per second, and the canonical 33.3 leaves a
residual under 2.2 boost at every snap -- so the reconstruction is good to
about one fiftieth of a tank.

The same snaps give a better pad record than the "pickup" events do. Those
events name no player two thirds of the time and fire up to 34 times on a
single timestamp, whereas a snap that jumps *upward* is unambiguously a pad.
It also catches pads taken mid-burst, which a raw diff of the replicated
value misses entirely: collect 12 while burning 33 a second and the number
goes down.

Big pads are told from small ones by where the car was, not by how much it
gained -- a big pad taken on 70 boost only yields 30. The six big pads sit
1283 uu from the nearest small pad, so a 500 uu radius separates them
cleanly.

Time is weighted by the real gap between samples, and dead time is dropped:
the multi-second jumps where the replay skips a goal celebration, the frozen
kickoff countdowns, and the frames after the ball has crossed a goal line.
On the sample replay that leaves 359 s of live play out of 482 s of samples.
"""

from __future__ import annotations

import math

TITLE = "Boost economy"

# Boost drains at 33.3 units per second: a full tank lasts three seconds.
BOOST_DRAIN = 33.3
SUPERSONIC = 2200.0

# The six large pads. Everything else on the field is a 12-boost small pad,
# and the closest small pad to any of these is 1283 uu away.
BIG_PADS = (
    (-3072.0, -4096.0), (3072.0, -4096.0),
    (-3584.0, 0.0), (3584.0, 0.0),
    (-3072.0, 4096.0), (3072.0, 4096.0),
)
BIG_PAD_RADIUS = 500.0
SMALL_PAD_VALUE = 12.0

# A gap longer than this between samples is the replay skipping dead time,
# not a slow frame. Real frames run at about 26 Hz.
DT_CAP = 0.25

# A snap upward smaller than this is reconstruction noise, not a pad. The
# smallest real pad seen is +7.1 (a small pad taken mid-burst, partly eaten
# by the drain before the value replicates); the worst noise residual is -2.2.
PAD_MIN_GAIN = 4.0

GOAL_Y = 5120.0


def _speed(v):
    try:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    except Exception:
        return 0.0


def _live_mask(samples):
    """(dt, is_live) per sample, with dead time zeroed out.

    Dead means: the replay jumped over a goal celebration, every car is
    frozen on a kickoff countdown, or the ball is already in the net.
    """
    out = []
    n = len(samples)
    for i, s in enumerate(samples):
        dt = 0.0
        if i + 1 < n:
            try:
                dt = samples[i + 1]["t"] - s["t"]
            except Exception:
                dt = 0.0
        if not (0.0 < dt <= DT_CAP):
            out.append((0.0, False))
            continue
        ok = True
        try:
            ball = s.get("ball") or (0.0, 0.0, 0.0)
            if abs(ball[1]) > GOAL_Y:
                ok = False
            elif (abs(ball[0]) < 15.0 and abs(ball[1]) < 15.0
                    and ball[2] < 130.0
                    and _speed(s.get("ball_vel") or (0.0, 0.0, 0.0)) < 5.0):
                # Ball parked on the spot. Live only once somebody moves.
                fastest = 0.0
                for c in (s.get("cars") or {}).values():
                    fastest = max(fastest, _speed(c.get("vel") or (0.0, 0.0, 0.0)))
                if fastest < 25.0:
                    ok = False
        except Exception:
            ok = True
        out.append((dt, ok))
    return out


def _blank(name=None):
    return {
        "player": name, "ok": False,
        "live_s": 0.0, "known_s": 0.0,
        "avg_held": 0.0, "starved_share": 0.0, "low_share": 0.0,
        "hoard_share": 0.0, "dry_count": 0,
        "collected": 0.0, "spent": 0.0,
        "collected_per_min": 0.0, "spent_per_min": 0.0,
        "big_pads": 0, "small_pads": 0,
        "big_per_min": 0.0, "small_per_min": 0.0,
        "big_pad_tank": 0.0, "big_pads_topped_up": 0,
        "boost_time_s": 0.0, "boost_share": 0.0,
        "ss_time_s": 0.0, "ss_boost": 0.0, "ss_share": 0.0,
        "peers": {}, "lobby_avg_held": 0.0, "rank": 0, "lobby_n": 0,
        "opp_spend_lo": 0.0, "opp_spend_hi": 0.0,
    }


class _Tank:
    """Reconstructed boost state and accumulators for one car."""

    __slots__ = ("hat", "rep", "actor", "held", "known", "live", "starved",
                 "low", "hoard", "dry", "coll", "spent", "boost_t", "ss_t",
                 "ss_boost", "big", "small", "big_tank")

    def __init__(self):
        self.hat = None       # reconstructed tank, 0-100
        self.rep = None       # last replicated value seen
        self.actor = None
        self.held = 0.0
        self.known = 0.0
        self.live = 0.0
        self.starved = 0.0
        self.low = 0.0
        self.hoard = 0.0
        self.dry = 0
        self.coll = 0.0
        self.spent = 0.0
        self.boost_t = 0.0
        self.ss_t = 0.0
        self.ss_boost = 0.0
        self.big = 0
        self.small = 0
        self.big_tank = []


def _is_big_pad(pos, gain):
    """Where the car was beats how much it gained: a big pad taken on a
    part-full tank yields less than 100, but it is still a big pad."""
    try:
        near = min(math.hypot(pos[0] - bx, pos[1] - by) for bx, by in BIG_PADS)
        if near < BIG_PAD_RADIUS:
            return True
    except Exception:
        pass
    # No small pad can hand out more than 12, so a bigger jump had to be big.
    return gain >= 2.0 * SMALL_PAD_VALUE


def _walk(match):
    """One pass over the timeline, reconstructing every car's tank."""
    samples = getattr(match, "samples", None) or []
    mask = _live_mask(samples)
    tanks = {}

    for i, s in enumerate(samples):
        dt, live = mask[i]
        cars = s.get("cars") or {}
        for name, c in cars.items():
            st = tanks.get(name)
            if st is None:
                st = tanks[name] = _Tank()
            try:
                rep = c.get("boost")
                on = bool(c.get("boosting"))
                actor = c.get("actor")
                pos = c.get("pos") or (0.0, 0.0, 0.0)
                spd = _speed(c.get("vel") or (0.0, 0.0, 0.0))
            except Exception:
                continue

            if actor != st.actor:
                # New car body: kickoff respawn or a demolition. Start clean
                # so the jump to the spawn tank is not scored as a pad.
                st.actor = actor
                st.hat = rep
                st.rep = rep
            else:
                if st.hat is not None and dt > 0.0 and on:
                    burn = min(st.hat, BOOST_DRAIN * dt)
                    before = st.hat
                    st.hat -= burn
                    if live:
                        st.spent += burn
                        if spd >= SUPERSONIC:
                            st.ss_t += dt
                            st.ss_boost += burn
                        if before > 0.5 and st.hat <= 0.0:
                            st.dry += 1
                if rep is not None:
                    if st.rep is None or st.hat is None:
                        st.hat = rep
                    elif rep != st.rep:
                        gain = rep - st.hat
                        if gain > PAD_MIN_GAIN and live:
                            st.coll += gain
                            if _is_big_pad(pos, gain):
                                st.big += 1
                                st.big_tank.append(st.hat)
                            else:
                                st.small += 1
                        st.hat = rep
                    st.rep = rep

            if live and dt > 0.0:
                st.live += dt
                if on:
                    st.boost_t += dt
                if st.hat is not None:
                    st.known += dt
                    st.held += st.hat * dt
                    if st.hat <= 10.0:
                        st.starved += dt
                    if st.hat < 30.0:
                        st.low += dt
                    if st.hat >= 90.0:
                        st.hoard += dt
    return tanks


def _summarise(name, st):
    r = _blank(name)
    live = st.live
    known = st.known
    mins = live / 60.0 if live > 0 else 0.0
    r["live_s"] = live
    r["known_s"] = known
    r["ok"] = live > 20.0 and known > 0.5 * live
    if known > 0:
        r["avg_held"] = st.held / known
        r["starved_share"] = st.starved / known
        r["low_share"] = st.low / known
        r["hoard_share"] = st.hoard / known
    r["dry_count"] = st.dry
    r["collected"] = st.coll
    r["spent"] = st.spent
    if mins > 0:
        r["collected_per_min"] = st.coll / mins
        r["spent_per_min"] = st.spent / mins
        r["big_per_min"] = st.big / mins
        r["small_per_min"] = st.small / mins
    r["big_pads"] = st.big
    r["small_pads"] = st.small
    if st.big_tank:
        r["big_pad_tank"] = sum(st.big_tank) / len(st.big_tank)
        r["big_pads_topped_up"] = sum(1 for x in st.big_tank if x > 50.0)
    r["boost_time_s"] = st.boost_t
    if live > 0:
        r["boost_share"] = st.boost_t / live
    r["ss_time_s"] = st.ss_t
    r["ss_boost"] = st.ss_boost
    if st.spent > 0:
        r["ss_share"] = st.ss_boost / st.spent
    return r


def compute(match, who) -> dict:
    """Boost economy for one player, with the rest of the lobby for scale."""
    try:
        name = match.resolve(who) if hasattr(match, "resolve") else who
    except Exception:
        name = who
    if not name:
        name = who

    try:
        tanks = _walk(match)
    except Exception:
        return _blank(name)
    if not tanks:
        return _blank(name)

    rows = {}
    for nm, st in tanks.items():
        try:
            rows[nm] = _summarise(nm, st)
        except Exception:
            rows[nm] = _blank(nm)

    out = rows.get(name) or _blank(name)

    # Everyone else in the lobby, as a same-lobby yardstick. Rank 1 holds the
    # most boost.
    out["peers"] = {
        nm: {"avg_held": r["avg_held"],
             "spent_per_min": r["spent_per_min"],
             "hoard_share": r["hoard_share"],
             "starved_share": r["starved_share"]}
        for nm, r in rows.items() if nm != name and r["ok"]}

    usable = [r for r in rows.values() if r["ok"]]
    if usable:
        out["lobby_n"] = len(usable)
        out["lobby_avg_held"] = sum(r["avg_held"] for r in usable) / len(usable)
        if out["ok"]:
            better = sum(1 for r in usable if r["avg_held"] > out["avg_held"])
            out["rank"] = better + 1
    try:
        opp = [rows[n]["spent_per_min"] for n in match.opponents(name)
               if n in rows and rows[n]["ok"]]
        if opp:
            out["opp_spend_lo"] = min(opp)
            out["opp_spend_hi"] = max(opp)
    except Exception:
        pass
    return out


def render(result) -> list[str]:
    r = result or {}
    if not r.get("ok"):
        if r.get("live_s", 0.0) <= 0.0:
            return ["  no live play found in this replay"]
        return ["  boost telemetry too sparse to trust (%.0f s known of %.0f s)"
                % (r.get("known_s", 0.0), r.get("live_s", 0.0))]

    lines = [
        "  live play                  %6.0f s" % r["live_s"],
        "  average boost held         %6.1f" % r["avg_held"],
        "  starved (0-10)             %6.1f %% of live play%s" % (
            r["starved_share"] * 100.0,
            ("   ran dry %dx" % r["dry_count"]) if r["dry_count"] else ""),
        "  below 30 (no aerials)      %6.1f %%" % (r["low_share"] * 100.0),
        "  hoarding (90-100)          %6.1f %%" % (r["hoard_share"] * 100.0),
        "  boost collected            %6.0f /min" % r["collected_per_min"],
        "  boost spent                %6.0f /min" % r["spent_per_min"],
        "  pads taken                 %6d big, %d small" % (
            r["big_pads"], r["small_pads"]),
    ]
    if r["big_pads"]:
        lines.append("  tank when taking a big pad %6.0f   (%d taken above 50)"
                     % (r["big_pad_tank"], r["big_pads_topped_up"]))
    lines.append("  on boost                   %6.1f s  %.1f %% of live play"
                 % (r["boost_time_s"], r["boost_share"] * 100.0))
    lines.append("  wasted at supersonic       %6.1f s  %.0f boost, %.1f %% of spend"
                 % (r["ss_time_s"], r["ss_boost"], r["ss_share"] * 100.0))
    if r.get("lobby_n", 0) > 1 and r.get("rank"):
        lines.append("  lobby average held         %6.1f   (you rank %d of %d)"
                     % (r["lobby_avg_held"], r["rank"], r["lobby_n"]))
    return lines


def tips(result, match, who) -> list[str]:
    r = result or {}
    if not r.get("ok"):
        return []

    out = []
    starved = r["starved_share"] * 100.0
    low = r["low_share"] * 100.0
    hoard = r["hoard_share"] * 100.0

    if starved > 28.0:
        if r["big_per_min"] < 2.5:
            fix = (" You only touched %.1f big pads a minute, so the refills"
                   " are not on your route -- rotate back through a corner"
                   " boost instead of cutting straight to the net."
                   % r["big_per_min"])
        else:
            fix = (" You are collecting fine at %.1f big pads a minute, so this"
                   " is a spending problem: stop burning boost through neutral"
                   " and keep a tank for the challenge." % r["big_per_min"])
        out.append("You were under 10 boost for %.0f%% of live play%s -- that is"
                   " a third of the match with no aerial and no fast recovery.%s"
                   % (starved,
                      (" and ran dry %d times" % r["dry_count"])
                      if r["dry_count"] >= 6 else "",
                      fix))

    if low > 50.0 and starved <= 28.0:
        out.append("You sat below 30 boost for %.0f%% of live play, the line"
                   " under which you cannot go up for a ball. Take the small"
                   " pad line on the way back rather than the straight line --"
                   " you are already on %.1f small pads a minute, so the habit"
                   " is there, it just needs to cover your own half too."
                   % (low, r["small_per_min"]))

    banked = False
    if hoard > 25.0:
        out.append("You were at 90-100 boost for %.0f%% of live play. Every pad"
                   " you crossed in that state was thrown away, and you still"
                   " only spent %.0f a minute. Boost sitting in the tank scores"
                   " nothing -- use it to arrive at the challenge first."
                   % (hoard, r["spent_per_min"]))
        banked = True

    if (not banked and r["avg_held"] >= 58.0 and r["spent_per_min"] < 380.0
            and r["opp_spend_hi"] > 0.0):
        out.append("You averaged %.0f boost and spent %.0f a minute while the"
                   " opponents spent %.0f-%.0f. You are banking boost you never"
                   " cash in -- on a full tank, challenge first instead of"
                   " shadowing, and take the aerial you are currently passing"
                   " up." % (r["avg_held"], r["spent_per_min"],
                             r["opp_spend_lo"], r["opp_spend_hi"]))

    if r["ss_share"] > 0.12 and r["ss_boost"] > 150.0:
        out.append("You burned %.0f boost (%.0f%% of everything you spent) while"
                   " already supersonic, which buys no speed at all. That is"
                   " %.1f big pads thrown away -- let go of the button once the"
                   " trail turns blue."
                   % (r["ss_boost"], r["ss_share"] * 100.0,
                      r["ss_boost"] / 100.0))

    if (r["big_pads"] >= 8 and r["big_pads_topped_up"] >= 4
            and r["big_pads_topped_up"] >= 0.30 * r["big_pads"]):
        out.append("%d of your %d big pads were taken with more than 50 already"
                   " in the tank. A full pad is worth 100 to a teammate on empty"
                   " and maybe 30 to you -- leave those and take the small line."
                   % (r["big_pads_topped_up"], r["big_pads"]))

    if (r["collected_per_min"] < 260.0 and starved > 20.0
            and r["small_per_min"] < 7.0):
        out.append("You collected only %.0f boost a minute (%.1f small pads a"
                   " minute) and spent %.0f%% of the match under 10. You are"
                   " driving over bare floor -- the pad lines cost almost no"
                   " time if you plan the rotation through them."
                   % (r["collected_per_min"], r["small_per_min"], starved))

    return out
