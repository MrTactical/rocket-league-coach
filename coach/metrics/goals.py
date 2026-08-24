"""
What happened around every goal -- yours and theirs.

Season averages tell you that you concede. This tells you WHERE YOU WERE when
each one went in, which is the part you can change on Monday. Four numbers per
goal: how far you were from your own net, whether you were goal-side of the
ball, how much boost you had, and whether you were the last man. Averaged over
the concedes, that is about the most directly actionable line in the analyser.

Three things about the data are worth knowing, because each one silently
produces a wrong answer if you ignore it:

  * `goals_meta` frames index the RAW frame list, not `samples`. They are
    matched here by looking the frame's TIME up in `match.frames` and finding
    the sample carrying that time -- exact, not proportional. A proportional
    fallback covers a Match with no raw frames attached.

  * "Goal-side of the ball" at the exact frame the ball crosses the line is
    meaningless. The ball is behind everyone by definition: at a concede
    nobody is goal-side, at a goal for, everybody is. So goal-side is measured
    LEAD seconds before the crossing, when it still described a decision.

  * The replay sometimes records several seconds of live play with no position
    updates at all -- the ball and every car sit frozen while the match clock
    keeps running. Those frames are not a stationary game, they are missing
    data, and reading a "2 seconds before" position out of one gives you a
    position from six seconds earlier. Frozen stretches where the ball is NOT
    on the kickoff spot and NOT already in a net are flagged and kept out of
    the lead-time averages.
"""

from __future__ import annotations

import bisect
import math

TITLE = "Goals: where you were when it mattered"

# Soccar geometry, unreal units.
GOAL_Y = 5120.0
SIDE_X = 4096.0

LEAD = 2.0            # seconds before the crossing, for goal-side / recovery
BUILDUP = 3.0         # seconds of build-up scanned for "first man"
PAUSE_GAP = 0.8       # a jump in sample time this big is a stoppage
FREEZE_MIN = 0.35     # ball unmoved this long is a frozen stretch
DT_CAP = 0.25         # cap one sample's dt so a stoppage cannot dominate
TOUCH_DV = 350.0      # change in ball velocity that counts as a touch
TOUCH_RADIUS = 400.0  # car centre to ball centre, to own that touch
TOUCH_WINDOW = 6.0    # how far back to look for the shot that scored
LOW_BOOST = 20.0


# -- small helpers -------------------------------------------------------

def _flat(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _d3(a, b):
    try:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                         + (a[2] - b[2]) ** 2)
    except Exception:
        return float("inf")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _clock_str(sec, ot=False):
    if sec is None:
        return "--"
    try:
        sec = int(sec)
    except Exception:
        return "--"
    if ot:
        return "+%d:%02d" % (sec // 60, sec % 60)
    return "%d:%02d" % (sec // 60, sec % 60)


def _dead_ball(b):
    """True where a motionless ball is the game, not a hole in the recording."""
    try:
        x, y, z = b[0], b[1], b[2]
    except Exception:
        return True
    if abs(y) > GOAL_Y - 30.0:      # sitting in or behind a net after a goal
        return True
    if abs(x) < 80.0 and abs(y) < 80.0 and z < 250.0:   # on the kickoff spot
        return True
    return False


def _stale_flags(samples):
    """
    Mark samples whose positions are a stale copy of an earlier frame.

    A run of frames with an identical ball position is either a dead ball
    (fine, the positions are real) or the recording dropping updates while the
    match plays on (not fine). `_dead_ball` separates the two.
    """
    n = len(samples)
    stale = [False] * n
    stall_secs = 0.0
    i = 0
    while i < n:
        j = i
        while (j + 1 < n
               and samples[j + 1].get("ball") == samples[i].get("ball")
               and (samples[j + 1]["t"] - samples[j]["t"]) < PAUSE_GAP):
            j += 1
        dur = samples[j]["t"] - samples[i]["t"]
        if dur >= FREEZE_MIN and not _dead_ball(samples[i].get("ball")):
            for k in range(i + 1, j + 1):
                stale[k] = True
            stall_secs += dur
        i = j + 1
    return stale, stall_secs


def _rewind(samples, j, lead):
    """
    Step back `lead` seconds from index j, stopping at a stoppage.

    Never walks through a gap in sample time: the frames on the far side of one
    belong to the previous phase of play, and reading a position out of them
    would report a kickoff line-up as a defensive shape.
    """
    t0 = samples[j]["t"]
    k = j
    while k > 0:
        if samples[k]["t"] - samples[k - 1]["t"] > PAUSE_GAP:
            break
        if t0 - samples[k - 1]["t"] > lead:
            break
        k -= 1
    return k, t0 - samples[k]["t"]


def _touches(samples):
    """
    Every ball touch we can attribute, from velocity discontinuities.

    A step in ball velocity is a hit, a wall or the floor; the nearest car
    within TOUCH_RADIUS separates a player touch from a bounce.
    """
    out = []
    prev = None
    for s in samples:
        v = s.get("ball_vel") or (0.0, 0.0, 0.0)
        if prev is not None and _d3(v, prev) > TOUCH_DV:
            b = s.get("ball")
            best, bd = None, float("inf")
            for nm, c in (s.get("cars") or {}).items():
                d = _d3(c.get("pos") or (0.0, 0.0, 0.0), b)
                if d < bd:
                    bd, best = d, nm
            if best is not None and bd <= TOUCH_RADIUS:
                out.append({"t": s["t"], "player": best, "ball": b})
        prev = v
    return out


def _ot_start(samples):
    """Time the clock hit zero, if the match then carried on into overtime."""
    zero = None
    for s in samples:
        c = s.get("clock")
        if c is None:
            continue
        if c == 0 and zero is None:
            zero = s["t"]
        elif zero is not None and c > 0:
            return zero
    return None


def _goal_time(match, meta, samples):
    """Raw frame index -> sample time. Exact via frame times where possible."""
    f = meta.get("frame")
    frames = getattr(match, "frames", None) or []
    if isinstance(f, int) and 0 <= f < len(frames):
        t = frames[f].get("time")
        if isinstance(t, (int, float)):
            return float(t), "frame time"
    if isinstance(f, int) and len(frames) > 1:
        frac = max(0.0, min(1.0, f / float(len(frames) - 1)))
        return (samples[0]["t"]
                + frac * (samples[-1]["t"] - samples[0]["t"])), "proportional"
    return None, "unmatched"


def _nearest(times, t):
    i = bisect.bisect_left(times, t)
    best, bd = 0, float("inf")
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(times):
            d = abs(times[j] - t)
            if d < bd:
                bd, best = d, j
    return best


# -- the metric ----------------------------------------------------------

def compute(match, who):
    res = {"ok": False, "player": who, "goals": [], "note": None}
    try:
        return _compute(match, who, res)
    except Exception as exc:                              # must never raise
        res["error"] = "%s: %s" % (type(exc).__name__, exc)
        return res


def _compute(match, who, res):
    samples = getattr(match, "samples", None) or []
    if not samples:
        res["note"] = "no frames in this replay"
        return res

    teams = getattr(match, "teams", None) or {}
    name = who if who in teams else None
    if name is None:
        try:
            name = match.resolve(who)
        except Exception:
            name = None
    if name is None:
        res["note"] = "player %r not found in this replay" % (who,)
        return res
    res["player"] = name

    my_team = teams.get(name)
    sign = match.attack_sign(name)
    own_y = -sign * GOAL_Y          # the net this player defends
    opp_y = sign * GOAL_Y
    team = [n for n, t in teams.items() if t == my_team]

    times = [s["t"] for s in samples]
    stale, stall_secs = _stale_flags(samples)
    touches = _touches(samples)
    ot_t = _ot_start(samples)

    metas = list(getattr(match, "goals_meta", None) or [])
    mapping = "none"
    records = []

    for meta in metas:
        if not isinstance(meta, dict):
            continue
        tg, how = _goal_time(match, meta, samples)
        if tg is None:
            continue
        mapping = how if mapping in ("none", how) else "mixed"
        j = _nearest(times, tg)

        # Snap to the first frame the ball is actually past the line, if one is
        # close by. Costs nothing when the mapping is already exact and rescues
        # it when it is a frame or two out.
        for k in range(max(0, j - 20), min(len(samples), j + 20)):
            b = samples[k].get("ball")
            if (b and abs(b[1]) >= GOAL_Y
                    and abs(samples[k]["t"] - tg) <= 0.6):
                j = k
                break

        s = samples[j]
        ball = s.get("ball") or (0.0, 0.0, 0.0)

        # Which net took it. The ball's own position is ground truth and gets
        # own goals right; PlayerTeam is the fallback.
        if abs(ball[1]) > GOAL_Y - 100.0:
            side = "for" if sign * ball[1] > 0 else "against"
        else:
            side = "for" if meta.get("PlayerTeam") == my_team else "against"

        scorer = meta.get("PlayerName")
        by_you = False
        if scorer:
            try:
                by_you = (match.resolve(scorer) == name)
            except Exception:
                by_you = (scorer == name)

        rec = {
            "t": s["t"], "clock": s.get("clock"),
            "ot": bool(ot_t is not None and s["t"] >= ot_t),
            "side": side, "scorer": scorer, "by_you": by_you,
            "missing": False,
        }

        car = (s.get("cars") or {}).get(name)
        if not car or not car.get("pos"):
            rec["missing"] = True
            records.append(rec)
            continue

        px, py, pz = car["pos"]
        rec["pos"] = (px, py, pz)
        rec["dist_own"] = _flat(px, py, 0.0, own_y)
        rec["dist_opp"] = _flat(px, py, 0.0, opp_y)
        rec["up_pitch"] = GOAL_Y + sign * py       # 0 own line, 5120 halfway
        rec["off_centre"] = abs(px)
        rec["x"] = px
        rec["dist_ball"] = _d3((px, py, pz), ball)
        rec["boost"] = car.get("boost")

        # Last man: nearest of your side to your own net, right now.
        here = {}
        for n in team:
            c = (s.get("cars") or {}).get(n)
            if c and c.get("pos"):
                here[n] = _flat(c["pos"][0], c["pos"][1], 0.0, own_y)
        rec["last_man"] = bool(here) and min(here, key=here.get) == name
        rec["last_man_was"] = min(here, key=here.get) if here else None
        rec["mates_back"] = sum(1 for n, d in here.items()
                                if n != name and d < rec["dist_own"])

        # LEAD seconds earlier -- where the decision actually was.
        k, got = _rewind(samples, j, LEAD)
        s2 = samples[k]
        c2 = (s2.get("cars") or {}).get(name)
        rec["lead_got"] = got
        rec["lead_stale"] = bool(stale[k]) or got < LEAD * 0.6
        rec["lead_dist_own"] = rec["lead_goalside"] = rec["recovery"] = None
        rec["lead_dist_ball"] = None
        if c2 and c2.get("pos"):
            b2 = s2.get("ball") or (0.0, 0.0, 0.0)
            qx, qy = c2["pos"][0], c2["pos"][1]
            rec["lead_dist_own"] = _flat(qx, qy, 0.0, own_y)
            rec["lead_goalside"] = bool(sign * (b2[1] - qy) > 0)
            rec["lead_dist_ball"] = _d3(c2["pos"], b2)
            rec["recovery"] = rec["lead_dist_own"] - rec["dist_own"]

        # Build-up: share of the preceding BUILDUP seconds spent as the
        # teammate closest to the ball. Weighted by real sample spacing --
        # frames are ~26 Hz on average but irregular, and stale frames
        # contribute nothing at all.
        k3, _got3 = _rewind(samples, j, BUILDUP)
        tw = fm = 0.0
        for i in range(k3, j):
            if stale[i]:
                continue
            dt = samples[i + 1]["t"] - samples[i]["t"]
            if dt <= 0.0:
                continue
            dt = min(dt, DT_CAP)
            ss = samples[i]
            b = ss.get("ball")
            db = {}
            for n in team:
                c = (ss.get("cars") or {}).get(n)
                if c and c.get("pos"):
                    db[n] = _d3(c["pos"], b)
            if not db:
                continue
            tw += dt
            if min(db, key=db.get) == name:
                fm += dt
        rec["first_man_pct"] = (100.0 * fm / tw) if tw > 0 else None
        rec["buildup_secs"] = tw

        # Your own shot, for the goals you scored.
        if by_you:
            mine = [x for x in touches
                    if x["player"] == name
                    and 0.0 <= rec["t"] - x["t"] <= TOUCH_WINDOW]
            if mine:
                lb = mine[-1]["ball"]
                rec["shot_dist"] = _flat(lb[0], lb[1], 0.0, opp_y)
                rec["shot_lead"] = rec["t"] - mine[-1]["t"]

        records.append(rec)

    res["goals"] = records
    res["mapping"] = mapping
    res["stall_secs"] = stall_secs
    res["team_size"] = getattr(match, "team_size", 0)

    against = [r for r in records
               if r["side"] == "against" and not r["missing"]]
    forr = [r for r in records if r["side"] == "for" and not r["missing"]]
    mine = [r for r in forr if r["by_you"]]

    res["n_for"] = sum(1 for r in records if r["side"] == "for")
    res["n_against"] = sum(1 for r in records if r["side"] == "against")
    res["n_you"] = sum(1 for r in records if r["by_you"])

    con = {"n": len(against)}
    if against:
        con["dist_own"] = _mean([r.get("dist_own") for r in against])
        con["up_pitch"] = _mean([r.get("up_pitch") for r in against])
        con["off_centre"] = _mean([r.get("off_centre") for r in against])
        con["x"] = _mean([r.get("x") for r in against])
        con["boost"] = _mean([r.get("boost") for r in against])
        con["dist_ball"] = _mean([r.get("dist_ball") for r in against])
        con["first_man_pct"] = _mean([r.get("first_man_pct") for r in against])
        con["boost_known"] = sum(1 for r in against
                                 if r.get("boost") is not None)
        con["low_boost"] = sum(1 for r in against
                               if r.get("boost") is not None
                               and r["boost"] < LOW_BOOST)
        con["last_man"] = sum(1 for r in against if r.get("last_man"))
        con["last_man_of"] = len(against)
        con["in_opp_half"] = sum(1 for r in against
                                 if (r.get("up_pitch") or 0.0) > GOAL_Y)
        clean = [r for r in against
                 if not r.get("lead_stale") and r.get("lead_goalside") is not None]
        con["lead_n"] = len(clean)
        con["goalside"] = sum(1 for r in clean if r["lead_goalside"])
        con["lead_dist_own"] = _mean([r.get("lead_dist_own") for r in clean])
        con["lead_dist_ball"] = _mean([r.get("lead_dist_ball") for r in clean])
        con["clean_dist_own"] = _mean([r.get("dist_own") for r in clean])
        con["recovery"] = _mean([r.get("recovery") for r in clean])
        con["worst"] = max(against, key=lambda r: r.get("dist_own") or 0.0)
        tally = {}
        for r in against:
            lm = r.get("last_man_was")
            if lm:
                tally[lm] = tally.get(lm, 0) + 1
        con["last_man_tally"] = tally
        xs = [r.get("x") for r in against if r.get("x") is not None]
        con["one_flank"] = bool(len(xs) >= 3
                                and (all(x > 0 for x in xs)
                                     or all(x < 0 for x in xs)))
    res["conceded"] = con

    sc = {"n_team": len(forr), "n_you": len(mine)}
    if mine:
        sc["shot_dist"] = _mean([r.get("shot_dist") for r in mine])
        sc["shot_known"] = sum(1 for r in mine if r.get("shot_dist") is not None)
        sc["first_man_pct"] = _mean([r.get("first_man_pct") for r in mine])
        sc["boost"] = _mean([r.get("boost") for r in mine])
    mate_goals = [r for r in forr if not r["by_you"]]
    if mate_goals:
        sc["n_mate"] = len(mate_goals)
        sc["mate_dist_own"] = _mean([r.get("dist_own") for r in mate_goals])
        sc["mate_first_man_pct"] = _mean([r.get("first_man_pct")
                                          for r in mate_goals])
    res["scored"] = sc

    # Cross-check against the replay header, which is computed by the game and
    # is independent of everything above.
    try:
        res["header_goals"] = match.stat_line(name).get("Goals")
    except Exception:
        res["header_goals"] = None
    try:
        s0, s1 = match.score
        res["header_for"] = s1 if my_team == 1 else s0
        res["header_against"] = s0 if my_team == 1 else s1
    except Exception:
        res["header_for"] = res["header_against"] = None

    res["ok"] = True
    return res


# -- output --------------------------------------------------------------

def _fmt_boost(b):
    return "%3.0f" % b if b is not None else " ??"


def render(result):
    if not result.get("ok"):
        return ["  %s" % (result.get("error") or result.get("note")
                          or "no goal data")]

    out = []
    n_for = result.get("n_for", 0)
    n_ag = result.get("n_against", 0)
    n_you = result.get("n_you", 0)
    hf, ha = result.get("header_for"), result.get("header_against")
    hg = result.get("header_goals")

    tail = ""
    if hf is not None and (hf != n_for or ha != n_ag):
        tail = "   (header says %s-%s)" % (hf, ha)
    out.append("  goals for / against  %3d / %-3d%s" % (n_for, n_ag, tail))
    chk = ""
    if hg is not None:
        chk = ("   (header agrees)" if hg == n_you
               else "   (header says %s -- MISMATCH)" % hg)
    out.append("  scored by you        %3d%s" % (n_you, chk))

    con = result.get("conceded") or {}
    out.append("")
    out.append("  CONCEDED  (%d)" % con.get("n", 0))
    if not con.get("n"):
        out.append("    nothing went in at your end")
    else:
        out.append("    distance from own net   %6.0f uu    average, straight line"
                   % (con.get("dist_own") or 0.0))
        out.append("    up-pitch along field    %6.0f uu    (own line 0, halfway %d)"
                   % (con.get("up_pitch") or 0.0, int(GOAL_Y)))
        out.append("    off centre              %6.0f uu    (side wall %d)"
                   % (con.get("off_centre") or 0.0, int(SIDE_X)))
        b = con.get("boost")
        out.append("    boost in hand           %6s       %d of %d under %d"
                   % ("%.0f" % b if b is not None else "??",
                      con.get("low_boost", 0), con.get("boost_known", 0),
                      int(LOW_BOOST)))
        out.append("    last man back           %3d of %d"
                   % (con.get("last_man", 0), con.get("last_man_of", 0)))
        out.append("    caught in their half    %3d of %d"
                   % (con.get("in_opp_half", 0), con.get("n", 0)))
        fm = con.get("first_man_pct")
        out.append("    first man, last %ds      %5s %%     share of the build-up"
                   % (int(BUILDUP), "%.0f" % fm if fm is not None else "??"))

        ln = con.get("lead_n", 0)
        if ln:
            out.append("    goal-side %.0fs earlier   %3d of %d    (at the crossing"
                       " nobody is: the ball is behind everyone)"
                       % (LEAD, con.get("goalside", 0), ln))
            ld, cd = con.get("lead_dist_own"), con.get("clean_dist_own")
            db = con.get("lead_dist_ball")
            if ld is not None and cd is not None:
                out.append("      on those %d you were %.0f uu out %.0fs earlier "
                           "and %.0f uu out at the goal," % (ln, ld, LEAD, cd))
                out.append("      so you closed %.0f uu%s"
                           % (ld - cd,
                              " while %.0f uu from the ball" % db
                              if db is not None else ""))
        else:
            out.append("    goal-side %.0fs earlier    -- no clean frames to read"
                       % LEAD)

        tally = con.get("last_man_tally") or {}
        if tally:
            me = result.get("player")
            bits = ["%s x%d" % ("you" if k == me else k, v)
                    for k, v in sorted(tally.items(), key=lambda kv: -kv[1])]
            out.append("    last man on those was   %s" % ", ".join(bits))

        out.append("")
        out.append("      clock   own net  up-pitch  off-ctr  boost  mates back"
                   "  last man  g-side")
        for r in [x for x in result["goals"] if x["side"] == "against"]:
            cs = _clock_str(r.get("clock"), r.get("ot"))
            if r.get("missing"):
                out.append("      %-6s  you were not on the field" % cs)
                continue
            gs = r.get("lead_goalside")
            gs_s = ("--" if (gs is None or r.get("lead_stale"))
                    else ("yes" if gs else "no"))
            out.append("      %-6s %8.0f  %8.0f  %7.0f    %s  %8d  %-8s  %-4s%s"
                       % (cs, r.get("dist_own") or 0.0,
                          r.get("up_pitch") or 0.0,
                          r.get("off_centre") or 0.0,
                          _fmt_boost(r.get("boost")),
                          r.get("mates_back") or 0,
                          "yes" if r.get("last_man") else "no",
                          gs_s,
                          " *stale" if r.get("lead_stale") else ""))

    sc = result.get("scored") or {}
    out.append("")
    out.append("  SCORED  (%d of your team's %d)"
               % (sc.get("n_you", 0), sc.get("n_team", 0)))
    if not sc.get("n_you"):
        out.append("    you did not score in this match")
    else:
        sd = sc.get("shot_dist")
        if sd is not None:
            out.append("    scored from             %6.0f uu    average of %d "
                       "traced shots" % (sd, sc.get("shot_known", 0)))
        fm = sc.get("first_man_pct")
        out.append("    first man, last %ds      %5s %%     share of the build-up"
                   % (int(BUILDUP), "%.0f" % fm if fm is not None else "??"))
        out.append("")
        out.append("      clock   shot from  first man 3s  boost  own net")
        for r in [x for x in result["goals"] if x["by_you"] and not x["missing"]]:
            sdi, fmi = r.get("shot_dist"), r.get("first_man_pct")
            out.append("      %-6s %10s  %12s    %s %8.0f"
                       % (_clock_str(r.get("clock"), r.get("ot")),
                          "%.0f uu" % sdi if sdi is not None else "--",
                          "%.0f %%" % fmi if fmi is not None else "--",
                          _fmt_boost(r.get("boost")),
                          r.get("dist_own") or 0.0))
    if sc.get("n_mate"):
        out.append("    on your mates' %d goals you were %.0f uu from your own "
                   "net" % (sc["n_mate"], sc.get("mate_dist_own") or 0.0))

    ss = result.get("stall_secs") or 0.0
    stale_n = sum(1 for r in result["goals"] if r.get("lead_stale"))
    if ss > 1.0 or stale_n:
        out.append("")
        out.append("    data: goals matched by %s; %.1fs of live play recorded "
                   "with frozen" % (result.get("mapping"), ss))
        out.append("    positions, so %d goal(s) have no readable %.0fs snapshot "
                   "(*stale)" % (stale_n, LEAD))
    return [ln.rstrip() for ln in out]


def tips(result, match, who):
    if not result.get("ok"):
        return []
    out = []
    con = result.get("conceded") or {}
    sc = result.get("scored") or {}
    n = con.get("n", 0)
    if not n:
        return out

    dist = con.get("dist_own")
    up = con.get("up_pitch")
    off = con.get("off_centre")
    ln = con.get("lead_n", 0)
    gs = con.get("goalside", 0)
    fm = con.get("first_man_pct")

    # 1. Caught up-pitch. Halfway is 5120 uu from your own net, so a competent
    #    player averaging past ~4200 is living too far forward.
    if dist is not None and dist > 4200.0:
        out.append(
            "You are on average %.0f uu from your own net at the moment the "
            "ball crosses your line, against a halfway line at %.0f uu. Over "
            "%d concedes you were never in a position to make the save; the "
            "fix is leaving the attack a beat earlier, not defending harder."
            % (dist, GOAL_Y, n))
    elif up is not None and up > GOAL_Y and n >= 2:
        out.append(
            "Your average up-pitch position when conceding is %.0f uu, past "
            "the halfway line at %.0f. You are being scored on from the "
            "counter, so look at when you commit rather than at your defending."
            % (up, GOAL_Y))

    # 2. Wide. A defender near the wall cannot cover the net however deep they
    #    are. Half-width is 4096, so a 2200+ average is genuinely off to one side.
    if off is not None and off > 2200.0 and n >= 3:
        flank = (", every one of them on the same side of the pitch"
                 if con.get("one_flank") else "")
        out.append(
            "On your %d concedes you averaged %.0f uu off centre against a "
            "%.0f uu half-width%s. You are defending from the wing: shade back "
            "toward the middle of your own third so you are between the ball "
            "and the net rather than beside it." % (n, off, SIDE_X, flank))

    # 3. Goal-side, measured LEAD seconds out, when it was still a choice.
    if ln >= 3 and gs == 0:
        out.append(
            "Two seconds before the ball crossed, you were ahead of it on all "
            "%d readable concedes. Being up-pitch of the ball as the attack "
            "starts is what turns a shot into a goal -- get behind it before "
            "you challenge." % ln)
    elif ln >= 3 and gs <= ln * 0.34:
        out.append(
            "You were goal-side of the ball two seconds before only %d of %d "
            "concedes. That is the last moment you can still change the "
            "outcome; after it you are chasing." % (gs, ln))

    # 4. Neither challenging nor covering -- the no-man's-land pattern.
    if (fm is not None and fm < 25.0 and n >= 3
            and ln >= 2 and gs <= ln * 0.5):
        out.append(
            "In the three seconds before your concedes you were first man only "
            "%.0f%% of the time, and goal-side %d times out of %d. You are "
            "neither pressuring the ball nor covering the net. Pick one -- the "
            "middle third does neither job." % (fm, gs, ln))

    # 5. Boost.
    known, low = con.get("boost_known", 0), con.get("low_boost", 0)
    if known >= 3 and low >= max(2, known * 0.5):
        out.append(
            "You had under %d boost on %d of %d concedes. Empty in your own "
            "half means you cannot close, cannot clear and cannot recover; "
            "take the small pads on the way back."
            % (int(LOW_BOOST), low, known))

    # 6. Last man -- only speak up at the extremes, since both are real but
    #    opposite problems.
    lm, lmo = con.get("last_man", 0), con.get("last_man_of", 0)
    size = result.get("team_size") or 3
    if lmo >= 3 and lm >= lmo * 0.75:
        out.append(
            "You were the last man back on %d of %d concedes. You are the one "
            "being beaten, so the work is in the shadow-defence retreat rather "
            "than in the challenge itself." % (lm, lmo))
    elif lmo >= 3 and lm == 0 and size >= 3 and (dist or 0.0) > 3000.0:
        out.append(
            "You were never the last man on any of your %d concedes -- a "
            "teammate was covering every time, while you sat %.0f uu from your "
            "own net. Fine if you were creating; check you are not simply "
            "playing as a third attacker." % (lmo, dist or 0.0))

    # 7. Your own goals: created end to end, or finished off a teammate?
    if sc.get("n_you", 0) >= 2 and sc.get("first_man_pct") is not None:
        if sc["first_man_pct"] >= 75.0:
            out.append(
                "You were first man for %.0f%% of the build-up to your own "
                "goals. Every goal you score you also carry, which is worth "
                "checking against how often your teammates get the ball."
                % sc["first_man_pct"])

    # 8. One concede far worse than the rest is a specific clip to rewatch.
    worst = con.get("worst")
    if worst and (worst.get("dist_own") or 0.0) > 6000.0:
        out.append(
            "Your worst concede was at %s on the clock, %.0f uu from your own "
            "net with %s boost. Rewatch that one clip: at that distance the "
            "goal was decided several seconds before the shot."
            % (_clock_str(worst.get("clock"), worst.get("ot")),
               worst["dist_own"],
               "%.0f" % worst["boost"] if worst.get("boost") is not None
               else "unknown"))

    return out
