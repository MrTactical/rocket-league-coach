"""
Kickoff performance.

A kickoff is the one moment of a match that repeats with the whole board reset:
same ball, same five spawns, same countdown. Ten of them in a normal game, each
deciding who attacks for the next ten seconds -- and almost nobody ever reviews
theirs.

Nothing in the replay flags a kickoff, so it is reconstructed from three facts:

  * The ball is teleported to the exact centre spot and put to sleep. Nothing
    else in a match parks it dead still on (0, 0, 92.75), so a run of frames
    with the ball inside 25 uu of centre and under 5 uu/s is a kickoff setting
    up. In the sample replay those runs last 5.7-6.0 s each.

  * During the countdown the cars are frozen on five canonical spawns -- but
    they are NOT motionless in the data. Each car is dropped in from about
    z = 35 and its suspension settles, so the reported speed reads 50-140 uu/s
    for the first half second and never falls below 7. The HORIZONTAL speed,
    however, is exactly 0.0 for every car until the whistle and jumps to 200+
    on the very next frame. That transition is the release, and it is the only
    clean way to time one. Using full speed instead puts the release up to
    six seconds early.

  * First contact is the first frame the ball leaves the spot. The exact moment
    comes from walking the ball back along its own velocity -- it has moved
    only 7-9 uu by the frame it is first seen moving, so this lands within a
    hundredth of a second.

Spawn names use the usual convention, seen from the player's own end:

    right corner (-2048, -2560)      left corner (2048, -2560)
    back right   ( -256, -3840)      back left   ( 256, -3840)
    back centre  (    0, -4608)

Orange mirrors blue through the origin, so multiplying both x and y by
match.attack_sign() puts every car in its own terms. In the sample replay all
sixty car placements across ten kickoffs land on one of those five points to
within 0.0 uu, which is a strong check that the mirroring is right.

What the numbers mean, in order of how much they change a game:

  went for it   -- you were inside 800 uu of the ball when it was struck. The
                   split is not marginal: in the sample replay a player who
                   drove at it was 250-620 uu away and a player who did not was
                   1300-4900 uu away.
  took it       -- of your team, you were the closest when it was struck.
  won           -- three seconds after first contact the ball is more than
                   1000 uu into the opponents' half. Inside that band either
                   way is neutral, because a ball still near halfway has not
                   been won by anybody.
  first flip    -- when your first dodge came after the whistle. A speed-flip
                   fires inside the first second; a plain forward flip at
                   1.5 s means you arrived without one.
"""

from __future__ import annotations

import bisect
import math

TITLE = "Kickoffs"

# -- geometry ---------------------------------------------------------------

SPAWNS = (
    (-2048.0, -2560.0, "right corner"),
    (2048.0, -2560.0, "left corner"),
    (-256.0, -3840.0, "back right"),
    (256.0, -3840.0, "back left"),
    (0.0, -4608.0, "back centre"),
)

# -- detection --------------------------------------------------------------

CENTRE_XY = 25.0        # uu, ball's distance from the centre spot
BALL_Z_LO, BALL_Z_HI = 80.0, 115.0
BALL_ASLEEP = 5.0       # uu/s
RUN_GAP = 0.5           # s, break that separates two set-ups
MIN_SET = 0.4           # s, the ball must sit still at least this long
FROZEN_HSPEED = 8.0     # uu/s of horizontal car speed still counts as frozen
SPAWN_TOL = 80.0        # uu, how close a car must sit to a canonical spawn

# -- judgement --------------------------------------------------------------

COMMIT_UU = 800.0       # inside this of the ball at first touch = you went
NEUTRAL_UU = 1000.0     # ball inside this of halfway at +3 s = neither half
OUTCOME_DT = 3.0        # s after first contact
GOAL_WINDOW = 10.0      # s, a goal this soon after counts as kickoff-driven
FLIP_WINDOW = 2.5       # s after release, how far to look for a first flip
EARLY_FLIP = 1.00       # s, a flip this early is a speed-flip


def _sp(v):
    try:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    except Exception:
        return 0.0


def _hsp(v):
    try:
        return math.hypot(v[0], v[1])
    except Exception:
        return 0.0


def _d3(a, b):
    try:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                         + (a[2] - b[2]) ** 2)
    except Exception:
        return float("inf")


def _spawn(pos, sign):
    """Nearest canonical spawn: (name, its distance to the ball, fit error)."""
    ox, oy = pos[0] * sign, pos[1] * sign
    name, reach, err = None, 0.0, float("inf")
    for sx, sy, nm in SPAWNS:
        e = math.hypot(ox - sx, oy - sy)
        if e < err:
            name, reach, err = nm, math.hypot(sx, sy), e
    return name, reach, err


def _clock(sec):
    if sec is None:
        return "  -  "
    sec = max(0, int(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _goal_times(match):
    """(time, team, scorer) per goal. goals_meta carries a frame, not a time."""
    out = []
    frames = getattr(match, "frames", None)
    for g in (getattr(match, "goals_meta", None) or []):
        try:
            t = frames[g["frame"]].get("time")
            if t is not None:
                out.append((float(t), g.get("PlayerTeam"), g.get("PlayerName")))
        except Exception:
            continue
    out.sort()
    return out


def _find_kickoffs(match):
    """(frozen index, first-contact index, cars on spawns, cars seen) each."""
    S = match.samples
    settled = []
    for i, s in enumerate(S):
        try:
            b = s["ball"]
            if (math.hypot(b[0], b[1]) < CENTRE_XY
                    and BALL_Z_LO < b[2] < BALL_Z_HI
                    and _sp(s.get("ball_vel") or (0.0, 0.0, 0.0)) < BALL_ASLEEP):
                settled.append(i)
        except Exception:
            continue

    runs = []
    for i in settled:
        if not runs or S[i]["t"] - S[runs[-1][-1]]["t"] > RUN_GAP:
            runs.append([i])
        else:
            runs[-1].append(i)

    out = []
    for run in runs:
        if S[run[-1]]["t"] - S[run[0]]["t"] < MIN_SET:
            continue

        frozen = None
        for i in run:
            cars = S[i].get("cars") or {}
            if cars and max(_hsp(c.get("vel")) for c in cars.values()) \
                    < FROZEN_HSPEED:
                frozen = i
        if frozen is None or frozen + 1 >= len(S):
            continue

        cars = S[frozen]["cars"]
        on_spawn = sum(
            1 for n, c in cars.items()
            if _spawn(c["pos"], match.attack_sign(n))[2] < SPAWN_TOL)
        if on_spawn < 2 or on_spawn < 0.6 * len(cars):
            continue

        hit = run[-1] + 1
        while hit < len(S):
            b = S[hit]["ball"]
            if (_sp(S[hit].get("ball_vel") or (0.0, 0.0, 0.0)) >= BALL_ASLEEP
                    or math.hypot(b[0], b[1]) >= CENTRE_XY):
                break
            hit += 1
        if hit >= len(S):
            continue

        out.append((frozen, hit, on_spawn, len(cars)))
    return out


def compute(match, who) -> dict:
    res = {
        "ok": False, "player": who, "note": None, "kickoffs": [],
        "n": 0, "n_expected": None,
    }
    try:
        S = getattr(match, "samples", None) or []
        if not S:
            res["note"] = "no frames in this replay"
            return res
        name = match.resolve(who) if who else None
        if not name:
            res["note"] = "player %r not found in this replay" % (who,)
            return res
        res["player"] = name

        mine = match.teams.get(name)
        sign = match.attack_sign(name)
        squad = set(match.mates(name)) | {name}
        goals = _goal_times(match)
        res["n_expected"] = (len(goals) + 1) if goals else None

        flips = {}
        for e in (getattr(match, "events", None) or []):
            if e.get("kind") == "dodge" and e.get("player"):
                flips.setdefault(e["player"], []).append(e["t"])
        for v in flips.values():
            v.sort()

        times = [s["t"] for s in S]

        def sample_at(t):
            j = bisect.bisect_left(times, t)
            if j <= 0:
                return S[0]
            if j >= len(S):
                return S[-1]
            return S[j] if (times[j] - t) < (t - times[j - 1]) else S[j - 1]

        found = _find_kickoffs(match)
        res["n"] = len(found)

        for k, (frozen, hit, on_spawn, seen) in enumerate(found):
            f, b = S[frozen], S[hit]
            release = (f["t"] + S[frozen + 1]["t"]) / 2.0
            speed = _sp(b.get("ball_vel") or (0.0, 0.0, 0.0))
            drift = math.hypot(b["ball"][0], b["ball"][1])
            contact = max(release,
                          b["t"] - (drift / speed if speed > 1.0 else 0.0))

            dist = {n: _d3(c["pos"], b["ball"])
                    for n, c in (b.get("cars") or {}).items()}
            ours = {n: d for n, d in dist.items() if n in squad}
            toucher = min(dist, key=dist.get) if dist else None
            lead = min(ours, key=ours.get) if ours else None
            my_d = dist.get(name)
            cut = max(COMMIT_UU, 1.5 * ours[lead]) if lead else COMMIT_UU
            went = my_d is not None and my_d <= cut
            with_mate = sorted(n for n, d in ours.items()
                               if n != name and d <= cut) if went else []

            car = (f.get("cars") or {}).get(name)
            spawn, reach = (None, None)
            if car:
                spawn, reach, _err = _spawn(car["pos"], sign)
            reaches = []
            for n in squad:
                c = (f.get("cars") or {}).get(n)
                if c:
                    reaches.append(_spawn(c["pos"], match.attack_sign(n))[1])
            front = bool(reach is not None and reaches
                         and reach <= min(reaches) + 1.0)

            def first_flip(pn):
                for t in flips.get(pn, ()):
                    if release + 0.05 <= t <= release + FLIP_WINDOW:
                        return t - release
                return None

            # Where the ball stands three seconds later. A goal inside that
            # window resets the ball to centre, so score it by the goal.
            horizon = contact + OUTCOME_DT
            scored = next((g for g in goals if contact < g[0] <= horizon), None)
            if scored is not None:
                ball_y = 6000.0 if scored[1] == mine else -6000.0
                shot = sample_at(min(horizon, scored[0]))
            else:
                if k + 1 < len(found):
                    horizon = min(horizon, S[found[k + 1][0]]["t"] - 0.2)
                shot = sample_at(horizon)
                ball_y = shot["ball"][1] * sign
            outcome = ("won" if ball_y > NEUTRAL_UU
                       else "lost" if ball_y < -NEUTRAL_UU else "neutral")
            me3 = (shot.get("cars") or {}).get(name)
            after = next((g for g in goals
                          if contact < g[0] <= contact + GOAL_WINDOW), None)

            res["kickoffs"].append({
                "i": k + 1,
                "t": contact,
                "clock": f.get("clock"),
                "release": release,
                "ttc": contact - release,
                "spawn": spawn,
                "reach": reach,
                "front": front,
                # Where the SECOND man stood when the ball was struck --
                # "cheating up". Bimodal in practice: about 2000 uu (cheated)
                # or about 4800 (stayed home).
                "mate2_d": (sorted(ours.values())[1]
                            if len(ours) > 1 else None),
                "went": bool(went),
                "took": bool(lead == name and went),
                "with_mate": with_mate,
                "my_dist": my_d,
                "toucher": toucher,
                "gap_uu": (my_d - dist[toucher])
                          if (my_d is not None and toucher) else None,
                "my_speed": _sp((((b.get("cars") or {}).get(name)) or {})
                                .get("vel") or (0.0, 0.0, 0.0)),
                "my_flip": first_flip(name),
                "winner_flip": first_flip(toucher) if toucher else None,
                "ball_y3": ball_y,
                "outcome": outcome,
                "possession": shot.get("hit_team") == mine,
                "boost3": (me3 or {}).get("boost"),
                "my_y3": (me3["pos"][1] * sign) if me3 else None,
                "goal_dt": (after[0] - contact) if after else None,
                "goal_for": (after[1] == mine) if after else None,
                "on_spawn": "%d/%d" % (on_spawn, seen),
            })

        ks = res["kickoffs"]
        res["ok"] = bool(ks)
        gone = [k for k in ks if k["went"]]
        res["went"] = len(gone)

        # Cheating up: did having the second man advanced actually win the
        # kickoff? Outcome is where the ball sits three seconds later.
        CHEAT_UU = 2200.0
        up = [k for k in ks if k.get("mate2_d") is not None
              and k["mate2_d"] < CHEAT_UU and k.get("ball_y3") is not None]
        back = [k for k in ks if k.get("mate2_d") is not None
                and k["mate2_d"] >= CHEAT_UU and k.get("ball_y3") is not None]
        res["cheat_n"] = len(up)
        res["cheat_won"] = sum(1 for k in up if k["ball_y3"] > 0)
        res["back_n"] = len(back)
        res["back_won"] = sum(1 for k in back if k["ball_y3"] > 0)
        res["took"] = sum(1 for k in ks if k["took"])
        res["first_touch"] = sum(1 for k in gone if k["toucher"] == name)
        res["beaten"] = sum(1 for k in gone if k["toucher"] != name)
        res["double"] = sum(1 for k in gone if k["with_mate"])
        res["front_spawns"] = sum(1 for k in ks if k["front"])
        res["went_off_role"] = sum(1 for k in gone if not k["front"])
        res["skipped_front"] = sum(1 for k in ks if k["front"] and not k["went"])

        for tag, pool in (("", ks), ("_mine", gone)):
            for label in ("won", "neutral", "lost"):
                res[label + tag] = sum(1 for k in pool
                                       if k["outcome"] == label)
        res["win_rate"] = (res["won_mine"] / len(gone)) if gone else None
        res["team_win_rate"] = (res["won"] / len(ks)) if ks else None
        res["poss_rate"] = ((sum(1 for k in gone if k["possession"]) / len(gone))
                            if gone else None)

        res["ttc_mine"] = _mean([k["ttc"] for k in gone])
        res["ttc_all"] = _mean([k["ttc"] for k in ks])
        res["my_speed"] = _mean([k["my_speed"] for k in gone])
        res["gap_uu"] = _mean([k["gap_uu"] for k in gone
                               if k["gap_uu"] and k["gap_uu"] > 0.0])
        res["gap_s"] = _mean([k["gap_uu"] / k["my_speed"] for k in gone
                              if k["gap_uu"] and k["my_speed"] > 200.0])
        res["flip_mine"] = _mean([k["my_flip"] for k in gone])
        res["flip_winner"] = _mean([k["winner_flip"] for k in ks])
        res["early_flips"] = sum(1 for k in gone
                                 if k["my_flip"] is not None
                                 and k["my_flip"] <= EARLY_FLIP)
        res["boost3_back"] = _mean([k["boost3"] for k in ks if not k["went"]])
        res["y3_back"] = _mean([k["my_y3"] for k in ks if not k["went"]])
        res["goals_for"] = sum(1 for k in ks if k["goal_for"] is True)
        res["goals_against"] = sum(1 for k in ks if k["goal_for"] is False)

        partners = {}
        for k in gone:
            for n in k["with_mate"]:
                partners[n] = partners.get(n, 0) + 1
        res["partner"] = max(partners, key=partners.get) if partners else None

        spawns = {}
        for k in ks:
            row = spawns.setdefault(k["spawn"] or "unknown",
                                    {"n": 0, "went": 0, "won": 0})
            row["n"] += 1
            row["went"] += int(k["went"])
            row["won"] += int(k["outcome"] == "won")
        res["by_spawn"] = spawns
    except Exception as exc:                    # never raise on odd data
        res["ok"] = False
        res["note"] = "kickoff analysis failed: %s" % (exc,)
    return res


def render(result) -> list[str]:
    r = result or {}
    if not r.get("ok"):
        return ["  %s" % (r.get("note") or "no kickoffs found in this replay")]

    ks = r["kickoffs"]
    lines = ["   #  clock  your spawn    your move   1st touch"
             "  ball @+3s  outcome  goal"]
    for k in ks:
        move = ("took it" if k["took"]
                else "also went" if k["went"] else "hung back")
        goal = "-"
        if k["goal_dt"] is not None:
            goal = "%s %4.1fs" % ("for" if k["goal_for"] else "vs ",
                                  k["goal_dt"])
        lines.append("  %2d  %5s  %-12s  %-10s %8.2f s  %+9.0f  %-7s  %s"
                     % (k["i"], _clock(k["clock"]), k["spawn"] or "?", move,
                        k["ttc"], k["ball_y3"], k["outcome"], goal))
    lines.append("")

    n, gone = r["n"], r["went"]
    tail = ""
    if r.get("n_expected"):
        tail = "   %d goals + 1 start" % (r["n_expected"] - 1)
    lines.append("  kickoffs detected          %6d%s" % (n, tail))
    lines.append("  you drove at the ball      %6d   of %d" % (gone, n))
    lines.append("  first of your team there   %6d" % r["took"])

    if gone:
        lines.append("  you touched it first       %6d   of %d"
                     % (r["first_touch"], gone))
        if r["gap_uu"] is not None:
            lines.append("  short at the first touch   %6.0f uu about %.2f s "
                         "late at %.0f uu/s"
                         % (r["gap_uu"], r["gap_s"] or 0.0, r["my_speed"] or 0.0))
        lines.append("  won / neutral / lost       %6s   on the ones you went for"
                     % ("%d/%d/%d" % (r["won_mine"], r["neutral_mine"],
                                      r["lost_mine"])))
        if r["win_rate"] is not None:
            lines.append("  your kickoff win rate      %6.0f %%" %
                         (100.0 * r["win_rate"]))
        if r["poss_rate"] is not None:
            lines.append("  last touch yours at +3 s   %6.0f %%" %
                         (100.0 * r["poss_rate"]))
    lines.append("  team won / neutral / lost  %6s   over all %d"
                 % ("%d/%d/%d" % (r["won"], r["neutral"], r["lost"]), n))

    if r["ttc_mine"] is not None:
        lines.append("  time to first contact      %6.2f s  on yours, %.2f s "
                     "match-wide" % (r["ttc_mine"], r["ttc_all"]))
    elif r["ttc_all"] is not None:
        lines.append("  time to first contact      %6.2f s  match-wide"
                     % r["ttc_all"])
    if r["flip_mine"] is not None:
        lines.append("  your first flip            %6.2f s  after the whistle,"
                     " %d of %d inside %.2f s"
                     % (r["flip_mine"], r["early_flips"], gone, EARLY_FLIP))
    if r["flip_winner"] is not None:
        lines.append("  first flip of the winner   %6.2f s  after the whistle"
                     % r["flip_winner"])
    if r["double"]:
        lines.append("  both of you charged it     %6d   with %s"
                     % (r["double"], r["partner"] or "a team mate"))
    lines.append("  front spawn, and you went  %6s"
                 % ("%d of %d" % (gone - r["went_off_role"],
                                  r["front_spawns"])))
    if r["boost3_back"] is not None:
        lines.append("  hanging back, boost at 3 s %6.0f    %.0f uu from halfway"
                     % (r["boost3_back"], r["y3_back"] or 0.0))
    lines.append("  goals inside %.0f s of one   %6s"
                 % (GOAL_WINDOW, "%d for, %d against"
                    % (r["goals_for"], r["goals_against"])))
    if r.get("n_expected") and n != r["n_expected"]:
        lines.append("  note: expected %d kickoffs from the goal list, found %d"
                     % (r["n_expected"], n))
    return lines


def tips(result, match, who) -> list[str]:
    r = result or {}
    out = []
    if not r.get("ok"):
        return out
    ks, n, gone = r["kickoffs"], r["n"], r["went"]

    mine, winner = r.get("flip_mine"), r.get("flip_winner")
    if gone >= 3 and mine is not None and mine > 1.15 \
            and r["early_flips"] == 0:
        line = ("Your first flip on the %d kickoffs you drove at came %.2f s "
                "after the whistle and never once inside %.2f s."
                % (gone, mine, EARLY_FLIP))
        if winner is not None and winner < mine - 0.3:
            line += (" Whoever reached the ball first across this match's "
                     "kickoffs flipped at %.2f s." % winner)
        line += (" Jump and diagonal-dodge in the first half second: the "
                 "speed-flip is what buys the extra car length.")
        out.append(line)

    if gone >= 3 and r["beaten"] >= max(2, 0.6 * gone) and r.get("gap_uu"):
        out.append(
            "You were beaten to the ball on %d of the %d kickoffs you drove "
            "at, still %.0f uu short when it was struck -- about %.2f s late "
            "at the %.0f uu/s you were carrying."
            % (r["beaten"], gone, r["gap_uu"], r.get("gap_s") or 0.0,
               r.get("my_speed") or 0.0))

    if r["double"] >= 2:
        when = ", ".join(_clock(k["clock"]) for k in ks
                         if k["went"] and k["with_mate"])
        out.append(
            "You and %s both charged the ball on %d kickoffs (at %s on the "
            "clock). When your team holds both corner spawns only one of you "
            "can win it -- call it before the whistle and let the other peel "
            "for the big pad." % (r["partner"] or "a team mate", r["double"],
                                  when))

    if gone >= 4 and r["won_mine"] == 0:
        worst = min((k["ball_y3"] for k in ks if k["went"]), default=0.0)
        out.append(
            "Three seconds after every one of your %d kickoffs the ball was "
            "still out of the opponents' half, and on the worst it sat %.0f uu "
            "inside your own. Angle the first touch wide toward the far corner "
            "instead of straight back up the middle." % (gone, abs(worst)))

    if r["goals_against"] >= 3:
        out.append(
            "%d of your %d kickoffs led to a goal against inside %.0f s. The "
            "restart is leaking goals, not just possession -- make sure one of "
            "you is genuinely back before the whistle."
            % (r["goals_against"], n, GOAL_WINDOW))

    if r["skipped_front"] >= 3:
        out.append(
            "On %d kickoffs you had the spawn closest to the ball and did not "
            "go for it, which hands the opponents a free first touch. The "
            "front spawn takes the kickoff." % r["skipped_front"])

    # NOTE: no cheating-up tip here. One match holds about seven
    # kickoffs, so any threshold strong enough to trust cannot be
    # reached inside a single match -- the comparison is made over
    # the whole record in page.py instead, where it has hundreds.

    if r["went_off_role"] >= 3:
        out.append(
            "You drove at the ball on %d kickoffs where a team mate spawned "
            "closer to it. Leave those to the front spawn and spend the head "
            "start on boost instead." % r["went_off_role"])

    back = n - gone
    boost = r.get("boost3_back")
    if back >= 3 and boost is not None and boost < 45.0:
        out.append(
            "On the %d kickoffs you hung back you held only %.0f boost three "
            "seconds in. From a back spawn the corner big pad is directly on "
            "your way -- take it every time." % (back, boost))
    return out
