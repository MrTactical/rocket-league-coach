"""
Playback track and per-moment coaching for the in-page replay viewer.

Carries height and heading as well as position, so the viewer can draw a real
perspective view rather than a plan. Heading comes from velocity -- the
timeline does not keep the orientation matrix, and a car points where it is
going closely enough for a replay at this scale.

Frame layout, flat for size:

    [t, ball_x, ball_y, ball_z, (car_x, car_y, car_z, car_heading) * N]

Everything is rounded to whole units. At ~10 Hz a full match is a few hundred
kilobytes; at native rate with floats it is megabytes, for precision nobody can
see in a replay.

The moments carry the actual coaching. A marker saying "conceded" is a
bookmark; one saying "you were 3,400 uu out and moving away from your net while
your team-mate closed 1,300" is the lesson.
"""

from __future__ import annotations

import math

from coach.metrics.recovery import _homeward

PLAY_HZ = 10
GOAL_LEAD = 4.0        # the window the recovery metric judges
SLOW_FOR = 6.0         # seconds a guided run-through dwells on a moment


def build_track(match, who):
    """{names, teams, frames, moments, ...} or None if unusable."""
    if not match.samples or not match.teams:
        return None

    names = sorted(match.teams, key=lambda n: (match.teams[n], n))
    dur = max(match.duration(), 1e-9)
    step = max(1, round(len(match.samples) / dur / PLAY_HZ))
    t0 = match.samples[0]["t"]

    frames = []
    for s in match.samples[::step]:
        row = [round(s["t"] - t0, 1),
               round(s["ball"][0]), round(s["ball"][1]), round(s["ball"][2])]
        for n in names:
            c = s["cars"].get(n)
            if c:
                vx, vy = c["vel"][0], c["vel"][1]
                head = round(math.degrees(math.atan2(vy, vx))) if (vx or vy) else 0
                row += [round(c["pos"][0]), round(c["pos"][1]),
                        round(c["pos"][2]), head]
            else:
                row += [0, 0, 0, 0]
        frames.append(row)

    return {
        "names": names,
        "teams": [match.teams[n] for n in names],
        "me": names.index(who) if who in names else 0,
        "my_team": match.teams.get(who, 0),
        "frames": frames,
        "stride": 4,
        "lead": GOAL_LEAD,
        "slow": SLOW_FOR,
        "moments": _moments(match, who, t0),
    }


def _at(match, t):
    """The sample nearest a match time."""
    best, bd = None, 1e18
    for s in match.samples:
        d = abs(s["t"] - t)
        if d < bd:
            best, bd = s, d
        elif s["t"] > t:
            break
    return best


def _moments(match, who, t0):
    """Timestamps worth reviewing, each with what actually went wrong."""
    out = []
    team = match.teams.get(who)
    if team is None:
        return out
    sgn = match.attack_sign(who)
    own_y = -5120.0 * sgn
    n_frames = len(match.frames) or 1

    for g in match.goals_meta:
        idx = max(0, min(len(match.samples) - 1,
                         int(len(match.samples) * (g.get("frame", 0) / n_frames))))
        t_goal = match.samples[idx]["t"]
        scorer = g.get("PlayerName") or "someone"

        if g.get("PlayerTeam") == team:
            out.append({
                "t": round(max(0.0, t_goal - t0 - 3.0), 1),
                "kind": "scored", "title": "Goal for you",
                "text": "%s scores. Watch the three seconds of build-up -- what "
                        "made the space." % scorer,
            })
            continue

        # A conceded goal: replay the run-up and say what went wrong in it.
        j = idx
        while j > 0 and t_goal - match.samples[j]["t"] < GOAL_LEAD:
            j -= 1
        seg = match.samples[j:idx + 1]
        bits = []
        if len(seg) >= 5 and who in seg[0]["cars"]:
            share, closed = _homeward(seg, who, own_y, sgn)
            mates = [n for n in seg[0]["cars"]
                     if n != who and match.teams.get(n) == team]
            mate_closed = [c for c in
                           (_homeward(seg, n, own_y, sgn)[1] for n in mates)
                           if c is not None]

            me_end = seg[-1]["cars"].get(who)
            if me_end:
                bits.append("%.0f uu from your net when it went in"
                            % abs(me_end["pos"][1] - own_y))
            if share is not None:
                if share < 0:
                    bits.append("you were driving AWAY from your net")
                elif share < 25:
                    bits.append("only %.0f%% of your movement was homeward" % share)
                else:
                    bits.append("%.0f%% homeward, which is the right reaction"
                                % share)
            # Only contrast with team-mates when YOU did badly. Otherwise the
            # note praises the reaction and criticises it in the same sentence.
            poor = (share is not None and share < 25) or (closed is not None
                                                          and closed < 0)
            if poor and closed is not None and mate_closed:
                mc = sum(mate_closed) / len(mate_closed)
                if closed < mc - 300:
                    bits.append("you recovered %+.0f uu, your team-mates %+.0f"
                                % (closed, mc))
            me0, ball0 = seg[0]["cars"].get(who), seg[0]["ball"]
            if me0 is not None:
                gs = (me0["pos"][1] * sgn) < (ball0[1] * sgn)
                bits.append("you started this %s the ball"
                            % ("goal-side of" if gs else "ahead of"))
                if me0.get("boost") is not None:
                    bits.append("%.0f boost in hand" % me0["boost"])

        out.append({
            "t": round(max(0.0, t_goal - t0 - GOAL_LEAD), 1),
            "kind": "conceded",
            "title": "Conceded in %.0fs" % GOAL_LEAD,
            "text": ("%s scores. " % scorer) + ("; ".join(bits) + "." if bits
                                                else "Watch the run-up."),
        })

    for e in match.events:
        t = e["t"] - t0
        if e["kind"] == "demoed" and e.get("player") == who:
            s = _at(match, e["t"])
            where = ""
            if s and who in s["cars"]:
                where = " at %.0f uu up-pitch" % abs(
                    s["cars"][who]["pos"][1] - own_y)
            out.append({"t": round(max(0.0, t - 2.0), 1), "kind": "demoed",
                        "title": "You get demoed",
                        "text": "Demoed%s. You are out of the play for about "
                                "three seconds -- watch what your team does "
                                "without you." % where})
        elif e["kind"] == "demolish" and e.get("player") == who:
            out.append({"t": round(max(0.0, t - 2.0), 1), "kind": "demo",
                        "title": "You demo someone",
                        "text": "Free pressure. Note whether it actually bought "
                                "your team the ball."})

    was_ko = False
    for s in match.samples:
        is_ko = (abs(s["ball"][0]) < 100 and abs(s["ball"][1]) < 100
                 and math.dist((0, 0, 0), s["ball_vel"]) < 50)
        if is_ko and not was_ko:
            out.append({"t": round(s["t"] - t0, 1), "kind": "kickoff",
                        "title": "Kickoff",
                        "text": "Who goes, and where is everyone else."})
        was_ko = is_ko

    # Collapse the duplicate kickoff frames the pause produces.
    out.sort(key=lambda m: m["t"])
    kept = []
    for m in out:
        if kept and m["kind"] == "kickoff" and kept[-1]["kind"] == "kickoff" \
                and m["t"] - kept[-1]["t"] < 3.0:
            continue
        kept.append(m)
    return kept
