"""
A compact playback track for the in-page replay viewer.

Top-down 2D, not 3D. Everything this analyser has found is positional --
who was goal-side, who retreated, who double-committed -- and a plan view
shows all of it. A 3D viewer would be a lot of work to show less.

Downsampled to PLAY_HZ and rounded to whole unreal units: a full match is
about 250 KB of JSON at 10 Hz, against 3 MB at native rate for detail no
human can see in a top-down view.

Moments are the point. A scrubber with no markers means hunting; a scrubber
that jumps to "the four seconds before you conceded" is the coaching.
"""

from __future__ import annotations

import math

PLAY_HZ = 10
GOAL_LEAD = 4.0        # the window the recovery metric judges


def build_track(match, who):
    """{names, teams, frames, moments, t0} or None if unusable."""
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
            row += [round(c["pos"][0]), round(c["pos"][1])] if c else [0, 0]
        frames.append(row)

    return {
        "names": names,
        "teams": [match.teams[n] for n in names],
        "me": names.index(who) if who in names else 0,
        "my_team": match.teams.get(who, 0),
        "frames": frames,
        "hz": PLAY_HZ,
        "moments": _moments(match, who, t0),
    }


def _moments(match, who, t0):
    """Timestamps worth jumping to, most instructive first in the list."""
    out = []
    team = match.teams.get(who)
    n_frames = len(match.frames) or 1

    for g in match.goals_meta:
        idx = max(0, min(len(match.samples) - 1,
                         int(len(match.samples) * (g.get("frame", 0) / n_frames))))
        t = match.samples[idx]["t"] - t0
        scorer = g.get("PlayerName") or "?"
        if g.get("PlayerTeam") == team:
            out.append({"t": round(t, 1), "kind": "scored",
                        "text": "Goal for you - %s scored" % scorer})
        else:
            # Start the jump before the goal: the mistake is in the run-up,
            # not the shot.
            out.append({"t": round(max(0.0, t - GOAL_LEAD), 1),
                        "kind": "conceded",
                        "text": "Conceded in %.0fs - %s scores. Watch whether "
                                "you turn for your net." % (GOAL_LEAD, scorer)})

    for e in match.events:
        t = e["t"] - t0
        if e["kind"] == "demoed" and e.get("player") == who:
            out.append({"t": round(max(0.0, t - 1.5), 1), "kind": "demoed",
                        "text": "You get demoed here"})
        elif e["kind"] == "demolish" and e.get("player") == who:
            out.append({"t": round(max(0.0, t - 1.5), 1), "kind": "demo",
                        "text": "You demo someone here"})

    was_ko = False
    for s in match.samples:
        is_ko = (abs(s["ball"][0]) < 100 and abs(s["ball"][1]) < 100
                 and math.dist((0, 0, 0), s["ball_vel"]) < 50)
        if is_ko and not was_ko:
            out.append({"t": round(s["t"] - t0, 1), "kind": "kickoff",
                        "text": "Kickoff"})
        was_ko = is_ko

    out.sort(key=lambda m: m["t"])
    return out
