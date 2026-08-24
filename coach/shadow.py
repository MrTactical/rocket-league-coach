"""
Where the covering defender stands, and whether it held.

One implementation, used for BOTH your replays and the benchmark replays
pulled from ballchasing. Measuring your matches one way and a higher rank
another way produces a difference that is an artefact of the two methods, not
of the two ranks -- and it would look exactly like a finding.

The measurement, per frame:

  * a team is UNDER ATTACK when the ball is in their half and an opponent is
    nearer to it than any of them are
  * the COVERING DEFENDER is whichever of them is goal-side of the ball and
    closest to their own net
  * their DEPTH is the fraction of the way from that net to the ball -- 0 is
    on the goal line, 1 is level with the ball
  * the frame is CONCEDED if a goal goes in against them within 6 seconds,
    otherwise HELD

Correlational, and the caveat travels with the numbers: a deeper defender may
partly reflect a less dangerous attack rather than better positioning.
"""

from __future__ import annotations

import math

WINDOW = 6.0          # a goal within this long counts the frame as conceded
STEP = 4              # sample every Nth frame; the signal is not per-tick
MIN_SPAN = 1200.0     # ball too close to the net for a fraction to mean much
LATERAL_MIN_X = 600.0 # ball too central for a lateral ratio to mean much


def shadow_stats(match):
    """(held, conceded) -- each a list of {depth, lateral} for one match."""
    held, conceded = [], []
    if not match.samples or not match.goals_meta or not match.teams:
        return held, conceded

    n_frames = len(match.frames) or 1
    goals_by_team = {}
    for g in match.goals_meta:
        idx = max(0, min(len(match.samples) - 1,
                         int(len(match.samples) * (g.get("frame", 0) / n_frames))))
        goals_by_team.setdefault(g.get("PlayerTeam"), []).append(
            match.samples[idx]["t"])

    for team in (0, 1):
        sgn = 1.0 if team == 0 else -1.0
        own_y = -5120.0 * sgn
        against = goals_by_team.get(1 - team, [])

        for i in range(0, len(match.samples), STEP):
            s = match.samples[i]
            ball = s["ball"]
            if (ball[1] * sgn) > -800.0:
                continue                      # ball is not in our half
            mine = [c for n, c in s["cars"].items() if match.teams.get(n) == team]
            opp = [c for n, c in s["cars"].items()
                   if match.teams.get(n) == (1 - team)]
            if len(mine) < 2 or not opp:
                continue
            if min(math.dist(c["pos"], ball) for c in opp) > \
               min(math.dist(c["pos"], ball) for c in mine):
                continue                      # not actually under attack

            cover = [c for c in mine if (c["pos"][1] * sgn) < (ball[1] * sgn)]
            if not cover:
                continue
            c = min(cover, key=lambda k: abs(k["pos"][1] - own_y))

            span = abs(ball[1] - own_y)
            if span < MIN_SPAN:
                continue
            row = {"depth": abs(c["pos"][1] - own_y) / span}
            if abs(ball[0]) > LATERAL_MIN_X:
                lat = c["pos"][0] / ball[0]
                if -3.0 < lat < 3.0:
                    row["lateral"] = lat

            bad = any(0.0 < (gt - s["t"]) < WINDOW for gt in against)
            (conceded if bad else held).append(row)

    return held, conceded


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def summarise(held, conceded):
    """Medians in the shape the page and the benchmark file both expect."""
    hd = [r["depth"] for r in held]
    cd = [r["depth"] for r in conceded]
    hl = [r["lateral"] for r in held if "lateral" in r]
    cl = [r["lateral"] for r in conceded if "lateral" in r]
    return {
        "depth_held": _median(hd) or 0.0,
        "depth_conceded": _median(cd) or 0.0,
        "lateral_held": _median(hl) or 0.0,
        "lateral_conceded": _median(cl) or 0.0,
        "n_held": len(hd),
        "n_conceded": len(cd),
        # How far apart holding and conceding actually are. If this is small
        # the measurement does not discriminate, and a marker drawn from it
        # would be decoration wearing the clothes of advice.
        #
        # Reported for BOTH axes because they disagree. On 24 matches depth
        # separated by 0.10 and looked like the finding; at 100 matches it fell
        # to 0.04 and stopped meaning anything, while lateral held up at 0.12.
        # A single "separation" number would have hidden that.
        "separation": abs((_median(cd) or 0.0) - (_median(hd) or 0.0)),
        "separation_lateral": abs((_median(cl) or 0.0) - (_median(hl) or 0.0)),
    }


# --- CLI: measure your own, so the marker is not a constant in the source ---

ME_OUT = None      # set in main(), keeps the import side-effect free


def measure_local(limit=60, team_size=None, verbose=True):
    """Run the same measurement over your own recent replays."""
    import glob
    import os
    from coach.timeline import DEMOS, load

    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime)
    held, conceded, used = [], [], 0
    for f in files[-limit:]:
        try:
            m = load(f)
        except Exception:
            continue
        if team_size and m.team_size != team_size:
            continue
        h, c = shadow_stats(m)
        if not h and not c:
            continue
        held += h
        conceded += c
        used += 1
    if verbose:
        print("measured %d of your matches" % used)
    out = summarise(held, conceded)
    out["matches"] = used
    out["team_size"] = team_size
    return out


def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Measure where your covering defender stands.")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--team-size", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = measure_local(args.limit, args.team_size)
    (root / "coach" / "shadow-me.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print("  depth   held %.2f (n=%d)   conceded %.2f (n=%d)"
          % (out["depth_held"], out["n_held"],
             out["depth_conceded"], out["n_conceded"]))
    print("  lateral held %.2f            conceded %.2f"
          % (out["lateral_held"], out["lateral_conceded"]))
    print("wrote coach/shadow-me.json")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
