"""
Positional coaching from the replay network stream.

    python coach/positional.py                    # your most recent replay
    python coach/positional.py --last 5           # aggregate the last N
    python coach/positional.py --player Name

This is the part `coach/form.py` cannot do. Stat lines say what happened;
frames say WHERE you were when it happened, which is where nearly all ranked
improvement actually lives.

Needs tools/rrrocket/rrrocket.exe (nickbabcock/rrrocket, built on boxcars).
Rattletrap was tried first and rejected: its newest release predates the
current season and dies on `TAGame.Default__ViralItemActor_TA`, so it parses
old replays and not the recent ones that matter.

The replay gives rigid-body state per actor per frame. Cars link to a
PlayerReplicationInfo actor which carries the name and team, so every position
can be attributed to a named player.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMOS = Path(os.path.expanduser("~/Documents/My Games/Rocket League/TAGame/Demos"))
RRROCKET = ROOT / "tools" / "rrrocket" / "rrrocket.exe"

# Field geometry (uu).
GOAL_Y = 5120.0
THIRD = GOAL_Y * 2 / 3.0
SUPERSONIC = 2200.0
BOOST_LOW = 20.0
# Two cars closer than this to each other, both near the ball, is a double
# commit -- the single most common 2s mistake.
CROWD_DIST = 1200.0
NEAR_BALL = 1800.0


def parse_replay(path) -> dict:
    if not RRROCKET.is_file():
        raise SystemExit("rrrocket not found at %s" % RRROCKET)
    r = subprocess.run([str(RRROCKET), "-n", str(path)],
                       capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b"").decode(errors="replace")[:300])
    return json.loads(r.stdout)


def build_timeline(d: dict):
    """
    Walk the frames and return (players, samples).

    players: actor_id of a car -> {"name", "team"}
    samples: list of {"t", "ball": (x,y,z), "cars": {name: {...}}}
    """
    objects = d["objects"]
    frames = d["network_frames"]["frames"]

    actor_obj = {}          # actor_id -> archetype name
    pri_name = {}           # pri actor_id -> player name
    pri_team = {}           # pri actor_id -> team number
    car_pri = {}            # car actor_id -> pri actor_id
    car_team = {}           # car actor_id -> team (from TeamPaint, a fallback)
    # Boost is NOT an attribute of the car. It lives on a separate
    # CarComponent_Boost actor that points back at its car through
    # TAGame.CarComponent_TA:Vehicle. Reading ReplicatedBoost off the car
    # actor silently yields nothing, which is why the boost section was blank.
    comp_car = {}           # component actor_id -> car actor_id
    car_boost = {}          # car actor_id -> current boost 0-100
    state = defaultdict(dict)   # actor_id -> latest rigid body / boost
    ball_actors = set()

    samples = []
    for f in frames:
        for a in f.get("new_actors", []):
            actor_obj[a["actor_id"]] = objects[a["object_id"]]
            nm = actor_obj[a["actor_id"]]
            if "Ball_" in nm and "Default__" not in nm:
                ball_actors.add(a["actor_id"])

        for u in f.get("updated_actors", []):
            aid = u["actor_id"]
            attr_name = objects[u["object_id"]]
            val = u["attribute"]

            if attr_name == "Engine.PlayerReplicationInfo:PlayerName":
                pri_name[aid] = val.get("String")
            elif attr_name == "Engine.PlayerReplicationInfo:Team":
                act = (val.get("ActiveActor") or {}).get("actor")
                if act is not None and act >= 0:
                    pri_team[aid] = act
            elif attr_name == "Engine.Pawn:PlayerReplicationInfo":
                act = (val.get("ActiveActor") or {}).get("actor")
                if act is not None and act >= 0:
                    car_pri[aid] = act
            elif attr_name == "TAGame.Car_TA:TeamPaint":
                tp = val.get("TeamPaint") or {}
                if "team" in tp:
                    car_team[aid] = tp["team"]
            elif attr_name == "TAGame.RBActor_TA:ReplicatedRBState":
                rb = val.get("RigidBody") or {}
                loc, vel = rb.get("location"), rb.get("linear_velocity")
                if loc:
                    state[aid]["pos"] = (loc["x"], loc["y"], loc["z"])
                if vel:
                    state[aid]["vel"] = (vel["x"], vel["y"], vel["z"])
                state[aid]["sleeping"] = rb.get("sleeping", False)
            elif attr_name == "TAGame.CarComponent_TA:Vehicle":
                act = (val.get("ActiveActor") or {}).get("actor")
                if act is not None and act >= 0:
                    comp_car[aid] = act
            elif attr_name == "TAGame.CarComponent_Boost_TA:ReplicatedBoost":
                b = val.get("ReplicatedBoost") or {}
                amt = b.get("boost_amount") if isinstance(b, dict) else None
                if amt is None and isinstance(val.get("Byte"), int):
                    amt = val["Byte"]
                if amt is not None:
                    car = comp_car.get(aid)
                    if car is not None:
                        car_boost[car] = amt * 100.0 / 255.0

        for aid in f.get("deleted_actors", []):
            state.pop(aid, None)

        ball = None
        for b in ball_actors:
            if b in state and "pos" in state[b]:
                ball = state[b]["pos"]
                break
        if ball is None:
            continue

        cars = {}
        for aid, obj in actor_obj.items():
            if obj != "Archetypes.Car.Car_Default":
                continue
            s = state.get(aid)
            if not s or "pos" not in s:
                continue
            pri = car_pri.get(aid)
            name = pri_name.get(pri) if pri is not None else None
            if not name:
                continue
            cars[name] = {
                "pos": s["pos"],
                "vel": s.get("vel", (0.0, 0.0, 0.0)),
                "boost": car_boost.get(aid),
                "team": car_team.get(aid, pri_team.get(pri)),
            }
        if cars:
            samples.append({"t": f.get("time", 0.0), "ball": ball, "cars": cars})

    return samples


def dist(a, b):
    return math.dist(a, b)


def flat_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalise_teams(samples):
    """
    TeamPaint gives 0/1 directly; the PRI Team attribute is an actor id, which
    differs per replay. Collapse whatever we have into two stable groups.
    """
    seen = defaultdict(set)
    for s in samples:
        for name, c in s["cars"].items():
            if c["team"] is not None:
                seen[name].add(c["team"])
    fixed = {}
    for name, vals in seen.items():
        fixed[name] = sorted(vals)[0] if vals else None
    groups = sorted({v for v in fixed.values() if v is not None})
    remap = {g: i for i, g in enumerate(groups[:2])}
    return {n: remap.get(v) for n, v in fixed.items()}


def canon(name):
    """Lowercase alphanumerics only, so decoration does not break matching."""
    return "".join(c for c in (name or "").lower() if c.isalnum())


def resolve_name(teams, who):
    """
    Match a player across replays whose display name has changed.

    Real case: recent replays say "MrTactical", 2023 ones say "MrTactical ^-^".
    An exact match silently drops every older game.
    """
    if who in teams:
        return who
    target = canon(who)
    if not target:
        return None
    exact = [n for n in teams if canon(n) == target]
    if exact:
        return exact[0]
    partial = [n for n in teams
               if target and (target in canon(n) or canon(n) in target)]
    if len(partial) == 1:
        return partial[0]
    return None


def analyse(samples, who):
    teams = normalise_teams(samples)
    who = resolve_name(teams, who)
    if who is None:
        return None
    my_team = teams.get(who)
    if my_team is None:
        return None

    mates = [n for n, t in teams.items() if t == my_team and n != who]
    # An even rotation share is 1/N, not 50%. Judging a 3v3 against a 2s
    # baseline reports a perfectly even 33% as "playing second fiddle".
    team_size = len(mates) + 1
    # Attacking direction: team 0 shoots toward +y, team 1 toward -y.
    attack_sign = 1.0 if my_team == 0 else -1.0

    m = {
        "n": 0, "dist_ball": [], "speed": [], "boost": [], "low_boost": 0,
        "supersonic": 0, "third_def": 0, "third_mid": 0, "third_att": 0,
        "ahead_of_ball": 0, "first_man": 0, "mate_dist": [], "crowded": 0,
        "both_near_ball": 0, "offence_frames": 0, "airborne": 0,
        "team_size": team_size, "size_frames": 0,
    }
    for s in samples:
        me = s["cars"].get(who)
        if me is None:
            continue
        m["n"] += 1
        m["size_frames"] += team_size
        ball = s["ball"]
        p, v = me["pos"], me["vel"]
        speed = math.dist((0, 0, 0), v)

        m["dist_ball"].append(flat_dist(p, ball))
        m["speed"].append(speed)
        if me["boost"] is not None:
            m["boost"].append(me["boost"])
            if me["boost"] < BOOST_LOW:
                m["low_boost"] += 1
        if speed > SUPERSONIC:
            m["supersonic"] += 1
        if p[2] > 200.0:
            m["airborne"] += 1

        # Thirds, from our own goal outward.
        y = p[1] * attack_sign
        if y < -THIRD / 2:
            m["third_def"] += 1
        elif y > THIRD / 2:
            m["third_att"] += 1
        else:
            m["third_mid"] += 1

        # Being goal-side of the ball is the whole game defensively.
        if p[1] * attack_sign > ball[1] * attack_sign:
            m["ahead_of_ball"] += 1

        mate_pos = [s["cars"][n]["pos"] for n in mates if n in s["cars"]]
        if mate_pos:
            near = min(mate_pos, key=lambda q: flat_dist(p, q))
            md = flat_dist(p, near)
            m["mate_dist"].append(md)
            my_bd = flat_dist(p, ball)
            mate_bd = min(flat_dist(q, ball) for q in mate_pos)
            if my_bd < mate_bd:
                m["first_man"] += 1
            if md < CROWD_DIST and my_bd < NEAR_BALL and mate_bd < NEAR_BALL:
                m["crowded"] += 1
            if my_bd < NEAR_BALL and mate_bd < NEAR_BALL:
                m["both_near_ball"] += 1
    return m, mates, teams


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report(who, agg, n_matches, minutes):
    n = agg["n"] or 1
    print("=" * 64)
    print("  %s  --  %dv%d, %d match(es), %.0f min"
          % (who, agg.get("team_size", 2), agg.get("team_size", 2),
             n_matches, minutes))
    print("=" * 64)

    print()
    print("Where you are")
    print("-" * 48)
    print("  distance to ball    %5.0f uu median   (%.0f uu mean)"
          % (st.median(agg["dist_ball"]), st.mean(agg["dist_ball"])))
    print("  defensive third     %5.1f%%" % pct(agg["third_def"], n))
    print("  middle third        %5.1f%%" % pct(agg["third_mid"], n))
    print("  attacking third     %5.1f%%" % pct(agg["third_att"], n))
    print("  ahead of the ball   %5.1f%%   <- goal-side is the other %.1f%%"
          % (pct(agg["ahead_of_ball"], n), 100 - pct(agg["ahead_of_ball"], n)))

    print()
    print("With your teammate")
    print("-" * 48)
    if agg["mate_dist"]:
        print("  distance apart      %5.0f uu median" % st.median(agg["mate_dist"]))
        even = 100.0 / max(agg.get("team_size") or 2, 1)
        print("  you are first man   %5.1f%%   (%.0f%% is even for %dv%d)"
              % (pct(agg["first_man"], n), even,
                 agg.get("team_size", 2), agg.get("team_size", 2)))
        print("  both near the ball  %5.1f%%" % pct(agg["both_near_ball"], n))
        print("  DOUBLE COMMITTED    %5.1f%%   <- both near the ball AND within "
              "%d uu of each other" % (pct(agg["crowded"], n), CROWD_DIST))
    else:
        print("  no teammate found in these replays")

    print()
    print("Speed and boost")
    print("-" * 48)
    print("  average speed       %5.0f uu/s" % st.mean(agg["speed"]))
    print("  supersonic          %5.1f%%" % pct(agg["supersonic"], n))
    print("  airborne (>200uu)   %5.1f%%" % pct(agg["airborne"], n))
    if agg["boost"]:
        print("  average boost       %5.0f" % st.mean(agg["boost"]))
        print("  starved (<%d)       %5.1f%%" % (BOOST_LOW, pct(agg["low_boost"], n)))

    print()
    print("Pointers")
    print("-" * 48)
    tips = []
    size = agg.get("team_size") or 2
    crowd = pct(agg["crowded"], n)
    ahead = pct(agg["ahead_of_ball"], n)
    fm = pct(agg["first_man"], n)
    med_mate = st.median(agg["mate_dist"]) if agg["mate_dist"] else 9e9
    att = pct(agg["third_att"], n)

    if crowd > 12:
        tips.append(
            "You are double committed %.1f%% of the time -- both of you near "
            "the ball and within %d uu of each other. This is the single most "
            "expensive habit in 2s: when it goes wrong the whole pitch is open. "
            "Aim under 8%%." % (crowd, CROWD_DIST))
    # How much of the match you should spend ahead of the ball depends on the
    # format: in 3v3 you are the last man a third of the time by design, so the
    # healthy band sits lower than it does in 2s.
    hi, lo = (55.0, 35.0) if size <= 2 else (45.0, 22.0)
    if ahead > hi:
        tips.append(
            "You are ahead of the ball %.1f%% of the time (over %.0f%% for "
            "%dv%d). You are consistently caught up-pitch on the turnover, "
            "which usually explains conceding better than any mechanical "
            "weakness does." % (ahead, hi, size, size))
    elif ahead < lo:
        tips.append(
            "You are goal-side %.1f%% of the time, which is passive even for "
            "%dv%d. You are safe but you are not pressuring, so expect long "
            "defensive sieges and few chances." % (100 - ahead, size, size))
    # An even share is 1/N. Comparing a 3v3 against a 2s baseline reports a
    # textbook 33% rotation as a fault.
    even = 100.0 / size
    if fm > even * 1.35:
        tips.append(
            "You are first man %.1f%% of the time against an even share of "
            "%.0f%% for %dv%d. You are taking most of the touches -- either "
            "your teammates are passive or you are not letting them rotate in."
            % (fm, even, size, size))
    elif fm < even * 0.7:
        tips.append(
            "You are first man only %.1f%% of the time against an even %.0f%% "
            "for %dv%d. You are playing second fiddle; more of the play should "
            "go through you given your stat lines." % (fm, even, size, size))
    if med_mate < 1600:
        tips.append(
            "Median distance to your nearest teammate is %.0f uu. You are "
            "sitting on top of each other -- the team should be spread across "
            "the thirds, not sharing one." % med_mate)
    if agg["boost"] and pct(agg["low_boost"], n) > 35:
        tips.append(
            "You are under %d boost %.1f%% of the time. That is a rotation "
            "problem, not a pickup problem: you are burning it chasing rather "
            "than collecting on the way back."
            % (BOOST_LOW, pct(agg["low_boost"], n)))
    if att > 45:
        tips.append(
            "You spend %.1f%% of the match in the attacking third. That is "
            "committed; make sure the goals justify it." % att)
    if not tips:
        tips.append("Positioning looks sound on these numbers.")
    for t in tips:
        print("  * " + t)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Positional coaching from replays.")
    ap.add_argument("--player", help="your in-game name")
    ap.add_argument("--last", type=int, default=1, help="how many recent replays")
    args = ap.parse_args()

    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime)
    if not files:
        print("No replays in %s" % DEMOS)
        return 1
    files = files[-args.last:]

    who = args.player
    if who is None:
        # Resolve from headers before parsing anything. Doing it lazily inside
        # the loop meant an older replay with no PlayerName left `who` as None
        # and silently skipped every file until one happened to carry it.
        from coach.replay import read_header
        from collections import Counter
        votes = Counter()
        for f in reversed(files):
            try:
                n = read_header(f).get("PlayerName")
            except Exception:
                continue
            if n:
                votes[n] += 1
        if votes:
            who = votes.most_common(1)[0][0]
    if who is None:
        print("Could not work out your player name -- pass --player.")
        return 1

    agg = None
    minutes = 0.0
    used = 0
    sizes = set()
    for f in files:
        try:
            d = parse_replay(f)
        except Exception as e:
            print("skipped %s: %s" % (os.path.basename(f), str(e)[:90]))
            continue
        if who is None:
            who = (d.get("properties") or {}).get("PlayerName")
        props = d.get("properties") or {}
        if props.get("bForfeit"):
            print("note: %s ended in a FORFEIT, so it is short and skewed"
                  % os.path.basename(f))
        samples = build_timeline(d)
        if not samples:
            print("skipped %s: no usable frames" % os.path.basename(f))
            continue
        got = analyse(samples, who)
        if got is None:
            print("skipped %s: %r not found" % (os.path.basename(f), who))
            continue
        m, mates, teams = got
        minutes += (samples[-1]["t"] - samples[0]["t"]) / 60.0
        used += 1
        sizes.add(m["team_size"])
        if agg is None:
            agg = m
        else:
            for k, v in m.items():
                if k == "team_size":
                    continue          # a count, not a total -- never sum it
                if isinstance(v, list):
                    agg[k].extend(v)
                else:
                    agg[k] += v

    if agg is None or not agg["n"]:
        print("Nothing usable to analyse.")
        return 1
    if len(sizes) > 1:
        print("note: mixing %s -- rotation shares are not comparable across "
              "formats, so filter to one playlist for a clean read."
              % ", ".join("%dv%d" % (x, x) for x in sorted(sizes)))
    report(who, agg, used, minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
