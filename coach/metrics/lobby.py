"""
Everyone in the lobby -- including the opponents, and what to steal from them.

The rest of the analyser asks "how did you play". This one asks "who played
this match best, and what did they do that you did not". Every player in the
replay is measured on the same axes, you are ranked against them, and the
highest-scoring player in the lobby is diffed against you.

Opponents are the useful half. They are at your rank by definition -- Champion
lobbies are full of Champion players -- so a habit that separates the best of
them from you is a habit at exactly the level you are trying to leave.

Also reports kickoff spawns for everyone, since who takes which spawn and what
they do from it is visible and copyable.

Scored by Rocket League's own points, which is crude -- it rewards touching
things -- so it is used only to pick who to look at, never as the verdict.
"""

from __future__ import annotations

import math

TITLE = "The lobby -- and what to steal"

MAX_GAP = 0.5
SPAWN_TOL = 400.0
# The five standard kickoff spawns, mirrored per side.
SPAWNS = {"back centre": (0.0, 4608.0), "back left": (-256.0, 3840.0),
          "back right": (256.0, 3840.0), "left corner": (-2048.0, 2560.0),
          "right corner": (2048.0, 2560.0)}


def _spawn_name(pos, sgn):
    """Which of the five spawns this is, in the player's own frame."""
    x, y = pos[0], pos[1] * sgn * -1.0     # flip so own half is +y
    best, bd = None, 1e9
    for name, (sx, sy) in SPAWNS.items():
        for mirror in (1.0, -1.0):
            d = math.hypot(x - sx * mirror, y - sy)
            if d < bd:
                best, bd = name, d
    return best if bd < SPAWN_TOL * 3 else None


def compute(match, who):
    r = {"ok": False, "players": {}, "me": who, "best": None, "spawns": {}}
    if not match.samples or not match.teams:
        return r

    acc = {}
    for i in range(len(match.samples) - 1):
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        ball = s["ball"]
        order = sorted(s["cars"].items(),
                       key=lambda kv: math.dist(kv[1]["pos"], ball))
        for rank, (name, car) in enumerate(order):
            a = acc.setdefault(name, {"t": 0.0, "dist": 0.0, "speed": 0.0,
                                      "ss": 0.0, "air": 0.0, "slide": 0.0,
                                      "boost": 0.0, "bt": 0.0, "low": 0.0,
                                      "first": 0.0, "updown": 0.0})
            sgn = 1.0 if match.teams.get(name) == 0 else -1.0
            own_y = -5120.0 * sgn
            v = math.dist((0, 0, 0), car["vel"])
            a["t"] += dt
            a["speed"] += v * dt
            a["dist"] += v * dt
            a["updown"] += abs(car["pos"][1] - own_y) * dt
            if v > 2200:
                a["ss"] += dt
            if car["pos"][2] > 300:
                a["air"] += dt
            if car.get("handbrake"):
                a["slide"] += dt
            if car.get("boost") is not None:
                a["boost"] += car["boost"] * dt
                a["bt"] += dt
                if car["boost"] < 20:
                    a["low"] += dt
            if rank == 0:
                a["first"] += dt

    demos = {}
    for e in match.events:
        if e["kind"] == "demolish" and e.get("player"):
            demos[e["player"]] = demos.get(e["player"], 0) + 1

    for name, a in acc.items():
        if a["t"] < 30.0:
            continue
        mins = a["t"] / 60.0
        stat = match.stat_line(name) or {}
        r["players"][name] = {
            "team": match.teams.get(name),
            "mine": match.teams.get(name) == match.teams.get(who),
            "score": int(stat.get("Score") or 0),
            "goals": int(stat.get("Goals") or 0),
            "saves": int(stat.get("Saves") or 0),
            "speed": a["speed"] / a["t"],
            "supersonic": 100.0 * a["ss"] / a["t"],
            "airborne": 100.0 * a["air"] / a["t"],
            "powerslide": 100.0 * a["slide"] / a["t"],
            "boost_held": (a["boost"] / a["bt"]) if a["bt"] else 0.0,
            "starved": (100.0 * a["low"] / a["bt"]) if a["bt"] else 0.0,
            "first_man": 100.0 * a["first"] / a["t"],
            "up_pitch": a["updown"] / a["t"],
            "demos": demos.get(name, 0),
            "km": a["dist"] / 100000.0 / max(mins, 1e-9),
        }

    # Kickoff spawns, counted ONCE per kickoff. The ball sits on the centre
    # spot for the entire pre-kickoff pause -- 712 frames of it in one match --
    # so counting every matching frame reports 330 kickoffs instead of 7. Only
    # the frame where the game ENTERS that state is a kickoff.
    was_kickoff = False
    for s in match.samples:
        is_kickoff = (abs(s["ball"][0]) < 100 and abs(s["ball"][1]) < 100
                      and math.dist((0, 0, 0), s["ball_vel"]) < 50)
        if is_kickoff and not was_kickoff:
            for name, car in s["cars"].items():
                sgn = 1.0 if match.teams.get(name) == 0 else -1.0
                nm = _spawn_name(car["pos"], sgn)
                if nm:
                    d = r["spawns"].setdefault(name, {})
                    d[nm] = d.get(nm, 0) + 1
        was_kickoff = is_kickoff
    # Pick the lobby's best by the game's own score, excluding the player.
    others = [(n, p) for n, p in r["players"].items() if n != who]
    if others:
        r["best"] = max(others, key=lambda kv: kv[1]["score"])[0]
    r["ok"] = who in r["players"] and len(r["players"]) > 1
    return r


AXES = [
    ("speed", "avg speed", "%.0f uu/s", True),
    ("supersonic", "supersonic", "%.1f %%", True),
    ("airborne", "airborne", "%.1f %%", True),
    ("powerslide", "powerslide", "%.1f %%", True),
    ("boost_held", "boost held", "%.0f", True),
    ("starved", "starved", "%.1f %%", False),
    ("first_man", "first to the ball", "%.1f %%", None),
    ("up_pitch", "avg up-pitch", "%.0f uu", None),
    ("demos", "demos given", "%d", True),
]


def render(res):
    if not res.get("ok"):
        return ["  not enough of the lobby to compare"]
    me = res["players"][res["me"]]
    out = ["  %-20s %5s %6s %7s %7s %6s %6s"
           % ("player", "score", "speed", "first%", "boost", "air%", "demos")]
    for name, p in sorted(res["players"].items(), key=lambda kv: -kv[1]["score"]):
        tag = "you " if name == res["me"] else ("mate" if p["mine"] else "opp ")
        out.append("  %-20s %5d %6.0f %6.1f%% %7.0f %5.1f%% %6d"
                   % ((tag + " " + name)[:20], p["score"], p["speed"],
                      p["first_man"], p["boost_held"], p["airborne"], p["demos"]))

    best = res.get("best")
    if best and best in res["players"]:
        b = res["players"][best]
        side = "teammate" if b["mine"] else "OPPONENT"
        out.append("")
        out.append("  top of the lobby: %s (%s, %d pts) vs you"
                   % (best, side, b["score"]))
        for key, label, fmt, higher in AXES:
            mv, bv = me[key], b[key]
            if higher is None or abs(bv - mv) < max(abs(mv) * 0.12, 1e-9):
                continue
            better = (bv > mv) if higher else (bv < mv)
            if better:
                out.append("    %-18s you " % label + fmt % mv +
                           "   them " + fmt % bv)

    if res.get("spawns"):
        out.append("")
        out.append("  kickoff spawns taken")
        for name, d in sorted(res["spawns"].items()):
            tag = "you " if name == res["me"] else ""
            out.append("    %-20s %s" % ((tag + name)[:20],
                       ", ".join("%s x%d" % (k, v) for k, v in d.items())))
    return out


def tips(res, match, who):
    if not res.get("ok") or not res.get("best"):
        return []
    me, b = res["players"][who], res["players"][res["best"]]
    side = "your teammate" if b["mine"] else "an opponent"
    out = []

    if b["speed"] > me["speed"] + 150:
        out.append(
            "%s was the best-scoring player in that lobby and drove %.0f uu/s "
            "faster than you on average (%.0f vs %.0f). At your rank that gap "
            "is not mechanics, it is time spent stationary or coasting."
            % (res["best"], b["speed"] - me["speed"], b["speed"], me["speed"]))
    if b["airborne"] > me["airborne"] * 1.6 and b["airborne"] > 8:
        out.append(
            "%s spent %.1f%% of the match airborne against your %.1f%%. That is "
            "the single clearest thing to copy from %s -- they are contesting "
            "balls you are conceding."
            % (res["best"], b["airborne"], me["airborne"], side))
    if b["boost_held"] > me["boost_held"] + 12:
        out.append(
            "%s held %.0f boost on average to your %.0f. More boost is more "
            "options; watch their replay for where they pick pads up on the way "
            "back rather than detouring for them."
            % (res["best"], b["boost_held"], me["boost_held"]))
    return out
