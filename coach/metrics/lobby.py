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

from coach import possession as P

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

    acc, pair, live = {}, {}, 0.0
    for i in range(len(match.samples) - 1):
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        ball = s["ball"]
        live += dt
        # Co-occupation, measured SIMULTANEOUSLY. Comparing average positions
        # cannot see this: two team-mates rotating correctly -- one deep while
        # the other attacks, then swapping -- both average mid-pitch, so an
        # average-similarity test flags textbook rotation as a conflict.
        names = list(s["cars"])
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                an, bn = names[x], names[y]
                if match.teams.get(an) != match.teams.get(bn):
                    continue
                ca, cb = s["cars"][an], s["cars"][bn]
                k = (an, bn) if an < bn else (bn, an)
                d = pair.setdefault(k, {"close": 0.0, "both_up": 0.0})
                if math.dist(ca["pos"], cb["pos"]) < 1500.0:
                    d["close"] += dt
                sg = 1.0 if match.teams.get(an) == 0 else -1.0
                if (ca["pos"][1] * sg) > (ball[1] * sg) and                         (cb["pos"][1] * sg) > (ball[1] * sg):
                    d["both_up"] += dt
        order = sorted(s["cars"].items(),
                       key=lambda kv: math.dist(kv[1]["pos"], ball))
        for rank, (name, car) in enumerate(order):
            a = acc.setdefault(name, {"t": 0.0, "dist": 0.0, "speed": 0.0,
                                      "ss": 0.0, "air": 0.0, "slide": 0.0,
                                      "boost": 0.0, "bt": 0.0, "low": 0.0,
                                      "first": 0.0, "updown": 0.0,
                                      "exposed": 0.0, "committed": 0.0})
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
            # Ahead of the ball, split by who actually has it. The same
            # position is a commit during your own attack (measurably SAFER
            # than average) and a hole while they have it in your half (~2.8x
            # the concede rate). "Lives forward" as a single number could not
            # tell those apart and criticised both.
            my_team = match.teams.get(name)
            if my_team is not None:
                ph = P.phase(s, my_team, sgn)
                if (car["pos"][1] * sgn) > (ball[1] * sgn):
                    if P.is_exposed(ph):
                        a["exposed"] += dt
                    elif P.is_committed(ph):
                        a["committed"] += dt

            if rank == 0:
                a["first"] += dt

    demos = {}
    for e in match.events:
        if e["kind"] == "demolish" and e.get("player"):
            demos[e["player"]] = demos.get(e["player"], 0) + 1

    n_cars = max(1, len({n for n, t in match.teams.items() if t in (0, 1)}))
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
            # Relative to an even share of the WHOLE lobby. first_man ranks
            # all cars by distance to the ball, so an even share is 1/4 in
            # 2v2 and 1/6 in 3v3 -- a single absolute threshold calls every
            # 2v2 player ball-dominant purely because there are fewer cars.
            "first_man_rel": (100.0 * a["first"] / a["t"]) / (100.0 / n_cars),
            "up_pitch": a["updown"] / a["t"],
            "exposed": 100.0 * a["exposed"] / a["t"],
            "committed": 100.0 * a["committed"] / a["t"],
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
    r["pairs"] = {"%s|%s" % k: {"close": 100.0 * v["close"] / live,
                                "both_up": 100.0 * v["both_up"] / live}
                  for k, v in pair.items() if live > 0
                  and k[0] in r["players"] and k[1] in r["players"]}

    # The even share of "first to the ball" is over the WHOLE lobby, not your
    # team: first_man ranks all six cars by distance, so even is 100/6 in 3v3,
    # not 100/3. Using team size made every player in every match read as
    # "passive" -- a label that fires for everyone tells you nothing.
    size = len(r["players"]) or 1
    for p in r["players"].values():
        p["read"] = read_player(p, size)
        p["improve"] = improve_player(p, size,
                                      list(r["players"].values()))
    r["clashes"] = clashes(r["players"], who, size, r["pairs"])

    # Pick the lobby's best by the game's own score, excluding the player.
    others = [(n, p) for n, p in r["players"].items() if n != who]
    if others:
        r["best"] = max(others, key=lambda kv: kv[1]["score"])[0]
    r["ok"] = who in r["players"] and len(r["players"]) > 1
    return r


# Percentiles over 1008 player-observations from 168 3v3 lobbies at Champion
# rank, opponents included -- so it is a rank population, not one player.
# A user at a very different rank will see these labels fire more or less than
# one player in ten; re-derive them by dumping these fields across their own
# replays if that matters. Labels fire on the tails, so "notable" means
# roughly the top or bottom tenth of players actually seen -- not a number
# picked by feel. The first version of this file guessed: it put "lives
# forward" at 5200 uu when the real 90th percentile is 4627, so the label
# never fired once, and "plays the air" at 14% when the median is 12.4, so it
# fired for all six players in a lobby. A label that fires for everyone and a
# label that fires for no one are equally useless.
P90 = {"first_man_rel": 1.30, "up_pitch": 4627.0, "airborne": 16.7,
       "exposed": 17.3, "committed": 11.4,
       "starved": 31.5, "boost_held": 64.4, "speed": 1530.0,
       "supersonic": 17.0, "powerslide": 13.9}
# Share of live time spent ahead of the ball while the OPPONENT touched it
# last in your half. Measured over 246 player-observations from 41 3v3
# lobbies: p10 5.8, median 10.9, p90 17.3. Set at the p90, because a label
# that fires for five of six players in a lobby says nothing -- 9.0 was a
# guess and did exactly that.
EXPOSED_HI = 17.3

P10 = {"first_man_rel": 0.713, "up_pitch": 3350.0, "airborne": 8.3,
       "exposed": 5.8, "committed": 3.9,
       "starved": 11.0, "boost_held": 44.2, "speed": 1261.0,
       "supersonic": 5.7, "powerslide": 1.6}


def _tail(p, axis):
    """How far into a tail this value sits, in p10-p90 widths. 0 = ordinary."""
    v = p.get(axis)
    if v is None:
        return 0.0
    hi, lo = P90[axis], P10[axis]
    span = (hi - lo) or 1.0
    if v > hi:
        return (v - hi) / span
    if v < lo:
        return (v - lo) / span
    return 0.0


def read_player(p, size=None):
    """One line on what this player was doing, from wherever they are extreme."""
    notes = []
    for axis, high, low in (
            ("first_man_rel", "ball-dominant", "passive"),
            ("up_pitch", None, "anchors deep"),
            ("airborne", "plays the air", "ground only"),
            ("starved", "boost starved", None),
            ("boost_held", "boost hoarder", None),
            ("speed", "fast", "slow"),
            ("powerslide", "heavy powerslide", None)):
        t = _tail(p, axis)
        if t > 0 and high:
            notes.append(high)
        elif t < 0 and low:
            notes.append(low)
    # Forward is not one habit. Ahead of the ball during your own attack and
    # ahead of it while they have it in your half are opposite behaviours that
    # produce the same average up-pitch, so name the one that is actually
    # happening rather than the average of the two.
    exp, com = p.get("exposed") or 0.0, p.get("committed") or 0.0
    if exp > EXPOSED_HI:
        notes.append("caught upfield")
    elif com > exp and com > 11.4:
        notes.append("attacks hard")
    if p.get("demos", 0) >= 2:
        notes.append("demos")
    return ", ".join(notes) if notes else "nothing unusual"


# Each fault is (axis, direction, template). Direction +1 means "too high is
# the problem", -1 means "too low is". Whichever axis a player is furthest
# into the wrong tail on becomes their one note.
FAULTS = [
    ("starved", 1, "runs empty %.0f%% of the time -- take the small pads on "
                   "the way back instead of arriving with nothing"),
    ("first_man_rel", 1, "first to the ball %.1fx an even share -- ball "
                         "chasing, which leaves nobody behind it"),
    ("exposed", 1, "is ahead of the ball %.0f%% of the match while the other "
                   "team has it in this half -- the state that precedes goals, "
                   "about 2.8x the normal concede rate"),
    ("first_man_rel", -1, "first to the ball only %.2fx an even share -- "
                          "waiting for the play instead of making one"),
    ("up_pitch", -1, "averages %.0f uu from their own net -- anchored so deep "
                     "the team plays a man short"),
    ("airborne", -1, "airborne %.1f%% of the match -- anything above head "
                     "height is a free ball for the other team"),
    ("boost_held", 1, "sits on %.0f boost -- hoarding it rather than spending "
                      "it on position"),
    ("powerslide", -1, "powerslides %.1f%% of the time -- turning in wide "
                       "arcs, which is where the lost seconds are"),
    ("speed", -1, "averages %.0f uu/s -- slow enough that plays arrive "
                  "without them"),
    ("supersonic", -1, "supersonic only %.1f%% -- moves constantly but rarely "
                       "fast enough to beat anyone to a ball"),
]


def _lobby_note(p, peers):
    """
    Fallback when nothing is unusual for the rank: worst axis in THIS lobby.

    Clearly a weaker claim than a population tail, and labelled as one on the
    page. Without it a third of players get "nothing to fix", which is true
    and useless.
    """
    best, worst = None, 0.0
    for axis, direction, template in FAULTS:
        xs = [q.get(axis) for q in peers if q.get(axis) is not None]
        if len(xs) < 3 or p.get(axis) is None:
            continue
        span = (max(xs) - min(xs)) or 1.0
        d = direction * (p[axis] - (sum(xs) / len(xs))) / span
        if d > worst:
            best, worst = template, d
    if best is None or worst < 0.30:
        return None
    axis = next(a for a, _, t in FAULTS if t == best)
    # Neutral phrasing on purpose. The population templates assert a cause --
    # "ball chasing", "hoarding" -- which is a fair call for someone in the
    # top tenth and an overclaim for someone merely above their lobby's mean.
    # This branch fired on a player at 1.1x an even share and called it ball
    # chasing, which is not what 1.1x means.
    return NEUTRAL[axis] % p[axis]


NEUTRAL = {
    "exposed": "was ahead of the ball while the other team had it more than "
               "anyone else here, %.0f%% of the match",
    "starved": "spent the most time on empty in this lobby, %.0f%%",
    "first_man_rel": "went for the ball more than anyone else here, %.1fx an "
                     "even share",
    "up_pitch": "played the furthest up the pitch here, %.0f uu",
    "airborne": "was the least airborne in this lobby, %.1f%%",
    "boost_held": "held the most boost in this lobby, %.0f",
    "powerslide": "powerslid the least here, %.1f%%",
    "speed": "was the slowest in this lobby, %.0f uu/s",
    "supersonic": "spent the least time supersonic here, %.1f%%",
}


def improve_player(p, size=None, peers=None):
    """
    The single biggest thing this player could fix.

    Picks whichever axis they are furthest into the wrong tail on, so the note
    is the most unusual thing about them rather than the first rule that
    happened to match. Deliberately ONE item: a list of six faults per player
    is a scoreboard, not coaching, and nobody acts on the sixth.
    """
    best, worst = None, 0.0
    for axis, direction, template in FAULTS:
        t = _tail(p, axis)
        if direction > 0 and t > worst:
            best, worst = (axis, template), t
        elif direction < 0 and -t > worst:
            best, worst = (axis, template), -t
    if not best:
        note = _lobby_note(p, peers or [])
        if note:
            return "nothing unusual for the rank; worst in this lobby: " + note
        return "nothing unusual on these axes -- a solid, unremarkable game"
    return best[1] % p[best[0]]


# Set from measured distributions over this player's own 3v3 matches, and
# kept only where the measurement separates a won match from a lost one.
#
#   both ahead of the ball   won 10.4%   lost 15.6%   gap 5.1   -> kept
#   bunched within 1500 uu   won 13.6%   lost 15.6%   gap 2.0   -> dropped
#
# The bunching check sounded like the more obvious fault and measured almost
# nothing. Being near your team-mate is not the problem; both of you being on
# the wrong side of the ball at the same moment is. 0 disables a check.
#
# 32 matches is a small sample -- the shadow-depth finding looked solid at 24
# and collapsed at 100 -- so the surviving check was re-tested on a separate
# population: 380 team-observations from downloaded Grand Champion replays,
# different players, different rank, neither team the user.
#
#   GC both ahead of ball    won 11.4%   lost 14.8%   gap 3.4
#   GC bunched <1500 uu      won 16.7%   lost 17.1%   gap 0.4
#
# Same direction, smaller effect, and split-half within that population gives
# 5.2 and 2.3 -- so it is real but modest, and the page says so. The bunching
# check measured nothing on either population, which is why it is not here.
CLOSE_P90 = 0.0
BOTH_UP_P90 = 16.5


def clashes(players, me, size, pairs=None):
    """
    Where two team-mates want the same job, measured per frame.

    Only flags pairs on the SAME team: two opponents with identical styles is
    their problem, not a rotation you can do anything about.
    """
    out = []
    mine = [(n, p) for n, p in players.items() if p.get("mine")]
    even = 100.0 / max(size, 1)
    for i in range(len(mine)):
        for j in range(i + 1, len(mine)):
            an, a = mine[i]
            bn, b = mine[j]
            pair = "%s and %s" % (an, bn)
            pk = "%s|%s" % ((an, bn) if an < bn else (bn, an))
            pv = (pairs or {}).get(pk) or {}
            if CLOSE_P90 and pv.get("close", 0) > CLOSE_P90:
                out.append("%s spent %.0f%% of the match within 1500 uu of "
                           "each other -- bunched, so one challenge beats two "
                           "players" % (pair, pv["close"]))
            if BOTH_UP_P90 and pv.get("both_up", 0) > BOTH_UP_P90:
                out.append("%s were both ahead of the ball %.0f%% of the time "
                           "-- for that share of the match nobody was covering"
                           % (pair, pv["both_up"]))
            if a["first_man"] > even * 1.25 and b["first_man"] > even * 1.25:
                out.append("%s both wanted the ball (%.0f%% and %.0f%% first "
                           "to it, even is %.0f%%) -- someone has to give it up"
                           % (pair, a["first_man"], b["first_man"], even))
    return out


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

    size = max(1, sum(1 for p in res["players"].values() if p.get("mine")))
    out.append("")
    out.append("  what each of them was doing")
    for name, p in sorted(res["players"].items(), key=lambda kv: -kv[1]["score"]):
        tag = "you " if name == res["me"] else ("mate" if p["mine"] else "opp ")
        out.append("    %-4s %-18s %s" % (tag, name[:18], read_player(p, size)))

    cl = res.get("clashes") or []
    if cl:
        out.append("")
        out.append("  where your team pulled against itself")
        for c in cl:
            out.append("    - " + c)

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
    if not res.get("ok"):
        return []
    out = []
    for c in (res.get("clashes") or [])[:2]:
        out.append("Team shape: " + c + ".")
    if not res.get("best"):
        return out
    me, b = res["players"][who], res["players"][res["best"]]
    side = "your teammate" if b["mine"] else "an opponent"

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
