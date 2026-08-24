"""
What the car physically does: speed, flips, air time, powerslide, throttle.

The one subtlety worth stating. Height alone does not mean "airborne" in
Rocket League -- a car driving up the side wall is at z = 900 with all four
wheels planted, and counting that as an aerial inflates the number badly. This
module measures distance to the NEAREST surface, walls, corners and ceiling
included, so a real aerial is only counted when the car is genuinely off
everything.

All shares are weighted by the real time between samples. Frames arrive
irregularly and cluster around busy moments, so counting frames silently
over-weights whatever was happening during a scramble.
"""

from __future__ import annotations

import math

TITLE = "Mechanics -- speed, flips and air"

# Soccar geometry.
WALL_X = 4096.0
WALL_Y = 5120.0
CEILING = 2044.0
CORNER = 8064.0
ROOT2 = 1.4142135623730951

SUPERSONIC = 2200.0
FAST = 1400.0            # roughly the no-boost ceiling
CRAWL = 300.0            # slower than this is, in effect, standing still
OFF_SURFACE = 160.0      # clear of everything by this much is a real aerial
MAX_GAP = 0.5            # ignore sample gaps longer than this (goal replays)


def surface_gap(pos):
    """Distance to the nearest arena surface, corners included."""
    x, y, z = pos
    best = min(z, CEILING - z, WALL_X - abs(x), WALL_Y - abs(y))
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            best = min(best, (CORNER - (sx * x + sy * y)) / ROOT2)
    return best


def compute(match, who):
    r = {
        "live": 0.0, "samples": 0,
        "t_super": 0.0, "t_fast": 0.0, "t_crawl": 0.0,
        "t_air": 0.0, "t_wall": 0.0, "t_ground": 0.0,
        "t_handbrake": 0.0, "t_full_throttle": 0.0, "t_coast": 0.0,
        "t_reverse": 0.0, "t_boosting": 0.0,
        "speeds": [], "peak_z": 0.0, "peak_air_z": 0.0,
        "dodges": 0, "double_jumps": 0,
        "air_episodes": 0, "air_lengths": [],
        "dist": 0.0,
        "lobby_speed": {},
    }

    samples = match.samples
    in_air = False
    air_start = 0.0
    lobby_time = {}
    lobby_dist = {}

    for i in range(len(samples) - 1):
        s, nxt = samples[i], samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        me = s["cars"].get(who)

        # Everyone, for a lobby comparison.
        for nm, c in s["cars"].items():
            sp = math.dist((0.0, 0.0, 0.0), c["vel"])
            lobby_time[nm] = lobby_time.get(nm, 0.0) + dt
            lobby_dist[nm] = lobby_dist.get(nm, 0.0) + sp * dt

        if me is None:
            continue
        r["live"] += dt
        r["samples"] += 1

        speed = math.dist((0.0, 0.0, 0.0), me["vel"])
        r["speeds"].append(speed)
        r["dist"] += speed * dt

        if speed > SUPERSONIC:
            r["t_super"] += dt
        elif speed > FAST:
            r["t_fast"] += dt
        elif speed < CRAWL:
            r["t_crawl"] += dt

        gap = surface_gap(me["pos"])
        r["peak_z"] = max(r["peak_z"], me["pos"][2])
        if gap > OFF_SURFACE:
            r["t_air"] += dt
            r["peak_air_z"] = max(r["peak_air_z"], me["pos"][2])
            if not in_air:
                in_air = True
                air_start = s["t"]
        else:
            if in_air:
                in_air = False
                r["air_episodes"] += 1
                r["air_lengths"].append(s["t"] - air_start)
            if me["pos"][2] > 200.0:
                r["t_wall"] += dt
            else:
                r["t_ground"] += dt

        if me.get("handbrake"):
            r["t_handbrake"] += dt
        if me.get("boosting"):
            r["t_boosting"] += dt

        thr = me.get("throttle") or 0.0
        if thr > 0.9:
            r["t_full_throttle"] += dt
        elif thr < -0.2:
            r["t_reverse"] += dt
        elif abs(thr) < 0.1:
            r["t_coast"] += dt

    for e in match.events:
        if e.get("player") != who:
            continue
        if e["kind"] == "dodge":
            r["dodges"] += 1
        elif e["kind"] == "double_jump":
            r["double_jumps"] += 1

    r["lobby_speed"] = {
        nm: (lobby_dist[nm] / lobby_time[nm]) if lobby_time.get(nm) else 0.0
        for nm in lobby_time
    }
    return r


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def render(res):
    live = res["live"] or 1.0
    mins = live / 60.0
    speeds = res["speeds"]
    avg = sum(speeds) / len(speeds) if speeds else 0.0

    lob = res["lobby_speed"]
    rank = ""
    if lob:
        order = sorted(lob.values(), reverse=True)
        mine = None
        for nm, v in lob.items():
            if abs(v - avg) < 1.0:
                mine = v
                break
        if mine is not None:
            rank = "   (you rank %d of %d)" % (order.index(mine) + 1, len(order))
        lobby_avg = sum(lob.values()) / len(lob)
    else:
        lobby_avg = 0.0

    out = [
        "  live play               %5.0f s" % live,
        "  average speed           %5.0f uu/s   (lobby %.0f)%s"
        % (avg, lobby_avg, rank),
        "  supersonic              %5.1f %%" % _pct(res["t_super"], live),
        "  fast (1400+)            %5.1f %%" % _pct(res["t_fast"], live),
        "  crawling (<300)         %5.1f %%   standing still is dead time"
        % _pct(res["t_crawl"], live),
        "  distance driven         %5.1f km" % (res["dist"] / 100000.0),
        "",
        "  genuinely airborne      %5.1f %%   clear of every surface"
        % _pct(res["t_air"], live),
        "  on a wall               %5.1f %%   high up but still driving"
        % _pct(res["t_wall"], live),
        "  on the floor            %5.1f %%" % _pct(res["t_ground"], live),
        "  air episodes            %5d     avg %.2f s, peak height %.0f uu"
        % (res["air_episodes"],
           (sum(res["air_lengths"]) / len(res["air_lengths"]))
           if res["air_lengths"] else 0.0,
           res["peak_air_z"]),
        "",
        "  dodges / flips          %5d     %.1f per min" % (
            res["dodges"], res["dodges"] / mins if mins else 0.0),
        "  double jumps            %5d     %.1f per min" % (
            res["double_jumps"], res["double_jumps"] / mins if mins else 0.0),
        "",
        "  full throttle           %5.1f %%" % _pct(res["t_full_throttle"], live),
        "  coasting                %5.1f %%" % _pct(res["t_coast"], live),
        "  reversing               %5.1f %%" % _pct(res["t_reverse"], live),
        "  powerslide              %5.1f %%" % _pct(res["t_handbrake"], live),
    ]
    return out


def tips(res, match, who):
    live = res["live"] or 1.0
    mins = live / 60.0
    out = []

    crawl = _pct(res["t_crawl"], live)
    if crawl > 18.0:
        out.append(
            "You are crawling or stopped %.0f%% of the match. That is the "
            "cheapest thing on this whole page to fix: keep the car moving, "
            "even when repositioning, and you arrive at every play earlier."
            % crawl)

    air = _pct(res["t_air"], live)
    if air < 4.0:
        out.append(
            "Only %.1f%% of your time is genuinely off a surface. You are a "
            "ground player -- fine at most ranks, but it means any ball above "
            "about 500uu is conceded for free." % air)

    slide = _pct(res["t_handbrake"], live)
    if slide < 3.0:
        out.append(
            "Powerslide is on for %.1f%% of the match. Almost every fast "
            "direction change should use it; without it you are turning in "
            "wide arcs and losing half a second every rotation." % slide)

    dpm = res["dodges"] / mins if mins else 0.0
    if dpm < 6.0:
        out.append(
            "%.1f flips a minute is low. Flipping is free speed on the ground "
            "and the only way to add power to a touch -- if you are driving "
            "everywhere flat you are slower than the lobby for no reason."
            % dpm)

    super_pct = _pct(res["t_super"], live)
    if super_pct < 4.0 and crawl < 18.0:
        out.append(
            "You are supersonic only %.1f%% of the time. You have the boost "
            "for it (see the boost section) -- being top speed into a "
            "challenge is most of what wins it." % super_pct)

    rev = _pct(res["t_reverse"], live)
    if rev > 12.0:
        out.append(
            "You are in reverse %.0f%% of the match. Reversing out of a "
            "position usually means you over-committed getting into it; a "
            "half-flip or a wider arc is faster than backing up." % rev)

    return out
