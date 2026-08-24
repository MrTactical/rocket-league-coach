"""
Demolitions given and taken, and what you do in the five seconds after one.

Two signals of very different quality, labelled honestly:

  GIVEN  exact. The game replicates a per-player MatchDemolishes counter. It is
         re-sent constantly rather than only on change -- one match carried 92
         events for 5 actual demos -- so only a RISE in the count is counted.

  TAKEN  estimated, roughly +/-30%. Being demoed is not replicated at all; it
         is inferred from the respawn teleport (a jump over 1000uu landing past
         |y| = 4000). Kickoffs teleport everyone, so a frame moving three or
         more cars is discarded as the whistle. Across six matches this gave 48
         against 37 known demolitions, so treat it as a trend, not a count.

The five seconds after a respawn are the coaching part: a demo costs you three
seconds of nothing, and what you do with the fourth and fifth decides whether
it cost a goal.
"""

from __future__ import annotations

import math

TITLE = "Demolitions"

AFTER = 5.0        # seconds after a respawn to watch
MAX_GAP = 0.5


def compute(match, who):
    r = {"ok": False, "given": 0, "taken": 0, "mins": 0.0,
         "after_up_pitch": [], "after_boost": [], "rejoin_s": [],
         "lobby_given": {}}
    if not match.samples:
        return r
    team = match.teams.get(who)
    sgn = match.attack_sign(who)
    own_y = -5120.0 * sgn
    r["mins"] = max(match.duration(), 1.0) / 60.0

    for e in match.events:
        if e["kind"] == "demolish":
            if e.get("player"):
                r["lobby_given"][e["player"]] = r["lobby_given"].get(e["player"], 0) + 1
            if e.get("player") == who:
                r["given"] += 1
        elif e["kind"] == "demoed" and e.get("player") == who:
            r["taken"] += 1
            # Where were you, and with what, once you were driving again?
            t_end = e["t"] + AFTER
            seg = [s for s in match.samples if e["t"] < s["t"] <= t_end
                   and who in s["cars"]]
            if not seg:
                continue
            last = seg[-1]["cars"][who]
            r["after_up_pitch"].append(abs(last["pos"][1] - own_y))
            if last.get("boost") is not None:
                r["after_boost"].append(last["boost"])
            # How long until you were the closest of your team to the ball
            # again -- a rough "back in the game" clock.
            for s in seg:
                me = s["cars"][who]
                mates = [c for n, c in s["cars"].items()
                         if n != who and match.teams.get(n) == team]
                if not mates:
                    continue
                mine = math.dist(me["pos"], s["ball"])
                if mine < min(math.dist(c["pos"], s["ball"]) for c in mates):
                    r["rejoin_s"].append(s["t"] - e["t"])
                    break

    r["ok"] = (r["given"] + r["taken"]) > 0
    return r


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def render(res):
    if not res.get("ok"):
        return ["  no demolitions either way"]
    mins = res["mins"] or 1.0
    out = [
        "  demos given             %5d     %.1f per min (exact)"
        % (res["given"], res["given"] / mins),
        "  demos taken             %5d     %.1f per min (estimated, +/-30%%)"
        % (res["taken"], res["taken"] / mins),
    ]
    if res["after_up_pitch"]:
        out.append("  %.0fs after a respawn    %5.0f uu   up-pitch from your net"
                   % (AFTER, _mean(res["after_up_pitch"])))
    if res["after_boost"]:
        out.append("  boost by then           %5.0f" % _mean(res["after_boost"]))
    if res["rejoin_s"]:
        out.append("  back to first man       %5.1f s   on %d of %d respawns"
                   % (_mean(res["rejoin_s"]), len(res["rejoin_s"]), res["taken"]))
    top = sorted(res["lobby_given"].items(), key=lambda kv: -kv[1])[:3]
    if top:
        out.append("  most demos in the lobby       " +
                   ", ".join("%s %d" % (n, c) for n, c in top))
    return out


def tips(res, match, who):
    if not res.get("ok"):
        return []
    out = []
    mins = res["mins"] or 1.0
    taken_rate = res["taken"] / mins

    if taken_rate > 1.0:
        out.append(
            "You are being demoed about %.1f times a minute. At that rate it is "
            "not bad luck -- you are holding still or driving predictable "
            "straight lines near opponents with boost. Vary your approach and "
            "keep a jump in hand when someone is closing." % taken_rate)
    if res["given"] == 0 and res["taken"] >= 3:
        out.append(
            "You were demoed %d times and gave none back. Demos are free "
            "pressure in 3s and you are only on the receiving end of them."
            % res["taken"])
    if res["rejoin_s"] and _mean(res["rejoin_s"]) < 2.0 and res["taken"] >= 2:
        out.append(
            "After a respawn you are back to being first man in %.1f s. That is "
            "fast -- too fast. You just re-entered the play without boost or "
            "position; let the rotation absorb you instead of rejoining at the "
            "front." % _mean(res["rejoin_s"]))
    return out
