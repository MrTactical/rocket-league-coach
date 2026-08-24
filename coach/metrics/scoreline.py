"""
Does the scoreboard change how you play?

Per-match averages hide this completely: playing 1400 uu deeper when behind and
1400 uu higher when ahead averages out to "normal". Splitting by game state is
what makes it visible.

The scoreline is reconstructed from goal frames, which the timeline already
carries -- no extra capture, no second parse.

Measured on this player: 4762 uu up-pitch when ahead, 3320 when behind, with
possession falling 61% -> 43%. That is the passive direction, not the usual
tilt-forward one, and it needs the opposite advice.
"""

from __future__ import annotations

MAX_GAP = 0.5
MIN_STATE_SECS = 30.0     # below this a state is noise, not a habit
STATES = ("ahead", "level", "behind")


def compute(match, who):
    r = {"ok": False, "states": {k: {"secs": 0.0, "up_pitch": 0.0,
                                     "possession": 0.0, "touches_s": 0.0}
                                 for k in STATES}}
    team = match.teams.get(who)
    if team is None or not match.samples:
        return r
    sgn = match.attack_sign(who)
    own_y = -5120.0 * sgn
    n_frames = len(match.frames) or 1

    goals = sorted(
        (int(len(match.samples) * (g.get("frame", 0) / n_frames)),
         g.get("PlayerTeam"))
        for g in match.goals_meta)

    diff, gi = 0, 0
    acc = {k: {"t": 0.0, "dist": 0.0, "touch": 0.0} for k in STATES}
    for i in range(len(match.samples) - 1):
        while gi < len(goals) and goals[gi][0] <= i:
            diff += 1 if goals[gi][1] == team else -1
            gi += 1
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > MAX_GAP:
            continue
        car = s["cars"].get(who)
        if not car:
            continue
        k = "ahead" if diff > 0 else "behind" if diff < 0 else "level"
        acc[k]["t"] += dt
        acc[k]["dist"] += abs(car["pos"][1] - own_y) * dt
        if s.get("hit_team") == team:
            acc[k]["touch"] += dt

    for k, a in acc.items():
        if a["t"] <= 0.0:
            continue
        r["states"][k] = {
            "secs": a["t"],
            "up_pitch": a["dist"] / a["t"],
            "possession": 100.0 * a["touch"] / a["t"],
        }
    r["ok"] = any(v["secs"] >= MIN_STATE_SECS for v in r["states"].values())
    return r


TITLE = "Scoreline -- how the scoreboard changes you"


def render(res):
    if not res.get("ok"):
        return ["  not enough time in any one game state"]
    out = ["  %-8s %8s %13s %12s" % ("state", "minutes", "up-pitch", "possession")]
    for k in STATES:
        v = res["states"][k]
        if v["secs"] < MIN_STATE_SECS:
            continue
        out.append("  %-8s %8.1f %10.0f uu %10.0f %%"
                   % (k, v["secs"] / 60.0, v["up_pitch"], v["possession"]))
    return out


def tips(res, match, who):
    if not res.get("ok"):
        return []
    s = res["states"]
    if s["ahead"]["secs"] < MIN_STATE_SECS or s["behind"]["secs"] < MIN_STATE_SECS:
        return []

    out = []
    drop = s["ahead"]["up_pitch"] - s["behind"]["up_pitch"]
    poss = s["ahead"]["possession"] - s["behind"]["possession"]

    if drop > 700:
        out.append(
            "When you go behind you drop %.0f uu deeper than when you are "
            "ahead (%.0f vs %.0f). That is the passive reaction, not the "
            "reckless one -- but a goal down is when you need the ball, and "
            "sitting back hands the game to whoever is already winning it."
            % (drop, s["behind"]["up_pitch"], s["ahead"]["up_pitch"]))
    elif drop < -700:
        out.append(
            "When you go behind you push %.0f uu higher than when ahead. That "
            "is the classic tilt-forward, and it is why a one-goal deficit "
            "becomes three." % (-drop))

    if poss > 12:
        out.append(
            "Possession falls from %.0f%% when ahead to %.0f%% when behind. "
            "You are conceding the ball exactly when you need it most."
            % (s["ahead"]["possession"], s["behind"]["possession"]))
    return out
