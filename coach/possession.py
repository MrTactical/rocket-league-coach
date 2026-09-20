"""
Who has the ball, and therefore what a position MEANS.

The rest of the analyser measures states. A state does not carry an intention:
being ahead of the ball is a deliberate commit when your team has possession in
the opponent's half, and being beaten when the opponent has it in yours. The
same number comes out of both, so advice built on the raw number criticises the
attack and the mistake in identical words.

`hit_team` (TAGame HitTeamNum, parsed in timeline.py) is sticky: it names the
team that touched the ball last and holds until someone else touches it. That
is a possession signal, and it was sitting unused by every positional metric.
touches.py calibrated it independently -- 95 of the 96 handovers in the sample
match line up with a detected touch by the team taking over -- so it is is good
enough to condition on.

This is a PROXY for intent, not a reading of it. A player can be ahead of the
ball during their own possession and still be badly placed. What it buys is the
ability to stop calling every advanced position a mistake.
"""

from __future__ import annotations

# Phases, from the point of view of one team.
ATTACK = "attack"        # we touched last, ball in their half
BUILD = "build"          # we touched last, ball in our half
DEFEND = "defend"        # they touched last, ball in our half
CONTAIN = "contain"      # they touched last, ball in their half
UNKNOWN = "unknown"      # nobody has touched it yet this kickoff

# Ball this close to the halfway line belongs to neither half in any meaningful
# sense; forcing it into one produces phase flapping around midfield.
NEUTRAL_Y = 900.0


def phase(sample, team, sgn):
    """
    One of the five constants above, for `team` at this frame.

    `sgn` is match.attack_sign(who): +1 when the team attacks toward +y, so
    `ball_y * sgn > 0` means the ball is in the opponent's half.
    """
    ht = sample.get("hit_team")
    if ht is None:
        return UNKNOWN
    fwd = sample["ball"][1] * sgn
    if abs(fwd) < NEUTRAL_Y:
        return UNKNOWN
    ours = (ht == team)
    if ours:
        return ATTACK if fwd > 0 else BUILD
    return CONTAIN if fwd > 0 else DEFEND


def ours(sample, team):
    """True when this team touched the ball most recently."""
    ht = sample.get("hit_team")
    return None if ht is None else (ht == team)


def is_committed(phase_name):
    """Phases where being ahead of the ball is the point, not a fault."""
    return phase_name in (ATTACK, CONTAIN)


def is_exposed(phase_name):
    """Phases where being ahead of the ball leaves your net open."""
    return phase_name == DEFEND


def split(match, who):
    """
    Live seconds spent in each phase. Weighted by real elapsed time, because
    replay frames are irregular and counting them over-weights dense stretches.
    """
    team = match.teams.get(who)
    if team is None or not match.samples:
        return {}
    sgn = match.attack_sign(who)
    out = {}
    for i in range(len(match.samples) - 1):
        s, nxt = match.samples[i], match.samples[i + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0.0 or dt > 0.5:
            continue
        out[phase(s, team, sgn)] = out.get(phase(s, team, sgn), 0.0) + dt
    return out


def demo():
    """Self-check on hand-built frames -- no replay needed."""
    def s(ball_y, ht):
        return {"t": 0.0, "hit_team": ht, "ball": (0.0, ball_y, 93.0)}

    # team 0 attacks toward +y (sgn +1)
    assert phase(s(3000, 0), 0, 1.0) == ATTACK
    assert phase(s(-3000, 0), 0, 1.0) == BUILD
    assert phase(s(-3000, 1), 0, 1.0) == DEFEND
    assert phase(s(3000, 1), 0, 1.0) == CONTAIN
    assert phase(s(0, 0), 0, 1.0) == UNKNOWN        # neutral band
    assert phase(s(3000, None), 0, 1.0) == UNKNOWN

    # team 1 attacks toward -y (sgn -1): the same world position flips meaning
    assert phase(s(-3000, 1), 1, -1.0) == ATTACK
    assert phase(s(-3000, 0), 1, -1.0) == CONTAIN
    assert phase(s(3000, 0), 1, -1.0) == DEFEND

    assert is_exposed(DEFEND) and not is_exposed(ATTACK)
    assert is_committed(ATTACK) and not is_committed(DEFEND)
    print("possession: all assertions pass")


if __name__ == "__main__":
    demo()
