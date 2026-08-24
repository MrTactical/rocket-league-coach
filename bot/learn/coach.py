"""
Coaching.

Two halves. `LiveCoach` speaks during the match through quick chat -- sparse,
because a teammate who comments on everything is noise. `analyse` runs after
the match over the recorded tally and produces specific, checkable pointers.

The bar for an insight: it must name a number, say what to do differently, and
suggest somewhere to practise it. "Rotate better" is not coaching. "You stayed
forward after 71% of your attacking touches; the ball came back past you 4
times" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Severity drives ordering in the report.
CRITICAL = 3
MAJOR = 2
MINOR = 1
GOOD = 0


@dataclass
class Insight:
    severity: int
    title: str
    detail: str
    drill: str = ""
    stat: str = ""


# --- live, in-match callouts ---------------------------------------------


class LiveCoach:
    """
    Occasional in-game callouts.

    Rate-limited hard: at most one message every `min_gap` seconds, and each
    distinct message type has its own cooldown so the same line cannot repeat.
    """

    def __init__(self, min_gap: float = 9.0, enabled: bool = True):
        self.enabled = enabled
        self.min_gap = min_gap
        self._last_any = -99.0
        self._last_of: dict[str, float] = {}

    def _can(self, key: str, now: float, cooldown: float) -> bool:
        if not self.enabled:
            return False
        if now - self._last_any < self.min_gap:
            return False
        if now - self._last_of.get(key, -99.0) < cooldown:
            return False
        return True

    def _fire(self, key: str, now: float, text: str) -> str:
        self._last_any = now
        self._last_of[key] = now
        return text

    def check(self, state, debug, model, my_time: float, ally_time: float) -> str | None:
        """Return a line to say this tick, or None. Called every tick."""
        if not self.enabled or not state.is_active or state.ally is None:
            return None
        now = state.time
        ally = state.ally

        # Taking the ball -- tell them so they don't double-commit.
        if debug.role == "attack" and my_time < 1.2 and ally_time < 2.0:
            if self._can("mine", now, 12.0):
                return self._fire("mine", now, "I got it!")

        # They're taking it and we're setting up behind.
        if debug.role == "support" and ally_time < 1.0:
            if self._can("yours", now, 14.0):
                return self._fire("yours", now, "All yours - I'm behind you")

        # Empty and staying forward.
        if ally.boost < 8.0 and ally.pos.y * state.goal_sign < -1500.0:
            if self._can("boost", now, 25.0):
                return self._fire("boost", now, "Grab boost, I'll cover")

        # We're last man and they're deep in the corner too.
        if (
            debug.role == "defend"
            and debug.threat > 0.5
            and ally.pos.y * state.goal_sign < -2000.0
        ):
            if self._can("rotate", now, 20.0):
                return self._fire("rotate", now, "Rotate back!")

        # Both of us bearing down on the same ball.
        if abs(my_time - ally_time) < 0.35 and my_time < 1.5 and debug.role == "attack":
            if self._can("double", now, 18.0):
                return self._fire("double", now, "You take it")

        return None


# --- post-match analysis --------------------------------------------------


def _pct(a: float, b: float) -> float:
    return (100.0 * a / b) if b > 0 else 0.0


def analyse(model, score=(0, 0)) -> list[Insight]:
    """
    Turn a finished match into coaching points, most important first.
    """
    t = model.traits
    m = model.tally
    out: list[Insight] = []
    dur = max(m.duration, 1.0)

    # --- boost economy ----------------------------------------------------
    low_pct = _pct(m.ally_low_boost_time, dur)
    dry_pct = _pct(m.ally_no_boost_time, dur)
    if low_pct > 45.0:
        out.append(Insight(
            CRITICAL if low_pct > 60 else MAJOR,
            "You're running on empty",
            f"You spent {low_pct:.0f}% of the match below 20 boost "
            f"({dry_pct:.0f}% completely dry). Low boost is why you lose "
            "50/50s and can't challenge -- it isn't a mechanical problem, "
            "it's a routing one.",
            "Free play: drive a full lap collecting only the small pads along "
            "the side walls, without touching the ball. The line through them "
            "is the line you should already be driving on every rotation.",
            f"{low_pct:.0f}% under 20 boost",
        ))
    elif low_pct < 18.0:
        out.append(Insight(GOOD, "Boost economy is solid",
                           f"Only {low_pct:.0f}% of the match under 20 boost. "
                           "That's what lets you challenge when you want to.",
                           stat=f"{low_pct:.0f}% under 20 boost"))

    # --- ball chasing -----------------------------------------------------
    # Two separate symptoms, reported differently because they need different
    # fixes: contesting balls you should leave, versus never leaving the ball
    # at all.
    glue_pct = t.ball_glue * 100.0
    if t.ball_glue > 0.72:
        out.append(Insight(
            CRITICAL if t.ball_glue > 0.85 else MAJOR,
            "You never leave the ball",
            f"You were within about two car lengths of the ball {glue_pct:.0f}% "
            "of the match. At this level that number wants to be nearer 50% -- "
            "the other half is you rotating out, collecting boost and coming "
            "back with speed. Right now there is no second man, because you "
            "are always the first.",
            "Play three games where, after every single touch, you drive to "
            "your own back post before looking at the ball again. It will feel "
            "like you are doing nothing. Your win rate will go up.",
            f"{glue_pct:.0f}% of match on the ball",
        ))
    elif t.chase_rate > 0.55:
        out.append(Insight(
            CRITICAL if t.chase_rate > 0.72 else MAJOR,
            "Ball chasing",
            f"In contested situations you drove at the ball {t.chase_rate * 100:.0f}% "
            "of the time when I was the better-placed player. Every one of "
            "those is a moment where we had two players on the ball and none "
            "covering.",
            "Play a few games where you are not allowed to touch the ball "
            "twice in a row. It feels awful and it fixes rotation faster than "
            "anything else.",
            f"chase rate {t.chase_rate * 100:.0f}%",
        ))
    elif t.ball_glue < 0.35 and t.rotation_discipline > 0.5:
        out.append(Insight(GOOD, "Good spacing",
                           f"You were on the ball {glue_pct:.0f}% of the match, "
                           "which means you were actually rotating rather than "
                           "following it around.",
                           stat=f"{glue_pct:.0f}% on the ball"))

    # --- double commits ---------------------------------------------------
    if m.double_commits >= 4:
        per_min = m.double_commits / (dur / 60.0)
        out.append(Insight(
            MAJOR if per_min > 1.5 else MINOR,
            "We keep going for the same ball",
            f"We both committed to the same ball {m.double_commits} times "
            f"({per_min:.1f} per minute). When you hear me call it, peel off "
            "and take the space behind instead.",
            "Watch the replay and pause at each one. Usually one of us was "
            "half a second later and should have read it.",
            f"{m.double_commits} double commits",
        ))

    # --- rotation discipline ----------------------------------------------
    if t.rotation_discipline < 0.45 and m.ally_touches > 5:
        stay_pct = (1.0 - t.rotation_discipline) * 100.0
        out.append(Insight(
            MAJOR,
            "You stay forward after your touches",
            f"After roughly {stay_pct:.0f}% of your attacking touches you held "
            "your position instead of retreating. That's how counters happen: "
            "the ball comes back past you and nobody's home.",
            "After every touch in their half, make yourself drive toward your "
            "own back post before you look at the ball again.",
            f"rotated out after {t.rotation_discipline * 100:.0f}% of touches",
        ))

    # --- conceding while upfield ------------------------------------------
    if m.conceded_while_ally_upfield >= 2:
        out.append(Insight(
            MAJOR,
            "Goals conceded while you were upfield",
            f"{m.conceded_while_ally_upfield} of {m.goals_against} goals went "
            "in while you were in the attacking third. Being high isn't wrong, "
            "but it has to be paired with me being back -- check where I am "
            "before you commit.",
            "Custom training: defending set shots. Learn what a save looks "
            "like from a bad starting position.",
            f"{m.conceded_while_ally_upfield}/{m.goals_against} conceded while high",
        ))

    # --- field balance -----------------------------------------------------
    own = m.ally_time_in_own_third
    att = m.ally_time_in_attacking_third
    if own + att > 30.0:
        att_pct = _pct(att, own + att)
        if att_pct > 72.0:
            out.append(Insight(
                MINOR,
                "You live in their half",
                f"{att_pct:.0f}% of your time in a third was the attacking one. "
                "It puts pressure on, but it means I'm defending alone.",
                "Try consciously touching your own back post once per minute.",
                f"{att_pct:.0f}% attacking third",
            ))
        elif att_pct < 28.0:
            out.append(Insight(
                MINOR,
                "You're playing too passively",
                f"Only {att_pct:.0f}% of your time was in the attacking third. "
                "I can hold the back -- you can afford to press higher when "
                "we win possession.",
                "When I clear the ball upfield, follow it rather than resetting.",
                f"{att_pct:.0f}% attacking third",
            ))

    # --- aerials ------------------------------------------------------------
    if m.ally_touches > 8:
        aerial_pct = _pct(m.ally_aerial_touches, m.ally_touches)
        if aerial_pct < 6.0:
            out.append(Insight(
                MINOR,
                "Almost everything you hit is on the ground",
                f"Only {aerial_pct:.0f}% of your touches were above the ground "
                "game. At this level a lot of balls are winnable in the air "
                "that you're currently letting bounce.",
                "Custom training: 'Aerial Shots' packs, plus free play where "
                "you only hit the ball above head height.",
                f"{aerial_pct:.0f}% aerial touches",
            ))

    # --- positives ---------------------------------------------------------
    if m.ally_saves >= 3:
        out.append(Insight(GOOD, "Good defending",
                           f"{m.ally_saves} saves. You were reading the danger early.",
                           stat=f"{m.ally_saves} saves"))
    if t.rotation_discipline > 0.65 and m.ally_touches > 5:
        out.append(Insight(GOOD, "Rotation is clean",
                           f"You rotated out after {t.rotation_discipline * 100:.0f}% "
                           "of your attacking touches. That's the habit that "
                           "carries teams.",
                           stat=f"{t.rotation_discipline * 100:.0f}% rotate-out"))

    out.sort(key=lambda i: -i.severity)
    return out


def format_report(model, insights: list[Insight], score=(0, 0), rank: str = "") -> str:
    """Plain-text report for the console."""
    m = model.tally
    lines = []
    lines.append("=" * 66)
    lines.append(f"  MATCH REPORT     {score[0]} - {score[1]}     "
                 f"({m.duration / 60.0:.1f} min, partner playing vs {rank})")
    lines.append("=" * 66)

    lines.append("")
    lines.append(f"  Your touches: {m.ally_touches:<5} shots: {m.ally_shots:<4} "
                 f"saves: {m.ally_saves:<4} goals: {m.ally_goals}")
    lines.append(f"  Double commits: {m.double_commits:<4} "
                 f"kickoffs taken: {m.kickoffs_taken_by_ally}/{m.kickoffs_total}")
    lines.append("")

    if not insights:
        lines.append("  Nothing stood out this match.")
    labels = {CRITICAL: "[!!]", MAJOR: "[! ]", MINOR: "[ ~]", GOOD: "[ +]"}
    for ins in insights:
        lines.append(f"  {labels[ins.severity]} {ins.title}"
                     + (f"   ({ins.stat})" if ins.stat else ""))
        for chunk in _wrap(ins.detail, 62):
            lines.append(f"        {chunk}")
        if ins.drill:
            lines.append(f"        -> Drill: {_wrap(ins.drill, 54)[0]}")
            for chunk in _wrap(ins.drill, 54)[1:]:
                lines.append(f"                  {chunk}")
        lines.append("")

    lines.append("-" * 66)
    lines.append(f"  Learned profile: {model.summary()}")
    lines.append("=" * 66)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]
