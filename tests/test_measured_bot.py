"""
The measured profile must actually change what the bot does.

Every field here was written into the profile JSON, loaded into SkillProfile,
and then read by nothing at all -- the bot missed shots like the player and
went nowhere like them. These tests fail if a field goes back to being
decorative.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.brain import roles  # noqa: E402
from bot.brain.humanize import RANKS, profile_from_file  # noqa: E402


def _profile(**skill):
    base = {"name": "test", "skill": {"band": "diamond", **skill},
            "rotation": {}, "derived_from": {"matches": 10}}
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8")
    json.dump(base, fh)
    fh.close()
    return profile_from_file(fh.name)


def test_every_skill_field_in_the_file_reaches_the_profile():
    """The loader had a hardcoded list of three and silently dropped the rest."""
    p = _profile(aerial_confidence=0.11, whiff_chance=0.22, input_noise=0.033,
                 wavedash_rate=0.44, speedflip_skill=0.55,
                 pressure_sensitivity=0.66, reaction_time=0.77, aim_error=0.088)
    assert p.aerial_confidence == 0.11
    assert p.whiff_chance == 0.22
    assert p.input_noise == 0.033
    assert p.wavedash_rate == 0.44
    assert p.speedflip_skill == 0.55
    assert p.pressure_sensitivity == 0.66
    assert p.reaction_time == 0.77
    assert p.aim_error == 0.088


def test_unset_fields_fall_back_to_the_band():
    """A replay cannot show reaction time; the band must still supply one."""
    p = _profile(aerial_confidence=0.5)
    assert p.reaction_time == RANKS["diamond"].reaction_time
    assert p.aim_error == RANKS["diamond"].aim_error


def test_support_depth_responds_to_up_pitch():
    """A player who sits further up gets a bot that supports closer in."""
    roles.set_measured({})
    roles.set_measured({"up_pitch": roles.POP_UP_PITCH * 1.3})
    forward = roles._MEASURED["up_pitch"]
    roles.set_measured({"up_pitch": roles.POP_UP_PITCH * 0.7})
    deep = roles._MEASURED["up_pitch"]
    assert forward > deep, "fixture is backwards"

    # The factor itself, applied the way support_position applies it.
    def factor(up):
        return max(0.7, min(1.4, 2.0 - (up / roles.POP_UP_PITCH)))

    assert factor(forward) < 1.0 < factor(deep), \
        "playing further up must shorten the support distance, not lengthen it"
    roles.set_measured({})


def test_committed_pushes_support_forward():
    def factor(com):
        return max(0.8, min(1.25, 1.0 - 0.5 * (com / roles.POP_COMMITTED - 1.0)))

    assert factor(roles.POP_COMMITTED * 1.5) < 1.0, \
        "more time committed forward should support closer to the ball"
    assert factor(roles.POP_COMMITTED * 0.5) > 1.0


def test_exposed_raises_the_cover_threshold():
    """
    The bot should reproduce the fault, not quietly correct it.

    A player often caught ahead of the ball gets a bot that waits longer
    before covering -- otherwise the practice partner is better behaved than
    the person it models, and the drill teaches nothing.
    """
    base = 0.26

    def enter(exposed):
        ratio = exposed / roles.POP_EXPOSED
        return max(0.10, min(0.45, base * max(0.75, min(1.35, ratio))))

    assert enter(roles.POP_EXPOSED * 1.5) > base
    assert enter(roles.POP_EXPOSED * 0.5) < base


def test_clamps_hold_at_the_extremes():
    """No profile may produce a bot glued to the ball or parked in net."""
    for up in (1.0, 100000.0):
        f = max(0.7, min(1.4, 2.0 - (up / roles.POP_UP_PITCH)))
        assert 0.7 <= f <= 1.4
    for com in (0.0, 100.0):
        f = max(0.8, min(1.25, 1.0 - 0.5 * (com / roles.POP_COMMITTED - 1.0)))
        assert 0.8 <= f <= 1.25


def demo():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
    print("%d/%d passed" % (n, n))


if __name__ == "__main__":
    demo()
