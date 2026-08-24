"""
Config validation.

Checking that a TOML file parses is not the same as checking RLBot will accept
it. Enum values like `match_length` are plain strings to a TOML parser and are
only rejected later, by the server, at match start -- which means a typo
surfaces as a failed launch rather than a failed test.

This suite runs every config through RLBot's own loader, so the same code that
would reject it at launch rejects it here instead.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlbot.config import load_match_config, load_player_config  # noqa: E402

MATCH_DIR = ROOT / "matches"
BOT_CONFIG = ROOT / "bot" / "ally.bot.toml"


def raw_cars(path: Path) -> list[dict]:
    """The `[[cars]]` tables as written, before RLBot resolves them."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f).get("cars", [])
    except Exception:
        return []


def check_match_configs() -> list[str]:
    errors = []
    configs = sorted(MATCH_DIR.glob("*.toml"))
    if not configs:
        return ["no match configs found"]

    for path in configs:
        try:
            mc = load_match_config(path)
        except Exception as e:
            errors.append(f"{path.name}: {type(e).__name__}: {e}")
            continue

        players = mc.player_configurations
        if not players:
            errors.append(f"{path.name}: no cars defined")
            continue

        varieties = [type(p.variety).__name__ for p in players]

        # Self-play, training and measurement configs deliberately have no
        # human in them -- they exist to be left running and scored. Everything
        # else is meant to be played, so a missing human seat is a real mistake
        # there.
        self_play = any(
            tag in path.stem
            for tag in ("ally-vs-ally", "train", "selfplay", "measure")
        )
        if not self_play and "Human" not in varieties:
            errors.append(f"{path.name}: no human player")
        if self_play and "Human" in varieties:
            errors.append(f"{path.name}: self-play config should not seat a human")
        if "CustomBot" not in varieties:
            errors.append(f"{path.name}: Ally is not in the match")

        # Every Ally in a match needs its own agent_id.
        #
        # RLBot groups cars by agent_id and hands the whole group to a single
        # process as a hivemind. Our Bot base class drives only the first
        # controllable, so duplicate ids silently leave cars parked -- a six
        # car match plays out as a 3v1 with no error anywhere.
        agent_ids = []
        for car in raw_cars(path):
            cfg_file = car.get("config_file")
            if not cfg_file:
                continue
            bot_path = (path.parent / cfg_file).resolve()
            if not bot_path.exists():
                errors.append(f"{path.name}: missing bot config {cfg_file}")
                continue
            try:
                with bot_path.open("rb") as f:
                    agent_ids.append(tomllib.load(f)["settings"]["agent_id"])
            except Exception as e:
                errors.append(f"{path.name}: could not read {cfg_file}: {e}")

        dupes = {i for i in agent_ids if agent_ids.count(i) > 1}
        if dupes:
            errors.append(
                f"{path.name}: duplicate agent_id(s) {sorted(dupes)} across "
                f"{len(agent_ids)} bots -- all but one car will sit parked. "
                "Run tools/gen_ally_slots.py and use ally-N.bot.toml."
            )

        # A training config is useless without state setting, since the
        # director works by teleporting the ball into fresh scenarios.
        if self_play and path.stem.startswith("train"):
            if not mc.script_configurations:
                errors.append(f"{path.name}: training config has no director script")
            if not mc.enable_state_setting:
                errors.append(
                    f"{path.name}: training needs enable_state_setting = true, "
                    "or the director cannot reset drills"
                )

        # Teams must be 0 or 1, and not everyone on one side.
        teams = {p.team for p in players}
        if not teams.issubset({0, 1}):
            errors.append(f"{path.name}: invalid team numbers {teams}")

    return errors


def check_bot_config() -> list[str]:
    errors = []
    try:
        with BOT_CONFIG.open("rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        return [f"ally.bot.toml: {e}"]

    settings = raw.get("settings", {})
    cmd = settings.get("run_command", "")
    if not cmd:
        errors.append("ally.bot.toml: no run_command")
    else:
        parts = cmd.split(" ")
        exe = (BOT_CONFIG.parent / parts[0]).resolve()
        if not exe.exists():
            errors.append(f"ally.bot.toml: interpreter not found at {exe}")
        if len(parts) > 1:
            script = (BOT_CONFIG.parent / parts[1]).resolve()
            if not script.exists():
                errors.append(f"ally.bot.toml: script not found at {script}")

    loadout = settings.get("loadout_file")
    if loadout and not (BOT_CONFIG.parent / loadout).exists():
        errors.append(f"ally.bot.toml: loadout not found: {loadout}")

    # The agent id must match what bot/ally.py passes to super().__init__.
    agent_id = settings.get("agent_id", "")
    source = (ROOT / "bot" / "ally.py").read_text(encoding="utf-8")
    if agent_id and f'"{agent_id}"' not in source:
        errors.append(
            f"ally.bot.toml: agent_id {agent_id!r} does not appear in ally.py -- "
            "the bot will not be matched to its config"
        )

    try:
        load_player_config(BOT_CONFIG, 0)
    except Exception as e:
        errors.append(f"ally.bot.toml: RLBot rejected it: {e}")

    return errors


def main() -> int:
    errors = check_match_configs() + check_bot_config()
    n = len(sorted(MATCH_DIR.glob("*.toml"))) + 1
    if errors:
        print(f"{len(errors)} problem(s) across {n} config(s):\n")
        for e in errors:
            print(f"  FAIL  {e}")
        return 1
    print(f"all {n} configs valid (checked against RLBot's own loader)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
