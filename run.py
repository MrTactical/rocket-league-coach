"""
Launcher.

    python run.py                 # default match (2v2 vs All-Star)
    python run.py 3v3             # any file in matches/, by prefix
    python run.py --list          # show available matches
    python run.py --rank champion # override the bot's skill for this run
    python run.py 1v1 --rank ssl

Requires RLBotServer, which ships with the RLBot installer from
https://rlbot.org (Windows MSI). If it is installed somewhere unusual, set
the RLBOT_SERVER_PATH environment variable to the executable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rlbot import flat  # noqa: E402
from rlbot.managers import MatchManager  # noqa: E402
from rlbot.utils.os_detector import RLBOT_SERVER_NAME  # noqa: E402

MATCH_DIR = ROOT / "matches"
DEFAULT_MATCH = "2v2-vs-allstar"

# Parent directories the installer might drop an RLBot folder into. The folder
# itself is matched by glob rather than named exactly: the v5 installer uses
# "RLBot5", older ones used "RLBot" and "RLBotGUIX", and that naming has
# already changed once. Globbing survives it changing again.
SEARCH_PARENTS = [
    Path(os.environ.get("LOCALAPPDATA", "")),
    Path(os.environ.get("APPDATA", "")),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
    Path(os.environ.get("PROGRAMFILES", "C:/Program Files")),
    Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")),
    Path.home(),
    ROOT,
]


def _newest(paths: list[Path]) -> Path | None:
    """Prefer the most recently modified binary if several are installed."""
    real = [p for p in paths if p.is_file()]
    if not real:
        return None
    return max(real, key=lambda p: p.stat().st_mtime)


def find_server() -> Path | None:
    explicit = os.environ.get("RLBOT_SERVER_PATH")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        if p.is_dir():
            found = next(p.glob(f"**/{RLBOT_SERVER_NAME}"), None)
            if found:
                return found

    # If a server is already running, its own path is the most reliable answer.
    try:
        import psutil

        for proc in psutil.process_iter(["name", "exe"]):
            if proc.info["name"] == RLBOT_SERVER_NAME and proc.info["exe"]:
                p = Path(proc.info["exe"])
                if p.is_file():
                    return p
    except Exception:
        pass

    hits: list[Path] = []
    for parent in SEARCH_PARENTS:
        try:
            if not parent or not parent.exists():
                continue
            for base in parent.glob("RLBot*"):
                if not base.is_dir():
                    continue
                hits.extend(base.glob(f"**/{RLBOT_SERVER_NAME}"))
        except Exception:
            continue
    return _newest(hits)


def list_matches() -> list[Path]:
    return sorted(MATCH_DIR.glob("*.toml"))


def resolve_match(name: str | None) -> Path | None:
    options = list_matches()
    if not options:
        return None
    if not name:
        name = DEFAULT_MATCH
    # Exact stem first, then prefix, then substring.
    for p in options:
        if p.stem == name:
            return p
    for p in options:
        if p.stem.startswith(name):
            return p
    for p in options:
        if name.lower() in p.stem.lower():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch a Rocket League match with Ally.")
    ap.add_argument("match", nargs="?", help="match name (see --list)")
    ap.add_argument("--list", action="store_true", help="list available matches")
    ap.add_argument("--rank", help="override Ally's skill level for this run")
    args = ap.parse_args()

    if args.list:
        print("Available matches:\n")
        for p in list_matches():
            print(f"  {p.stem}")
        print(f"\nDefault: {DEFAULT_MATCH}")
        return 0

    match_path = resolve_match(args.match)
    if match_path is None:
        print(f"No match config matching {args.match!r}. Try --list.")
        return 1

    if args.rank:
        # ally.py reads this in preference to config/ally.toml.
        os.environ["ALLY_RANK"] = args.rank

    server = find_server()
    if server is None:
        print(
            "Could not find RLBotServer.\n\n"
            "Install RLBot from https://rlbot.org (the Windows MSI), or if it is\n"
            "already installed somewhere unusual, point at it directly:\n\n"
            '    $env:RLBOT_SERVER_PATH = "C:\\path\\to\\RLBotServer.exe"\n'
        )
        return 2

    print(f"Server : {server}")
    print(f"Match  : {match_path.name}")
    if args.rank:
        print(f"Rank   : {args.rank} (override)")
    print("\nStarting Rocket League. This takes a moment the first time.\n")

    with MatchManager(server) as man:
        man.start_match(match_path)
        if man.packet is None:
            print("Match failed to start.")
            return 3

        # Waiting only on MatchPhase.Ended is not enough. Observed: a five
        # minute match played its full 364s, the bots wrote complete
        # telemetry, and this loop then span for another eight minutes
        # because the phase never flipped -- the server had already let go
        # of the game. So also give up when the clock stops advancing.
        STALL_LIMIT = 20.0
        last_seen = -1.0
        stalled_for = 0.0
        try:
            while man.packet.match_info.match_phase != flat.MatchPhase.Ended:
                sleep(1.0)
                now = man.packet.match_info.seconds_elapsed
                if now == last_seen:
                    stalled_for += 1.0
                    if stalled_for >= STALL_LIMIT:
                        print(
                            f"Match clock stopped at {now:.1f}s and has not "
                            f"advanced for {STALL_LIMIT:.0f}s. Treating it as over."
                        )
                        break
                else:
                    stalled_for = 0.0
                    last_seen = now
        except KeyboardInterrupt:
            print("Stopping.")

    print("\nMatch over. Report and telemetry are in data/telemetry/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
