"""
Verify that RocketSim can build an arena from the dumped collision meshes.

Run this after dumping. It checks the directory name (the dumper and RocketSim
disagree about the separator, which is the usual reason a correct dump appears
not to work), then actually constructs an arena and steps it.

    python tools/check_meshes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# RocketSim looks for this, relative to the working directory.
WANTED = ROOT / "collision_meshes"
# The dumper writes this. Same content, different separator.
DUMPED = ROOT / "collision-meshes"


def main() -> int:
    print("looking in:", ROOT)

    if DUMPED.is_dir() and not WANTED.is_dir():
        n = len(list(DUMPED.rglob("*")))
        print(f"\nFound '{DUMPED.name}' ({n} files) but RocketSim wants "
              f"'{WANTED.name}' -- note the underscore.")
        print("Renaming it for you...")
        try:
            DUMPED.rename(WANTED)
            print("  renamed OK")
        except Exception as e:
            print(f"  could not rename ({e}); do it by hand")
            return 1

    if not WANTED.is_dir():
        print(f"\nNo '{WANTED.name}' directory here yet.")
        print("Dump the meshes first -- see the walkthrough. In short:")
        print("  1. start Rocket League and go into free play")
        print("  2. run RLArenaCollisionDumper.exe from this folder")
        print("  3. run this script again")
        return 1

    files = sorted(p.name for p in WANTED.rglob("*") if p.is_file())
    print(f"\n'{WANTED.name}' contains {len(files)} files")
    for name in files[:12]:
        print("   ", name)
    if len(files) > 12:
        print(f"    ... and {len(files) - 12} more")

    if not files:
        print("\nDirectory is empty -- the dump did not produce anything.")
        return 1

    print("\nbuilding a RocketSim arena...")
    try:
        import RocketSim as rs
    except ImportError:
        print("  RocketSim is not installed:  pip install RocketSim")
        return 1

    try:
        arena = rs.Arena(rs.GameMode.SOCCAR)
    except Exception as e:
        print(f"  FAILED: {e}")
        print("\n  RocketSim resolves the mesh path relative to the WORKING")
        print("  directory, so run it from the project root.")
        return 1

    car = arena.add_car(rs.Team.BLUE)
    ball = arena.ball.get_state()
    ball.pos = rs.Vec(0, 0, 1200)
    # Nudge the velocity. Bullet puts a resting body to sleep, and teleporting
    # it does not wake it -- without this the ball hangs motionless at 1200 and
    # the check passes on a simulation that never ran.
    ball.vel = rs.Vec(0.0, 0.0, -1.0)
    arena.ball.set_state(ball)

    # Drop the ball and confirm it bounces off real floor geometry.
    heights = []
    for _ in range(300):
        arena.step(1)
        heights.append(arena.ball.get_state().pos.z)

    lowest = min(heights)
    rebound = max(heights[150:])
    print("  arena OK, car id", car.id)
    print(f"  ball dropped from 1200: fell to {lowest:.1f}, rebounded to {rebound:.0f}")

    if lowest > 1100:
        print("\n  FAILED: the ball never fell -- physics is not stepping.")
        return 1
    if not (80.0 < lowest < 110.0):
        print(f"\n  SUSPECT: floor contact at {lowest:.1f}, expected ~92 "
              "(the ball radius). Geometry may be wrong.")
        return 1
    if rebound < 200.0:
        print("\n  SUSPECT: the ball barely rebounded; check the meshes.")
        return 1
    print("\nRocketSim is usable. tools/optimise.py can now be given "
          "collision-accurate targets (walls, corners, ceiling, car-to-car).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
