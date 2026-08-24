"""
Compare two 3v3 measurement runs on the metrics that matter.

    python tools/compare_runs.py <before-stamp> <after-stamp>
    python tools/compare_runs.py 20260823-114345 20260823-131500

Clash rate is reported first and deliberately: any change that buys aerials by
letting more than one car hold the attack slot has not bought anything.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict

# b0-b2 are team 0, b3-b5 team 1 (matches/3v3-measure.toml).
TEAM = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}


def load(stamp):
    per_bot = {}
    for f in sorted(glob.glob("data/telemetry/%s-b*-trace.jsonl" % stamp)):
        b = int(f.split("-b")[1].split("-")[0])
        rows = []
        for line in open(f, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        per_bot[b] = rows
    return per_bot


def metrics(per_bot):
    rows = [r for rs in per_bot.values() for r in rs]
    n = len(rows) or 1

    # Team-instants: all three cars of one team at the same match time.
    inst = defaultdict(list)
    for b, rs in per_bot.items():
        for r in rs:
            inst[(TEAM.get(b, 0), round(r["t"], 2))].append(r)
    full = [v for v in inst.values() if len(v) == 3]

    clash = sum(1 for v in full
                if len({r.get("ord") for r in v}) < 3)
    two_attacking = sum(
        1 for v in full
        if sum(1 for r in v if (r.get("act") or "").startswith("attack")) >= 2)

    # Role changes per bot-minute.
    changes = 0
    minutes = 0.0
    for rs in per_bot.values():
        if len(rs) < 2:
            continue
        minutes += (rs[-1]["t"] - rs[0]["t"]) / 60.0
        prev = None
        for r in rs:
            cur = r.get("role")
            if prev is not None and cur != prev:
                changes += 1
            prev = cur

    ballz = sorted(r["ball"][2] for r in rows if r.get("ball"))
    high = [r for r in rows if r.get("ball") and r["ball"][2] > 500]
    kh = Counter(r.get("k") or "(none)" for r in high)

    return {
        "ticks": len(rows),
        "instants": len(full),
        "clash%": 100.0 * clash / max(len(full), 1),
        "two_attack%": 100.0 * two_attacking / max(len(full), 1),
        "role_changes_per_min": changes / max(minutes, 1e-9),
        "ball_p90": ballz[int(0.9 * len(ballz)) - 1] if ballz else 0,
        "ball>500%": 100.0 * len(high) / n,
        "aerial_when_high%": 100.0 * kh.get("aerial", 0) / max(len(high), 1),
        "dodge_when_high%": 100.0 * kh.get("dodge", 0) / max(len(high), 1),
        "dj_when_high%": 100.0 * kh.get("double_jump", 0) / max(len(high), 1),
        "aerial_exec_ticks": sum(1 for r in rows if r.get("mech") == "aerial"),
        "car>300uu%": 100.0 * sum(1 for r in rows if r["me"][2] > 300) / n,
        "peak_car_z": max((r["me"][2] for r in rows), default=0),
        "airborne%": 100.0 * sum(1 for r in rows if not r.get("gnd")) / n,
    }


ORDER = [
    ("clash%", "slot clashes (LOWER is the constraint)", "lower"),
    ("two_attack%", "two+ attacking at once", "lower"),
    ("role_changes_per_min", "role changes / bot-min", "lower"),
    ("aerial_when_high%", "aerial chosen, ball >500uu", "higher"),
    ("dj_when_high%", "double_jump chosen, ball >500uu", "lower"),
    ("dodge_when_high%", "dodge chosen, ball >500uu", "lower"),
    ("aerial_exec_ticks", "aerial ticks actually executed", "higher"),
    ("car>300uu%", "car above 300uu", "higher"),
    ("peak_car_z", "peak car height", "higher"),
    ("ball_p90", "ball height p90", "higher"),
    ("ball>500%", "ball above 500uu", "higher"),
    ("airborne%", "airborne", "info"),
]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    before, after = metrics(load(sys.argv[1])), metrics(load(sys.argv[2]))
    print("before: %s (%d ticks, %d team-instants)"
          % (sys.argv[1], before["ticks"], before["instants"]))
    print("after : %s (%d ticks, %d team-instants)"
          % (sys.argv[2], after["ticks"], after["instants"]))
    print()
    print("%-34s %10s %10s %12s" % ("metric", "before", "after", "change"))
    print("-" * 70)
    for key, label, want in ORDER:
        b, a = before[key], after[key]
        delta = a - b
        if want == "info":
            flag = ""
        elif want == "lower":
            flag = "  ok" if delta <= 0.5 else "  WORSE"
        else:
            flag = "  ok" if delta >= 0 else "  worse"
        print("%-34s %10.2f %10.2f %+11.2f%s" % (label, b, a, delta, flag))
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
