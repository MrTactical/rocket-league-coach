"""
Automated match analysis.

Reads the telemetry from a match and reports what is wrong with the bots, with
a number attached to each complaint. This is the feedback loop: run a match,
run this, fix the worst thing, run again.

    python tools/analyse.py              # newest match
    python tools/analyse.py 20260822-191507
    python tools/analyse.py --list

Each check has an explicit target and tolerance, so "is this better than last
time" is answerable rather than a matter of impression. Checks are ordered by
how badly they are failing, not by how interesting they are.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = ROOT / "data" / "telemetry"

BAD, WARN, OK, INFO = 3, 2, 1, 0
LABEL = {BAD: "[BAD ]", WARN: "[WARN]", OK: "[ ok ]", INFO: "[info]"}


@dataclass
class Finding:
    level: int
    title: str
    value: str
    target: str = ""
    detail: str = ""


@dataclass
class Match:
    stamp: str
    bots: dict[int, list[dict]] = field(default_factory=dict)
    events: dict[int, list[dict]] = field(default_factory=dict)

    @property
    def n_bots(self) -> int:
        return len(self.bots)

    @property
    def span(self) -> float:
        spans = [r[-1]["t"] - r[0]["t"] for r in self.bots.values() if len(r) > 1]
        return max(spans) if spans else 0.0

    def aligned_length(self) -> int:
        return min((len(r) for r in self.bots.values()), default=0)


def load_match(stamp: str) -> Match:
    m = Match(stamp)
    for p in sorted(TELEMETRY.glob(f"{stamp}*-trace.jsonl")):
        slot = _slot_of(p.name)
        m.bots[slot] = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    for p in sorted(TELEMETRY.glob(f"{stamp}*-events.jsonl")):
        slot = _slot_of(p.name)
        m.events[slot] = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return m


def _slot_of(name: str) -> int:
    # "20260822-191507-b3-trace.jsonl" -> 3 ; unslotted files -> 0
    for part in name.split("-"):
        if part.startswith("b") and part[1:].isdigit():
            return int(part[1:])
    return 0


def list_matches() -> list[str]:
    stamps = set()
    for p in TELEMETRY.glob("*-trace.jsonl"):
        parts = p.name.split("-")
        if len(parts) >= 2:
            stamps.add(f"{parts[0]}-{parts[1]}")
    return sorted(stamps)


def teams_of(m: Match) -> dict[str, list[int]]:
    """
    Split bot slots into teams by car index. RLBot allocates indices in match
    config order, and our self-play configs list team 0 first.
    """
    slots = sorted(m.bots)
    if len(slots) < 2:
        return {"team": slots}
    half = len(slots) // 2
    return {"team0": slots[:half], "team1": slots[half:]}


# --- individual checks ----------------------------------------------------


def check_simultaneous_attackers(m: Match) -> list[Finding]:
    """The headline number. More than one attacker at a time IS ball chasing."""
    out = []
    n = m.aligned_length()
    if n < 20 or m.n_bots < 2:
        return out
    for team, members in teams_of(m).items():
        if len(members) < 2:
            continue
        counts = collections.Counter()
        for i in range(n):
            counts[sum(1 for b in members if m.bots[b][i]["role"] == "attack")] += 1
        tot = sum(counts.values())
        multi = 100.0 * sum(v for k, v in counts.items() if k >= 2) / tot
        none = 100.0 * counts.get(0, 0) / tot
        level = BAD if multi > 20 else WARN if multi > 8 else OK
        out.append(Finding(
            level,
            f"{team}: two or more attacking at once",
            f"{multi:.1f}%",
            "under 8%",
            f"exactly one attacking {100.0 * counts.get(1, 0) / tot:.0f}% of the time, "
            f"nobody attacking {none:.0f}%",
        ))
    return out


def check_clumping(m: Match) -> list[Finding]:
    out = []
    n = m.aligned_length()
    if n < 20:
        return out
    for team, members in teams_of(m).items():
        if len(members) < 2:
            continue
        ds = []
        for i in range(0, n, 3):
            pts = [m.bots[b][i]["me"][:2] for b in members]
            for x in range(len(pts)):
                for y in range(x + 1, len(pts)):
                    ds.append(math.dist(pts[x], pts[y]))
        if not ds:
            continue
        ds.sort()
        close = 100.0 * sum(1 for d in ds if d < 900) / len(ds)
        med = ds[len(ds) // 2]
        level = BAD if close > 30 else WARN if close > 18 else OK
        out.append(Finding(
            level, f"{team}: teammates crowded together", f"{close:.0f}% under 900uu",
            "under 18%", f"median separation {med:.0f}uu",
        ))
    return out


def check_comms(m: Match) -> list[Finding]:
    """Silent comms make every coordination fix a no-op, so check it first."""
    out = []
    if m.n_bots < 2:
        return out
    have = [r for rows in m.bots.values() for r in rows if "peers" in r]
    if not have:
        out.append(Finding(INFO, "peer comms", "not recorded",
                           detail="telemetry predates the peers field"))
        return out
    expected = max(1, (m.n_bots // 2) - 1)
    per_bot = {}
    for b, rows in m.bots.items():
        vals = [r.get("peers", 0) for r in rows if r.get("role") != "kickoff"]
        per_bot[b] = sum(vals) / len(vals) if vals else 0.0
    avg = sum(per_bot.values()) / len(per_bot)
    silent = [b for b, v in per_bot.items() if v < 0.5]
    level = BAD if avg < 0.5 else WARN if avg < expected * 0.7 else OK
    out.append(Finding(
        level, "peer comms visibility", f"{avg:.2f} peers seen",
        f"{expected} expected",
        ("no peers visible at all -- coordination cannot work"
         if avg < 0.5 else
         f"silent bots: {silent}" if silent else "all bots hearing each other"),
    ))
    return out


def check_ordinals(m: Match) -> list[Finding]:
    """Two bots claiming first man at once is the direct cause of double commits."""
    out = []
    n = m.aligned_length()
    if n < 20 or m.n_bots < 2:
        return out
    if not any("ord" in r for rows in m.bots.values() for r in rows):
        return out
    for team, members in teams_of(m).items():
        if len(members) < 2:
            continue
        clashes = 0
        for i in range(n):
            ords = [m.bots[b][i].get("ord", 0) for b in members]
            if len(ords) != len(set(ords)):
                clashes += 1
        pct = 100.0 * clashes / n
        level = BAD if pct > 25 else WARN if pct > 10 else OK
        spread = collections.Counter()
        for b in members:
            for r in m.bots[b]:
                spread[r.get("ord", 0)] += 1
        tot = sum(spread.values())
        out.append(Finding(
            level, f"{team}: teammates claiming the same slot", f"{pct:.0f}% of ticks",
            "under 10%",
            "ordinal spread " + " ".join(f"#{k + 1}={100 * v / tot:.0f}%"
                                         for k, v in sorted(spread.items())),
        ))
    return out


def check_role_thrash(m: Match) -> list[Finding]:
    out = []
    for b, rows in sorted(m.bots.items()):
        if len(rows) < 30:
            continue
        changes = sum(1 for i in range(1, len(rows)) if rows[i]["role"] != rows[i - 1]["role"])
        span = max(rows[-1]["t"] - rows[0]["t"], 1.0)
        per_min = changes / span * 60.0
        if per_min > 40:
            out.append(Finding(BAD, f"b{b}: role thrashing", f"{per_min:.0f} changes/min",
                               "under 25", "the car will twitch instead of committing"))
        elif per_min > 25:
            out.append(Finding(WARN, f"b{b}: role changes frequent", f"{per_min:.0f}/min", "under 25"))
    return out[:3]


def check_boost(m: Match) -> list[Finding]:
    out = []
    allb = [r["me"][3] for rows in m.bots.values() for r in rows]
    if not allb:
        return out
    zero = 100.0 * sum(1 for x in allb if x <= 1) / len(allb)
    avg = sum(allb) / len(allb)
    level = BAD if zero > 30 else WARN if zero > 18 else OK
    out.append(Finding(level, "boost economy", f"{zero:.0f}% at zero", "under 18%",
                       f"average boost {avg:.0f}"))
    return out


def check_ball_glue(m: Match) -> list[Finding]:
    out = []
    ds = []
    for rows in m.bots.values():
        for r in rows:
            if r.get("ball"):
                ds.append(math.dist(r["me"][:2], r["ball"][:2]))
    if not ds:
        return out
    ds.sort()
    near = 100.0 * sum(1 for d in ds if d < 800) / len(ds)
    level = BAD if near > 55 else WARN if near > 40 else OK
    out.append(Finding(level, "everyone sitting on the ball", f"{near:.0f}% within 800uu",
                       "under 40%", f"median distance {ds[len(ds) // 2]:.0f}uu"))
    return out


def check_speed(m: Match) -> list[Finding]:
    out = []
    sp = []
    for rows in m.bots.values():
        for i in range(1, len(rows)):
            dt = rows[i]["t"] - rows[i - 1]["t"]
            if 0 < dt < 0.5:
                sp.append(math.dist(rows[i]["me"][:2], rows[i - 1]["me"][:2]) / dt)
    if not sp:
        return out
    sp.sort()
    med = sp[len(sp) // 2]
    slow = 100.0 * sum(1 for s in sp if s < 400) / len(sp)
    level = WARN if med < 900 else OK
    out.append(Finding(level, "pace", f"median {med:.0f} uu/s", "over 900",
                       f"{slow:.0f}% of time under 400 uu/s"))
    return out


def check_touches(m: Match) -> list[Finding]:
    out = []
    total = 0
    for evs in m.events.values():
        total += sum(1 for e in evs if e.get("kind") == "my_touch")
    if m.span < 10:
        return out
    per_min = total / (m.span / 60.0)
    out.append(Finding(INFO, "bot touches", f"{per_min:.1f}/min across {m.n_bots} bots",
                       detail=f"{total} total in {m.span:.0f}s"))
    return out


def check_actions(m: Match) -> list[Finding]:
    counts = collections.Counter()
    tot = 0
    for rows in m.bots.values():
        for r in rows:
            counts[r.get("act", "")] += 1
            tot += 1
    if not tot:
        return []
    top = "  ".join(f"{k}={100 * v / tot:.0f}%" for k, v in counts.most_common(6))
    findings = [Finding(INFO, "action mix", "", detail=top)]
    clear = counts.get("attack/clear", 0) / tot * 100
    shot = counts.get("attack/shot", 0) / tot * 100
    if clear > shot * 1.8 and clear > 15:
        findings.append(Finding(
            WARN, "clearing far more than shooting", f"clear {clear:.0f}% vs shot {shot:.0f}%",
            "clears should not dominate",
            "the clear threshold may be too conservative",
        ))
    return findings



def check_aerials(m: Match) -> list[Finding]:
    """
    Are the bots ever leaving the ground on purpose?

    Distinguishes real air play from jumps and recoveries. A bot that never
    goes above ~250uu is not aerialling, it is hopping -- and if the ball also
    never gets high, the two facts are probably the same fact.
    """
    out = []
    heights, ball_heights, airborne = [], [], 0
    total = 0
    for rows in m.bots.values():
        for r in rows:
            total += 1
            z = r["me"][2]
            heights.append(z)
            if z > 60:
                airborne += 1
            if r.get("ball"):
                ball_heights.append(r["ball"][2])
    if not total:
        return out

    peak = max(heights)
    high = 100.0 * sum(1 for z in heights if z > 400) / total
    level = BAD if peak < 300 else WARN if high < 1.0 else OK
    out.append(Finding(
        level, "air play", f"peak {peak:.0f}uu, {high:.1f}% above 400uu",
        "peak over 600uu",
        f"{100.0 * airborne / total:.1f}% off the ground at all "
        "(includes jumps and recoveries)",
    ))

    if ball_heights:
        ball_heights.sort()
        bh_med = ball_heights[len(ball_heights) // 2]
        bh_high = 100.0 * sum(1 for z in ball_heights if z > 400) / len(ball_heights)
        out.append(Finding(
            INFO, "ball height", f"median {bh_med:.0f}uu, {bh_high:.1f}% above 400uu",
            detail=("the ball rarely gets up, so there may simply be nothing to "
                    "aerial for" if bh_high < 8 else "there are high balls available"),
        ))
    return out


def check_boost_contention(m: Match) -> list[Finding]:
    """
    Do teammates chase the same pad?

    The trace records note=="boost" while a pad detour is active, so two
    teammates detouring at once and converging on the same point is visible
    without knowing which pad each picked.
    """
    out = []
    n = m.aligned_length()
    if n < 20 or m.n_bots < 2:
        return out

    for team, members in teams_of(m).items():
        if len(members) < 2:
            continue
        both, contended = 0, 0
        for i in range(n):
            detouring = [b for b in members
                         if "boost" in (m.bots[b][i].get("note") or "")]
            if len(detouring) < 2:
                continue
            both += 1
            # Converging: two detouring bots within a pad's catchment.
            for x in range(len(detouring)):
                for y in range(x + 1, len(detouring)):
                    a = m.bots[detouring[x]][i]["me"][:2]
                    c = m.bots[detouring[y]][i]["me"][:2]
                    if math.dist(a, c) < 1400:
                        contended += 1
                        break
                else:
                    continue
                break
        pct = 100.0 * contended / n
        level = BAD if pct > 12 else WARN if pct > 6 else OK
        out.append(Finding(
            level, f"{team}: teammates chasing the same boost", f"{pct:.1f}% of ticks",
            "under 6%",
            f"two or more detouring for boost simultaneously on "
            f"{100.0 * both / n:.0f}% of ticks",
        ))
    return out


CHECKS = [
    check_comms,
    check_ordinals,
    check_simultaneous_attackers,
    check_clumping,
    check_ball_glue,
    check_aerials,
    check_boost_contention,
    check_role_thrash,
    check_boost,
    check_speed,
    check_actions,
    check_touches,
]


def analyse(m: Match) -> list[Finding]:
    found: list[Finding] = []
    for fn in CHECKS:
        try:
            found.extend(fn(m))
        except Exception as e:
            found.append(Finding(INFO, f"check {fn.__name__} failed", str(e)))
    found.sort(key=lambda f: -f.level)
    return found


def main() -> int:
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        for s in list_matches():
            print(s)
        return 0

    stamps = list_matches()
    if not stamps:
        print("no telemetry found")
        return 1
    stamp = args[0] if args else stamps[-1]

    m = load_match(stamp)
    if not m.bots:
        print(f"no trace data for {stamp}")
        return 1

    print("=" * 74)
    print(f"  MATCH {stamp}   {m.n_bots} bot(s)   {m.span:.0f}s")
    print("=" * 74)
    print()

    findings = analyse(m)
    worst = [f for f in findings if f.level >= WARN]

    for f in findings:
        line = f"  {LABEL[f.level]} {f.title}"
        if f.value:
            line += f": {f.value}"
        if f.target:
            line += f"  (target {f.target})"
        print(line)
        if f.detail:
            print(f"         {f.detail}")

    print()
    print("-" * 74)
    if worst:
        print(f"  {len(worst)} issue(s) worth fixing. Worst first:")
        for f in worst[:3]:
            print(f"    - {f.title}: {f.value}")
    else:
        print("  Nothing above the warning threshold.")
    print("-" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
