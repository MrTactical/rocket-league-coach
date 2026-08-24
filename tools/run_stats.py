"""
Per-run metrics across many runs, with mean and spread.

    python tools/run_stats.py                    # every six-bot run
    python tools/run_stats.py 20260823-1         # runs matching a prefix

A single match cannot tell a real change from run-to-run noise. Measured on
this bot, clash rate on UNMODIFIED code varied 2.49%-5.30% across seven runs,
so any conclusion drawn from one match against one baseline is worthless.
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict

TEAM = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}


def six_bot_runs(prefix=""):
    seen = defaultdict(set)
    for f in glob.glob("data/telemetry/*-b*-trace.jsonl"):
        m = re.search(r"(\d{8}-\d{6})-b(\d)-trace", f)
        if m and m.group(1).startswith(prefix):
            seen[m.group(1)].add(int(m.group(2)))
    return sorted(s for s, b in seen.items() if b == {0, 1, 2, 3, 4, 5})


def stats_for(stamp):
    inst = defaultdict(list)
    rows = []
    for b in range(6):
        try:
            fh = open("data/telemetry/%s-b%d-trace.jsonl" % (stamp, b),
                      encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append(d)
            inst[(TEAM[b], round(d["t"], 2))].append(d)

    full = [v for v in inst.values() if len(v) == 3]
    if len(full) < 400 or not rows:
        return None
    clash = sum(1 for v in full if len({r.get("ord") for r in v}) < 3)
    high = [r for r in rows if r.get("ball") and r["ball"][2] > 500]
    kh = Counter(r.get("k") or "" for r in high)
    rej = Counter(r.get("arej") or "" for r in rows if r.get("arej"))
    return {
        "instants": len(full),
        "clash": 100.0 * clash / len(full),
        "aerial_high": 100.0 * kh.get("aerial", 0) / max(len(high), 1),
        "dj_high": 100.0 * kh.get("double_jump", 0) / max(len(high), 1),
        "aerial_exec": sum(1 for r in rows if r.get("mech") == "aerial"),
        "peak_z": max(r["me"][2] for r in rows),
        "top_reject": rej.most_common(1)[0][0] if rej else "-",
    }


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    runs = six_bot_runs(prefix)
    print("%-17s %9s %8s %8s %8s %8s  %s"
          % ("run", "instants", "clash%", "aerial%", "dj%", "aer_exec", "top reject"))
    print("-" * 78)
    acc = defaultdict(list)
    for s in runs:
        m = stats_for(s)
        if m is None:
            continue
        print("%-17s %9d %7.2f%% %7.2f%% %7.2f%% %8d  %s"
              % (s, m["instants"], m["clash"], m["aerial_high"], m["dj_high"],
                 m["aerial_exec"], m["top_reject"]))
        for k in ("clash", "aerial_high", "dj_high", "aerial_exec"):
            acc[k].append(m[k])
    print("-" * 78)
    if len(acc["clash"]) > 1:
        for k, label in (("clash", "clash%"), ("aerial_high", "aerial% (ball>500)"),
                         ("dj_high", "double_jump%"), ("aerial_exec", "aerial ticks")):
            v = acc[k]
            print("%-22s mean %7.2f   sd %6.2f   range %.2f - %.2f"
                  % (label, st.mean(v), st.pstdev(v), min(v), max(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
