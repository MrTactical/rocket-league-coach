"""
Turn measured play into a bot skill profile.

The coach measures how someone actually plays across every replay it has. The
bot has a SkillProfile with a handful of rank presets. This is the bridge: it
reads the coach's cache, builds a behavioural fingerprint for one player, and
writes a profile the bot loads instead of a generic rank label.

WHY THIS EXISTS. "diamond" as a preset is an average of everyone at that rank.
Nobody plays like the average. A practice partner that rotates like your actual
team-mates is worth more than one that rotates like a statistical composite,
because the habits you build against it transfer.

WHAT IS DERIVED AND WHAT IS NOT. Reaction time and aim error are not
observable in a replay -- there is no ground truth for what the player
intended, so a touch that went 8 degrees off cannot be separated from a touch
that was aimed there. Those stay tied to the rank band. Everything the replay
does show -- speed, boost discipline, air time, how often they are first to
the ball, how often they are caught ahead of it -- is measured and written
through.

    python coach/profile_export.py --player MrTactical
    python coach/profile_export.py --player TEAMMATE --out bot/profile-mate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("profile_export")


def _key(name):
    """
    Canonical form for matching one human across name changes.

    coach.timeline.canon strips non-alphanumerics, which already merges
    "MrTactical" with "MrTactical ^-^". It does not merge
    "Mate(Chat off)" with "Mate", because the suffix is
    alphanumeric -- so a parenthesised tag is removed first. Without this a
    player's history splits across two entries and each half looks like a
    different, thinner sample.
    """
    import re
    base = re.sub(r"\(.*?\)", "", name or "")
    key = "".join(c for c in base.lower() if c.isalnum())
    # A name that is entirely punctuation -- censored names show as "*******",
    # ".", ":)" -- canons to the empty string, which would merge every one of
    # them into a single fictional player. Keep those distinct.
    return key or ("raw:" + (name or ""))

# Rank bands by MMR, for the axes a replay cannot show. Boundaries are the
# published Season 23 standard-playlist cutoffs.
MMR_BANDS = (
    (0, "bronze"), (400, "silver"), (555, "gold"), (715, "platinum"),
    (875, "diamond"), (1035, "champion"), (1275, "grand_champion"),
    (1535, "ssl"),
)

# Population medians over 1008 player-observations from 168 Champion 3v3
# lobbies. Used to express a player as a RATIO to their peers rather than an
# absolute, so the profile still means something at another rank.
POP = {
    "speed": 1387.9, "supersonic": 10.3, "airborne": 12.4, "powerslide": 3.9,
    "boost_held": 54.7, "starved": 20.2, "first_man_rel": 1.0,
    "exposed": 10.9, "committed": 7.0,
}


def band_for_mmr(mmr):
    """The rank band an MMR falls in."""
    name = MMR_BANDS[0][1]
    for floor, label in MMR_BANDS:
        if mmr is None or mmr < floor:
            break
        name = label
    return name


def _median(xs):
    return statistics.median(xs) if xs else None


def fingerprint(cache, player, team_size=None, limit=None):
    """
    Median of every measured axis for one player, across their matches.

    Median, not mean: one 40-second forfeit or one blowout should not drag an
    axis. Returns None if the player never appears with enough data.
    """
    axes = ("speed", "supersonic", "airborne", "powerslide", "boost_held",
            "starved", "first_man", "first_man_rel", "up_pitch", "exposed",
            "committed")
    got = {a: [] for a in axes}
    tq = {}
    extra = {}
    matches = 0

    entries = list(cache.get("entries", {}).values())
    entries.sort(key=lambda v: v.get("date") or "")
    if limit:
        entries = entries[-limit:]

    for e in entries:
        secs = e.get("sections") or {}
        if team_size and e.get("team_size") != team_size:
            continue
        lob = (secs.get("lobby") or {}).get("data") or {}
        players = lob.get("players") or {}
        pl = players.get(player)
        if pl is None:
            want = _key(player)
            pl = next((v for k, v in players.items() if _key(k) == want), None)
        if not pl:
            continue
        matches += 1
        for a in axes:
            v = pl.get(a)
            if v is not None:
                got[a].append(float(v))
        # These sections only describe the replay's OWN player, so they are
        # available for you and not for a team-mate seen from your replays.
        if _key(e.get("resolved_name") or "") == _key(player):
            td = (secs.get("techniques") or {}).get("data") or {}
            live = td.get("live") or 0.0
            if live > 60:
                for k, v in (td.get("counts") or {}).items():
                    tq.setdefault(k, []).append(v * 600.0 / live)

            ko = (secs.get("kickoffs") or {}).get("data") or {}
            if (ko.get("flip_mine") or 0) >= 2:
                extra.setdefault("speedflip", []).append(
                    (ko.get("early_flips") or 0) / float(ko["flip_mine"]))

            wf = (secs.get("whiffs") or {}).get("data") or {}
            if (wf.get("commits") or 0) >= 8:
                extra.setdefault("whiff", []).append(
                    (wf.get("whiffs") or 0) / float(wf["commits"]))

            # Pressure: flip rate with an opponent close versus far. A player
            # whose rate collapses under pressure is a different practice
            # partner from one whose does not. Weak signal -- the indecision
            # module is not validated for advice -- but it only shapes a bot
            # here, it does not tell the player anything.
            ind = (secs.get("indecision") or {}).get("data") or {}
            ns, fs = ind.get("near_secs") or 0.0, ind.get("far_secs") or 0.0
            if ns > 20 and fs > 20:
                near = (ind.get("near_flips") or 0) / ns
                far = (ind.get("far_flips") or 0) / fs
                if far > 0:
                    extra.setdefault("pressure", []).append(near / far)

    if matches < 3:
        return None
    out = {a: _median(v) for a, v in got.items()}
    out["matches"] = matches
    out["per10"] = {k: _median(v) for k, v in tq.items()}
    out["extra"] = {k: _median(v) for k, v in extra.items()}
    return out


def to_skill_profile(fp, mmr=None, name="measured"):
    """
    A SkillProfile dict the bot can load, plus the rotation tendencies.

    Ratios against the population median drive each axis, so a player who is
    20% below median on air time gets a bot 20% less willing to go up,
    regardless of which rank band they sit in.
    """
    band = band_for_mmr(mmr)

    def ratio(key, default=1.0):
        pop = POP.get(key)
        val = fp.get(key)
        if not pop or val is None:
            return default
        return val / pop

    air = ratio("airborne")
    # Aerial confidence tracks air time directly. Clamped: a bot that never
    # leaves the floor is not a useful partner even for a player who does not.
    aerial_conf = max(0.20, min(0.90, 0.55 * air))

    # Whiffs, measured directly where the data exists: whiffs per committed
    # touch. The earlier version inferred this from air time, which is a proxy
    # for how HARD the touches were rather than how often they were missed.
    ex = fp.get("extra") or {}
    if ex.get("whiff") is not None:
        whiff = max(0.02, min(0.35, ex["whiff"]))
    else:
        whiff = max(0.03, min(0.20, 0.07 * (0.6 + 0.4 * air)))

    # Steering noise from powerslide use: a player who never slides is
    # turning in wide arcs, which the bot should imitate rather than correct.
    slide = ratio("powerslide")
    noise = max(0.02, min(0.09, 0.035 * (1.6 - 0.6 * slide)))

    # Wave dashes are measured directly. The population figure is this
    # player's own lobbies, so the ratio says "relative to people at my rank"
    # rather than to an absolute nobody agrees on.
    per10 = fp.get("per10") or {}
    wd = per10.get("wave_dash")
    wavedash = None if wd is None else max(0.05, min(0.95, wd / 22.0))

    # Speed-flip skill from how early the first kickoff flip comes. Not
    # directly in the fingerprint, so it falls back to the band default rather
    # than being invented.
    return {
        "name": name,
        "derived_from": {
            "matches": fp["matches"],
            "mmr": mmr,
            "band": band,
            "note": "reaction_time and aim_error come from the rank band -- "
                    "a replay cannot show what a player intended, so those "
                    "two are not measurable from it.",
        },
        "skill": {
            "name": name,
            "band": band,
            "aerial_confidence": round(aerial_conf, 3),
            "whiff_chance": round(whiff, 3),
            "input_noise": round(noise, 3),
            **({"wavedash_rate": round(wavedash, 3)}
               if wavedash is not None else {}),
            # Share of this player's own kickoff flips that came inside the
            # speed-flip window.
            **({"speedflip_skill": round(max(0.02, min(0.98, ex["speedflip"])), 3)}
               if ex.get("speedflip") is not None else {}),
            # 1.0 means the flip rate is unchanged under pressure. Mapped so
            # a bigger deviation either way means more sensitivity.
            **({"pressure_sensitivity":
                round(max(0.1, min(1.2, abs(1.0 - ex["pressure"]) * 1.6)), 3)}
               if ex.get("pressure") is not None else {}),
        },
        # How the bot should ROTATE, which is the half that actually decides
        # whether practice against it transfers.
        "rotation": {
            "first_man_share": round(fp.get("first_man") or 0.0, 1),
            "first_man_rel": round(fp.get("first_man_rel") or 1.0, 3),
            "up_pitch": round(fp.get("up_pitch") or 0.0, 0),
            "exposed_pct": round(fp.get("exposed") or 0.0, 2),
            "committed_pct": round(fp.get("committed") or 0.0, 2),
        },
        "movement": {
            "avg_speed": round(fp.get("speed") or 0.0, 0),
            "supersonic_pct": round(fp.get("supersonic") or 0.0, 2),
            "airborne_pct": round(fp.get("airborne") or 0.0, 2),
            "powerslide_pct": round(fp.get("powerslide") or 0.0, 2),
            "boost_held": round(fp.get("boost_held") or 0.0, 1),
            "starved_pct": round(fp.get("starved") or 0.0, 2),
        },
        "mechanics_per_10min": {k: round(v, 2)
                                for k, v in sorted((fp.get("per10") or {}).items())
                                if v},
    }


def load_cache(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOG.error("no cache at %s -- run coach/watch.py --once first", path)
    except json.JSONDecodeError as exc:
        LOG.error("cache at %s is not valid JSON: %s", path, exc)
    return None


def current_mmr(rank_path):
    try:
        d = json.loads(Path(rank_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for pl in d.get("playlists") or []:
        if pl.get("main"):
            return pl.get("mmr")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Derive a bot skill profile from measured play.")
    ap.add_argument("--player", help="in-game name (default: the cached one)")
    ap.add_argument("--cache", default=str(ROOT / "coach" / ".replay-cache.json"))
    ap.add_argument("--rank-file", default=str(ROOT / "coach" / "rank.json"))
    ap.add_argument("--team-size", type=int, choices=(1, 2, 3),
                    help="only use matches of this size")
    ap.add_argument("--limit", type=int, help="only the most recent N matches")
    ap.add_argument("--mmr", type=int, help="override the MMR used for banding")
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    ap.add_argument("--list", action="store_true",
                    help="list players seen in the cache and exit")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    # Player names carry emoji and non-Latin digits, and the Windows console
    # is cp1252 by default -- printing one raises UnicodeEncodeError and kills
    # the run. Same fix the analyser already uses.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    cache = load_cache(args.cache)
    if cache is None:
        return 2

    if args.list:
        seen, labels = {}, {}
        for e in cache.get("entries", {}).values():
            lob = ((e.get("sections") or {}).get("lobby") or {}).get("data") or {}
            for nm in lob.get("players") or {}:
                k = _key(nm)
                seen[k] = seen.get(k, 0) + 1
                labels.setdefault(k, set()).add(nm)
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1])[:40]:
            names = sorted(labels[k])
            extra = ("  [also: %s]" % ", ".join(names[1:])) if len(names) > 1 else ""
            print("%5d  %s%s" % (n, names[0], extra))
        return 0

    player = args.player or cache.get("player")
    if not player:
        LOG.error("no player given and none cached -- pass --player")
        return 2

    fp = fingerprint(cache, player, args.team_size, args.limit)
    if fp is None:
        LOG.error("not enough data for %r (need 3+ matches). "
                  "Try --list to see who is in the cache.", player)
        return 1

    mmr = args.mmr if args.mmr is not None else current_mmr(args.rank_file)
    prof = to_skill_profile(fp, mmr, name=player)
    text = json.dumps(prof, indent=2)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        LOG.info("wrote %s (%d matches, band %s)",
                 args.out, fp["matches"], prof["skill"]["band"])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
