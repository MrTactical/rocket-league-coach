# Rocket League Replay Coach

Reads the replays Rocket League already saves on your PC and tells you what to
work on. Runs entirely offline — no account, no API key, no uploads, nothing
injected into the game.

Leave it running while you play. Every match you save gets analysed and a
report page rewrites itself.

```
  ==============================================================
  WON 5-2   3v3   2026-08-23 20-17-28
  ==============================================================
  568 points   1 goals   3 assists   1 saves   4 shots

                                   this match   your last 12
  ----------------------------------------------------------
  heading home when conceding            +18%           +18%   same
  deepest of your team                    38%            42%   worse
  ball speed off touch              1729 uu/s      2088 uu/s   worse
  distance out when conceding         2593 uu        3233 uu   better
```

Every number is compared against **your own last 12 matches in the same
playlist**, so it reads as better or worse than usual rather than as a figure
with no context. 2v2 and 3v3 are never mixed — an even rotation share is 50%
in 2s and 33% in 3s, so comparing them would compare the format, not you.

## Running it

Download `RocketLeagueCoach.exe` from Releases, put it in a folder of its own,
and double-click. It waits for Rocket League, watches while you play, and
writes `coach-report.html` next to itself.

**Save your replays** at the end of each match. The tool reads your actual
keybind out of `TAInput.ini` and tells you what it is on startup, because the
default differs by device — `Backspace` on keyboard, `Back`/`View` on a
controller — and you may have rebound it anyway.

From source instead:

```bash
python -m venv venv
venv\Scripts\pip install psutil
venv\Scripts\python coach/watch.py --auto
```

You also need [`rrrocket.exe`](https://github.com/nickbabcock/rrrocket/releases)
in `tools/rrrocket/`. It is the replay parser — 2 MB, self-contained.

## What it measures

Fourteen modules, each one a file in `coach/metrics/`:

| module | what it tells you |
|---|---|
| `kickoffs` | which spawn, who went, time to contact, outcome at +3s |
| `touches` | power imparted, direction, contested, weak dinks vs real hits |
| `mechanics` | flips, genuine air vs *wall driving*, powerslide, standing still |
| `boost` | starved vs hoarding, big/small pads, burnt at supersonic |
| `positioning` | thirds, goal-side, and a printed pitch heatmap |
| `rotation` | first/second/third man share, cycles, double commits |
| `recovery` | do you turn for home when the ball gets behind you |
| `scoreline` | how the scoreboard changes your play |
| `fifties` | contested balls won, lost, and at what speed |
| `whiffs` | committed to the ball and missed |
| `demos` | given, taken, and what you do for 5s after a respawn |
| `indecision` | steering reversals near the ball |
| `goals` | where you stood for every goal conceded |
| `lobby` | everyone in the match, opponents included, and what to copy |

Adding a fifteenth is dropping a file in `coach/metrics/` that exposes
`TITLE`, `compute(match, who)`, `render(result)` and `tips(result, match, who)`.
Modules are discovered, not registered.

## Rank and MMR

Rocket League writes your MMR to its own log while matchmaking:

```
[0624.24] Matchmaking: Post-divide PartyLeaderMMR: 49.1883
[0626.14] Matchmaking: PartyLeaderTier=(16)
```

`49.1883 × 20 + 100 = 1083.8`, which matches the public trackers exactly, and
tier 16 is Champion I. So MMR tracking needs no API and no scraping — it is
read from a text file the game already wrote.

Three caveats, all real:

- It is the **party leader's** MMR. Solo queue, that's you.
- A reading is taken **when you queue**, so your last match's result is never
  in it yet.
- Rotated `Launch-backup-*.log` files keep their readings, so history
  accumulates. Logs predating the version that started recording MMR have none.

## Accuracy

Signals are labelled by how much they can be trusted:

- **Demos given** is exact — a replicated counter. (It is re-sent constantly
  rather than only on change; one match carried 92 events for 5 real demos, so
  only a *rise* counts.)
- **Demos taken** is estimated, roughly ±30%. Being demoed is not replicated at
  all; it is inferred from the respawn teleport.
- **Indecision** emits no advice at all. It has not been shown to predict
  anything yet, and a metric that has not earned a claim should not make one.

Percentages are weighted by real elapsed time, not frame count — replay frames
arrive irregularly and cluster around scrambles, so counting frames
over-weights whatever was happening during chaos.

## Privacy

Everything is local. The tool reads two folders:

```
Documents\My Games\Rocket League\TAGame\Demos    your replays
Documents\My Games\Rocket League\TAGame\Logs     your MMR
```

It writes `coach-report.html`, `rank.json` and a cache next to itself, and
sends nothing anywhere. Your replays contain the in-game names of everyone in
your lobbies, so the generated files are gitignored — don't commit them.

## Also in this repo

`bot/` is a separate project — an [RLBot](https://rlbot.org) v5 bot that plays
Rocket League offline against itself. It shares nothing with the coach except
the repo and some physics constants. `tests/` and `tools/` mostly belong to it.
If you only want the coach, `coach/` and `coach-watch.bat` are self-contained.

## Credits

Replay parsing by [rrrocket](https://github.com/nickbabcock/rrrocket) and
[boxcars](https://github.com/nickbabcock/boxcars).
