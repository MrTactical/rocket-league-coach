# Rocket League Coach — working notes

Two projects share this repo. `coach/` is the live one: it reads the replays
Rocket League saves locally and writes a coaching report. `bot/` is an RLBot
teammate ("Ally") used offline as a practice partner, now driven by profiles
measured out of `coach/`.

`README.md` is for users. This file is for whoever edits the code, and it is
mostly a list of things that cost real time to find out.

## Out of scope, permanently

Anything that puts a bot or an overlay into an **online** match. No EAC
bypass, no memory reading, no synthetic input driving a ranked game, no live
in-match overlay. This has been asked several ways — including "build our own
instead of RLBot", which *is* the bypass, since RLBot is blocked online
precisely because EAC blocks it. Do not re-litigate it by implementing "just
the reading part".

Two arguments that come up and do not change the answer: that opponents cheat
too (retaliation lands on strangers, not on the smurfs from earlier matches),
and that the bot fills only one slot (your friend can consent for himself;
neither of you can consent for the three opponents whose MMR is the thing
being taken).

Offline, LAN, Exhibition and post-match replay analysis are all fine — that is
what everything here is. Steam's own "Play with Anti-Cheat Disabled" launch
option is Psyonix-supported and unrelated to the above.

## Run it

```bash
venv/Scripts/python.exe coach/watch.py --auto     # or: coach-watch.bat
venv/Scripts/python.exe coach/watch.py --once     # sweep and exit
venv/Scripts/python.exe coach/watch.py --retrack  # rebuild only the viewer track
venv/Scripts/python.exe coach/watch.py --set-mmr 994 --set-rank "Diamond III Div I"
venv/Scripts/python.exe coach/watch.py --auto --refresh 30   # page self-reloads

venv/Scripts/python.exe coach/stack.py --with TEAMMATE    # you as a pair
venv/Scripts/python.exe coach/profile_export.py --player MrTactical \
       --team-size 3 --out bot/profile-me.json                # coach -> bot

ALLY_PROFILE=bot/profile-me.json venv/Scripts/python.exe run.py 3v3-measure
```

Tests are plain asserts, no framework:

```bash
for f in tests/test_*.py; do venv/Scripts/python.exe "$f"; done
```

## The traps

**Editing `coach/metrics/*.py` or `coach/timeline.py` wipes the analysis
cache.** `code_version()` in `watch.py` fingerprints exactly those files, so a
metric change re-measures all ~330 replays (~13 min). `page.py` and `watch.py`
are deliberately *not* fingerprinted — editing those only needs a rebuild.
Batch metric edits together; one re-measure, not four.

**`rebuild()` on a freshly-invalidated cache writes an empty page.** Edit a
metric, call `rebuild()` directly, and `load_cache()` returns zero entries:
"No matches found". Run a sweep first. The sweep rebuilds every 20 replays so
the page is never blank for the whole 13 minutes — but that also means a page
read mid-sweep shows a partial, oddly-ordered set. The final rebuild fixes it.

**A running watcher does not see your edits.** It exits with code 3 when a
fingerprinted file changes and `coach-watch.bat` loops to restart it. Changes
to `watch.py` or `page.py` need a manual restart — this has caused several
"it's not updating" reports where old code was running faithfully. `os.execv`
was tried for self-restart and does not survive the venv launcher shim on
Windows; the exit-code-plus-bat-loop is the version that works.

**Kill the watcher before starting another.** Repeated `Start-Process` on the
bat leaves a pile of console windows.

**The report does not auto-reload.** It used to, on an idle timer — but
*reading* is exactly the idle state, so it reloaded out from under you while
you were using it. A `file://` page cannot detect a new version without
reloading (CORS blocks fetching itself), so the honest default is off. Press
F5, or pass `--refresh N`.

**The watcher runs at BelowNormal priority.** It parses replays while the game
is running, which is the point, and a full re-measure is hundreds of replays
back to back. Note the ctypes detail: `GetCurrentProcess` needs
`restype = c_void_p` or the 64-bit pseudo-handle truncates and
`SetPriorityClass` silently fails, returning 0.

**Console output is cp1252 on Windows.** Player names carry emoji and
non-Latin digits; printing one raises `UnicodeEncodeError` and kills the run.
Every entry point reconfigures stdout/stderr to utf-8 with `errors="replace"`.

## Replay format, the hard-won bits

**Match time comes from `MatchStartEpoch`, never the `Date` property.** Saving
a replay from the in-game replay list writes a file whose `Date` is not when
the match was played. Six matches saved after a session came out ordered
15:30, 15:39, 15:47, 14:55, 15:01, 15:08 — so "your last match" showed a game
40 minutes earlier with the wrong lobby in it, and the ordering silently
affected the recent-8, then-vs-now and last-12 baselines too. The epoch
reproduces the in-game match history exactly, results and order. Do not
reintroduce a `Date` sort.

**Rocket League no longer logs MMR.** `mmrlog.py` looks for
`Matchmaking: Post-divide PartyLeaderMMR`. Current logs contain zero
occurrences — Psyonix removed it. Rank is hand-entered in `coach/rank.json`
via `--set-mmr` / `--set-rank`, which stamps a date so the page can admit its
age. Ranks are not in replay files at all. Ballchasing exposes per-player rank
but only as fresh as your last upload, which was staler than the hand-entered
value when checked.

**Parser is rrrocket 0.11.5** (`tools/rrrocket/`). rattletrap 14.1.4 cannot
read current-season replays.

**Two spellings of boost across eras.** 2023 replays use
`ReplicatedBoostAmount` as a bare Byte; current ones use `ReplicatedBoost`
wrapping a struct. Handling only the modern one reports every older match as
zero boost, which looks like a player who never picked up a pad.

**Component actors are separate from the car.** Boost, jump and dodge are
their own actors, linked by `TAGame.CarComponent_TA:Vehicle`. Camera settings
live on `CameraSettingsActor_TA` and link to the player through its own `PRI`
attribute — keying them by the car actor matches nothing at all, and ball-cam
read 0% of every frame.

**`bUsingSecondaryCamera` true means ball cam ON.** Reading it inverted put
every player in a Champion lobby at 5-19% ball cam. Real figures are 80-96%,
and the implausible number is what gave the sign away.

**Counters are re-sent, not edge-triggered.** `MatchDemolishes`,
`DodgesRefreshedCounter` and friends replicate constantly; only a *rise* is an
event. The same applies to anything proximity-based: bumps counted per-frame
reported 45 a match, against 16 counted on entry.

**Frames are irregular, ~10-30 Hz.** Weight everything by elapsed time, not
frame count. Anything defined by car attitude at the instant of a touch
(musty flicks, stalls, tornado spins) is not detectable at this rate — say so
rather than shipping a guess with a number attached.

**Names change and split.** `MrTactical` and `MrTactical ^-^` are one person;
so are `Mate` and `Mate(Chat off)`. `timeline.canon` strips
non-alphanumerics and merges the first pair but not the second, so
`profile_export._key` removes parenthesised tags first. A name that is *only*
punctuation (`*******`, `.`, `:)`) canons to the empty string — those must
stay distinct or every censored name merges into one fictional player.

## Field constants

Gravity −650. Max speed 2300; 1410 without boost. Field ±4096 x, ±5120 y,
ceiling 2044. Goal mouth 1786 × 642 (half-width 893). Ball radius 92.75; a car
sits ~17 above the floor. `(0, 0)` is a **real position** — the ball spawns
there and 50/50s happen there, so it can never be an "absent" sentinel. Using
zeros for that erased cars from the viewer for a whole release; absent is
`null`.

## Metric modules

Each file in `coach/metrics/` exposes `TITLE`, `compute(match, who)`,
`render(result)`, `tips(result, match, who)`. They are discovered
automatically. `compute` returns a dict that gets cached, so anything the page
needs must be *in that dict* — `page.py` renders from the cache, not by
calling modules.

`page.py` hand-picks which sections it renders. **Adding a module does not
make it appear**; wire it in explicitly. `techniques.py` and `shooting.py` both
computed correctly for a while without being visible anywhere.

Season-wide aggregates belong on the overview tab; the lobby tab is about the
six people in one match. Putting the kickoff-cheat and named-mechanics
sections at the top of the lobby tab pushed the actual lobby below the fold
and read as a broken page.

**Per-match tips cannot see season-scale effects.** A match holds about seven
kickoffs, so a threshold strong enough to trust is unreachable inside one. The
cheat comparison lives in `page.py` where it has hundreds.

Beyond the metrics, `coach/` also has `possession.py` (phase from `hit_team`),
`shadow.py` (the "be here" marker), `stack.py` (you and a regular team-mate as
a unit), `pro.py` (ballchasing) and `profile_export.py` (coach → bot).

## The standard this project holds itself to

Findings are measured, not asserted, and they have to survive a test before
they become advice:

1. **Calibrate thresholds from measured distributions**, not taste. Guessed
   constants have repeatedly produced labels that fired for every player in a
   lobby, or for none: "plays the air" at 14% when the median is 12.4, "lives
   forward" at 5200 uu when the 90th percentile is 4627, "caught upfield" at
   9% when the median is 10.9. Dump the axis across a few hundred
   player-observations and cut at a percentile.
2. **A metric must separate outcomes.** If a number cannot tell a held attack
   from a conceded one, or a won match from a lost one, it does not get to
   give advice. Two candidate clash checks were measured; "both ahead of the
   ball" separated wins from losses by 5.1 points and shipped, "bunched within
   1500 uu" by 2.0 and did not.
3. **Re-test on a bigger sample, and on an independent population.** The
   shadow *depth* finding looked solid at 24 matches (0.10 separation) and
   collapsed at 100 (0.04). It is retracted in the page, in those words. The
   possession split was confirmed on 683k frames of downloaded Grand Champion
   replays before it was believed.
4. **Set the bar before seeing the answer, and let it rule against you.** The
   kickoff-cheat finding misses its own two-standard-error gate by 0.09 of a
   point (45.32% over n=523 against 50.77% over n=849, gap 5.45, threshold
   5.54). The page reports both rates and says the difference is inside the
   noise. Moving the bar after seeing the number is not an option; replacing a
   flat threshold with a sample-size-aware one *before* looking is.
5. **Say what is not known, and do not build what cannot be measured.**
   Passing accuracy is defined by intent and a replay records only outcome, so
   `shooting.py` reports *possession retained* and says plainly it is not
   passing accuracy. Reaction time and aim error are excluded from measured
   bot profiles for the same reason.

The flagship finding: being ahead of the ball is **0.41×** the base concede
risk during your own attack and **2.78×** while the opponent has it in your
half (0.33× and 2.93× on the Grand Champion population). Pooled, that is a
meaningless "1.7× fault" that criticised the attack and the mistake in
identical words. `coach/possession.py` exists to keep those apart.

## The bot

Ally is a practice partner, driven either by a rank preset or by a profile
measured from real play (`coach/profile_export.py` → `profile_file` in
`config/ally.toml`, or `ALLY_PROFILE`).

**What a replay can and cannot set.** Speed, boost discipline, air time, ball
share, how often someone is caught ahead of the ball, wave-dash rate,
speed-flip rate, whiffs per committed touch — all measured. Reaction time and
aim error are not: separating "aimed there" from "aimed elsewhere and missed"
needs intent, which is not in the file. Those come from the rank band.

**Wiring a field into the profile is not the same as using it.** Rotation data
was written to JSON, loaded into `SkillProfile.measured`, and read by nothing
but a log line — so the bot missed shots like the player and went nowhere like
them. Check that something downstream actually consumes a new field;
`tests/test_measured_bot.py` exists to fail when one goes decorative.

**Log through the bot's own logger.** A load failure was visible only because
warnings pass a default root logger, while the success logged at INFO through
a bare module logger and vanished — which made "did the measured profile
actually load" unanswerable from the match output.

**The skill preset is not the lever for aerials.** A champion preset flew
*less* than a profile measured from a Diamond player. See
`notes/why-the-bots-never-fly.md`: the ordinal discards 88.4% of aerial
proposals before the skill layer is ever consulted, and that is still the
largest open item.

## Privacy and distribution

The exe is built for friends, so:

- **Never commit `coach/ballchasing.key`.** Gitignored, along with `data/`,
  `rank.json`, `.replay-cache.json`, `coach-report.html`, `*.replay` and
  `coach/.pro-replays/` — these hold MMR, match history, and the in-game names
  of everyone in the user's lobbies.
- **Do not parse `UniqueId`, `EpicPUID` or `PlayerID`.** They are stable
  platform account identifiers for other people, useless for coaching, and
  this tool gets distributed. Everything else in the network stream is fair
  game.
- **Player names are untrusted input.** They come from other people's Steam
  profiles and land in an inline `<script>` and two `innerHTML` sinks. A name
  containing `</script>` blanks the entire replay tab and injects markup.
  Escaping is at the sink in `page.py`; the build-time `esc()` is undone by
  reading values back through `dataset`.
- **The report's reference figures are not the reader's own.** Anything
  phrased as "your 30 matches" is false for a friend running the build.
  Calibration percentiles are Champion-rank; label them as such.

Verify a build before shipping: scan `dist/RocketLeagueCoach.exe` for the key,
in-game names, home paths and rank data. The spec has `datas=[]` precisely so
nothing personal is bundled. Confirmed clean at 9.3 MB.

## Working style that fits this repo

Comments explain *why*, especially where a previous approach was wrong and
what its symptom looked like — most comments here are load-bearing bug
history, and several were written specifically so the same wrong turn is not
taken twice. Prefer measuring over guessing. Delete a finding that does not
hold rather than softening it. When a number looks implausible, that is the
signal: 5% ball cam, 45 bumps a match, a shot missing by 2023 uu, and 94%
turnover were all found by noticing the value could not be true.
