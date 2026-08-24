# Why the bots never commit to leaving the ground

Dated 2026-08-23. Five gates investigated in parallel, each finding then
attacked by a skeptic. 13 agents, ~500 tool calls.

## Cleared: two plausible causes that are provably innocent

### The `len(found) >= 4` early break -- NOT a cause

This was the prime suspect from reading the code. There are five mechanics and
the loop breaks at four, so in principle it can exit before the aerial band is
searched. It does not.

A/B over 88,171 real `find_intercept` calls (states replayed from telemetry,
ball predictions from RocketSim with the real meshes), break removed:

  * chosen kind changed: **0 (0.0000%)**
  * aerial candidate gained that was missing: **0**
  * extra `aerial_feasible` calls unlocked corpus-wide: 25, none feasible

The premise was backwards. The aerial band is reached FIRST (median loop
iteration 0), not last -- ground and dodge are found around iteration 76-78.
The break can only be aerial-blind if a WALL candidate exists, which requires
|up.z| < 0.866, true on 3.41% of ticks; a WALL candidate is actually recorded
on 0.11% of calls.

### The double_jump band stealing aerials -- NOT a cause

In `_arbitrate` the aerial clause is first and compares against `dodge or
ground`. double_jump is not in the reference set, so a found+affordable aerial
can never be outranked by one. Exhaustive sweep of 71,680 synthetic candidate
sets: zero cases where double_jump beat a passing aerial.

Counterfactual, 4,000 real high-ball states, double_jump band deleted outright:
aerial count **163 -> 163, change of exactly zero**. The 1,325 freed decisions
went 98.9% to dodge. The band competes with the dodge, not the aerial.

(The band's "wait for it to land" failure is real, but the aerial is not its
victim -- see the full agent output.)

## Confirmed cause 1: the rotation system discards aerials wholesale

`bot/brain/decide.py:305`. Survived both refutation lenses.

**When the solver DOES return an aerial, the brain throws it away 88.4% of the
time.** Of 2,611 ticks with `k=aerial`, only 302 were in a state that can even
call `Strike.step`. The rest died before the executor:

| where it died | ticks | share |
|---|---|---|
| `_do_support`, decide.py:598 `self.strike = None` | 1,129 | 43.2% |
| `_do_defend` non-imminent, decide.py:582 `self.strike = None` | 923 | 35.4% |
| recovery | 176 | 6.7% |
| humanizer lapse | 37 | 1.4% |
| everything else | 44 | 1.7% |

The mechanism is the ordinal, not the humanizer. With `len(state.teammates) >=
2` -- true in every 3v3 -- the ordinal drives the role outright: ord==0 forced
to ATTACK, everyone else forced to SUPPORT/DEFEND, neither of which ever builds
a Strike.

    P(reach Strike.step | k=aerial):
      ord=0 :  47.44%
      ord=1 :   0.16%
      ord=2 :   0.54%

**76.1% of all aerial proposals are found by a car structurally forbidden from
touching the ball.** Team-level: across 33 six-bot matches there were 2,340
instants where at least one car proposed an aerial; the team acted on it in
300 (12.8%).

This is an over-correction from earlier work. The ordinal was added to fix "no
rotation, driving next to each other" and it did -- slot clashes went from
17-26% to 3.6-8.6%. But it now forbids anyone but first man from ever striking,
including when an aerial is the only right answer.

## Confirmed cause 2: the aerial is charged for the drive

`bot/core/physics.py:191`. Survived 1 of 2 lenses; the surviving objection
sharpened it rather than overturning it.

Unlike GROUND, DODGE and DOUBLE_JUMP -- which all subtract a `_drivable` drive
segment first -- the aerial branch calls `aerial_feasible(car, target, dt)`
from the car's CURRENT position with no drive-up phase. It therefore prices the
ENTIRE displacement as powered flight.

Consequence: median priced cost is **47.4 boost** and it is barely
distance-sensitive (40.8 under 500uu horizontal, only 58.2 beyond 4000uu).
With the 12-boost reserve a car needs ~59 boost to buy the median aerial, and
only 33.8% of high-ball moments have that.

Replay over 107,342 band ticks:

| stage | surviving | share |
|---|---|---|
| kinematically reachable | 40,763 | 37.97% |
| ...that pass the silent cost gate (physics.py:191) | 16,441 | 15.32% |
| ...that pass the arbiter reserve | 12,426 | 11.58% |

**69.5% of every physically flyable aerial is destroyed by boost pricing**, and
59.7% of that is the silent gate. The off-axis case is a total blind spot: a
car at 1200 uu/s whose velocity is 45deg, 90deg or 150deg off the ball line is
called infeasible at EVERY T from 0.5 to 4.0s, where a drive-then-fly model
succeeds in all three.

`AERIAL_MIN_BOOST = 25` is measurably inert -- sweeping it 0/10/20/25 changes
the outcome by 5 ticks in 12,431 (0.04%). `AERIAL_BOOST_RESERVE = 12` costs
23.4%.

## Correction to an earlier claim of mine

I twice said "aerials are never declined, only 9-58 ticks, so they are never
found". That reasoning was built on a misread. `adecl` is set at
`bot/brain/strike.py:389`, reachable only inside `if ic.kind == AERIAL:` --
i.e. AFTER the arbiter has already chosen an aerial. It records a humanizer
decline of an already-chosen aerial, not a boost rejection. All 66 adecl ticks
have `k == "aerial"` and a median boost of 88.

There is zero instrumentation on any boost rejection: `grep -rn "debug\."
bot/core/intercept.py bot/core/physics.py` returns nothing. The boost gates
fail invisibly, which is why this took so long to find.

## Order of work

1. Ordinal gating (decide.py:305, 582, 598) -- biggest, and it is a policy
   change not a physics change.
2. Drive-then-fly pricing (intercept.py aerial branch + physics.py:191).
3. Instrument the silent rejections, and rename `adecl` so it stops reading as
   exoneration.
4. `AERIAL_MIN_BOOST` is inert -- remove or lower it so it stops looking like a
   live constraint. Reserve 12 -> 4 is worth +20.4% candidates.

---

# What was implemented, 2026-08-23

Constraint from the user: fix it, **but keep the clash rate low**. That ruled
out the obvious fix.

## 1. Aerial pre-emption of the attack slot (not "break rotation")

The tempting fix is to let a non-first-man ignore its role and go for a high
ball. That reintroduces exactly the pack behaviour the ordinal was built to
kill, and the clash rate is the thing it bought.

Instead the aerial competes for the **attack slot itself**, through the same
ranking that already guarantees exactly one car holds slot 0. Nobody strikes
who is not first man. We only change who first man is.

`AERIAL_BONUS` in `bot/brain/decide.py` works exactly like `INCUMBENT_BONUS`:
seconds of advantage in the time-to-ball ranking. It is **broadcast** over
comms (`"a"` in the intent message, `PeerIntent.aerial`) so every car applies
it to the same car. A self-applied bonus is what once made all three bots
believe they were first man, and that fault must not be reintroduced.

Sized from measurement, not taste. Across 6,064 three-car team-instants the car
proposing an aerial is usually ALREADY the fastest -- gap behind the team's
best time-to-ball is 0.00s at p75 and +0.12s at p90. It is not behind, it
simply is not the incumbent. So the bonus only has to beat incumbency:

| bonus | aerial proposals converted (of 164 not already on slot 0) |
|---|---|
| 0.45 | 90.9% |
| **0.55** | **95.1%** |
| 0.65 | 96.3% |
| 0.75+ | 97.6%, flat |

0.55 takes the knee, and lets a car within (0.55 - 0.30) = 0.25s of the team's
fastest take the slot. Tight enough that an out-of-position car still cannot.

## 2. Drive-then-fly pricing

`_aerial_after_drive` in `bot/core/intercept.py`. GROUND, DODGE and DOUBLE_JUMP
all subtract a drive segment before their mechanic; the aerial branch did not,
so it priced the entire displacement as powered flight.

The ground phase runs on a **no-boost** `ReachCurve` -- boost spent driving is
boost unavailable for the climb, and the point is to arrive under the ball with
a full tank, which is what a player does. `ReachCurve.speed_at()` was added to
get the launch velocity from the same table `distance_at` reads.

`DRIVE_FRACTIONS` includes 1.0, which IS the old model, so the new search is a
strict superset and can never find less. That invariant is a test
(`tests/test_aerial_supply.py::test_drive_then_fly_is_a_strict_superset`).

Measured over a 10,800-cell envelope: feasible rate 13.86% -> 22.15% (1.60x).
The audit's headline blind spot is largely gone -- a car at 1200 uu/s angled
45deg or 150deg off the ball line to a z=900 ball 2000uu away went from
infeasible at EVERY window to feasible. 90deg off is still infeasible.

## 3. Instrumentation, so this is never guessed at again

The aerial gates failed completely silently: an unaffordable aerial returned
feasible=False and was never recorded, so nothing downstream could distinguish
"too expensive" from "out of reach" from "never searched".

`Intercept.aerial_reject` / `.aerial_cost` now carry the reason, logged as
`arej` / `acost`: `no_lead`, `unreachable`, `min_boost`, `reserve`, or
`slower_than_<kind>`. `adecl` keeps its name for continuity with older traces
but now carries a comment saying plainly what it actually means, since reading
it as exoneration is what cost the detour.

## 4. Launcher bug found along the way

`run.py` waited only on `MatchPhase.Ended`. A five minute match played its full
364s, wrote complete telemetry, and the launcher then span for another eight
minutes because the phase never flipped. It now also gives up when the match
clock stops advancing for 20s.

## Not done

`AERIAL_MIN_BOOST = 25` is measurably inert (5 ticks in 12,431) and
`AERIAL_BOOST_RESERVE = 12 -> 4` is worth +20.4% candidates. Both left alone
deliberately: two behavioural changes are already in this run, and shipping a
third would make the measurement unattributable.

---

# Open, paused 2026-08-23

Both fixes are in and the full suite is green (103 tests, 15 files). One
verification match ran. The unresolved question is the clash rate.

    metric                      before    after      baseline range (unmodified)
    slot clashes                 3.25%    4.46%      2.49% - 5.30%, mean 3.46
    aerial chosen, ball >500uu   2.95%    6.64%
    double_jump, ball >500uu    40.42%   29.24%
    aerial ticks executed           12       18

4.46% is 1.21 sd above the unmodified mean and BELOW the worst the unmodified
code produced on its own, so a single match cannot separate it from noise.

The split says the change is probably not the cause: near an aerial, clash FELL
4.72% -> 0.82%. The rise is entirely in instants with no aerial involved, where
the change should be a no-op.

TO RESUME: run three or more matches and compare distributions, not points.

    ./venv/Scripts/python.exe run.py 3v3-measure     # repeat 3-4 times
    ./venv/Scripts/python.exe tools/run_stats.py 20260823
    ./venv/Scripts/python.exe tools/compare_runs.py <before-stamp> <after-stamp>

If the new mean lands inside 2.49-5.30 it is clean. If above, the lever is
making the aerial claim stickier -- raise AERIAL_CLAIM_HOLD (currently 0.6) or
require the claim to persist before it counts. Every claim toggle re-ranks the
team, and re-ranking is where clashes happen.

Also unresolved and probably worth a look first: ball height p90 fell 739 ->
644 and time above 500uu fell 17.9% -> 14.6%. That may be good (balls being
intercepted in the air rather than arcing) or bad, but it shrinks the sample
the aerial metrics are computed over, so it affects how much the numbers above
can be trusted.

Deliberately NOT shipped, so the next measurement stays attributable:
AERIAL_MIN_BOOST is inert, and AERIAL_BOOST_RESERVE 12 -> 4 is worth +20.4%
candidates.
