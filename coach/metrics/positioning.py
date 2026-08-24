"""
Where you actually stand on the pitch.

Everything here is measured in the player's own frame of reference: y is
multiplied by `match.attack_sign(who)` so "defensive" always means their own
half and "up-pitch" always means toward the goal they are shooting at. x is
flipped with it, because rotating the field 180 degrees about z sends both x
and y to their negatives -- without that, "left" would swap sides depending on
which team you happened to be on.

Two things this module is fussy about, because both quietly wreck a heatmap:

  * TIME, NOT FRAMES. Samples arrive at roughly 30 Hz but the spacing is
    irregular and there are multi-second holes where the replay skipped a goal
    celebration. Every share below is a share of elapsed seconds, with each
    sample's weight capped so one 9-second hole cannot become 9 seconds of
    "standing in the corner".
  * LIVE PLAY ONLY. Nearly a third of a replay's wall time is dead: goal
    replays, celebrations and the kickoff countdown, during which cars sit
    still in fixed spots. Counting it puts a fat fake hotspot on the kickoff
    positions. Dead time is detected from the match clock -- if
    SecondsRemaining has not moved for over a second the game is paused --
    plus an explicit test for the ball frozen on the centre spot. On the
    sample replay that recovers 333 s of live play against a match clock that
    ran 300 s of regulation plus 33 s of overtime.

Reference numbers for the ball are computed over the same live window and
printed alongside, so "you were in the corners 25% of the time" can be read
against where the ball actually was rather than against a number pulled from
the air.
"""

from __future__ import annotations

import math

TITLE = "Positioning -- where you played"

# Soccar geometry (uu).
GOAL_Y = 5120.0
SIDE_X = 4096.0
THIRD_Y = GOAL_Y / 3.0          # +/-1706.7 splits the pitch into thirds
LANE_X = SIDE_X / 3.0           # +/-1365.3 splits it into three lanes

# Heatmap resolution: 5 columns across x, 3 rows along y. The rows ARE the
# thirds, so the printed grid and the printed third shares agree by
# construction, and "corners" is exactly the four corner cells of the picture.
COLS, ROWS = 5, 3
COL_W = 2 * SIDE_X / COLS       # 1638.4
ROW_H = 2 * GOAL_Y / ROWS       # 3413.3
CORNER_X = SIDE_X - COL_W       # 2457.6, the inner edge of the outer columns

# Inside this radius of your own goal centre you are in the crease: no angle
# left to cover and nowhere to accelerate to.
GOAL_AREA_R = 1100.0

# A sample's weight is the gap to the next one, capped here. Normal gaps are
# 0.033 s; anything larger is the replay skipping, not you standing still.
DT_CAP = 0.2
# The clock ticks once a second, so a value that has stood for longer than
# this means the game is paused.
CLOCK_HOLD = 1.2

COL_NAMES = ["far left", "left", "centre", "right", "far right"]
ROW_NAMES = ["defensive third", "middle third", "attacking third"]
SHADE = [(0.85, "#"), (0.60, "="), (0.35, ":"), (0.15, "."), (0.0, " ")]


# -- small safe helpers ---------------------------------------------------

def _xyz(v):
    """A 3-tuple of finite floats, or None. Replay data is not trustworthy."""
    try:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def _pct(part, whole):
    return 100.0 * part / whole if whole > 0 else 0.0


def _blank():
    return {
        "ok": False, "reason": "no data", "player": None, "team": None,
        "live_s": 0.0, "wall_s": 0.0, "tracked_s": 0.0, "samples": 0,
        "thirds": {"def": 0.0, "mid": 0.0, "att": 0.0},
        "ball_thirds": {"def": 0.0, "mid": 0.0, "att": 0.0},
        "lanes": {"left": 0.0, "centre": 0.0, "right": 0.0},
        "ball_lanes": {"left": 0.0, "centre": 0.0, "right": 0.0},
        "avg_dist_own_goal": 0.0, "avg_up_pitch": 0.0,
        "attack_support": None, "attack_support_s": 0.0,
        "mates_attack_support": None,
        "defensive_cover": None, "defensive_cover_s": 0.0,
        "mates_defensive_cover": None,
        "goal_side": 0.0, "goal_side_defending": 0.0, "defending_s": 0.0,
        "corners": 0.0, "ball_corners": 0.0,
        "goal_area": 0.0, "goal_area_ball_upfield": 0.0, "ball_upfield_s": 0.0,
        "behind_own_line": 0.0,
        "last_man": None, "first_man": None, "team_n": 0,
        "grid": [[0.0] * COLS for _ in range(ROWS)],
        "grid_max": 0.0, "hot_cell": None,
    }


# -- live play ------------------------------------------------------------

def _weights(samples):
    """Per-sample (dt, live) lists with dead time marked out.

    dt is the gap to the NEXT sample, capped: the replay drops frames during
    goal replays, and an uncapped gap would silently donate several seconds to
    whatever cell the player happened to be standing in.
    """
    n = len(samples)
    ts = []
    for s in samples:
        try:
            ts.append(float(s.get("t") or 0.0))
        except Exception:
            ts.append(ts[-1] if ts else 0.0)

    gaps = []
    for i in range(n):
        d = (ts[i + 1] - ts[i]) if i + 1 < n else 0.0
        gaps.append(min(d, DT_CAP) if d > 0.0 else 0.0)
    if n:
        typical = sorted(g for g in gaps if g > 0.0)
        gaps[-1] = typical[len(typical) // 2] if typical else 0.033

    live = []
    held = 0.0
    prev = object()          # a value no clock reading can equal
    have_clock = False
    for i, s in enumerate(samples):
        c = s.get("clock")
        if c is not None:
            have_clock = True
        if c != prev:
            held = 0.0
        elif i:
            held += max(0.0, min(ts[i] - ts[i - 1], DT_CAP))
        prev = c
        b = _xyz(s.get("ball"))
        bv = _xyz(s.get("ball_vel")) or (0.0, 0.0, 0.0)
        # The ball parked on the centre spot is a kickoff countdown.
        frozen = (b is not None and abs(b[0]) < 12.0 and abs(b[1]) < 12.0
                  and math.sqrt(bv[0] ** 2 + bv[1] ** 2 + bv[2] ** 2) < 8.0)
        live.append((c is None or held <= CLOCK_HOLD) and not frozen)

    wall = sum(gaps)
    got = sum(g for g, l in zip(gaps, live) if l)
    # Safety net: a replay whose clock never ticks (private match, broken
    # attribute) would come out almost entirely "dead". Over-count rather than
    # report nothing.
    if have_clock and got < max(30.0, 0.25 * wall):
        live = [_xyz(s.get("ball")) is not None for s in samples]
    return gaps, live


# -- the work -------------------------------------------------------------

def compute(match, who) -> dict:
    res = _blank()
    try:
        samples = list(getattr(match, "samples", None) or [])
        teams = getattr(match, "teams", None) or {}
        if who not in teams:
            try:
                who = match.resolve(who)
            except Exception:
                who = None
        if not who or who not in teams:
            res["reason"] = "player not in this match"
            return res
        res["player"] = who
        res["team"] = teams.get(who)
        if len(samples) < 2:
            res["reason"] = "replay has no usable frames"
            return res

        try:
            sign = float(match.attack_sign(who)) or 1.0
        except Exception:
            sign = 1.0
        try:
            mates = [n for n in (match.mates(who) or []) if n in teams]
        except Exception:
            mates = []
        res["team_n"] = 1 + len(mates)

        gaps, live = _weights(samples)
        res["wall_s"] = sum(gaps)
        res["live_s"] = sum(g for g, l in zip(gaps, live) if l)

        grid = [[0.0] * COLS for _ in range(ROWS)]
        T = 0.0                      # live time with this player on the pitch
        thirds = [0.0, 0.0, 0.0]     # def, mid, att
        lanes = [0.0, 0.0, 0.0]      # left, centre, right
        b_thirds = [0.0, 0.0, 0.0]
        b_lanes = [0.0, 0.0, 0.0]
        b_corner = 0.0
        dist_sum = up_sum = 0.0
        goal_side = defending = goal_side_def = 0.0
        corners = goal_area = behind_line = 0.0
        ball_upfield = deep_upfield = 0.0
        rank_t = last_man = first_man = 0.0
        n_used = 0
        # Raw thirds are confounded by territory: a team that is camped in its
        # own half looks "passive" no matter how each player behaves. These
        # two are conditioned on where the ball is, which takes the territory
        # out and leaves the habit -- when the ball IS up there, are you?
        ball_att_s = ball_def_s = 0.0
        with_att = with_def = 0.0
        mate_att = dict((n, [0.0, 0.0]) for n in mates)   # [numer, denom]
        mate_def = dict((n, [0.0, 0.0]) for n in mates)

        for s, dt, is_live in zip(samples, gaps, live):
            if not is_live or dt <= 0.0:
                continue
            cars = s.get("cars") or {}
            car = cars.get(who)
            pos = _xyz((car or {}).get("pos"))
            ball = _xyz(s.get("ball"))
            if pos is None or ball is None:
                continue

            n_used += 1
            T += dt
            ax, ay = pos[0] * sign, pos[1] * sign
            bx, by = ball[0] * sign, ball[1] * sign

            # thirds and lanes, for the player and for the ball
            thirds[0 if ay < -THIRD_Y else (2 if ay > THIRD_Y else 1)] += dt
            b_thirds[0 if by < -THIRD_Y else (2 if by > THIRD_Y else 1)] += dt
            lanes[0 if ax < -LANE_X else (2 if ax > LANE_X else 1)] += dt
            b_lanes[0 if bx < -LANE_X else (2 if bx > LANE_X else 1)] += dt

            # how far from home, and how far up the pitch
            home = math.hypot(pos[0], ay + GOAL_Y)
            dist_sum += dt * home
            up_sum += dt * ay

            # goal-side: nearer your own goal than the ball is, along the
            # attacking axis
            if ay < by:
                goal_side += dt
                if by < 0.0:
                    goal_side_def += dt
            if by < 0.0:
                defending += dt

            # corners = the four corner cells of the printed heatmap
            if abs(ax) > CORNER_X and abs(ay) > THIRD_Y:
                corners += dt
            if abs(bx) > CORNER_X and abs(by) > THIRD_Y:
                b_corner += dt

            # in the crease, and in the crease with nothing to defend against
            in_area = home < GOAL_AREA_R or ay < -GOAL_Y
            if in_area:
                goal_area += dt
            if ay < -GOAL_Y:
                behind_line += dt
            if by > -THIRD_Y:
                ball_upfield += dt
                if in_area:
                    deep_upfield += dt

            col = min(COLS - 1, max(0, int((ax + SIDE_X) / COL_W)))
            row = min(ROWS - 1, max(0, int((ay + GOAL_Y) / ROW_H)))
            grid[row][col] += dt

            # are you where the ball is, when it matters
            up = by > THIRD_Y
            back = by < -THIRD_Y
            if up:
                ball_att_s += dt
                if ay > THIRD_Y:
                    with_att += dt
            if back:
                ball_def_s += dt
                if ay < -THIRD_Y:
                    with_def += dt

            # team-mates: rank on the pitch, and the same two ball-relative
            # numbers so yours can be read against people in the same match
            others = []
            for n in mates:
                p = _xyz((cars.get(n) or {}).get("pos"))
                if p is None:
                    continue
                may = p[1] * sign
                others.append(may)
                if up:
                    mate_att[n][1] += dt
                    if may > THIRD_Y:
                        mate_att[n][0] += dt
                if back:
                    mate_def[n][1] += dt
                    if may < -THIRD_Y:
                        mate_def[n][0] += dt
            if others:
                rank_t += dt
                if ay <= min(others):
                    last_man += dt
                if ay >= max(others):
                    first_man += dt

        res["samples"] = n_used
        res["tracked_s"] = T
        if T <= 0.0:
            res["reason"] = "no live frames with this player on the pitch"
            return res

        res["ok"] = True
        res["reason"] = ""
        res["thirds"] = {"def": _pct(thirds[0], T), "mid": _pct(thirds[1], T),
                         "att": _pct(thirds[2], T)}
        res["ball_thirds"] = {"def": _pct(b_thirds[0], T),
                              "mid": _pct(b_thirds[1], T),
                              "att": _pct(b_thirds[2], T)}
        res["lanes"] = {"left": _pct(lanes[0], T), "centre": _pct(lanes[1], T),
                        "right": _pct(lanes[2], T)}
        res["ball_lanes"] = {"left": _pct(b_lanes[0], T),
                             "centre": _pct(b_lanes[1], T),
                             "right": _pct(b_lanes[2], T)}
        res["avg_dist_own_goal"] = dist_sum / T
        res["avg_up_pitch"] = up_sum / T

        # Below about ten seconds the conditional shares are one clearance
        # away from anything, so leave them out rather than print a number
        # nobody should act on.
        res["attack_support_s"] = ball_att_s
        res["defensive_cover_s"] = ball_def_s
        if ball_att_s >= 10.0:
            res["attack_support"] = _pct(with_att, ball_att_s)
            peers = [_pct(a, b) for a, b in mate_att.values() if b >= 10.0]
            if peers:
                res["mates_attack_support"] = sum(peers) / len(peers)
        if ball_def_s >= 10.0:
            res["defensive_cover"] = _pct(with_def, ball_def_s)
            peers = [_pct(a, b) for a, b in mate_def.values() if b >= 10.0]
            if peers:
                res["mates_defensive_cover"] = sum(peers) / len(peers)

        res["goal_side"] = _pct(goal_side, T)
        res["goal_side_defending"] = _pct(goal_side_def, defending)
        res["defending_s"] = defending
        res["corners"] = _pct(corners, T)
        res["ball_corners"] = _pct(b_corner, T)
        res["goal_area"] = _pct(goal_area, T)
        res["behind_own_line"] = _pct(behind_line, T)
        res["goal_area_ball_upfield"] = _pct(deep_upfield, ball_upfield)
        res["ball_upfield_s"] = ball_upfield
        if rank_t > 0.0:
            res["last_man"] = _pct(last_man, rank_t)
            res["first_man"] = _pct(first_man, rank_t)

        g = [[_pct(v, T) for v in row] for row in grid]
        res["grid"] = g
        flat = [(v, r, c) for r, row in enumerate(g) for c, v in enumerate(row)]
        top = max(flat)
        res["grid_max"] = top[0]
        res["hot_cell"] = {"row": top[1], "col": top[2], "pct": top[0],
                           "name": "%s, %s" % (ROW_NAMES[top[1]],
                                               COL_NAMES[top[2]])}
    except Exception as exc:                                # never raise
        res["ok"] = False
        res["reason"] = "positioning failed: %s" % (
            str(exc)[:120] or type(exc).__name__)
    return res


# -- printing -------------------------------------------------------------

def _shade(v, top):
    if top <= 0.0:
        return " "
    f = v / top
    for cut, ch in SHADE:
        if f >= cut:
            return ch
    return " "


def _heatmap(res):
    g, top = res["grid"], res["grid_max"]
    pad = " " * 10
    inner = COLS * 6 + (COLS - 1)          # cells plus separators
    rule = pad + "+" + "+".join(["-" * 6] * COLS) + "+"
    out = ["",
           "  share of live play, one cell = %.0f x %.0f uu"
           % (COL_W, ROW_H),
           pad + "opponent net".center(inner + 2),
           pad + " " + "left".ljust(inner - 5) + "right",
           rule]
    for row, label in ((2, "attack"), (1, "middle"), (0, "defend")):
        cells = ["%s%4.1f%s" % (_shade(g[row][c], top), g[row][c],
                                _shade(g[row][c], top))
                 for c in range(COLS)]
        out.append("  %-8s|%s|" % (label, "|".join(cells)))
    out.append(rule)
    out.append(pad + "your net".center(inner + 2))
    out.append(pad + " cold  .  :  =  #  hot   (blank = barely there)")
    return [ln.rstrip() for ln in out]


def render(result) -> list[str]:
    r = result or {}
    if not r.get("ok"):
        return ["  no positioning data -- %s" % (r.get("reason") or "unknown")]

    def line(label, val, unit="%", extra=""):
        fmt = "%5.0f" if unit == "uu" else "%5.1f"
        s = ("  %-24s" % label) + (fmt % val) + " " + unit
        return s + ("   " + extra if extra else "")

    out = ["  %-24s%5.0f s   of %.0f s of replay"
           % ("live play analysed", r["live_s"], r["wall_s"])]
    bt = r["ball_thirds"]
    out.append(line("defensive third", r["thirds"]["def"], "%",
                    "(ball %.1f %%)" % bt["def"]))
    out.append(line("middle third", r["thirds"]["mid"], "%",
                    "(ball %.1f %%)" % bt["mid"]))
    out.append(line("attacking third", r["thirds"]["att"], "%",
                    "(ball %.1f %%)" % bt["att"]))
    if r.get("attack_support") is not None:
        extra = "of the %.0f s the ball was up there" % r["attack_support_s"]
        if r.get("mates_attack_support") is not None:
            extra += "   (mates %.1f %%)" % r["mates_attack_support"]
        out.append(line("with the attack", r["attack_support"], "%", extra))
    if r.get("defensive_cover") is not None:
        extra = "of the %.0f s it was in your third" % r["defensive_cover_s"]
        if r.get("mates_defensive_cover") is not None:
            extra += "   (mates %.1f %%)" % r["mates_defensive_cover"]
        out.append(line("back on defence", r["defensive_cover"], "%", extra))
    out.append(line("avg dist from own goal", r["avg_dist_own_goal"], "uu",
                    "(halfway is %.0f)" % GOAL_Y))
    gsd = ("(%.1f %% with the ball in your half)"
           % r["goal_side_defending"]) if r["defending_s"] > 5.0 else ""
    out.append(line("goal-side of the ball", r["goal_side"], "%", gsd))
    bl = r["ball_lanes"]
    out.append(line("central lane", r["lanes"]["centre"], "%",
                    "(ball %.1f %%)" % bl["centre"]))
    out.append(line("left / right lane", r["lanes"]["left"], "%",
                    "/%5.1f %%   (ball %.1f / %.1f %%)"
                    % (r["lanes"]["right"], bl["left"], bl["right"])))
    out.append(line("corner cells", r["corners"], "%",
                    "(ball %.1f %%)" % r["ball_corners"]))
    ga = ("(%.1f %% with the ball out of your third)"
          % r["goal_area_ball_upfield"]) if r["ball_upfield_s"] > 5.0 else ""
    out.append(line("inside own goal area", r["goal_area"], "%", ga))
    if r.get("last_man") is not None:
        out.append(line("deepest of your team", r["last_man"], "%",
                        "(furthest forward %.1f %%)" % r["first_man"]))
    out.extend(_heatmap(r))
    return out


# -- coaching -------------------------------------------------------------

def tips(result, match, who) -> list[str]:
    r = result or {}
    # Under a minute and a half of live play is a forfeit or a broken parse,
    # and any share computed from it is noise dressed up as a habit.
    if not r.get("ok") or r.get("tracked_s", 0.0) < 90.0:
        return []

    lanes, blanes = r["lanes"], r["ball_lanes"]
    out = []   # (rank, sentence) -- lower rank is the more urgent habit

    if r["defending_s"] > 45.0 and r["goal_side_defending"] < 55.0:
        out.append((1,
                    "With the ball in your own half you were goal-side of it "
                    "only %.1f%% of the time -- recover through the middle of "
                    "your own half instead of chasing the ball round the "
                    "outside, so you are always the one between it and your "
                    "net." % r["goal_side_defending"]))

    if r["ball_upfield_s"] > 45.0 and r["goal_area_ball_upfield"] >= 6.0:
        out.append((2,
                    "You sat inside your own goal area for %.1f%% of the "
                    "%.0f s the ball spent out of your defensive third -- "
                    "there is nothing to save from in there, so hold the top "
                    "of the box with speed on instead of parking on the line."
                    % (r["goal_area_ball_upfield"], r["ball_upfield_s"])))
    elif r["goal_area"] >= 16.0:
        out.append((2,
                    "You spent %.1f%% of live play inside your own goal area "
                    "-- deep in the net you have no angle and no speed, so "
                    "meet the play at the edge of the box and clear it away "
                    "from the middle." % r["goal_area"]))

    # Raw thirds cannot carry a tip on their own: a team pinned in its own
    # half puts every one of its players at 15% attacking third, which says
    # nothing about any of them. Only the ball-conditioned versions are worth
    # acting on, and only against the team-mates who shared the same match.
    sup, msup = r.get("attack_support"), r.get("mates_attack_support")
    cov = r.get("defensive_cover")
    if sup is not None and r["attack_support_s"] >= 40.0:
        if sup < 30.0:
            out.append((3,
                        "When the ball was in your attacking third you were "
                        "up there with it only %.1f%% of the time -- you are "
                        "not part of your team's offence, so follow the "
                        "clear up the wing and take the far post instead of "
                        "stopping at halfway." % sup))
        elif msup is not None and sup - msup <= -18.0:
            out.append((3,
                        "You joined the attack %.1f%% of the time the ball "
                        "was in the attacking third while your team-mates "
                        "managed %.1f%% -- you are leaving them to score "
                        "without you; push up as third man once the ball is "
                        "settled in their half." % (sup, msup)))
        elif cov is not None and sup >= 65.0 and cov < 60.0:
            out.append((3,
                        "You were up with the ball %.1f%% of the time it was "
                        "in the attacking third but back for only %.1f%% of "
                        "the time it was in yours -- you are playing as a "
                        "permanent striker and your net is paying for it."
                        % (sup, cov)))
    if cov is not None and r["defensive_cover_s"] >= 40.0 and cov < 55.0:
        out.append((3,
                    "With the ball in your own defensive third you were only "
                    "there %.1f%% of the time -- you are getting caught "
                    "up-field, so start your recovery the moment the "
                    "challenge is lost rather than waiting to see it." % cov))

    if r["corners"] - r["ball_corners"] >= 8.0:
        out.append((4,
                    "You were in the four corner cells %.1f%% of the time "
                    "while the ball was only there %.1f%% -- you are "
                    "recovering into the corner rather than back through the "
                    "middle, which hands over the next touch."
                    % (r["corners"], r["ball_corners"])))
    elif blanes["centre"] - lanes["centre"] >= 8.0:
        out.append((4,
                    "You spent %.1f%% of live play in the central lane "
                    "against the ball's %.1f%% -- come off the wall as you "
                    "recover; from the middle you can cover either side, from "
                    "the wall you cannot."
                    % (lanes["centre"], blanes["centre"])))

    skew = abs(lanes["left"] - lanes["right"])
    ball_skew = abs(blanes["left"] - blanes["right"])
    if skew - ball_skew >= 15.0 and max(lanes["left"], lanes["right"]) >= 38.0:
        strong = "left" if lanes["left"] > lanes["right"] else "right"
        weak = "right" if strong == "left" else "left"
        out.append((5,
                    "You lived on the %s side, %.1f%% of live play against "
                    "%.1f%% on the %s, while the ball split %.1f/%.1f -- "
                    "drill your %s-side recovery, because an opponent only "
                    "has to watch you once to know where you will be."
                    % (strong, max(lanes["left"], lanes["right"]),
                       min(lanes["left"], lanes["right"]), weak,
                       blanes["left"], blanes["right"], weak)))

    hot = r.get("hot_cell") or {}
    if hot.get("pct", 0.0) >= 25.0:
        out.append((6,
                    "%.1f%% of your live play happened in one cell of the "
                    "heatmap (%s) -- that is a parking spot rather than a "
                    "position; make yourself leave it the moment the ball "
                    "does." % (hot["pct"], hot["name"])))

    lm, fm = r.get("last_man"), r.get("first_man")
    if lm is not None and lm >= 50.0:
        out.append((7,
                    "You were the deepest player on your team %.1f%% of live "
                    "play -- you have made yourself the permanent last man, "
                    "so take a turn in the attack when possession is settled "
                    "and let a team-mate hold the back post." % lm))
    elif fm is not None and fm >= 55.0:
        out.append((7,
                    "You were your team's furthest-forward player %.1f%% of "
                    "live play -- if you are always first man then somebody "
                    "else is always covering for you, and one lost challenge "
                    "becomes a fast break." % fm))

    d = r["avg_dist_own_goal"]
    if d < 3800.0:
        out.append((8,
                    "Your average position was only %.0f uu from your own "
                    "goal, well inside your own half (halfway is %.0f uu) -- "
                    "you are defending the whole match rather than playing "
                    "it." % (d, GOAL_Y)))
    elif d > 6500.0:
        out.append((8,
                    "Your average position was %.0f uu from your own goal, a "
                    "long way past halfway (%.0f uu) -- you are cheating up "
                    "the pitch, which is why the counter keeps arriving "
                    "before you do." % (d, GOAL_Y)))

    out.sort(key=lambda p: p[0])
    return [s for _, s in out[:4]]
