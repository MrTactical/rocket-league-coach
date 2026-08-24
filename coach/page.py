"""
Render the analysis as a self-contained page.

    python coach/analyse.py --last 4 --json out.json
    python coach/page.py out.json --out coach.html

Or in one step:

    python coach/page.py --last 4 --out coach.html

Built to be glanced at between games on a second monitor, so the shape is a
dashboard rather than a document: session state first, the single biggest fix
second, detail underneath.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def esc(x):
    return html.escape(str(x), quote=True)


def num(d, *path, default=0.0):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if isinstance(cur, (int, float)) else default


# The metric modules disagree about scale, and nothing enforces a convention.
# Measured directly off a real match:
#   boost    starved_share = 0.1188   (a FRACTION; render multiplies by 100)
#   kickoffs team_win_rate = 0.25     (a FRACTION)
#   rotation even_share    = 33.33    (already a PERCENTAGE)
#   touches  big_pct       = 64.58    (already a PERCENTAGE)
# Reading a fraction as a percentage prints 11.9% starved as "0.1%", which
# looks like a player with flawless boost discipline rather than a scaling bug.
# One list, one accessor, instead of x100 sprinkled at each call site.
FRACTION_KEYS = {
    "boost_share", "hoard_share", "low_share", "ss_share", "starved_share",
    "poss_rate", "team_win_rate", "win_rate",
}


def share(d, key, default=0.0):
    """A percentage, whatever scale the producing module happened to use."""
    v = num(d, key, default=default)
    return v * 100.0 if key in FRACTION_KEYS else v


def outcome(match):
    """(verdict, us, them) from this player's side."""
    sc = match.get("score") or [0, 0]
    # `analyse` stores raw [team0, team1]; the player's team comes from the
    # rotation section, which knows the roster.
    team = match.get("team")
    if team not in (0, 1):
        pos = (match.get("sections", {}).get("positioning") or {}).get("data") or {}
        team = pos.get("team")
    if team not in (0, 1):
        # No honest answer available -- say so rather than inventing a winner.
        return "?", sc[0], sc[1]
    us, them = sc[team], sc[1 - team]
    return ("W" if us > them else "L" if us < them else "D"), us, them



def sec(match, key):
    return ((match.get("sections") or {}).get(key) or {}).get("data") or {}


def aggregate(matches):
    """
    Roll a set of matches into one picture.

    Time-share metrics are weighted by the live seconds each match actually
    contributed. Averaging percentages match-by-match would let a 40-second
    forfeit count as heavily as a full game, which is how a handful of short
    matches quietly rewrite a season.
    """
    agg = {"n": len(matches), "w": 0, "l": 0, "secs": 0.0,
           "goals": 0, "assists": 0, "saves": 0, "shots": 0, "score": 0,
           "lobby_score": 0.0, "conceded": 0, "scored": 0}
    wsum, wtot = {}, {}
    csum = {}

    def wadd(key, val, weight):
        if val is None or weight <= 0:
            return
        wsum[key] = wsum.get(key, 0.0) + val * weight
        wtot[key] = wtot.get(key, 0.0) + weight

    def cadd(key, val):
        if val is not None:
            csum[key] = csum.get(key, 0.0) + val

    for m in matches:
        v, us, them = outcome(m)
        if v == "W":
            agg["w"] += 1
        elif v == "L":
            agg["l"] += 1
        agg["scored"] += us
        agg["conceded"] += them

        sl = m.get("stat_line") or {}
        for k, sk in (("goals", "Goals"), ("assists", "Assists"),
                      ("saves", "Saves"), ("shots", "Shots"), ("score", "Score")):
            try:
                agg[k] += int(sl.get(sk) or 0)
            except (TypeError, ValueError):
                pass

        mech = sec(m, "mechanics")
        live = num(mech, "live") or num(sec(m, "boost"), "live_s") or 0.0
        agg["secs"] += live

        if live > 0:
            wadd("air", 100.0 * num(mech, "t_air") / live, live)
            wadd("wall", 100.0 * num(mech, "t_wall") / live, live)
            wadd("super", 100.0 * num(mech, "t_super") / live, live)
            wadd("crawl", 100.0 * num(mech, "t_crawl") / live, live)
            wadd("slide", 100.0 * num(mech, "t_handbrake") / live, live)
            cadd("dodges", num(mech, "dodges"))
            cadd("djumps", num(mech, "double_jumps"))
            cadd("km", num(mech, "dist") / 100000.0)

        b = sec(m, "boost")
        bl = num(b, "live_s") or live
        wadd("boost_held", num(b, "avg_held"), bl)
        wadd("starved", share(b, "starved_share"), bl)
        wadd("hoard", share(b, "hoard_share"), bl)
        wadd("low", share(b, "low_share"), bl)
        wadd("lobby_boost", num(b, "lobby_avg_held"), bl)
        cadd("big_pads", num(b, "big_pads"))
        cadd("small_pads", num(b, "small_pads"))

        r = sec(m, "rotation")
        rl = num(r, "play_s") or live
        wadd("first_ball", num(r, "rank_share", "1"), rl)
        wadd("last_back", num(r, "last_back"), rl)
        wadd("no_cover", num(r, "no_cover"), rl)
        wadd("double_commit", num(r, "double_commit"), rl)
        wadd("mate_med", num(r, "mate_med"), rl)
        wadd("even", num(r, "even_share"), rl)

        po = sec(m, "positioning")
        pl = num(po, "live_s") or live
        wadd("third_def", num(po, "thirds", "def"), pl)
        wadd("third_mid", num(po, "thirds", "mid"), pl)
        wadd("third_att", num(po, "thirds", "att"), pl)
        wadd("goal_side", num(po, "goal_side"), pl)
        wadd("goal_area", num(po, "goal_area"), pl)
        wadd("dist_own", num(po, "avg_dist_own_goal"), pl)

        t = sec(m, "touches")
        n_t = num(t, "touches")
        wadd("exit", num(t, "avg_exit"), n_t)
        wadd("big_pct", num(t, "big_pct"), n_t)
        wadd("weak_pct", num(t, "weak_pct"), n_t)
        wadd("fwd_pct", num(t, "fwd_pct"), n_t)
        cadd("touches", n_t)

        con = (sec(m, "goals").get("conceded") or {})
        nc = num(con, "n")
        wadd("con_dist", num(con, "dist_own"), nc)
        wadd("con_boost", num(con, "boost"), nc)

        k = sec(m, "kickoffs")
        nk = num(k, "n")
        wadd("ko_win", share(k, "team_win_rate"), nk)
        wadd("ko_ttc", num(k, "ttc_mine"), num(k, "took"))
        cadd("ko_n", nk)
        cadd("ko_took", num(k, "took"))

        # Grid, summed by cell so the heatmap can cover a whole era.
        g = po.get("grid")
        if isinstance(g, list) and g and isinstance(g[0], list):
            acc = agg.setdefault("grid", [[0.0] * len(g[0]) for _ in g])
            for ri, row in enumerate(g):
                for ci, val in enumerate(row):
                    if ri < len(acc) and ci < len(acc[ri]):
                        acc[ri][ci] += (val or 0.0) * pl

    agg["avg"] = {k: (wsum[k] / wtot[k]) for k in wsum if wtot.get(k)}
    agg["tot"] = csum
    if agg.get("grid"):
        mx = max(max(r) for r in agg["grid"]) or 1.0
        tot = sum(sum(r) for r in agg["grid"]) or 1.0
        agg["grid"] = [[100.0 * v / tot for v in row] for row in agg["grid"]]
        agg["grid_max"] = max(max(r) for r in agg["grid"])
    return agg


def playlists(matches):
    """Split by team size. 2v2 and 3v3 rotation shares are not comparable."""
    groups = {}
    for m in matches:
        groups.setdefault(m.get("team_size") or 0, []).append(m)
    return dict(sorted(groups.items()))


def era_of(match):
    return (match.get("date") or "????")[:7]


CSS = """
:root{
  --ground:#F3F5F8; --surface:#FFFFFF; --surface-2:#EAEEF4; --line:#D5DCE6;
  --ink:#131820; --ink-2:#4A5567; --ink-3:#78849A;
  --accent:#C97A12; --accent-soft:rgba(201,122,18,.12);
  --blue:#2F6FD0; --orange:#D2551C;
  --good:#1E874B; --warn:#B47400; --bad:#C0392B;
  --grid-0:rgba(201,122,18,.06); --grid-1:rgba(201,122,18,1);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#10131A; --surface:#171B24; --surface-2:#1F2430; --line:#2A313E;
    --ink:#E9ECF2; --ink-2:#A3AEC0; --ink-3:#727E93;
    --accent:#F2A33C; --accent-soft:rgba(242,163,60,.14);
    --blue:#4C8DFF; --orange:#FF7A45;
    --good:#4ADE80; --warn:#F2A33C; --bad:#F87171;
    --grid-0:rgba(242,163,60,.05); --grid-1:rgba(242,163,60,1);
  }
}
:root[data-theme="dark"]{
  --ground:#10131A; --surface:#171B24; --surface-2:#1F2430; --line:#2A313E;
  --ink:#E9ECF2; --ink-2:#A3AEC0; --ink-3:#727E93;
  --accent:#F2A33C; --accent-soft:rgba(242,163,60,.14);
  --blue:#4C8DFF; --orange:#FF7A45;
  --good:#4ADE80; --warn:#F2A33C; --bad:#F87171;
  --grid-0:rgba(242,163,60,.05); --grid-1:rgba(242,163,60,1);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 72px}
h1,h2,h3{font-family:"Saira Condensed","Arial Narrow",system-ui,sans-serif;
  font-weight:600; text-wrap:balance; margin:0; letter-spacing:.01em}
h1{font-size:2.6rem;line-height:1.02;text-transform:uppercase;letter-spacing:.03em}
h2{font-size:1.35rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink)}
.mono,td.n,.stat b,.chip b{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums}

header.top{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end;
  justify-content:space-between;padding-bottom:18px;border-bottom:2px solid var(--line)}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--ink-3)}
.rec{font-family:"Saira Condensed",sans-serif;font-size:2.1rem;line-height:1}
.rec .w{color:var(--good)} .rec .l{color:var(--bad)} .rec sep{color:var(--ink-3)}

.strip{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 30px}
.chip{flex:1 1 150px;min-width:150px;background:var(--surface);border:1px solid var(--line);
  border-left:4px solid var(--line);border-radius:3px;padding:11px 13px}
.chip.w{border-left-color:var(--good)} .chip.l{border-left-color:var(--bad)}
.chip .k{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.chip b{display:block;font-size:1.5rem;line-height:1.15;margin:1px 0 2px}
.chip .sub{font-size:.78rem;color:var(--ink-2)}

.lead{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--accent);
  border-radius:3px;padding:22px 24px;margin-bottom:30px}
.lead .eyebrow{color:var(--accent)}
.lead p{margin:.5rem 0 0;font-size:1.12rem;line-height:1.5;max-width:68ch;color:var(--ink)}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:3px;
  padding:17px 19px 19px;display:flex;flex-direction:column;gap:11px}
.card h2{font-size:1.05rem;color:var(--ink-2)}
.hero{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.hero b{font-family:"Saira Condensed",sans-serif;font-size:2.5rem;line-height:.95;
  color:var(--accent);font-variant-numeric:tabular-nums}
.hero span{font-size:.85rem;color:var(--ink-2);max-width:26ch}

table{width:100%;border-collapse:collapse;font-size:.83rem}
.scroll{overflow-x:auto}
td{padding:3px 0;border-bottom:1px solid var(--line);color:var(--ink-2)}
td:last-child{text-align:right;color:var(--ink);white-space:nowrap;
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}

.pitch{display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin-top:4px}
.cell{aspect-ratio:1.5;border-radius:2px;display:flex;align-items:center;
  justify-content:center;font-family:"IBM Plex Mono",monospace;font-size:.74rem;
  color:var(--ink);border:1px solid var(--line)}
.pitchwrap{position:relative}
.goalline{display:flex;justify-content:space-between;font-size:.68rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin:5px 0}

ul.tips{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:11px}
ul.tips li{background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:3px;padding:12px 15px;
  font-size:.93rem;max-width:78ch;color:var(--ink)}
.sect{margin:34px 0 14px;display:flex;align-items:center;gap:12px}
.sect:after{content:"";flex:1;height:1px;background:var(--line)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
  font-size:.78rem;color:var(--ink-3);max-width:76ch}
.dim{color:var(--ink-3);font-size:.9em}
code{font-family:"IBM Plex Mono",monospace;background:var(--surface-2);
  padding:1px 5px;border-radius:2px;font-size:.85em}

.tabs{display:flex;gap:2px;margin:26px 0 0;border-bottom:2px solid var(--line)}
.tabs button{appearance:none;background:none;border:0;border-bottom:2px solid transparent;
  margin-bottom:-2px;padding:9px 16px;cursor:pointer;color:var(--ink-3);
  font-family:"Saira Condensed",sans-serif;font-size:1rem;letter-spacing:.06em;
  text-transform:uppercase}
.tabs button:hover{color:var(--ink-2)}
.tabs button[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.panel[hidden]{display:none}

.viewer{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:16px}
@media (max-width:820px){.viewer{grid-template-columns:1fr}}
#pitch{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);
  border-radius:3px;display:block}
.ctrl{display:flex;align-items:center;gap:10px;margin-top:10px}
.ctrl button{appearance:none;background:var(--accent);color:#10131A;border:0;
  border-radius:3px;padding:7px 15px;cursor:pointer;font-weight:600;
  font-family:"IBM Plex Sans",sans-serif}
.ctrl input[type=range]{flex:1;accent-color:var(--accent)}
.ctrl .clock{font-family:"IBM Plex Mono",monospace;color:var(--ink-2);font-size:.85rem;
  min-width:72px;text-align:right}
.moments{max-height:430px;overflow-y:auto;display:flex;flex-direction:column;gap:5px}
.moments button{appearance:none;text-align:left;background:var(--surface);
  border:1px solid var(--line);border-left:3px solid var(--ink-3);border-radius:3px;
  padding:7px 10px;cursor:pointer;color:var(--ink-2);font-size:.8rem;
  font-family:"IBM Plex Sans",sans-serif}
.moments button:hover{color:var(--ink);border-color:var(--accent)}
.moments button.conceded{border-left-color:var(--bad)}
.moments button.scored{border-left-color:var(--good)}
.moments button.kickoff{border-left-color:var(--ink-3)}
.moments button.demoed,.moments button.demo{border-left-color:var(--accent)}
.moments b{font-family:"IBM Plex Mono",monospace;color:var(--ink-3);
  margin-right:7px;font-weight:400}
.ctrl button.ghost{background:none;color:var(--ink-2);border:1px solid var(--line)}
.caption{margin-top:10px;padding:12px 15px;border-radius:3px;font-size:.9rem;
  background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--accent);color:var(--ink);max-width:78ch}
.caption[hidden]{display:none}
.caption b{display:block;font-family:"Saira Condensed",sans-serif;font-size:1.1rem;
  text-transform:uppercase;letter-spacing:.05em;color:var(--accent);margin-bottom:3px}
.caption.conceded{border-left-color:var(--bad)}
.caption.conceded b{color:var(--bad)}
.caption.scored{border-left-color:var(--good)}
.caption.scored b{color:var(--good)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 4px;
  font-size:.78rem;color:var(--ink-2)}
.legend span{display:flex;align-items:center;gap:6px}
.legend .sw{width:13px;height:13px;border-radius:3px;display:inline-block;flex:none}
.legend .sw.blue{background:var(--blue)}
.legend .sw.orange{background:var(--orange)}
.legend .sw.ring{border-radius:50%;background:var(--blue);
  box-shadow:0 0 0 2px var(--ink)}
.legend .sw.good{border-radius:50%;background:none;border:2px solid var(--good)}
.legend .sw.bad{border-radius:50%;background:none;border:2px solid var(--bad)}
.legend .sw.ghost{border-radius:50%;background:none;
  border:2px dashed var(--accent)}
.live{margin-top:10px;padding:10px 14px;border-radius:3px;font-size:.86rem;
  background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--ink-3);color:var(--ink-2);max-width:78ch}
.live[hidden]{display:none}
.live b{display:block;color:var(--ink);font-size:.95rem;margin-bottom:2px}
.live span{display:block;color:var(--ink-2)}
.live.sev1{border-left-color:var(--warn)}
.live.sev1 b{color:var(--warn)}
.live.sev2{border-left-color:var(--bad);background:var(--surface-2)}
.live.sev2 b{color:var(--bad)}
"""


VIEWER_JS = r"""
<script>
(function () {
  var T = window.__TRACK__;
  var tabs = [].slice.call(document.querySelectorAll('.tabs button'));
  var panels = [].slice.call(document.querySelectorAll('.panel'));
  var onShow = null, current = 'overview';
  function show(name) {
    current = name;
    tabs.forEach(function (b) {
      b.setAttribute('aria-selected', String(b.dataset.tab === name));
    });
    panels.forEach(function (p) { p.hidden = p.dataset.panel !== name; });
    // A hidden panel has clientWidth 0, so a canvas sized on load paints an
    // empty bitmap. Re-size whenever the tab actually becomes visible.
    if (name === 'replay' && onShow) onShow();
  }
  tabs.forEach(function (b) {
    b.addEventListener('click', function () { show(b.dataset.tab); });
  });
  show('overview');

  // --- reload without stealing the page ---------------------------------
  // This replaces <meta http-equiv="refresh">, which reloaded every 30s and
  // threw away the current tab and any playback in progress -- the page
  // appeared to "jump back to Your stats" on its own, mid-replay. A reload is
  // only useful when you are not using the page, so wait until you are idle.
  var RELOAD_AFTER = window.__RELOAD_SECS__ || 0;
  if (RELOAD_AFTER) {
    var idleSince = Date.now();
    ['click', 'input', 'keydown', 'wheel', 'touchstart'].forEach(function (ev) {
      document.addEventListener(ev, function () { idleSince = Date.now(); },
                                {passive: true});
    });
    setInterval(function () {
      var busy = playing || current === 'replay';
      if (!busy && Date.now() - idleSince > RELOAD_AFTER * 1000) location.reload();
    }, 5000);
  }

  if (!T || !T.frames || !T.frames.length) return;

  var W = 8192, H = 10240, STRIDE = T.stride || 4;
  var N = T.names.length;
  var cv = document.getElementById('pitch');
  var ctx = cv.getContext('2d');
  var range = document.getElementById('scrub');
  var play = document.getElementById('play');
  var guide = document.getElementById('guide');
  var clock = document.getElementById('clock');
  var caption = document.getElementById('caption');
  var live = document.getElementById('live');
  var pausedAt = -99, prevBall = null;
  var playing = false, guided = false, rate = 1, raf = null, last = 0;
  var T0 = T.frames[0][0], T1 = T.frames[T.frames.length - 1][0];
  var cursor = T0;                       // playback head, in seconds (float)
  // Always defend the near goal, whichever side was actually played --
  // otherwise half your replays render backwards.
  var flip = T.my_team === 1 ? -1 : 1;

  function css(v) {
    return getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  }

  // --- dynamic camera ----------------------------------------------------
  // A fixed camera cannot do both jobs: framed for the whole pitch the cars
  // are specks, framed for the action half the pitch falls off screen. So the
  // camera follows, and the distance is SOLVED each frame from the spread of
  // whatever matters right now -- the ball, your car, and anyone near enough
  // to the ball to be part of the play.
  //
  // Everything is smoothed toward its target. Snapping the camera to a
  // bouncing ball is unwatchable.
  // The camera must LOOK AT the focus, so its height is a function of its
  // distance: camZ = d * tan(pitch). Held at a fixed height it aims at the
  // horizon instead, which put the ball off the top of the frame ~100% of
  // the time while every other check (depth order, scale) still passed.
  var CAM = {pitch: 0.80, f: 900};
  var cam = {x: 0, y: 0, d: 11000};        // current, smoothed
  var want = {x: 0, y: 0, d: 11000};       // target for this frame
  var MIN_D = 5200, MAX_D = 17000, FOLLOW = 0.08;

  function planCamera(f) {
    var pts = [[f.ball[0], f.ball[1]]];
    var me = f.cars[T.me];
    if (me) pts.push([me[0], me[1]]);
    for (var c = 0; c < N; c++) {
      var k = f.cars[c];
      if (!k || c === T.me) continue;
      var near = Math.hypot(k[0] - f.ball[0], k[1] - f.ball[1]);
      if (near < 3400) pts.push([k[0], k[1]]);
    }
    var xs = pts.map(function (p) { return p[0]; });
    var ys = pts.map(function (p) { return p[1]; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);

    want.x = (minX + maxX) / 2;
    want.y = (minY + maxY) / 2;

    // Perspective makes "what distance fits this box" awkward in closed form
    // -- ground compresses toward the horizon, so the visible span is not
    // linear in distance. Binary search instead: ~18 cheap iterations a frame,
    // and correct by construction rather than by an approximation I would then
    // have to trust.
    var pad = 1300;
    var box = [[minX - pad, minY - pad], [maxX + pad, minY - pad],
               [minX - pad, maxY + pad], [maxX + pad, maxY + pad]];
    var lo = MIN_D, hi = MAX_D;
    for (var it = 0; it < 18; it++) {
      var mid = (lo + hi) / 2;
      if (fitsAt(box, want.x, want.y, mid)) hi = mid; else lo = mid;
    }
    want.d = hi;
  }

  function projectFrom(x, y, z, ox, oy, d) {
    var rx = (x - ox) * flip, ry = (y - oy) * flip;
    var dy = ry + d, dz = z - d * Math.tan(CAM.pitch);
    var c = Math.cos(CAM.pitch), s = Math.sin(CAM.pitch);
    var fwd = dy * c - dz * s;
    var up = dy * s + dz * c;
    if (fwd < 150) return null;
    var sc = CAM.f / fwd;
    return {x: cv.width / 2 + rx * sc, y: cv.height * 0.52 - up * sc,
            s: sc, d: fwd};
  }

  function fitsAt(box, ox, oy, d) {
    for (var k = 0; k < box.length; k++) {
      var q = projectFrom(box[k][0], box[k][1], 0, ox, oy, d);
      if (!q) return false;
      if (q.x < 4 || q.x > cv.width - 4 || q.y < 4 || q.y > cv.height - 4)
        return false;
    }
    return true;
  }

  function project(x, y, z) {
    var rx = (x - cam.x) * flip, ry = (y - cam.y) * flip;
    var dy = ry + cam.d, dz = z - cam.d * Math.tan(CAM.pitch);
    var c = Math.cos(CAM.pitch), s = Math.sin(CAM.pitch);
    var fwd = dy * c - dz * s;
    var up = dy * s + dz * c;
    if (fwd < 150) return null;
    var sc = CAM.f / fwd;
    return {x: cv.width / 2 + rx * sc, y: cv.height * 0.52 - up * sc,
            s: sc, d: fwd};
  }

  function line(a, b, col, w) {
    var p = project(a[0], a[1], a[2] || 0), q = project(b[0], b[1], b[2] || 0);
    if (!p || !q) return;
    ctx.strokeStyle = col; ctx.lineWidth = w || 1.5;
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  }

  function drawPitch() {
    var lin = css('--line');
    var hw = W / 2, hh = H / 2;
    line([-hw, -hh], [hw, -hh], lin, 2);
    line([-hw, hh], [hw, hh], lin, 2);
    line([-hw, -hh], [-hw, hh], lin, 2);
    line([hw, -hh], [hw, hh], lin, 2);
    line([-hw, 0], [hw, 0], lin, 2);
    var prev = null;
    for (var a = 0; a <= 32; a++) {
      var th = a / 32 * 6.2832;
      var pt = [Math.cos(th) * 920, Math.sin(th) * 920];
      if (prev) line(prev, pt, lin, 1);
      prev = pt;
    }
    [[-hh, css('--blue')], [hh, css('--orange')]].forEach(function (g) {
      var y = g[0], col = g[1];
      line([-893, y, 0], [-893, y, 642], col, 3);
      line([893, y, 0], [893, y, 642], col, 3);
      line([-893, y, 642], [893, y, 642], col, 3);
      line([-893, y, 0], [893, y, 0], col, 3);
    });
  }

  // --- interpolation ----------------------------------------------------
  // The track is ~9.5 Hz. Stepping frame to frame looks fine at 1x and awful
  // at 0.25x, which is exactly when you are studying a mistake -- 2.4 frames
  // a second. So playback carries a float time cursor and every draw lerps
  // between the two bracketing frames, giving smooth motion at any rate from
  // the same data.
  function frameAt(t) {
    var lo = 0, hi = T.frames.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (T.frames[mid][0] < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function lerpAngle(a, b, u) {
    var d = ((b - a + 540) % 360) - 180;   // shortest way round
    return a + d * u;
  }

  function sample(t) {
    var hi = frameAt(t), lo = Math.max(0, hi - 1);
    var A = T.frames[lo], B = T.frames[hi];
    var span = B[0] - A[0];
    var u = span > 1e-6 ? Math.min(1, Math.max(0, (t - A[0]) / span)) : 0;
    var out = {ball: [A[1] + (B[1] - A[1]) * u,
                      A[2] + (B[2] - A[2]) * u,
                      A[3] + (B[3] - A[3]) * u], cars: []};
    for (var c = 0; c < N; c++) {
      var o = 4 + c * STRIDE;
      // A car absent from a frame is stored as zeros; interpolating toward
      // that would slide it to the centre spot instead of leaving it be.
      var aOff = (A[o] === 0 && A[o + 1] === 0);
      var bOff = (B[o] === 0 && B[o + 1] === 0);
      if (aOff && bOff) { out.cars.push(null); continue; }
      var src = aOff ? B : A, dst = bOff ? A : B, uu = (aOff || bOff) ? 0 : u;
      out.cars.push([
        src[o] + (dst[o] - src[o]) * uu,
        src[o + 1] + (dst[o + 1] - src[o + 1]) * uu,
        src[o + 2] + (dst[o + 2] - src[o + 2]) * uu,
        lerpAngle(src[o + 3], dst[o + 3], uu)
      ]);
    }
    return out;
  }

  function drawCar(x, y, z, head, col, me, name) {
    var p = project(x, y, z + 17);
    if (!p) return;
    var len = 118 * p.s, wid = 84 * p.s;
    if (z > 120) {
      var g = project(x, y, 0);
      if (g) {
        ctx.globalAlpha = 0.22; ctx.fillStyle = css('--ink-3');
        ctx.beginPath(); ctx.ellipse(g.x, g.y, len / 2, wid / 3, 0, 0, 6.2832);
        ctx.fill(); ctx.globalAlpha = 1;
      }
    }
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-(head * Math.PI / 180) * flip * 0.55);
    ctx.fillStyle = col;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-len / 2, -wid / 2, len, wid, 3 * p.s);
    else ctx.rect(-len / 2, -wid / 2, len, wid);
    ctx.fill();
    ctx.globalAlpha = 0.5;
    ctx.fillRect(-len / 6, -wid / 2, len / 2.6, wid);
    ctx.restore();
    ctx.globalAlpha = 1;
    if (me) {
      // Two rings, light over dark, so it reads on any surface behind it.
      // Sized off the car, and kept clear of it. A fixed 3.5px ring is
      // thicker than the car itself when the camera is close, which read as
      // a white car rather than a ringed blue one.
      var rr = Math.max(len, wid) * 0.95 + 5;
      ctx.strokeStyle = css('--ink');
      ctx.lineWidth = Math.max(1.5, Math.min(3, rr * 0.09));
      ctx.beginPath(); ctx.arc(p.x, p.y, rr, 0, 6.2832);
      ctx.stroke();
    }
    ctx.fillStyle = me ? css('--ink') : css('--ink-3');
    if (me) ctx.font = 'bold 12px ui-monospace, monospace';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(name.slice(0, 12), p.x + len * 0.7, p.y - wid * 0.6);
  }

  // --- coaching overlay --------------------------------------------------
  // Only two marks, both tied to findings this analyser actually made rather
  // than to generic advice.
  //
  //   SHADOW    where you should be when the ball is not yours: on the line
  //             from your net to the ball, goal-side, about a third of the
  //             way out. Drawn only when you are NOT goal-side, which is the
  //             exact state the recovery metric says you fail to correct.
  //   HOME      an arrow from your car to your net while you are beaten,
  //             because the measured fault is not being out of position, it
  //             is not turning around.
  // --- live read of what is happening right now --------------------------
  // The markers show WHERE; this says WHY. Everything below is computed from
  // the frame on screen, and every rule is one this analyser measured on this
  // player rather than general advice.
  //
  // Severity 2 is reserved for the specific compound failure the numbers say
  // costs the goals: beaten, nobody covering, and the ball travelling toward
  // your net. That combination pauses the guided run.
  var GOAL_MOUTH = 893;

  function readFrame(f, prevBall) {
    var me = f.cars[T.me];
    if (!me) return null;
    var sgn = T.my_team === 1 ? -1 : 1;
    var ownY = -5120 * sgn;
    var notes = [], sev = 0;

    var goalSide = (me[1] * sgn) < (f.ball[1] * sgn);
    var distNet = Math.abs(me[1] - ownY);
    var ballToNet = Math.abs(f.ball[1] - ownY);

    // Who else is home, and who is nearest the ball.
    var coverBehind = 0, mateNames = [], closest = null, closestD = 1e9;
    for (var c = 0; c < N; c++) {
      var k = f.cars[c];
      if (!k) continue;
      var d = Math.hypot(k[0] - f.ball[0], k[1] - f.ball[1]);
      if (d < closestD) { closestD = d; closest = c; }
      if (c !== T.me && T.teams[c] === T.my_team &&
          (k[1] * sgn) < (f.ball[1] * sgn)) {
        coverBehind++; mateNames.push(T.names[c]);
      }
    }
    var iAmClosest = closest === T.me;

    // Is the ball being driven at your goal?
    var incoming = false, ballSpeed = 0;
    if (prevBall) {
      var vy = (f.ball[1] - prevBall[1]);
      incoming = (vy * sgn) < 0;                 // travelling toward our net
      ballSpeed = Math.hypot(f.ball[0] - prevBall[0], vy);
    }

    if (!goalSide) {
      notes.push('The ball is goal-side of you — you are beaten.');
      sev = Math.max(sev, 1);
      if (coverBehind === 0) {
        notes.push('Nobody on your team is between the ball and your net.');
        sev = Math.max(sev, incoming && ballToNet < 6000 ? 2 : 1);
      } else {
        notes.push(mateNames[0] + ' is covering behind the ball.');
      }
      if (incoming && ballSpeed > 12) {
        notes.push('And it is moving toward your goal.');
      }
      notes.push('You are ' + Math.round(distNet) +
                 ' uu from your net. Turn and drive at it — not at the ball.');
    } else if (iAmClosest && coverBehind === 0) {
      notes.push('You are first to the ball with no cover behind you.');
      notes.push('If this challenge does not win it, the net is open. ' +
                 'Delay and shepherd instead.');
      sev = Math.max(sev, 1);
    } else {
      // Double commit: you and a team-mate both on the ball.
      for (var c2 = 0; c2 < N; c2++) {
        var k2 = f.cars[c2];
        if (!k2 || c2 === T.me || T.teams[c2] !== T.my_team) continue;
        var dBall = Math.hypot(k2[0] - f.ball[0], k2[1] - f.ball[1]);
        var dMe = Math.hypot(k2[0] - me[0], k2[1] - me[1]);
        var myBall = Math.hypot(me[0] - f.ball[0], me[1] - f.ball[1]);
        if (dBall < 1800 && myBall < 1800 && dMe < 1200) {
          notes.push('You and ' + T.names[c2] +
                     ' are both on the ball — double committed.');
          notes.push('One of you should peel off; whoever is further from ' +
                     'your net has the easier exit.');
          sev = Math.max(sev, 1);
        }
      }
    }

    if (!notes.length) {
      notes.push(goalSide ? 'Goal-side of the ball. Shape is fine here.'
                          : 'Nothing notable.');
    }
    return {sev: sev, notes: notes, goalSide: goalSide, distNet: distNet};
  }

  function paintLive(r) {
    if (!live) return;
    if (!r) { live.hidden = true; return; }
    live.hidden = false;
    live.className = 'live sev' + r.sev;
    live.innerHTML = r.notes.map(function (n, k) {
      return k === 0 ? '<b>' + n + '</b>' : '<span>' + n + '</span>';
    }).join('');
  }

  function drawOverlay(f) {
    var me = f.cars[T.me];
    if (!me) return;
    var sgn = T.my_team === 1 ? -1 : 1;      // +y is their goal for team 0
    var ownY = -5120 * sgn;
    var goalSide = (me[1] * sgn) < (f.ball[1] * sgn);

    // Goal-side status ring under your car.
    var p = project(me[0], me[1], 0);
    if (p) {
      ctx.strokeStyle = goalSide ? css('--good') : css('--bad');
      ctx.globalAlpha = 0.75; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(p.x, p.y, 150 * p.s, 62 * p.s, 0, 0, 6.2832);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (goalSide) return;
    // No 'be here' marker unless the measurement actually separates holding
    // from conceding. In 2v2 it does not (0.35 vs 0.36), so drawing one there
    // would be decoration dressed as advice.
    if (SH.usable === false) return;

    // Where you should be instead.
    //
    // These two numbers are MEASURED, not taught. Across 40 of this player's
    // own matches, every frame where their team was under attack was recorded
    // with the covering defender's position, split by whether the attack ended
    // in a goal within six seconds:
    //
    //     depth from own net    held 0.23 (n=5811)   conceded 0.32 (n=687)
    //     lateral, as a share
    //     of the ball's x       held 0.35            conceded 0.63
    //
    // The original hand-written guess was 0.34 depth -- almost exactly the
    // CONCEDED figure. It was drawing the losing position and calling it the
    // right one. Successful defence sits deeper and stays more central.
    //
    // Correlational, not causal: a deeper defender may partly reflect a less
    // dangerous attack. It is still this player's own record of what held.
    var SH = T.shadow || {};
    var SHADOW_DEPTH = SH.depth || 0.23, SHADOW_LATERAL = SH.lateral || 0.35;
    var sx = f.ball[0] * SHADOW_LATERAL;
    var sy = ownY + (f.ball[1] - ownY) * SHADOW_DEPTH;
    var g = project(sx, sy, 0);
    if (g) {
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = css('--accent'); ctx.globalAlpha = 0.9; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(g.x, g.y, 150 * g.s, 62 * g.s, 0, 0, 6.2832);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = css('--accent');
      ctx.font = '11px "IBM Plex Sans", sans-serif';
      ctx.fillText('be here', g.x - 22, g.y - 70 * g.s - 4);
      ctx.globalAlpha = 1;
    }

    // And the direction to go, which is the thing you measurably do not do.
    if (p && g) {
      ctx.strokeStyle = css('--accent'); ctx.globalAlpha = 0.55;
      ctx.lineWidth = 2; ctx.setLineDash([4, 6]);
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(g.x, g.y); ctx.stroke();
      ctx.setLineDash([]);
      var a = Math.atan2(g.y - p.y, g.x - p.x);
      ctx.fillStyle = css('--accent');
      ctx.beginPath();
      ctx.moveTo(g.x, g.y);
      ctx.lineTo(g.x - 11 * Math.cos(a - 0.4), g.y - 11 * Math.sin(a - 0.4));
      ctx.lineTo(g.x - 11 * Math.cos(a + 0.4), g.y - 11 * Math.sin(a + 0.4));
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  function draw(t) {
    var f = sample(t);
    planCamera(f);
    cam.x += (want.x - cam.x) * FOLLOW;
    cam.y += (want.y - cam.y) * FOLLOW;
    cam.d += (want.d - cam.d) * FOLLOW;
    ctx.clearRect(0, 0, cv.width, cv.height);
    drawPitch();
    drawOverlay(f);

    // Painter's algorithm: furthest first, or near cars vanish behind far ones.
    var items = [];
    for (var c = 0; c < N; c++) {
      var k = f.cars[c];
      if (k) items.push({d: k[1] * flip, kind: 'car', c: c, p: k});
    }
    items.push({d: f.ball[1] * flip, kind: 'ball', p: f.ball});
    items.sort(function (a, b) { return b.d - a.d; });

    items.forEach(function (it) {
      if (it.kind === 'ball') {
        var p = project(it.p[0], it.p[1], it.p[2]);
        if (!p) return;
        if (it.p[2] > 120) {
          var g = project(it.p[0], it.p[1], 0);
          if (g) {
            ctx.globalAlpha = 0.22; ctx.fillStyle = css('--ink-3');
            ctx.beginPath();
            ctx.ellipse(g.x, g.y, 92 * p.s, 40 * p.s, 0, 0, 6.2832);
            ctx.fill(); ctx.globalAlpha = 1;
          }
        }
        ctx.fillStyle = css('--ink');
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(3, 92 * p.s), 0, 6.2832);
        ctx.fill();
      } else {
        // Colour says WHICH TEAM, never who you are. You used to be drawn in
        // the accent amber while opponents were orange -- two warm hues a
        // glance apart, and it broke the team encoding as well: your own car
        // did not match your own side. You are now your team's colour with a
        // ring around you.
        var mine = T.teams[it.c] === T.my_team;
        drawCar(it.p[0], it.p[1], it.p[2], it.p[3],
                mine ? css('--blue') : css('--orange'),
                it.c === T.me, T.names[it.c]);
      }
    });

    var read = readFrame(f, prevBall);
    prevBall = f.ball.slice();
    paintLive(read);
    // A severity-2 read is the compound failure the numbers say costs goals.
    // Stop on it once, so it can be read rather than scrolled past -- but only
    // during the guided run, and never twice for the same incident.
    if (read && read.sev === 2 && guided && playing && t - pausedAt > 8) {
      pausedAt = t;
      setCaption({kind: 'conceded', title: 'Stop here — this is the mistake',
                  text: read.notes.join(' ')});
      pause();
    }

    clock.textContent = t.toFixed(1) + 's' + (rate !== 1 ? '   ' + rate + 'x' : '');
    range.value = String(Math.round((t - T0) * 100));
  }

  function seekTime(t, snap) {
    cursor = Math.max(T0, Math.min(T1, t));
    if (snap !== false) {
      // Jumping to a moment should cut, not glide the camera across the pitch.
      planCamera(sample(cursor));
      cam.x = want.x; cam.y = want.y; cam.d = want.d;
    }
    draw(cursor);
  }

  // The cursor is a running sum of dt * rate, so it holds 22.299999 where it
  // reads 22.3. An exact `t >= m.t` misses the window by a microsecond, and a
  // separate "next moment is more than 0.05 ahead" test misses it too -- a
  // dead zone in which an imminent moment is invisible to both, so the
  // skip-ahead leaps clean over it. Four of nine moments were being skipped,
  // including the conceded goal.
  //
  // So: one function returns the next moment whose window has NOT yet ended,
  // and the skip is computed from that same moment. It is then structurally
  // impossible to jump past one.
  var SLOW = T.slow || 6, EPS = 0.15;
  function pendingMoment(t) {
    for (var k = 0; k < T.moments.length; k++) {
      if (T.moments[k].t + SLOW > t) return T.moments[k];
    }
    return null;
  }
  function momentAt(t) {
    var m = pendingMoment(t);
    return (m && t >= m.t - EPS) ? m : null;
  }
  function setCaption(m) {
    if (!m) { caption.hidden = true; return; }
    caption.hidden = false;
    caption.className = 'caption ' + m.kind;
    caption.innerHTML = '<b>' + m.title + '</b>' + m.text;
  }

  // Time-based, not frame-based: rAF gives ~60 fps regardless of the data
  // rate, and the cursor advances by real elapsed time times the play rate.
  function loop(now) {
    if (!playing) return;
    var dt = Math.min((now - last) / 1000, 0.25);   // ignore tab-away jumps
    last = now;

    if (guided) {
      var m = momentAt(cursor);
      if (m) { rate = 0.25; setCaption(m); }
      else {
        rate = 4; setCaption(null);
        var nm = pendingMoment(cursor);
        if (nm && nm.t - cursor > 3) cursor = nm.t - 1.5;
      }
    }

    cursor += dt * rate;
    if (cursor >= T1) { cursor = T1; draw(cursor); stop(); return; }
    draw(cursor);
    raf = requestAnimationFrame(loop);
  }

  function start() {
    if (playing) return;
    playing = true; last = performance.now();
    raf = requestAnimationFrame(loop);
  }
  function pause() {
    playing = false;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    guide.textContent = 'Resume run-through';
  }
  function stop() {
    playing = false; guided = false; rate = 1;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    play.textContent = 'Play';
    guide.textContent = 'Guided run-through';
    setCaption(null);
  }

  play.addEventListener('click', function () {
    if (playing && !guided) { stop(); return; }
    stop();
    play.textContent = 'Pause';
    start();
  });
  guide.addEventListener('click', function () {
    if (playing && guided) { stop(); return; }
    if (!playing && guided) {          // resume after an auto-pause
      guide.textContent = 'Stop'; setCaption(null); start(); return;
    }
    stop();
    guided = true; rate = 4;
    guide.textContent = 'Stop';
    if (T.moments.length) cursor = Math.max(T0, T.moments[0].t - 1.5);
    start();
  });

  // Centiseconds, so the slider is smooth rather than snapping to frames.
  range.min = 0;
  range.max = Math.round((T1 - T0) * 100);
  range.addEventListener('input', function () {
    stop();
    seekTime(T0 + (+range.value) / 100);
  });

  [].slice.call(document.querySelectorAll('.moments button')).forEach(function (b) {
    b.addEventListener('click', function () {
      stop();
      seekTime(+b.dataset.t);
      setCaption({kind: b.dataset.kind, title: b.dataset.title,
                  text: b.dataset.text});
    });
  });

  function size() {
    var w = cv.parentElement.clientWidth;
    if (!w) return;
    cv.width = w;
    cv.height = Math.round(Math.min(w * 0.72, 560));
    draw(cursor);
  }
  window.addEventListener('resize', size);
  onShow = size;
  size();
})();
</script>
"""


def cell_style(v, mx):
    frac = (v / mx) if mx else 0.0
    return ("background:color-mix(in srgb, var(--grid-1) %d%%, var(--grid-0));"
            % int(round(frac * 82)))


def stat_row(label, value, note=""):
    extra = (' <span style="color:var(--ink-3)">%s</span>' % esc(note)) if note else ""
    return "<tr><td>%s</td><td>%s%s</td></tr>" % (esc(label), esc(value), extra)


def playlist_card(size, ms):
    """
    One playlist, rolled up.

    Playlists are kept apart deliberately. An even rotation share is 50% in 2s
    and 33% in 3s, so a combined "career rotation" number describes nobody.
    The same goes for positioning thirds and double-commit rates.
    """
    g = aggregate(ms)
    av, tot = g["avg"], g["tot"]
    mins = g["secs"] / 60.0 or 1.0
    first = (ms[0].get("date") or "?")[:7]
    last = (ms[-1].get("date") or "?")[:7]
    span = first if first == last else "%s to %s" % (first, last)

    o = ['<div class="card">']
    o.append("<h2>%dv%d &middot; %d matches &middot; %s</h2>"
             % (size, size, g["n"], esc(span)))
    o.append('<div class="hero"><b>%d&ndash;%d</b><span>win&ndash;loss, '
             "%.0f minutes played</span></div>" % (g["w"], g["l"], mins))
    o.append("<table>")
    o.append(stat_row("goals / assists / saves", "%d / %d / %d"
                      % (g["goals"], g["assists"], g["saves"])))
    if g["shots"]:
        o.append(stat_row("shooting", "%.0f%%" % (100.0 * g["goals"] / g["shots"]),
                          "%d of %d" % (g["goals"], g["shots"])))
    o.append(stat_row("goals for / against", "%d / %d" % (g["scored"], g["conceded"]),
                      "%+d" % (g["scored"] - g["conceded"])))
    if "first_ball" in av:
        o.append(stat_row("first to the ball", "%.1f%%" % av["first_ball"],
                          "/ %.0f%% even" % av.get("even", 100.0 / max(size, 1))))
    if "last_back" in av:
        o.append(stat_row("deepest of your team", "%.1f%%" % av["last_back"]))
    if "double_commit" in av:
        o.append(stat_row("double committed", "%.1f%%" % av["double_commit"]))
    if "goal_side" in av:
        o.append(stat_row("goal-side of the ball", "%.1f%%" % av["goal_side"]))
    if "boost_held" in av:
        o.append(stat_row("boost held", "%.0f" % av["boost_held"],
                          "lobby %.0f" % av.get("lobby_boost", 0.0)))
    if "starved" in av:
        o.append(stat_row("starved (0-10)", "%.1f%%" % av["starved"],
                          "below 30: %.0f%%" % av.get("low", 0.0)))
    if "exit" in av:
        o.append(stat_row("ball speed off touch", "%.0f uu/s" % av["exit"],
                          "%.0f%% big hits" % av.get("big_pct", 0.0)))
    if "air" in av:
        o.append(stat_row("genuinely airborne", "%.1f%%" % av["air"],
                          "%.1f%% on walls" % av.get("wall", 0.0)))
    if tot.get("dodges"):
        o.append(stat_row("flips per minute", "%.1f" % (tot["dodges"] / mins)))
    if "crawl" in av:
        o.append(stat_row("crawling (<300 uu/s)", "%.1f%%" % av["crawl"]))
    if "con_dist" in av:
        o.append(stat_row("distance out when conceding", "%.0f uu" % av["con_dist"]))
    if tot.get("km"):
        o.append(stat_row("distance driven", "%.0f km" % tot["km"]))
    o.append("</table></div>")
    return "\n".join(o), g


TRANSFERABLE = [
    ("shooting %", lambda g: (100.0 * g["goals"] / g["shots"]) if g["shots"] else None),
    ("ball speed off touch", lambda g: g["avg"].get("exit")),
    ("big hits %", lambda g: g["avg"].get("big_pct")),
    ("boost held", lambda g: g["avg"].get("boost_held")),
    ("starved %", lambda g: g["avg"].get("starved")),
    ("below 30 boost %", lambda g: g["avg"].get("low")),
    ("genuinely airborne %", lambda g: g["avg"].get("air")),
    ("crawling %", lambda g: g["avg"].get("crawl")),
    ("supersonic %", lambda g: g["avg"].get("super")),
    ("flips / min", lambda g: (g["tot"].get("dodges", 0.0)
                               / (g["secs"] / 60.0)) if g["secs"] else None),
]


def then_vs_now(groups):
    """
    Compare only the traits that survive a format change.

    Rotation share, positioning thirds and double commits are all defined by
    how many players are on the pitch, so comparing them across 2v2 and 3v3
    would be comparing the format, not the player.
    """
    if len(groups) < 2:
        return ""
    keys = sorted(groups)
    old, new = groups[keys[0]], groups[keys[-1]]
    go, gn = aggregate(old), aggregate(new)
    lo = "%dv%d %s" % (keys[0], keys[0], (old[0].get("date") or "?")[:7])
    ln = "%dv%d %s" % (keys[-1], keys[-1], (new[-1].get("date") or "?")[:7])

    o = ['<div class="card scroll">']
    o.append("<h2>Then vs now</h2>")
    o.append('<p style="margin:0;font-size:.86rem;color:var(--ink-2);max-width:62ch">'
             "Rotation, thirds and double commits are excluded outright: an even "
             "share is 50% in 2s and 33% in 3s, so comparing them would compare "
             "the format, not you. Treat what is left as indicative rather than "
             "clean &mdash; 3s is more contested and gives less room to wind up, "
             "so touch power and flip counts drift downward on format alone."
             "</p>")
    o.append("<table><tr><td><b>trait</b></td><td><b>%s &rarr; %s</b></td></tr>"
             % (esc(lo), esc(ln)))
    for label, fn in TRANSFERABLE:
        a_, b_ = fn(go), fn(gn)
        if a_ is None or b_ is None:
            continue
        if b_ > a_ * 1.08:
            arrow, col = "&uarr;", "var(--good)"
        elif b_ < a_ * 0.92:
            arrow, col = "&darr;", "var(--bad)"
        else:
            arrow, col = "&ndash;", "var(--ink-3)"
        o.append('<tr><td>%s</td><td>%.1f &rarr; %.1f '
                 '<span style="color:%s">%s</span></td></tr>'
                 % (esc(label), a_, b_, col, arrow))
    o.append("</table></div>")
    return "\n".join(o)


def rank_block(rank):
    """
    Rank context, pasted in by hand from a tracker profile.

    Ranks are not in replay files -- they only exist on Psyonix's servers, and
    the public trackers block automated access. So this is manual, and it is
    worth the manual step: MMR is a far better estimate of how good someone is
    than anything derivable from a handful of replays, because it is fitted
    over hundreds of games against ranked opposition.
    """
    pls = rank.get("playlists") or []
    if not pls:
        return ""
    main = next((p for p in pls if p.get("main")), pls[0])
    gap = None
    if main.get("season_peak") and main.get("mmr"):
        gap = main["season_peak"] - main["mmr"]

    o = ['<div class="card">']
    o.append("<h2>Rank &middot; %s</h2>" % esc(rank.get("source", "tracker")))
    o.append('<div class="hero"><b>%s</b><span>%s &mdash; %d MMR over %d ranked '
             "matches this season</span></div>"
             % (esc(main.get("rank", "?")), esc(main.get("name", "")),
                main.get("mmr") or 0, main.get("matches") or 0))
    if gap:
        o.append('<p style="margin:0;font-size:.9rem;color:var(--ink);'
                 'max-width:60ch">You are <b style="color:var(--accent)">%d MMR '
                 "below your own peak this season</b> (%d), and %d below your "
                 "all-time best of %d in %s. Your rank is not the problem; the "
                 "gap to your own ceiling is.</p>"
                 % (gap, main["season_peak"],
                    (main.get("best") or main["season_peak"]) - main["mmr"],
                    main.get("best") or main["season_peak"],
                    esc(main.get("best_season") or "an earlier season")))
    # Prefer the within-session trace while the day-by-day history is thin.
    # Rocket League logs a reading every time you queue, so one evening already
    # gives a real movement chart where "one row per day" gives a single dot.
    sess = rank.get("session") or {}
    sess_r = [v for v in (sess.get("readings") or []) if isinstance(v, (int, float))]
    hist = [h for h in (rank.get("history") or []) if h.get("mmr_3v3")]
    if len(hist) < 2 and len(sess_r) > 1:
        lo, hi = min(sess_r), max(sess_r)
        rng = (hi - lo) or 1
        o.append('<div style="display:flex;align-items:flex-end;gap:4px;'
                 'height:56px;margin:8px 0 2px">')
        for v in sess_r:
            frac = (v - lo) / rng
            o.append('<div title="%d" style="flex:1;min-width:8px;height:%d%%;'
                     'background:var(--accent);opacity:%.2f;border-radius:1px">'
                     "</div>" % (v, int(20 + 80 * frac), 0.45 + 0.55 * frac))
        o.append("</div>")
        move = sess_r[-1] - sess_r[0]
        col = ("var(--good)" if move > 0 else
               "var(--bad)" if move < 0 else "var(--ink-3)")
        o.append('<div class="goalline"><span>%s &middot; %s</span>'
                 "<span>%d readings</span></div>"
                 % (esc(sess.get("date", "")), esc(sess.get("playlist", "")),
                    len(sess_r)))
        o.append('<p style="margin:0;font-size:.86rem;color:var(--ink-2)">'
                 'Last session: %d &rarr; %d, <b style="color:%s">%+d MMR</b>. '
                 "Peak in session %d, low %d. Read from the game&rsquo;s own "
                 "log at queue time, so the final match is not in it yet.</p>"
                 % (sess_r[0], sess_r[-1], col, move, hi, lo))
    elif len(hist) > 1:
        hist = sorted(hist, key=lambda h: h.get("date", ""))
        lo = min(h["mmr_3v3"] for h in hist)
        hi = max(h["mmr_3v3"] for h in hist)
        rng = (hi - lo) or 1
        o.append('<div style="display:flex;align-items:flex-end;gap:4px;'
                 'height:56px;margin:6px 0 2px">')
        for h in hist:
            frac = (h["mmr_3v3"] - lo) / rng
            o.append('<div title="%s: %d" style="flex:1;min-width:6px;'
                     "height:%d%%;background:var(--accent);opacity:%.2f;"
                     'border-radius:1px"></div>'
                     % (esc(h.get("date", "")), h["mmr_3v3"],
                        int(20 + 80 * frac), 0.45 + 0.55 * frac))
        o.append("</div>")
        o.append('<div class="goalline"><span>%s &middot; %d</span>'
                 "<span>%s &middot; %d</span></div>"
                 % (esc(hist[0].get("date", "")), hist[0]["mmr_3v3"],
                    esc(hist[-1].get("date", "")), hist[-1]["mmr_3v3"]))
        move = hist[-1]["mmr_3v3"] - hist[0]["mmr_3v3"]
        col = "var(--good)" if move > 0 else "var(--bad)" if move < 0 else "var(--ink-3)"
        o.append('<p style="margin:0;font-size:.86rem;color:var(--ink-2)">'
                 '<b style="color:%s">%+d MMR</b> across %d readings.</p>'
                 % (col, move, len(hist)))
    elif hist:
        o.append('<p style="margin:0;font-size:.84rem;color:var(--ink-3);'
                 'max-width:58ch">One reading so far. Run '
                 "<code>python coach/mmr.py &lt;your 3v3 MMR&gt;</code> after a "
                 "session and this becomes a movement chart.</p>")

    o.append('<table class="scroll">')
    for pl in pls:
        note = []
        if pl.get("matches"):
            note.append("%d games" % pl["matches"])
        if pl.get("streak"):
            note.append(pl["streak"])
        sub = " &middot; ".join(note)
        o.append('<tr><td>%s<br><span class="dim">%s</span></td>'
                 '<td>%s<br><span class="dim">%s</span></td></tr>'
                 % (esc(pl.get("name", "")), esc(pl.get("rank", "")),
                    pl.get("mmr") or "-", sub))
    o.append("</table></div>")
    return "\n".join(o)


def shadow_note(track):
    """Where the marker's numbers came from, and how a higher rank compares."""
    sh = (track or {}).get("shadow") or {}
    if not sh.get("depth"):
        return ('<p style="color:var(--ink-3);font-size:.78rem;max-width:66ch">'
                "&ldquo;Where you should be&rdquo; is using a fallback value. "
                "Run <code>python coach/shadow.py</code> to measure it from "
                "your own replays.</p>")

    size = sh.get("team_size") or 3
    bits = ["&ldquo;Where you should be&rdquo; is measured, not a coaching "
            "clich&eacute;. Across %d of your %dv%d matches the covering "
            "defender sat <b>%.0f%%</b> of the way from their net to the ball "
            "on attacks that were held (n=%s), against <b>%.0f%%</b> on "
            "attacks conceded (n=%s)."
            % (sh.get("matches", 0), size, size, 100 * sh["depth"],
               f"{sh.get('n_held', 0):,}", 100 * sh.get("depth_conceded", 0),
               f"{sh.get('n_conceded', 0):,}")]

    pro = sh.get("pro")
    if pro and pro.get("depth"):
        bits.append(
            "Grand Champion play, measured with the same code over %d "
            "downloaded replays, sits at <b>%.0f%%</b> deep and <b>%.0f%%</b> "
            "across &mdash; against your %.0f%% and %.0f%%."
            % (pro.get("matches", 0), 100 * pro["depth"],
               100 * (pro.get("lateral") or 0), 100 * sh["depth"],
               100 * (sh.get("lateral") or 0)))
        # The consistency gap is the real finding, and it is not about depth.
        mine_swing = abs((sh.get("lateral") or 0) -
                         (sh.get("lateral_conceded") or 0))
        pro_swing = abs((pro.get("lateral") or 0) -
                        (pro.get("lateral_conceded") or 0))
        if mine_swing > pro_swing * 2 and pro_swing >= 0:
            bits.append(
                "The gap that matters is consistency: their sideways position "
                "barely moves between holding (%.0f%%) and conceding (%.0f%%), "
                "while yours swings from %.0f%% to %.0f%%. They hold one "
                "position; you over-shift toward the ball when it goes wrong."
                % (100 * (pro.get("lateral") or 0),
                   100 * (pro.get("lateral_conceded") or 0),
                   100 * (sh.get("lateral") or 0),
                   100 * (sh.get("lateral_conceded") or 0)))
        if pro.get("separation") is not None and                 pro["separation"] < (sh.get("separation") or 1) * 0.7:
            bits.append(
                "Worth knowing: at their level this position predicts far less "
                "(separation %.2f against your %.2f). Position is costing you "
                "goals that it does not cost them."
                % (pro["separation"], sh.get("separation") or 0))
    else:
        bits.append(
            "No higher-rank benchmark yet &mdash; add a ballchasing.com API "
            "key and the watcher will fetch one.")
    return ('<p style="color:var(--ink-3);font-size:.78rem;max-width:72ch">'
            + " ".join(bits) + "</p>")


def heat_block(grid, grid_max, label_html):
    # label_html is trusted markup built here, NOT user text. Escaping it turned
    # "2v2 &middot; 86 matches" into a literal "2V2 &MIDDOT; 86 MATCHES" on the
    # page, because the entity got escaped a second time into &amp;middot;.
    o = ['<div class="card" style="max-width:560px">']
    o.append("<h2>%s</h2>" % label_html)
    o.append('<div class="goalline"><span>&larr; left</span>'
             '<span style="color:var(--ink-2)">their net</span>'
             "<span>right &rarr;</span></div>")
    o.append('<div class="pitch">')
    for row in reversed(grid):
        for v in row:
            o.append('<div class="cell" style="%s">%.1f</div>'
                     % (cell_style(v, grid_max), v))
    o.append("</div>")
    o.append('<div class="goalline"><span></span>'
             '<span style="color:var(--ink-2)">your net</span><span></span></div>')
    o.append("</div>")
    return "\n".join(o)


def build(payload, refresh=0) -> str:
    player = payload.get("player") or "Player"
    matches = payload.get("matches") or []
    if not matches:
        return "<title>No matches</title><h1>No matches found</h1>"

    matches = sorted(matches, key=lambda m: m.get("date") or "")
    latest = matches[-1]
    groups = playlists(matches)
    career = aggregate(matches)
    span = "%s to %s" % ((matches[0].get("date") or "?")[:10],
                         (matches[-1].get("date") or "?")[:10])

    out = []
    a = out.append
    a("<title>%s Match Coach</title>" % esc(player))
    if refresh:
        # NOT <meta http-equiv="refresh">. That reloaded unconditionally, so a
        # guided run-through was thrown away mid-replay and the page snapped
        # back to the first tab -- it read as the page changing tabs by itself.
        # The script reloads only once you are idle and not watching a replay.
        a("<script>window.__RELOAD_SECS__=%d;</script>" % refresh)
    a('<link rel="preconnect" href="https://fonts.googleapis.com">')
    a('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      "family=Saira+Condensed:wght@500;600;700&"
      "family=IBM+Plex+Sans:wght@400;500;600&"
      'family=IBM+Plex+Mono:wght@400;500&display=swap">')
    a("<style>%s</style>" % CSS)
    a('<div class="wrap">')

    a('<header class="top">')
    a('<div><div class="eyebrow">Rocket League &middot; replay coach</div>'
      '<h1>%s</h1><div class="eyebrow" style="margin-top:6px">%d matches '
      "&middot; %s</div></div>" % (esc(player), career["n"], esc(span)))
    a('<div style="text-align:right"><div class="eyebrow">all replays</div>'
      '<div class="rec mono"><span class="w">%dW</span> '
      '<span style="color:var(--ink-3)">/</span> '
      '<span class="l">%dL</span></div></div>' % (career["w"], career["l"]))
    a("</header>")

    tips = []
    for key, s in (latest.get("sections") or {}).items():
        for t in s.get("tips") or []:
            tips.append(t)
    if tips:
        a('<div class="lead" style="margin-top:26px">'
          '<div class="eyebrow">Fix this first</div><p>%s</p></div>' % esc(tips[0]))

    rank = payload.get("rank")
    if rank:
        a('<div class="sect"><h2>Where you actually sit</h2></div>')
        a('<div class="cards">%s</div>' % rank_block(rank))

    a('<div class="tabs" role="tablist">'
      '<button role="tab" data-tab="overview" aria-selected="true">Your stats</button>'
      '<button role="tab" data-tab="lobby">Lobby &amp; opponents</button>'
      '<button role="tab" data-tab="replay">Replay</button></div>')

    a('<div class="panel" data-panel="overview">')
    a('<div class="sect"><h2>By playlist</h2></div>')
    a('<div class="cards">')
    grids = []
    for size, ms in groups.items():
        card, g = playlist_card(size, ms)
        a(card)
        if g.get("grid"):
            grids.append((size, g))
    a("</div>")

    tvn = then_vs_now(groups)
    if tvn:
        a('<div class="sect"><h2>Has anything changed</h2></div>')
        a('<div class="cards">%s</div>' % tvn)

    recent = matches[-8:]
    a('<div class="sect"><h2>Most recent %d</h2></div>' % len(recent))
    a('<div class="strip">')
    for m in recent:
        v, us, them = outcome(m)
        sl = m.get("stat_line") or {}
        when = (m.get("date") or "")[5:16]
        a('<div class="chip %s"><div class="k">%s &middot; %dv%d</div>'
          '<b>%d&ndash;%d</b><div class="sub">%s pts &middot; %s g &middot; %s sv'
          "</div></div>"
          % (v.lower(), esc(when), m.get("team_size", 0), m.get("team_size", 0),
             us, them, sl.get("Score", 0), sl.get("Goals", 0), sl.get("Saves", 0)))
    a("</div>")

    v, us, them = outcome(latest)
    a('<div class="sect"><h2>Latest match &middot; %s %d&ndash;%d</h2></div>'
      % ({"W": "won", "L": "lost", "D": "drew"}.get(v, "played"), us, them))
    a('<div class="cards">')

    def D(k):
        return sec(latest, k)

    r = D("rotation")
    if r:
        a('<div class="card"><h2>Rotation</h2>')
        a('<div class="hero"><b>%.0f%%</b><span>of the match you were the '
          "deepest player on your team</span></div>" % num(r, "last_back"))
        a("<table>")
        a(stat_row("first to the ball", "%.1f%%" % num(r, "rank_share", "1"),
                   "/ %.0f%% even" % num(r, "even_share")))
        a(stat_row("first-man spell", "%.1f s mean" % num(r, "fm_mean")))
        a(stat_row("rotation cycles", "%d" % int(num(r, "cycles"))))
        a(stat_row("no cover behind you", "%.0f%%" % num(r, "no_cover")))
        a(stat_row("double committed", "%.1f%%" % num(r, "double_commit")))
        a(stat_row("nearest mate", "%.0f uu" % num(r, "mate_med")))
        a("</table></div>")

    b = D("boost")
    if b:
        a('<div class="card"><h2>Boost</h2>')
        a('<div class="hero"><b>%.0f</b><span>average held &mdash; lobby '
          "average %.0f</span></div>"
          % (num(b, "avg_held"), num(b, "lobby_avg_held")))
        a("<table>")
        a(stat_row("starved (0-10)", "%.1f%%" % share(b, "starved_share")))
        a(stat_row("below 30 (no aerials)", "%.1f%%" % share(b, "low_share")))
        a(stat_row("hoarding (90-100)", "%.1f%%" % share(b, "hoard_share")))
        a(stat_row("big / small pads", "%d / %d"
                   % (int(num(b, "big_pads")), int(num(b, "small_pads")))))
        a(stat_row("collected per min", "%.0f" % num(b, "collected_per_min")))
        a("</table></div>")

    t = D("touches")
    if t:
        a('<div class="card"><h2>Ball contact</h2>')
        a('<div class="hero"><b>%.0f</b><span>uu/s average off your touch, '
          "%d touches</span></div>"
          % (num(t, "avg_exit"), int(num(t, "touches"))))
        a("<table>")
        a(stat_row("big hits (>1500)", "%d" % int(num(t, "big")),
                   "%.0f%%" % num(t, "big_pct")))
        a(stat_row("weak dinks (<800)", "%d" % int(num(t, "weak")),
                   "%.0f%%" % num(t, "weak_pct")))
        a(stat_row("sent at their goal", "%.0f%%" % num(t, "fwd_pct")))
        a(stat_row("sent back at your own", "%.0f%%" % num(t, "back_pct")))
        a(stat_row("hardest touch", "%.0f uu/s" % num(t, "peak_exit")))
        a("</table></div>")

    mk = D("mechanics")
    if mk:
        live = num(mk, "live") or 1.0
        a('<div class="card"><h2>Mechanics</h2>')
        a('<div class="hero"><b>%.1f</b><span>flips per minute &mdash; %d in '
          "the match</span></div>"
          % (num(mk, "dodges") / (live / 60.0), int(num(mk, "dodges"))))
        a("<table>")
        a(stat_row("genuinely airborne", "%.1f%%" % (100.0 * num(mk, "t_air") / live)))
        a(stat_row("on a wall", "%.1f%%" % (100.0 * num(mk, "t_wall") / live)))
        a(stat_row("supersonic", "%.1f%%" % (100.0 * num(mk, "t_super") / live)))
        a(stat_row("crawling (<300 uu/s)", "%.1f%%"
                   % (100.0 * num(mk, "t_crawl") / live)))
        a(stat_row("powerslide", "%.1f%%" % (100.0 * num(mk, "t_handbrake") / live)))
        a(stat_row("peak height", "%.0f uu" % num(mk, "peak_air_z")))
        a("</table></div>")

    k = D("kickoffs")
    if k:
        a('<div class="card"><h2>Kickoffs</h2>')
        a('<div class="hero"><b>%d</b><span>of %d you actually drove at the '
          "ball</span></div>" % (int(num(k, "took")), int(num(k, "n"))))
        a("<table>")
        a(stat_row("your win rate", "%.0f%%" % share(k, "win_rate")))
        a(stat_row("team win rate", "%.0f%%" % share(k, "team_win_rate")))
        a(stat_row("time to first contact", "%.2f s" % num(k, "ttc_mine")))
        a(stat_row("short at first touch", "%.0f uu" % num(k, "gap_uu")))
        a(stat_row("goals within 10 s", "%d for / %d against"
                   % (int(num(k, "goals_for")), int(num(k, "goals_against")))))
        a("</table></div>")

    rec = D("recovery")
    if rec.get("ok"):
        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0
        # NOT `share` -- that is the module-level fraction/percent accessor,
        # and shadowing it here broke every boost row further down the page.
        home_pct = avg(rec.get("share") or [])
        mate_pct = avg(rec.get("mate_share") or [])
        closed = avg(rec.get("closed") or [])
        mclosed = avg(rec.get("mate_closed") or [])
        a('<div class="card"><h2>Recovery &middot; do you turn and go home</h2>')
        a('<div class="hero"><b>%+.0f%%</b><span>of your movement is homeward '
          "in the 4 s before a goal &mdash; teammates %+.0f%%</span></div>"
          % (home_pct, mate_pct))
        a("<table>")
        a(stat_row("ground you recovered", "%+.0f uu" % closed,
                   "teammates %+.0f" % mclosed))
        a(stat_row("drove away from your net", "%d of %d"
                   % (int(num(rec, "away")), int(num(rec, "n")))))
        a(stat_row("goal-side 4 s before", "%d of %d"
                   % (int(num(rec, "goalside_start")), int(num(rec, "n")))))
        a(stat_row("goal-side at the goal", "%d of %d"
                   % (int(num(rec, "goalside_end")), int(num(rec, "n")))))
        a(stat_row("boost you had", "%.0f" % avg(rec.get("boost") or []),
                   "not a resource problem"))
        a("</table></div>")

    con = (D("goals").get("conceded") or {})
    if con:
        a('<div class="card"><h2>When you concede</h2>')
        a('<div class="hero"><b>%.0f</b><span>uu from your own net, on '
          "average, at the moment it goes in</span></div>" % num(con, "dist_own"))
        a("<table>")
        a(stat_row("goals conceded", "%d" % int(num(con, "n"))))
        a(stat_row("you were last man", "%d of %d"
                   % (int(num(con, "last_man")), int(num(con, "last_man_of")))))
        a(stat_row("caught in their half", "%d" % int(num(con, "in_opp_half"))))
        a(stat_row("goal-side 2 s earlier", "%d of %d"
                   % (int(num(con, "goalside")), int(num(con, "lead_n")))))
        a(stat_row("boost in hand", "%.0f" % num(con, "boost")))
        a("</table></div>")

    a("</div>")

    a("</div>")   # end overview panel

    a('<div class="panel" data-panel="lobby" hidden>')
    lob = sec(latest, "lobby")
    if lob.get("players"):
        a('<div class="sect"><h2>Everyone in your last match</h2></div>')
        a('<div class="card scroll"><table>')
        a("<tr><td><b>player</b></td><td><b>score &middot; speed &middot; "
          "first-ball &middot; boost &middot; air &middot; demos</b></td></tr>")
        order = sorted(lob["players"].items(), key=lambda kv: -kv[1].get("score", 0))
        for nm, pl in order:
            side = ("you" if nm == lob.get("me")
                    else "mate" if pl.get("mine") else "opponent")
            col = ("var(--accent)" if side == "you"
                   else "var(--blue)" if side == "mate" else "var(--orange)")
            a('<tr><td><span style="color:%s">%s</span> %s</td>'
              "<td>%d &middot; %.0f &middot; %.1f%% &middot; %.0f &middot; "
              "%.1f%% &middot; %d</td></tr>"
              % (col, esc(side), esc(nm), pl.get("score", 0), pl.get("speed", 0),
                 pl.get("first_man", 0), pl.get("boost_held", 0),
                 pl.get("airborne", 0), pl.get("demos", 0)))
        a("</table></div>")
        for ln in ((latest.get("sections", {}).get("lobby") or {}).get("lines") or []):
            if "top of the lobby" in ln:
                a('<p style="color:var(--ink-2);max-width:70ch">%s</p>' % esc(ln.strip()))
    else:
        a('<p style="color:var(--ink-3)">No lobby data in the latest match.</p>')

    if grids:
        a('<div class="sect"><h2>Where you live</h2></div>')
        a('<div class="cards">')
        for size, g in grids:
            a(heat_block(g["grid"], g["grid_max"],
                         "%dv%d &middot; %d matches" % (size, size, g["n"])))
        a("</div>")

    a("</div>")   # end lobby panel

    a('<div class="panel" data-panel="replay" hidden>')
    track = payload.get("track")
    if track and track.get("frames"):
        a('<div class="sect"><h2>Replay &middot; %s</h2></div>'
          % esc(latest.get("date") or ""))
        a('<div class="viewer"><div>')
        a('<canvas id="pitch"></canvas>')
        a('<div class="ctrl">'
          '<button id="guide" type="button">Guided run-through</button>'
          '<button id="play" type="button" class="ghost">Play</button>'
          '<input id="scrub" type="range" min="0" value="0" '
          'aria-label="scrub the replay">'
          '<span class="clock" id="clock">0.0s</span></div>')
        a('<div class="live" id="live" hidden></div>')
        a('<div class="caption" id="caption" hidden></div>')
        a('<div class="legend">'
          '<span><i class="sw blue"></i>your team</span>'
          '<span><i class="sw orange"></i>opponents</span>'
          '<span><i class="sw ring"></i>you</span>'
          '<span><i class="sw good"></i>goal-side of the ball</span>'
          '<span><i class="sw bad"></i>beaten &mdash; get home</span>'
          '<span><i class="sw ghost"></i>where you should be</span>'
          "</div>")
        a(shadow_note(track))
        a('<p style="color:var(--ink-3);font-size:.82rem;max-width:66ch">'
          "Colour is the team, never the player &mdash; you are your side's "
          "colour with a ring around you. Guided run-through skips quiet play "
          "at 4x and drops to 0.25x on each moment, with what went wrong on "
          "screen. Your net is always the near one whichever side you played, "
          "and everything casts a shadow when airborne so height reads.</p>")
        a("</div>")
        a('<div><div class="eyebrow" style="margin-bottom:8px">Jump to</div>'
          '<div class="moments">')
        for mo in track.get("moments", []):
            a('<button type="button" class="%s" data-t="%s" data-kind="%s" '
              'data-title="%s" data-text="%s"><b>%d:%02d</b>%s</button>'
              % (esc(mo["kind"]), mo["t"], esc(mo["kind"]),
                 esc(mo.get("title", "")), esc(mo["text"]),
                 int(mo["t"] // 60), int(mo["t"] % 60),
                 esc(mo.get("title") or mo["text"])))
        a("</div></div></div>")
        a("<script>window.__TRACK__=%s;</script>" % json.dumps(track))
    else:
        a('<p style="color:var(--ink-3)">No replay track for the latest match.</p>')
    a("</div>")   # end replay panel

    if tips:
        a('<div class="sect"><h2>Everything worth working on</h2></div>')
        a('<ul class="tips">')
        for tx in tips:
            a("<li>%s</li>" % esc(tx))
        a("</ul>")

    a("<footer>Built from %d local replay files with <code>coach/analyse.py</code>. "
      "Percentages are weighted by the live seconds each match contributed, so a "
      "short forfeit cannot outweigh a full game. Playlists are never averaged "
      "together &mdash; an even rotation share is 50%% in 2s and 33%% in 3s. "
      "Pointers come from the most recent match only. Latest: %s.</footer>"
      % (career["n"], esc(latest.get("date") or "?")))
    a("</div>")
    # Last, so the elements it wires up already exist when it runs.
    a(VIEWER_JS)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", help="output of analyse.py --json")
    ap.add_argument("--last", type=int, default=4)
    ap.add_argument("--player")
    ap.add_argument("--out", default="coach.html")
    args = ap.parse_args()

    if args.json:
        payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        tmp = Path(tempfile.gettempdir()) / "rl_coach_payload.json"
        cmd = [sys.executable, str(ROOT / "coach" / "analyse.py"),
               "--last", str(args.last), "--json", str(tmp)]
        if args.player:
            cmd += ["--player", args.player]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:])
            return 1
        payload = json.loads(tmp.read_text(encoding="utf-8"))

    rank_path = ROOT / "coach" / "rank.json"
    if rank_path.is_file():
        try:
            payload["rank"] = json.loads(rank_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("could not read rank.json (%s); continuing without it" % e)

    Path(args.out).write_text(build(payload), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
