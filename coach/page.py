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
"""


VIEWER_JS = r"""
<script>
(function () {
  var T = window.__TRACK__;
  var tabs = [].slice.call(document.querySelectorAll('.tabs button'));
  var panels = [].slice.call(document.querySelectorAll('.panel'));
  var onShow = null;
  function show(name) {
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
  var i = 0, timer = null, guided = false, rate = 1;
  // Always defend the near goal, whichever side was actually played --
  // otherwise half your replays render backwards.
  var flip = T.my_team === 1 ? -1 : 1;

  function css(v) {
    return getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  }

  // --- a small hand-rolled perspective camera ---------------------------
  // No 3D library: the scene is a flat pitch plus a dozen boxes, and a
  // library would be 600 KB inlined to do what forty lines of projection do.
  // Solved numerically rather than eyeballed: a parameter search over camera
  // distance, height, pitch and focal length, maximising how much of the frame
  // the pitch fills while keeping all four corners, both goal frames and the
  // ceiling on screen. The first hand-picked values put the near goal line and
  // both near corners below the bottom edge.
  var CAM = {y: -9000, z: 8000, pitch: 0.80, f: 900};
  function project(x, y, z) {
    var rx = x * flip, ry = y * flip;
    var dy = ry - CAM.y, dz = z - CAM.z;
    var c = Math.cos(CAM.pitch), s = Math.sin(CAM.pitch);
    var fwd = dy * c - dz * s;          // depth into the screen
    var up = dy * s + dz * c;           // height on screen
    if (fwd < 200) return null;         // behind or too near the camera
    var sc = CAM.f / fwd;
    return {x: cv.width / 2 + rx * sc, y: cv.height * 0.50 - up * sc,
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
    // centre circle
    var prev = null;
    for (var a = 0; a <= 32; a++) {
      var th = a / 32 * 6.2832;
      var pt = [Math.cos(th) * 920, Math.sin(th) * 920];
      if (prev) line(prev, pt, lin, 1);
      prev = pt;
    }
    // goals, drawn as real 3D frames so the perspective reads
    [[-hh, css('--good'), 'your net'], [hh, css('--bad'), 'their net']]
      .forEach(function (g) {
        var y = g[0], col = g[1];
        line([-893, y, 0], [-893, y, 642], col, 3);
        line([893, y, 0], [893, y, 642], col, 3);
        line([-893, y, 642], [893, y, 642], col, 3);
        line([-893, y, 0], [893, y, 0], col, 3);
      });
  }

  function drawCar(x, y, z, head, col, me, name) {
    var p = project(x, y, z + 17);
    if (!p) return;
    var len = 118 * p.s, wid = 84 * p.s, hgt = 36 * p.s;
    ctx.save();
    ctx.translate(p.x, p.y);
    // Yaw only: the camera looks down the pitch, so heading is the rotation
    // that reads. Roll and pitch of the car are invisible at this scale.
    ctx.rotate(-(head * Math.PI / 180) * flip * 0.55);
    ctx.fillStyle = col;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-len / 2, -wid / 2, len, wid, 3 * p.s);
    else ctx.rect(-len / 2, -wid / 2, len, wid);
    ctx.fill();
    ctx.globalAlpha = 0.55;
    ctx.fillRect(-len / 6, -wid / 2, len / 2.6, wid);
    ctx.restore();
    ctx.globalAlpha = 1;
    if (me) {
      ctx.strokeStyle = css('--ink');
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(len, wid) * 0.75, 0, 6.2832);
      ctx.stroke();
    }
    if (z > 120) {                       // shadow, so height is readable
      var g = project(x, y, 0);
      if (g) {
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = css('--ink-3');
        ctx.beginPath(); ctx.ellipse(g.x, g.y, len / 2, wid / 3, 0, 0, 6.2832);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    ctx.fillStyle = me ? css('--accent') : css('--ink-3');
    ctx.font = Math.max(9, 11 * p.s * 40) + 'px ui-monospace, monospace';
    ctx.fillText(name.slice(0, 12), p.x + len * 0.7, p.y - wid * 0.6);
  }

  function draw() {
    var f = T.frames[i];
    ctx.clearRect(0, 0, cv.width, cv.height);
    drawPitch();

    // Painter's algorithm: furthest first, or near cars vanish behind far ones.
    var items = [];
    for (var c = 0; c < N; c++) {
      var o = 4 + c * STRIDE;
      var x = f[o], y = f[o + 1];
      if (x === 0 && y === 0) continue;
      items.push({d: (y * flip), kind: 'car', c: c, x: x, y: y,
                  z: f[o + 2], h: f[o + 3]});
    }
    items.push({d: f[2] * flip, kind: 'ball', x: f[1], y: f[2], z: f[3]});
    items.sort(function (a, b) { return b.d - a.d; });

    items.forEach(function (it) {
      if (it.kind === 'ball') {
        var p = project(it.x, it.y, it.z);
        if (!p) return;
        if (it.z > 120) {
          var g = project(it.x, it.y, 0);
          if (g) {
            ctx.globalAlpha = 0.25; ctx.fillStyle = css('--ink-3');
            ctx.beginPath(); ctx.ellipse(g.x, g.y, 92 * p.s, 40 * p.s, 0, 0, 6.2832);
            ctx.fill(); ctx.globalAlpha = 1;
          }
        }
        ctx.fillStyle = css('--ink');
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(3, 92 * p.s), 0, 6.2832);
        ctx.fill();
      } else {
        var mine = T.teams[it.c] === T.my_team;
        drawCar(it.x, it.y, it.z, it.h,
                it.c === T.me ? css('--accent')
                              : (mine ? css('--blue') : css('--orange')),
                it.c === T.me, T.names[it.c]);
      }
    });

    clock.textContent = f[0].toFixed(1) + 's' + (rate !== 1 ? '  ' + rate + 'x' : '');
  }

  function seek(n) {
    i = Math.max(0, Math.min(T.frames.length - 1, n));
    range.value = i;
    draw();
  }
  // Seek by the timestamp each frame carries, NOT t * hz: the downsample step
  // is an integer, so the real rate drifts from the nominal one and three
  // minutes in that is an 18 second miss.
  function frameAt(t) {
    var lo = 0, hi = T.frames.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (T.frames[mid][0] < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  var STEP_MS = 1000 * (T.frames[T.frames.length - 1][0] - T.frames[0][0]) /
                Math.max(T.frames.length - 1, 1);

  function nextMoment(t) {
    for (var k = 0; k < T.moments.length; k++) {
      if (T.moments[k].t > t + 0.05) return T.moments[k];
    }
    return null;
  }
  function momentAt(t) {
    for (var k = 0; k < T.moments.length; k++) {
      var m = T.moments[k];
      if (t >= m.t && t <= m.t + (T.slow || 6)) return m;
    }
    return null;
  }

  function tick() {
    if (i >= T.frames.length - 1) { stop(); return; }
    seek(i + 1);
    var t = T.frames[i][0];
    if (guided) {
      var m = momentAt(t);
      if (m) {
        rate = 0.25;
        caption.hidden = false;
        caption.innerHTML = '<b>' + m.title + '</b>' + m.text;
        caption.className = 'caption ' + m.kind;
      } else {
        rate = 4;
        caption.hidden = true;
        // Nothing is happening -- skip ahead to the next thing that is.
        var nm = nextMoment(t);
        if (nm && nm.t - t > 3) seek(frameAt(nm.t - 1.5));
      }
      restart();
    }
  }
  function restart() {
    if (!timer) return;
    clearInterval(timer);
    timer = setInterval(tick, STEP_MS / rate);
  }
  function stop() {
    clearInterval(timer); timer = null; guided = false; rate = 1;
    play.textContent = 'Play'; guide.textContent = 'Guided run-through';
    caption.hidden = true;
  }
  function start() {
    if (timer) clearInterval(timer);
    timer = setInterval(tick, STEP_MS / rate);
  }

  play.addEventListener('click', function () {
    if (timer && !guided) { stop(); return; }
    guided = false; rate = 1; caption.hidden = true;
    play.textContent = 'Pause'; guide.textContent = 'Guided run-through';
    start();
  });
  guide.addEventListener('click', function () {
    if (timer && guided) { stop(); return; }
    guided = true; rate = 4;
    play.textContent = 'Play'; guide.textContent = 'Stop';
    if (T.moments.length) seek(frameAt(Math.max(0, T.moments[0].t - 1.5)));
    start();
  });
  range.max = T.frames.length - 1;
  range.addEventListener('input', function () {
    stop(); seek(+range.value);
  });
  [].slice.call(document.querySelectorAll('.moments button')).forEach(function (b) {
    b.addEventListener('click', function () {
      stop();
      seek(frameAt(+b.dataset.t));
      caption.hidden = false;
      caption.className = 'caption ' + b.dataset.kind;
      caption.innerHTML = '<b>' + b.dataset.title + '</b>' + b.dataset.text;
    });
  });

  function size() {
    var w = cv.parentElement.clientWidth;
    if (!w) return;
    cv.width = w;
    cv.height = Math.round(Math.min(w * 0.72, 560));
    draw();
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
        # ponytail: meta refresh, not a websocket. The page is a file:// URL, so
        # fetch() to poll Last-Modified is CORS-blocked and any live-reload
        # needs a server. Browsers restore scroll position on reload, so this
        # costs nothing visible. Only the watcher sets it -- a published
        # snapshot would just re-request itself forever.
        a('<meta http-equiv="refresh" content="%d">' % refresh)
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
        a('<div class="caption" id="caption" hidden></div>')
        a('<p style="color:var(--ink-3);font-size:.82rem;max-width:64ch">'
          "Guided run-through skips the quiet stretches at 4x and drops to "
          "0.25x on every moment worth reviewing, with what went wrong on "
          "screen. Your net is always the near one whichever side you played. "
          "Cars and ball cast a shadow when airborne, so height reads.</p>")
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
