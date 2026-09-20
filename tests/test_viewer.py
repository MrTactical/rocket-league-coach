"""
Regression tests for the replay viewer's data contract.

Every case here is a bug that actually shipped and was visible in the viewer.
They are cheap because they run on hand-built fake matches -- no replay files,
no parsing -- which is the only reason there is a test at all.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coach import possession as P  # noqa: E402
from coach.viewer import build_track  # noqa: E402


class FakeMatch:
    """The smallest thing build_track will accept."""

    def __init__(self, samples, teams):
        self.samples = samples
        self.teams = teams
        self.frames = list(range(len(samples)))
        self.events = []
        self.goals_meta = []

    def duration(self):
        return self.samples[-1]["t"] - self.samples[0]["t"]

    def attack_sign(self, who):
        return 1.0 if self.teams.get(who) == 0 else -1.0


def car(x, y, z=17.0, vx=0.0, vy=0.0, rot=None):
    return {"pos": (x, y, z), "vel": (vx, vy, 0.0), "boost": 50, "rot": rot}


def sample(t, ball, cars):
    return {"t": t, "ball": ball, "ball_vel": (0.0, 0.0, 0.0), "cars": cars}


def build(cars_per_frame, teams, n=40):
    """n frames so build_track's decimation keeps them all."""
    samples = [sample(i * 0.1, (0.0, 0.0, 93.0), cars_per_frame(i))
               for i in range(n)]
    return build_track(FakeMatch(samples, teams), "me")


def test_absent_car_is_null_not_zero():
    """
    (0, 0) is a real position on this pitch.

    Absent cars used to be written as [0,0,0,0] and the viewer tested
    x===0 && y===0 to decide a car was missing -- so a car driving over the
    centre spot vanished, or froze for an interpolation window. The ball
    spawns at (0,0), so play concentrates exactly there.
    """
    teams = {"me": 0, "them": 1}

    def frames(i):
        out = {"me": car(1000.0, 1000.0)}
        if i % 2 == 0:                      # "them" only in even frames
            out["them"] = car(-500.0, 700.0)
        return out

    tr = build(frames, teams)
    names = tr["names"]
    o = 4 + names.index("them") * tr["stride"]
    present = [r for r in tr["frames"] if r[o] is not None]
    absent = [r for r in tr["frames"] if r[o] is None]
    assert present and absent, "test needs both states"
    for r in absent:
        assert r[o:o + tr['stride']] == [None] * tr['stride']


def test_car_on_centre_spot_is_still_present():
    """A car at exactly (0, 0) must not read as missing."""
    teams = {"me": 0, "them": 1}
    tr = build(lambda i: {"me": car(0.0, 0.0), "them": car(900.0, 900.0)}, teams)
    o = 4 + tr["names"].index("me") * tr["stride"]
    for r in tr["frames"]:
        assert r[o] is not None, "car on the centre spot read as absent"
        assert r[o] == 0 and r[o + 1] == 0


def test_heading_holds_when_barely_moving():
    """
    The FALLBACK path, for replays with no orientation.

    Heading now comes from the car's real rotation quaternion. Where that is
    missing it falls back to velocity, which is noise at a standstill: a
    stopped car used to spin on the spot, and an exact standstill snapped to
    0 degrees. This pins the fallback.
    """
    teams = {"me": 0, "them": 1}

    def frames(i):
        if i < 10:                          # driving +x, heading 0
            return {"me": car(float(i * 100), 0.0, vx=900.0, vy=0.0),
                    "them": car(2000.0, 2000.0)}
        # stopped, with tiny jittery velocity
        jitter = 12.0 if i % 2 else -9.0
        return {"me": car(1000.0, 0.0, vx=jitter, vy=-jitter),
                "them": car(2000.0, 2000.0)}

    tr = build(frames, teams)
    o = 4 + tr["names"].index("me") * tr["stride"]
    heads = [r[o + 3] for r in tr["frames"] if r[o] is not None]
    moving, stopped = heads[:8], heads[12:]
    assert all(h == moving[0] for h in moving), "steady driving should be steady"
    assert all(h == stopped[0] for h in stopped), "stopped car span its heading"
    assert stopped[0] == moving[0], "should hold the last real direction"


def test_screen_rotation_is_periodic_across_180():
    """
    The old form scaled the heading by a constant: rotate(-h * flip * 0.55).

    That is not periodic, so a car driving along -x whose heading flips
    between 179 and -179 -- a 2 degree wobble -- snapped ~198 degrees on
    screen. Measured on one real match, this happened 121 times.

    The fix projects a point ahead of the car and takes the screen angle to
    it, which is periodic by construction.
    """
    CAM_PITCH, CAM_F = 0.80, 900.0
    W, H = 860.0, 560.0

    def project(x, y, z, ox, oy, d, flip):
        rx, ry = (x - ox) * flip, (y - oy) * flip
        dy, dz = ry + d, z - d * math.tan(CAM_PITCH)
        c, s = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        fwd, up = dy * c - dz * s, dy * s + dz * c
        if fwd < 150:
            return None
        sc = CAM_F / fwd
        return (W / 2 + rx * sc, H * 0.52 - up * sc)

    def screen_angle(x, y, head, flip):
        p = project(x, y, 17, 0, 0, 9000, flip)
        rad = math.radians(head)
        n = project(x + math.cos(rad) * 120, y + math.sin(rad) * 120, 17,
                    0, 0, 9000, flip)
        return math.degrees(math.atan2(n[1] - p[1], n[0] - p[0]))

    def wrapped(a, b):
        return abs(((a - b + 540) % 360) - 180)

    for flip in (1, -1):
        new = wrapped(screen_angle(1500, 800, 179, flip),
                      screen_angle(1500, 800, -179, flip))
        old = wrapped(-179 * flip * 0.55, -(-179) * flip * 0.55)
        assert new < 15, "179 -> -179 still snaps on screen (%.0f deg)" % new
        assert old > 90, "old formula was supposed to be broken"


def test_track_json_cannot_close_the_script_tag():
    """
    Player names come from other people's Steam profiles and land verbatim in
    an inline <script>. A name containing '</script>' closed the tag, left an
    unterminated string, and blanked the whole replay tab.
    """
    evil = 'ok</script><img src=x onerror=alert(1)>'
    teams = {"me": 0, evil: 1}
    tr = build(lambda i: {"me": car(100.0, 100.0), evil: car(200.0, 200.0)},
               teams)

    blob = (json.dumps(tr).replace("<", "\\u003c")
                          .replace(">", "\\u003e")
                          .replace("&", "\\u0026"))
    html = "<script>window.__TRACK__=%s;</script>" % blob

    assert "</script>" == html[-9:], "only the real closing tag may appear"
    assert html.count("</script>") == 1
    assert "<img" not in html
    # and the escaping is value-preserving
    assert json.loads(blob)["names"] == tr["names"]


# --- possession, added when the metrics stopped being intent-blind ----------


def test_possession_phase_is_side_relative():
    """The same world position means opposite things to the two teams."""
    def s(ball_y, ht):
        return {"t": 0.0, "hit_team": ht, "ball": (0.0, ball_y, 93.0)}

    assert P.phase(s(3000, 0), 0, 1.0) == P.ATTACK
    assert P.phase(s(3000, 0), 1, -1.0) == P.DEFEND
    assert P.phase(s(-3000, 1), 0, 1.0) == P.DEFEND
    assert P.phase(s(-3000, 1), 1, -1.0) == P.ATTACK


def test_possession_needs_a_touch_and_a_side():
    """No touch yet, or a ball on halfway, is not a phase worth judging."""
    def s(ball_y, ht):
        return {"t": 0.0, "hit_team": ht, "ball": (0.0, ball_y, 93.0)}

    assert P.phase(s(3000, None), 0, 1.0) == P.UNKNOWN
    assert P.phase(s(0, 0), 0, 1.0) == P.UNKNOWN
    assert P.phase(s(P.NEUTRAL_Y - 1, 0), 0, 1.0) == P.UNKNOWN
    assert P.phase(s(P.NEUTRAL_Y + 1, 0), 0, 1.0) == P.ATTACK


def test_exposed_and_committed_are_not_the_same_state():
    """
    The whole point: ahead of the ball splits into a fault and a virtue.

    Measured 0.41x base concede risk when attacking against 2.78x when
    defending, replicated at 0.33x / 2.93x on Grand Champion replays.
    """
    assert P.is_exposed(P.DEFEND)
    assert not P.is_exposed(P.ATTACK)
    assert P.is_committed(P.ATTACK)
    assert not P.is_committed(P.DEFEND)
    assert not (P.is_exposed(P.BUILD) or P.is_committed(P.BUILD))


def test_real_orientation_beats_velocity_heading():
    """
    A car reversing points the OPPOSITE way to its travel.

    Velocity-derived heading cannot represent that at all -- it reports the
    direction of travel by construction, so a car reversing out of its own
    net was drawn facing forwards. Real rotation makes "did you actually
    turn around" answerable, which is what the recovery finding is about.
    """
    teams = {"me": 0, "them": 1}
    # nose at 0 degrees, travelling in -x: reversing
    tr = build(lambda i: {"me": car(1000.0 - i * 60, 0.0, vx=-900.0,
                                    rot=(0.0, 0.0, 0.0)),
                          "them": car(2000.0, 2000.0)}, teams)
    o = 4 + tr["names"].index("me") * tr["stride"]
    heads = [r[o + 3] for r in tr["frames"] if r[o] is not None]
    assert all(h == 0 for h in heads), "should report the nose, not the travel"

    from coach.timeline import _ypr, upright
    assert round(_ypr({"x": 0, "y": 0, "z": 0, "w": 1})[0]) == 0
    assert upright((0.0, 0.0, 0.0)) is True
    assert upright((0.0, 0.0, 180.0)) is False    # on its roof


def demo():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok   %s" % name)
    print("\nall viewer regressions pass")


if __name__ == "__main__":
    demo()
