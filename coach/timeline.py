"""
Turn a replay into a frame-by-frame timeline every analyser can share.

One parse, one pass, one structure. Metric modules read this rather than each
re-walking the network stream, so a fix to the actor plumbing lands everywhere
at once.

Actor plumbing worth knowing, because none of it is obvious and all of it
silently yields nothing when you get it wrong:

  * A car's boost, jump, dodge and double-jump are NOT attributes of the car.
    Each is a separate component actor that points back at its car through
    `TAGame.CarComponent_TA:Vehicle`. Which component you are looking at comes
    from the archetype it spawned as.
  * A car finds its player through `Engine.Pawn:PlayerReplicationInfo`, and the
    PRI carries the name and team.
  * `Engine.PlayerReplicationInfo:Team` is an ACTOR ID, not 0/1. It differs per
    replay. `TAGame.Car_TA:TeamPaint` gives the real 0/1.
  * Byte-valued inputs are 0-255 with 128 as centre for steer, and 0-255 for
    throttle where 128 is neutral.
"""

from __future__ import annotations

import json
import datetime as _dt
import math
import os
import re
import subprocess
import sys
from pathlib import Path

def _ypr(q):
    """
    Quaternion -> (yaw, pitch, roll) in degrees.

    Yaw is the compass direction the nose points, measured the same way as
    atan2(vel_y, vel_x) so the two are directly comparable. Pitch is nose up,
    roll is barrel rotation -- together they say whether the car is on its
    wheels, which is what "am I actually able to drive right now" comes down
    to.
    """
    x, y, z, w = (q.get("x", 0.0), q.get("y", 0.0),
                  q.get("z", 0.0), q.get("w", 1.0))
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                  1.0 - 2.0 * (y * y + z * z)))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sp))
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z),
                                   1.0 - 2.0 * (x * x + y * y)))
    return (yaw, pitch, roll)


def upright(rot):
    """True when the car is on its wheels enough to drive normally."""
    if not rot:
        return None
    return abs(rot[1]) < 45.0 and abs(rot[2]) < 45.0


ROOT = Path(__file__).resolve().parents[1]

# PyInstaller unpacks bundled data to a temp dir and points sys._MEIPASS at it,
# so the source-tree layout does not exist inside a built exe.
_BUNDLE = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None
RRROCKET = ((_BUNDLE / "rrrocket.exe") if _BUNDLE
            else ROOT / "tools" / "rrrocket" / "rrrocket.exe")
DEMOS = Path(os.path.expanduser("~/Documents/My Games/Rocket League/TAGame/Demos"))

CAR_ARCHETYPE = "Archetypes.Car.Car_Default"
COMPONENTS = {
    "Archetypes.CarComponents.CarComponent_Boost": "boost",
    "Archetypes.CarComponents.CarComponent_Jump": "jump",
    "Archetypes.CarComponents.CarComponent_DoubleJump": "double_jump",
    "Archetypes.CarComponents.CarComponent_Dodge": "dodge",
    "Archetypes.CarComponents.CarComponent_FlipCar": "flip_car",
}

# Soccar geometry, in unreal units.
GOAL_Y = 5120.0
SIDE_X = 4096.0
CEILING = 2044.0
BALL_RADIUS = 92.75
SUPERSONIC = 2200.0


def parse(path) -> dict:
    """Run rrrocket over a replay and return the decoded JSON."""
    if not RRROCKET.is_file():
        raise SystemExit(
            "rrrocket not found at %s -- see coach/README for the download"
            % RRROCKET)
    r = subprocess.run([str(RRROCKET), "-n", str(path)],
                       capture_output=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b"").decode(errors="replace")[:300])
    return json.loads(r.stdout)


def recent_replays(n=1):
    import glob
    files = sorted(glob.glob(str(DEMOS / "*.replay")), key=os.path.getmtime)
    return files[-n:] if n else files


def canon(name):
    return "".join(c for c in (name or "").lower() if c.isalnum())


def vlen(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def flat_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Match:
    """A parsed replay: metadata, per-frame samples, and discrete events."""

    def __init__(self, d: dict, path=None):
        self.path = path
        self.props = d.get("properties") or {}
        self.objects = d["objects"]
        self.frames = d["network_frames"]["frames"]

        self.name = self.props.get("PlayerName")
        # MatchStartEpoch first, the "Date" string only as a fallback.
        #
        # They disagree, and Date is the one that lies. Saving a replay from
        # the in-game replay list writes a file whose Date property is NOT
        # when the match was played, so six matches saved after a session
        # came out ordered 15:30, 15:39, 15:47, 14:55, 15:01, 15:08 -- and
        # "your last match" showed a game from forty minutes earlier, with the
        # wrong lobby in it. The epoch reproduces the in-game match history
        # exactly, results and order.
        self.date = self.props.get("Date")
        ep = self.props.get("MatchStartEpoch")
        if ep:
            try:
                self.date = _dt.datetime.fromtimestamp(int(ep)).strftime(
                    "%Y-%m-%d %H-%M-%S")
            except (ValueError, OSError, OverflowError):
                pass          # keep the header string rather than lose a date
        self.team_size = int(self.props.get("TeamSize") or 0)
        self.score = (self.props.get("Team0Score") or 0,
                      self.props.get("Team1Score") or 0)
        self.forfeit = bool(self.props.get("bForfeit"))
        self.stats = self.props.get("PlayerStats") or []
        self.goals_meta = self.props.get("Goals") or []

        self.samples = []
        self.events = []          # {"t", "kind", "player", ...}
        self.teams = {}           # player name -> 0/1
        self._build()

    # -- construction ----------------------------------------------------

    def _build(self):
        objects = self.objects
        actor_arch = {}
        self.ball_live = False
        self.ctx = {}
        pri_name, pri_team = {}, {}
        pri_stat = {}          # live per-player counters (goals, saves, ping...)
        car_cam = {}           # air/flip counters, keyed by car actor
        cam_state = {}         # camera settings, keyed by CameraSettingsActor
        cam_pri = {}           # CameraSettingsActor -> PRI actor
        match_ctx = {}         # playlist, region, team size, server
        car_pri, car_team = {}, {}
        comp_car, comp_kind = {}, {}
        car_boost, car_input = {}, {}
        car_active = {}
        state = {}
        ball_actors = set()
        clock = None
        hit_team = None
        prev_boost = {}
        demo_count = {}
        prev_pos = {}

        for f in self.frames:
            t = f.get("time", 0.0)

            for a in f.get("new_actors", []):
                arch = objects[a["object_id"]]
                aid = a["actor_id"]
                actor_arch[aid] = arch
                if arch in COMPONENTS:
                    comp_kind[aid] = COMPONENTS[arch]
                elif arch.startswith("Archetypes.Ball."):
                    ball_actors.add(aid)

            for u in f.get("updated_actors", []):
                aid = u["actor_id"]
                full = objects[u["object_id"]]
                short = full.split(":")[-1]
                val = u["attribute"]

                if short == "PlayerName":
                    pri_name[aid] = val.get("String")
                elif short == "Team" and "PlayerReplicationInfo" in full:
                    act = (val.get("ActiveActor") or {}).get("actor")
                    if act is not None and act >= 0:
                        pri_team[aid] = act
                elif short == "PlayerReplicationInfo":
                    act = (val.get("ActiveActor") or {}).get("actor")
                    if act is not None and act >= 0:
                        car_pri[aid] = act
                elif short == "TeamPaint":
                    tp = val.get("TeamPaint") or {}
                    if "team" in tp:
                        car_team[aid] = tp["team"]
                elif short == "Vehicle":
                    act = (val.get("ActiveActor") or {}).get("actor")
                    if act is not None and act >= 0:
                        comp_car[aid] = act
                elif short == "ReplicatedRBState":
                    rb = val.get("RigidBody") or {}
                    loc, lv = rb.get("location"), rb.get("linear_velocity")
                    av = rb.get("angular_velocity")
                    rot = rb.get("rotation")
                    s = state.setdefault(aid, {})
                    if loc:
                        s["pos"] = (loc["x"], loc["y"], loc["z"])
                    if lv:
                        s["vel"] = (lv["x"], lv["y"], lv["z"])
                    if av:
                        s["ang"] = (av["x"], av["y"], av["z"])
                    if rot:
                        # Orientation, as a quaternion. Where a car POINTS is
                        # not where it is going: the difference between the
                        # two is the whole of "did you actually turn around,
                        # or are you just drifting backwards". Everything
                        # downstream used to infer heading from velocity,
                        # which cannot tell those apart and is pure noise at
                        # a standstill.
                        s["rot"] = _ypr(rot)
                    s["sleeping"] = rb.get("sleeping", False)
                elif short in ("ReplicatedBoost", "ReplicatedBoostAmount"):
                    # Two spellings across replay eras, and they carry
                    # different shapes. Replays from 2023 use
                    # `ReplicatedBoostAmount` as a bare Byte; current ones use
                    # `ReplicatedBoost` wrapping a struct. Handling only the
                    # modern one reports every older match as zero boost --
                    # which looks like a player who never picked up a pad
                    # rather than like missing data.
                    amt = None
                    b = val.get("ReplicatedBoost")
                    if isinstance(b, dict):
                        amt = b.get("boost_amount")
                    if amt is None and isinstance(val.get("Byte"), int):
                        amt = val["Byte"]
                    car = comp_car.get(aid)
                    if amt is not None and car is not None:
                        car_boost[car] = amt * 100.0 / 255.0
                elif short == "ReplicatedThrottle":
                    car_input.setdefault(aid, {})["throttle"] = (
                        (val.get("Byte", 128) - 128) / 127.0)
                elif short == "ReplicatedSteer":
                    car_input.setdefault(aid, {})["steer"] = (
                        (val.get("Byte", 128) - 128) / 127.0)
                elif short == "bReplicatedHandbrake":
                    car_input.setdefault(aid, {})["handbrake"] = bool(
                        val.get("Boolean"))
                elif short == "bDriving":
                    car_input.setdefault(aid, {})["driving"] = bool(
                        val.get("Boolean"))

                # --- camera. Ball cam on or off is a real coaching signal and
                # it is replicated per car: bUsingSecondaryCamera is TRUE when
                # ball cam is OFF. CameraYaw/Pitch say where the player is
                # actually looking, which is not where the car points.
                elif short == "bUsingSecondaryCamera":
                    # The "secondary" camera IS ball cam. Reading this
                    # inverted put every player in this lobby at 5-19% ball
                    # cam, which is not a thing a Champion does -- the true
                    # figures are 81-96%, and the implausible number is what
                    # gave the sign away.
                    cam_state.setdefault(aid, {})["ballcam"] = bool(
                        val.get("Boolean"))
                elif short == "bUsingBehindView":
                    cam_state.setdefault(aid, {})["behind"] = bool(
                        val.get("Boolean"))
                elif short == "CameraYaw":
                    cam_state.setdefault(aid, {})["cam_yaw"] = (
                        (val.get("Byte", 128) - 128) / 127.0 * 180.0)
                elif short == "CameraPitch":
                    cam_state.setdefault(aid, {})["cam_pitch"] = (
                        (val.get("Byte", 128) - 128) / 127.0 * 90.0)
                elif short == "PRI":
                    # Camera settings live on their own actor, which points at
                    # the player through this. Keying them by the car actor
                    # instead -- which is what the first attempt did -- simply
                    # never matched, and ball-cam usage read as 0% of every
                    # frame of every match.
                    act = (val.get("ActiveActor") or {}).get("actor")
                    if act is not None and act >= 0:
                        cam_pri[aid] = act

                # --- air state. DodgesRefreshedCounter rises on a flip reset;
                # AirActivateCount counts air-dodge activations. Both only
                # ever go up, so only a RISE is an event.
                elif short == "DodgesRefreshedCounter":
                    car = comp_car.get(aid, aid)
                    n = val.get("Int")
                    if isinstance(n, int):
                        prev = car_cam.setdefault(car, {}).get("resets_raw")
                        if prev is not None and n > prev:
                            self.events.append({"t": t, "kind": "flip_reset",
                                                "car": car})
                        car_cam[car]["resets_raw"] = n
                elif short == "AirActivateCount":
                    car = comp_car.get(aid, aid)
                    n = val.get("Int")
                    if isinstance(n, int):
                        car_cam.setdefault(car, {})["air_activations"] = n

                # --- live per-player counters. The header carries only final
                # totals; these are replicated as they happen, so they can be
                # placed on the timeline.
                elif short in ("MatchScore", "MatchGoals", "MatchSaves",
                               "MatchShots", "MatchAssists", "CarDemolitions",
                               "SelfDemolitions", "Ping", "TotalGameTimePlayed",
                               "SteeringSensitivity"):
                    v = val.get("Int")
                    if v is None:
                        v = val.get("Byte")
                    if v is None:
                        v = val.get("Float")
                    if v is not None:
                        pri_stat.setdefault(aid, {})[short] = v

                # --- match context, once per match ---
                elif short in ("ReplicatedGamePlaylist", "MaxTeamSize",
                               "ReplicatedServerRegion", "ServerName"):
                    match_ctx[short] = (val.get("Int") if val.get("Int")
                                        is not None else val.get("String"))
                elif short == "bBallHasBeenHit":
                    # Authoritative kickoff state. Everything downstream was
                    # inferring it from "ball within 100 uu of the centre spot
                    # and nearly stationary", which is a guess that also fires
                    # on a ball that happens to stop there in open play.
                    self.ball_live = bool(val.get("Boolean"))
                elif short == "RoundNum":
                    n = val.get("Int")
                    if isinstance(n, int):
                        match_ctx["round"] = n
                elif short == "ReplicatedActive":
                    kind = comp_kind.get(aid)
                    car = comp_car.get(aid)
                    if kind and car is not None:
                        # Odd values mean active; the low bit toggles per use.
                        car_active.setdefault(car, {})[kind] = bool(
                            val.get("Byte", 0) % 2)
                elif short == "DodgeTorque":
                    car = comp_car.get(aid)
                    if car is not None:
                        self.events.append(
                            {"t": t, "kind": "dodge", "car": car})
                elif short == "DoubleJumpImpulse":
                    car = comp_car.get(aid)
                    if car is not None:
                        self.events.append(
                            {"t": t, "kind": "double_jump", "car": car})
                elif short == "NewReplicatedPickupData":
                    pk = val.get("PickupNew") or {}
                    if pk.get("picked_up"):
                        self.events.append({"t": t, "kind": "pickup",
                                            "car": pk.get("instigator")})
                elif short == "HitTeamNum":
                    hit_team = val.get("Byte")
                elif short == "SecondsRemaining":
                    clock = val.get("Int")
                elif short == "MatchDemolishes":
                    # The counter is RE-SENT constantly, not only when it
                    # changes: one match produced 92 events for 5 actual demos.
                    # Only a rise in the count is a demolition.
                    n = val.get("Int")
                    if isinstance(n, int) and n > demo_count.get(aid, 0):
                        demo_count[aid] = n
                        self.events.append({"t": t, "kind": "demolish",
                                            "pri": aid})

            for aid in f.get("deleted_actors", []):
                state.pop(aid, None)

            ball = None
            for b in ball_actors:
                s = state.get(b)
                if s and "pos" in s:
                    ball = s
                    break
            if ball is None:
                continue

            pri_cam = {}
            for cam_aid, pri_aid in cam_pri.items():
                if cam_aid in cam_state:
                    pri_cam[pri_aid] = cam_state[cam_aid]

            cars = {}
            for aid, arch in actor_arch.items():
                if arch != CAR_ARCHETYPE:
                    continue
                s = state.get(aid)
                if not s or "pos" not in s:
                    continue
                pri = car_pri.get(aid)
                nm = pri_name.get(pri) if pri is not None else None
                if not nm:
                    continue
                team = car_team.get(aid, pri_team.get(pri))
                if team is not None and team in (0, 1):
                    self.teams[nm] = team
                inp = car_input.get(aid, {})
                act = car_active.get(aid, {})
                b = car_boost.get(aid)
                cars[nm] = {
                    "actor": aid,
                    "pos": s["pos"],
                    "vel": s.get("vel", (0.0, 0.0, 0.0)),
                    "ang": s.get("ang", (0.0, 0.0, 0.0)),
                    "rot": s.get("rot"),
                    "ballcam": pri_cam.get(pri, {}).get("ballcam"),
                    "cam_yaw": pri_cam.get(pri, {}).get("cam_yaw"),
                    "cam_pitch": pri_cam.get(pri, {}).get("cam_pitch"),
                    "flip_resets": car_cam.get(aid, {}).get("resets_raw"),
                    "air_acts": car_cam.get(aid, {}).get("air_activations"),
                    "live": dict(pri_stat.get(pri, {})),
                    "boost": b,
                    "throttle": inp.get("throttle", 0.0),
                    "steer": inp.get("steer", 0.0),
                    "handbrake": inp.get("handbrake", False),
                    "driving": inp.get("driving", True),
                    "boosting": act.get("boost", False),
                    "jumping": act.get("jump", False),
                }
                if b is not None:
                    prev = prev_boost.get(aid)
                    if prev is not None and b > prev + 5.0:
                        self.events.append({"t": t, "kind": "boost_gain",
                                            "car": aid, "amount": b - prev,
                                            "big": (b - prev) > 40.0})
                    prev_boost[aid] = b

            # A car cannot cross 1000uu between two samples under its own
            # power, so a jump that big is a respawn. But a KICKOFF respawns
            # everyone at once, and counting those gave 23 "demos taken" in a
            # match with 5 actual demolitions. One or two cars jumping is a
            # demo; three or more is the whistle.
            # ...and a demo respawns you at your own back wall, so require the
            # landing spot to be deep. Without this, mid-pitch replay stutters
            # counted as demolitions.
            jumped = [nm for nm, c in cars.items()
                      if prev_pos.get(nm) is not None
                      and math.dist(prev_pos[nm], c["pos"]) > 1000.0
                      and abs(c["pos"][1]) > 4000.0]
            if len(jumped) < 3:
                for nm in jumped:
                    self.events.append({"t": t, "kind": "demoed", "player": nm})
            for nm, c in cars.items():
                prev_pos[nm] = c["pos"]

            if cars:
                self.samples.append({
                    "t": t, "clock": clock, "hit_team": hit_team,
                    "ball_live": self.ball_live,
                    "ball": ball["pos"], "ball_vel": ball.get("vel", (0, 0, 0)),
                    "cars": cars,
                })

        self.ctx = dict(match_ctx)

        # Attribute car-indexed events to player names.
        actor_to_name = {}
        for s in self.samples:
            for nm, c in s["cars"].items():
                actor_to_name[c["actor"]] = nm
        for e in self.events:
            if "car" in e:
                e["player"] = actor_to_name.get(e["car"])
            elif "pri" in e:
                e["player"] = pri_name.get(e["pri"])

        if not self.team_size:
            counts = {}
            for t in self.teams.values():
                counts[t] = counts.get(t, 0) + 1
            self.team_size = max(counts.values()) if counts else 0

    # -- helpers ---------------------------------------------------------

    def resolve(self, who):
        """Match a player name tolerantly -- display names change over time."""
        if who in self.teams:
            return who
        target = canon(who)
        exact = [n for n in self.teams if canon(n) == target]
        if exact:
            return exact[0]
        part = [n for n in self.teams
                if target and (target in canon(n) or canon(n) in target)]
        return part[0] if len(part) == 1 else None

    def mates(self, who):
        t = self.teams.get(who)
        return [n for n, x in self.teams.items() if x == t and n != who]

    def opponents(self, who):
        t = self.teams.get(who)
        return [n for n, x in self.teams.items() if x != t]

    def attack_sign(self, who):
        """+1 if this player shoots toward +y, -1 otherwise."""
        return 1.0 if self.teams.get(who) == 0 else -1.0

    def stat_line(self, who):
        for p in self.stats:
            if canon(p.get("Name")) == canon(who):
                return p
        return {}

    def duration(self):
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1]["t"] - self.samples[0]["t"]


def load(path):
    return Match(parse(path), path=path)


# --- the player's own save-replay keybind ---------------------------------
#
# Telling everyone to "hold Backspace" is telling them YOUR keybind. Rocket
# League stores the real one per input device in TAInput.ini, so read it
# instead of guessing -- a controller player has never pressed Backspace in
# their life.

CONFIG_DIR = Path(os.path.expanduser(
    "~/Documents/My Games/Rocket League/TAGame/Config"))

_BIND = re.compile(
    r'(?P<dev>PC|Gamepad|SteamInput)Bindings=\(\s*Action="AutoSaveReplay",\s*'
    r'Key="(?P<key>[^"]+)"(?:,\s*PressType=(?P<press>\w+))?')

# The engine's key names are not what is printed on the hardware.
_PRETTY = {
    "XboxTypeS_Back": "Back / View",
    "XboxTypeS_Start": "Start / Menu",
    "XboxTypeS_LeftThumbStick": "left stick click",
    "XboxTypeS_RightThumbStick": "right stick click",
    "XboxTypeS_DPad_Up": "D-pad up",
    "XboxTypeS_DPad_Down": "D-pad down",
    "XboxTypeS_LeftShoulder": "LB",
    "XboxTypeS_RightShoulder": "RB",
}


def save_replay_binding():
    """
    How THIS player saves a replay: {"keyboard": "...", "gamepad": "..."}.

    Empty if the config cannot be read -- callers should fall back to naming
    both common defaults rather than asserting one.
    """
    path = CONFIG_DIR / "TAInput.ini"
    if not path.is_file():
        return {}
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    out = {}
    for m in _BIND.finditer(txt):
        dev = "keyboard" if m.group("dev") == "PC" else "gamepad"
        key = _PRETTY.get(m.group("key"), m.group("key"))
        if (m.group("press") or "").endswith("Hold"):
            key = "hold " + key
        out[dev] = key            # later entries win; profiles repeat
    return out


def save_replay_hint():
    """One printable line telling the player how to save a replay."""
    b = save_replay_binding()
    if not b:
        return ("save a replay at the end of a match "
                "(hold Backspace, or hold Back/View on a controller)")
    parts = [v for k, v in (("keyboard", b.get("keyboard")),
                            ("gamepad", b.get("gamepad"))) if v]
    return "save a replay at the end of a match: " + " or ".join(parts)
