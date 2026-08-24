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
import math
import os
import re
import subprocess
import sys
from pathlib import Path

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
        self.date = self.props.get("Date")
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
        pri_name, pri_team = {}, {}
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
                    s = state.setdefault(aid, {})
                    if loc:
                        s["pos"] = (loc["x"], loc["y"], loc["z"])
                    if lv:
                        s["vel"] = (lv["x"], lv["y"], lv["z"])
                    if av:
                        s["ang"] = (av["x"], av["y"], av["z"])
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
                    "ball": ball["pos"], "ball_vel": ball.get("vel", (0, 0, 0)),
                    "cars": cars,
                })

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
