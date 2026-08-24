"""
Peer coordination between Ally instances.

RLBot lets bots exchange arbitrary bytes through match comms, which means
several Allies in one match do not have to guess about each other. They can
say what they are doing.

Two channels, deliberately separated:

  INTENT  -- tactical, team-only. "I reach the ball in 1.2s and I am claiming
             it, and I am on my way to boost pad 14." Lets teammates resolve
             who takes a ball, and who takes which pad, deterministically
             rather than each estimating independently and both committing.
             Team-only because telling the opposition your plan is not
             coordination, it is a handicap.

             The boost claim rides on this message rather than having its own
             channel: pad decisions change several times a second, so they
             need the 10Hz intent rate, not a slow broadcast.

  CALIB   -- mechanical, broadcast to everyone. Every Ally runs identical
             controllers, so their self-calibration observations are
             interchangeable. Pooling them means six cars gather six times the
             evidence, and corrections that would take six matches to settle
             converge inside one.

Messages are small JSON objects. At 10Hz across six bots this is a few
kilobytes a second, which is nothing, and being human-readable makes the
protocol debuggable from a telemetry dump.

Anything arriving here is treated as data from an untrusted peer: malformed
messages are dropped silently rather than being allowed to break a match.
"""

from __future__ import annotations

import json
import time

PROTOCOL_VERSION = 1

# A tag identifying us as an Ally, so we ignore chatter from other bot types
# that happen to share the match.
MAGIC = "ally"

INTENT = "intent"
CALIB = "calib"
EXPERIENCE = "exp"

# Broadcast rates.
INTENT_HZ = 10.0
INTENT_INTERVAL = 1.0 / INTENT_HZ
CALIB_INTERVAL = 4.0
EXPERIENCE_INTERVAL = 7.0

# Peer state older than this is stale -- the bot may have been demolished,
# crashed, or left.
PEER_TTL = 0.6


class PeerIntent:
    """What one other Ally last told us it was doing."""

    __slots__ = ("index", "team", "role", "time_to_ball", "claiming", "ordinal",
                 "boost_pad", "boost_eta", "aerial", "at")

    def __init__(self, index, team, role, time_to_ball, claiming, at, ordinal=0,
                 boost_pad=-1, boost_eta=99.0, aerial=False):
        self.index = index
        self.team = team
        self.role = role
        self.time_to_ball = time_to_ball
        self.claiming = claiming
        # Which slot this peer currently believes it holds. Needed so all
        # bots agree on who the incumbent first man is: an incumbency
        # bonus applied only to oneself makes every bot the incumbent.
        self.ordinal = ordinal
        # This peer has a viable aerial on the ball. Broadcast for the same
        # reason as `ordinal`: the aerial bonus in the rotation ranking has to
        # be applied to the same car by every bot. A bonus each car applied
        # only to itself would make all three believe they were first man,
        # which is a fault this codebase has already had once.
        self.aerial = aerial
        # Which boost pad this peer is currently routing to, and how long
        # it expects to take. -1 means it is not going for boost.
        self.boost_pad = boost_pad
        self.boost_eta = boost_eta
        self.at = at

    def fresh(self, now: float) -> bool:
        return (now - self.at) < PEER_TTL

    def __repr__(self):
        return (
            f"Peer(idx={self.index} {self.role} tt={self.time_to_ball:.2f} "
            f"claim={self.claiming})"
        )


class CommsHub:
    """
    Outgoing broadcasts and incoming peer state.

    The bot owns one of these. `should_send_intent` and `should_send_calib`
    rate-limit; `note_intent` / `note_calib` fold in what arrives.
    """

    def __init__(self, my_index: int, my_team: int):
        self.my_index = my_index
        self.my_team = my_team
        self.peers: dict[int, PeerIntent] = {}
        self._last_intent = -99.0
        self._last_calib = -99.0
        self.calib_pool: list[dict] = []
        self.exp_pool: list[dict] = []
        self._last_exp = -99.0
        self.messages_in = 0
        self.messages_dropped = 0

    # --- outgoing ---------------------------------------------------------

    def should_send_intent(self, now: float) -> bool:
        if now - self._last_intent < INTENT_INTERVAL:
            return False
        self._last_intent = now
        return True

    def build_intent(self, role: str, time_to_ball: float, claiming: bool,
                     ordinal: int = 0, boost_pad: int = -1,
                     boost_eta: float = 99.0, aerial: bool = False) -> bytes:
        return json.dumps(
            {
                "m": MAGIC,
                "v": PROTOCOL_VERSION,
                "t": INTENT,
                "i": self.my_index,
                "r": role,
                "tt": round(float(time_to_ball), 3),
                "c": bool(claiming),
                "o": int(ordinal),
                "bp": int(boost_pad),
                "be": round(float(boost_eta), 2),
                "a": bool(aerial),
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def should_send_calib(self, now: float) -> bool:
        if now - self._last_calib < CALIB_INTERVAL:
            return False
        self._last_calib = now
        return True

    def build_calib(self, corrections) -> bytes:
        return json.dumps(
            {
                "m": MAGIC,
                "v": PROTOCOL_VERSION,
                "t": CALIB,
                "i": self.my_index,
                "aim": round(corrections.aim_bias, 4),
                "time": round(corrections.timing_bias, 4),
                "off": round(corrections.offset_scale, 4),
                "n": corrections.contacts,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def should_send_experience(self, now: float) -> bool:
        if now - self._last_exp < EXPERIENCE_INTERVAL:
            return False
        self._last_exp = now
        return True

    def build_experience(self, rows: list[dict]) -> bytes:
        return json.dumps(
            {"m": MAGIC, "v": PROTOCOL_VERSION, "t": EXPERIENCE,
             "i": self.my_index, "rows": rows},
            separators=(",", ":"),
        ).encode("utf-8")

    # --- incoming ---------------------------------------------------------

    def receive(self, index: int, team: int, content: bytes, now: float):
        """
        Handle one inbound message. Never raises: a peer sending nonsense must
        not be able to interrupt our match.
        """
        if index == self.my_index or not content:
            return
        try:
            msg = json.loads(content.decode("utf-8"))
        except Exception:
            self.messages_dropped += 1
            return

        if not isinstance(msg, dict) or msg.get("m") != MAGIC:
            return  # not one of ours; some other bot's chatter
        if msg.get("v") != PROTOCOL_VERSION:
            self.messages_dropped += 1
            return

        kind = msg.get("t")
        try:
            if kind == INTENT:
                self.peers[index] = PeerIntent(
                    index=index,
                    team=team,
                    role=str(msg.get("r", "")),
                    time_to_ball=float(msg.get("tt", 99.0)),
                    aerial=bool(msg.get("a", False)),
                    claiming=bool(msg.get("c", False)),
                    at=now,
                    ordinal=int(msg.get("o", 0)),
                    boost_pad=int(msg.get("bp", -1)),
                    boost_eta=float(msg.get("be", 99.0)),
                )
                self.messages_in += 1
            elif kind == EXPERIENCE:
                rows = msg.get("rows")
                if isinstance(rows, list):
                    # Bounded: a peer cannot flood us into a memory problem.
                    self.exp_pool.extend(rows[:32])
                    del self.exp_pool[:-128]
                    self.messages_in += 1
            elif kind == CALIB:
                self.calib_pool.append(
                    {
                        "aim": float(msg.get("aim", 0.0)),
                        "time": float(msg.get("time", 0.0)),
                        "off": float(msg.get("off", 1.0)),
                        "n": int(msg.get("n", 0)),
                        "from": index,
                    }
                )
                del self.calib_pool[:-16]
                self.messages_in += 1
        except (TypeError, ValueError):
            self.messages_dropped += 1

    # --- queries ----------------------------------------------------------

    def teammates(self, now: float) -> list[PeerIntent]:
        return [
            p for p in self.peers.values()
            if p.team == self.my_team and p.fresh(now)
        ]

    def better_placed_peer(self, now: float, my_time: float, margin: float = 0.12):
        """
        A teammate Ally that can reach the ball meaningfully sooner than us.

        Ties are broken by index so two bots with identical estimates never
        both defer (which would leave the ball to nobody) and never both
        commit.
        """
        best = None
        for p in self.teammates(now):
            if p.time_to_ball < my_time - margin:
                if best is None or p.time_to_ball < best.time_to_ball:
                    best = p
            elif abs(p.time_to_ball - my_time) <= margin and p.index < self.my_index:
                # Dead heat: the lower index takes it.
                if best is None or p.time_to_ball < best.time_to_ball:
                    best = p
        return best

    def someone_claiming(self, now: float, my_time: float) -> bool:
        """True if a teammate has claimed the ball and is better placed."""
        return any(
            p.claiming and p.time_to_ball < my_time
            for p in self.teammates(now)
        )

    def pad_claims(self, now: float) -> dict[int, tuple[int, float]]:
        """
        Boost pads teammates are currently routing to.

        Maps pad index -> (claimer car index, their ETA). Only the strongest
        claim on each pad is kept: better ETA wins, and an exact tie goes to
        the lower car index so every bot resolves it the same way.
        """
        claims: dict[int, tuple[int, float]] = {}
        for p in self.teammates(now):
            if p.boost_pad < 0:
                continue
            held = claims.get(p.boost_pad)
            if held is None:
                claims[p.boost_pad] = (p.index, p.boost_eta)
                continue
            other_index, other_eta = held
            if p.boost_eta < other_eta - 1e-6 or (
                abs(p.boost_eta - other_eta) <= 1e-6 and p.index < other_index
            ):
                claims[p.boost_pad] = (p.index, p.boost_eta)
        return claims

    def pad_is_taken(self, now: float, pad_index: int, my_eta: float) -> bool:
        """True if a teammate has a stronger claim on this pad than we do."""
        held = self.pad_claims(now).get(pad_index)
        if held is None:
            return False
        other_index, other_eta = held
        if other_eta < my_eta - 1e-6:
            return True
        if abs(other_eta - my_eta) <= 1e-6:
            return other_index < self.my_index
        return False

    def drain_experience(self) -> list[dict]:
        out = self.exp_pool[:]
        self.exp_pool.clear()
        return out

    def drain_calibration(self) -> list[dict]:
        """Take pending peer calibration reports and clear the buffer."""
        out = self.calib_pool[:]
        self.calib_pool.clear()
        return out

    def summary(self, now: float) -> str:
        peers = self.teammates(now)
        return (
            f"{len(peers)} peer(s) "
            + " ".join(f"[{p.index}:{p.role}:{p.time_to_ball:.1f}s]" for p in peers)
            + f" | in {self.messages_in} dropped {self.messages_dropped}"
        )
