"""
Pure-Python reader for the Rocket League replay header.

No dependencies and nothing to download. The full frame-by-frame physics lives
in the replay's network stream, which needs a heavyweight parser (boxcars,
rattletrap) -- but the header alone is a complete property tree carrying the
match metadata, every player's stat line, and the goal list with frame numbers.
That is enough to track form over time, which is what we want first.

Format, little-endian throughout:

    int32   header size
    int32   header CRC
    int32   engine version
    int32   licensee version
    int32   net version        (only if engine >= 868 and licensee >= 18)
    string  class name         ("TAGame.Replay_Soccar_TA")
    props   property tree

Strings are an int32 length then bytes. A NEGATIVE length means UTF-16, and the
byte count is -length * 2 -- getting that wrong is the usual way these parsers
break on replays containing non-ASCII player names.
"""

from __future__ import annotations

import struct
from pathlib import Path


class ReplayError(Exception):
    pass


class Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def _take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise ReplayError("read of %d past end at %d" % (n, self.pos))
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self._take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def u8(self) -> int:
        return self._take(1)[0]

    def text(self) -> str:
        n = self.i32()
        if n == 0:
            return ""
        if n < 0:
            raw = self._take(-n * 2)
            return raw.decode("utf-16-le", errors="replace").rstrip("\x00")
        raw = self._take(n)
        return raw.decode("windows-1252", errors="replace").rstrip("\x00")


def _read_property(r: Reader):
    """
    Return (name, value), or (None, None) at the terminator.

    Every value is read inside the byte window `size` declares, and the stream
    is advanced to the end of that window whatever happens inside it. This is
    the difference between a parser that works on the replays you happen to
    test and one that works on all of them: a Steam-only assumption in the
    ByteProperty branch desynced an Epic lobby's PlayerStats and silently lost
    every player's stat line, while the surrounding properties still looked
    perfectly healthy.

    BoolProperty is the exception -- it declares size 0 and its byte lives
    outside the window.
    """
    name = r.text()
    if name == "None" or name == "":
        return None, None

    kind = r.text()
    size = r.i64()

    if kind == "BoolProperty":
        return name, bool(r.u8())

    # StructProperty and ByteProperty both carry a LEADING NAME STRING that
    # their `size` does not cover -- size counts from after it. Both verified
    # against real bytes:
    #
    #   PlayerID  StructProperty size=708: value at 40, "UniqueNetId" takes 16
    #             bytes, next property begins at 40+16+708 = 764.
    #   Platform  ByteProperty   size=25:  "OnlinePlatform" takes 19 bytes and
    #             25 is exactly "OnlinePlatform_Steam", the value alone.
    #
    # Treating size as covering the name lands short, desyncs the rest of the
    # row, and silently loses every player's stat line while the surrounding
    # properties still look perfectly healthy. That is the whole reason this
    # parser reported 87/87 files "parsed" with zero usable stats.
    lead_name = None
    if kind in ("StructProperty", "ByteProperty"):
        lead_name = r.text()

    end = r.pos + size
    value = None
    try:
        if kind == "IntProperty":
            value = r.i32()
        elif kind == "QWordProperty":
            value = r.i64()
        elif kind == "FloatProperty":
            value = r.f32()
        elif kind in ("StrProperty", "NameProperty"):
            value = r.text()
        elif kind == "ByteProperty":
            # `lead_name` was the enum type; what remains is the value.
            value = r.text() if size > 0 else lead_name
        elif kind == "ArrayProperty":
            count = r.i32()
            inner = Reader(r.buf[r.pos:end])
            # Keep whatever parses. One malformed entry used to discard the
            # whole array, which turned a single unreadable player into six
            # missing stat lines.
            value = []
            for _ in range(count):
                try:
                    value.append(_read_properties(inner))
                except ReplayError:
                    break
        elif kind == "StructProperty":
            inner = Reader(r.buf[r.pos:end])
            value = {"__struct": lead_name, **_read_properties(inner)}
    except ReplayError:
        value = None

    r.pos = end
    return name, value


def _read_properties(r: Reader) -> dict:
    out = {}
    while True:
        name, value = _read_property(r)
        if name is None:
            return out
        out[name] = value


def read_header(path) -> dict:
    """Parse one .replay and return its property tree."""
    data = Path(path).read_bytes()
    r = Reader(data)

    r.i32()                       # header size
    r.i32()                       # header CRC
    engine = r.i32()
    licensee = r.i32()
    if engine >= 868 and licensee >= 18:
        r.i32()                   # net version
    r.text()                      # class name

    props = _read_properties(r)
    props["__engine"] = engine
    props["__licensee"] = licensee
    props["__file"] = str(path)
    return props
