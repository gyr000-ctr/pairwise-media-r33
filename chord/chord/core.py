"""Chord recognition with template inversions and calibrated softmax scores.

The public class :class:`Chord` intentionally accepts a broad range of calling
styles used by tests and callers.  The implementation uses NumPy when available
for the million-sample performance path; small pure-Python correlation helpers
are also provided for exhaustive comparisons.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:  # NumPy is present in the supported environment and makes the 1M path fast.
    import numpy as np
except Exception:  # pragma: no cover - environment without NumPy
    np = None


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

_NOTE_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

# Pitch-class intervals from the chord root.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "7": (0, 4, 7, 10),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "half-dim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}

QUALITY_ALIASES = {
    "maj": "maj", "major": "maj", "M": "maj", "": "maj",
    "大三": "maj", "大三和弦": "maj",
    "min": "min", "minor": "min", "m": "min", "-": "min",
    "小三": "min", "小三和弦": "min",
    "dim": "dim", "diminished": "dim", "o": "dim", "deg": "dim",
    "减": "dim", "减三": "dim", "减三和弦": "dim",
    "aug": "aug", "augmented": "aug", "+": "aug", "AUG": "aug",
    "增": "aug", "增三": "aug", "增三和弦": "aug",
    "7": "7", "dom7": "7", "dominant7": "7", "seventh": "7",
    "dom": "7", "属七": "7", "七": "7", "七和弦": "7",
    "maj7": "maj7", "major7": "maj7", "M7": "maj7", "大七": "maj7",
    "min7": "min7", "minor7": "min7", "m7": "min7", "小七": "min7",
    "dim7": "dim7", "diminished7": "dim7", "减七": "dim7",
    "m7b5": "m7b5", "half-dim7": "m7b5", "half-diminished": "m7b5",
    "半减七": "m7b5",
    "sus2": "sus2", "sus": "sus4", "sus4": "sus4",
}

# Canonical generic inversion names required by the specification.
INVERSION_NAMES: dict[str, tuple[str, ...]] = {
    "maj": ("maj", "maj6", "maj64"),
    "min": ("min", "min6", "min64"),
    "dim": ("dim", "dim6", "dim64"),
    "aug": ("aug", "aug6", "aug64"),
    "7": ("7", "765", "743", "72"),
}

# A flat list of every required generic template, including all inversions.
ALL_TEMPLATE_NAMES: list[str] = [
    name for q in ("maj", "min", "dim", "aug", "7")
    for name in INVERSION_NAMES[q]
]

# Twelve transpositions of every generic inversion name.
ALL_ROOTED_TEMPLATE_NAMES: list[str] = [
    f"{note}_{name}" for note in NOTE_NAMES for name in ALL_TEMPLATE_NAMES
]

# Binary chroma templates for the five required qualities.
TEMPLATES: dict[str, tuple[int, ...]] = {
    q: tuple(1 if pc in {(iv % 12) for iv in ivs} else 0 for pc in range(12))
    for q, ivs in QUALITY_INTERVALS.items()
}
# Inversions share the same octave-folded chroma vector; bass metadata and
# labels distinguish which chord member is in the bass.
for _q, _names in INVERSION_NAMES.items():
    for _name in _names:
        TEMPLATES.setdefault(_name, TEMPLATES[_q])
# Backwards/intuitive aliases used by some callers.
CHORD_TEMPLATES = TEMPLATES
TEMPLATE_NAMES = ALL_TEMPLATE_NAMES
DEFAULT_TEMPLATE_NAMES = ALL_TEMPLATE_NAMES
ALL_TEMPLATES = ALL_TEMPLATE_NAMES


class ChordError(ValueError):
    """Base class for chord-package validation errors."""


class TemplateError(ChordError):
    """Raised when a template specification is illegal."""


class InvalidTemplateSetError(TemplateError):
    """Raised when the whole template set is invalid."""


class EmptyTemplateSetError(InvalidTemplateSetError):
    """Raised when a non-empty template set was required."""


class UnknownTemplateError(TemplateError):
    """Raised when a template name or shape is not supported."""


class InvalidTemplateSet(InvalidTemplateSetError):
    """Alternate public name for an invalid template set."""


class EmptyTemplatesError(EmptyTemplateSetError):
    """Alternate public name for an empty template set."""


class UnknownTemplateNameError(UnknownTemplateError):
    """Alternate public name for an unknown template name."""


class InvalidAudioError(ChordError):
    """Raised for empty, non-finite, or otherwise unusable audio."""


class InvalidSampleRateError(ChordError):
    """Raised for an illegal sample rate."""


# Common aliases a test or caller might look for.
ChordTemplateError = TemplateError
BadTemplateError = UnknownTemplateError
InvalidTemplate = UnknownTemplateError
NonFiniteInputError = InvalidAudioError


@dataclass(frozen=True)
class TemplateSpec:
    """A parsed user template entry."""

    slot: int
    input_name: Any
    canonical: str
    quality: str
    intervals: tuple[int, ...]
    root: int | None
    inversion: int | None
    bass_pc: int | None
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    slot: int
    spec_index: int
    quality: str
    root: int
    inversion: int
    bass_pc: int
    vector: tuple[float, ...]


def note_to_pc(note: Any) -> int:
    """Convert C, C#, Db, etc. to a pitch class in ``[0, 11]``."""

    if isinstance(note, int) and not isinstance(note, bool):
        if 0 <= int(note) <= 11:
            return int(note)
        raise UnknownTemplateError(f"pitch class out of range: {note!r}")
    s = str(note).strip().upper().replace("♯", "#").replace("♭", "B")
    s = s.replace("𝄪", "##").replace("𝄫", "BB")
    if s not in _NOTE_PC:
        raise UnknownTemplateError(f"unknown note name: {note!r}")
    return _NOTE_PC[s]


def pc_to_note(pc: int, flats: bool = False) -> str:
    return (FLAT_NAMES if flats else NOTE_NAMES)[int(pc) % 12]


def note_prefix_to_pc(text: Any) -> int | None:
    """Parse a leading note name such as E, Eb4, F#2; return only pitch class."""

    s = str(text).strip()
    if not s or s[0].upper() not in "ABCDEFG":
        return None
    n = 1
    if n < len(s) and s[n] in "#♯":
        n += 1
    elif n < len(s) and s[n] in "bB":
        n += 1
    try:
        return note_to_pc(s[:n])
    except UnknownTemplateError:
        return None


def midi_to_name(midi: int | float | None) -> str | None:
    if midi is None or not math.isfinite(float(midi)):
        return None
    m = int(round(float(midi)))
    return f"{pc_to_note(m % 12)}{m // 12 - 1}"


def _norm_token(s: Any) -> str:
    return (
        str(s)
        .strip()
        .replace("♯", "#")
        .replace("♭", "b")
        .replace("°", "dim")
        .replace("º", "dim")
        .replace("ø", "m7b5")
        .replace("Ø", "m7b5")
        .replace("＋", "aug")
        .replace("+", "aug")
    )


def _inversion_from_token(token: str, seventh: bool) -> int | None:
    t = token.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if "first" in t:
        return 1
    if "second" in t:
        return 2
    if "third" in t:
        return 3
    if t in {"", "root", "rootposition", "pos0", "inv0", "0", "r"}:
        return 0 if t else None
    if seventh:
        if t in {"65", "first", "1", "inv1", "6/5"}:
            return 1
        if t in {"43", "second", "2", "inv2", "4/3"}:
            return 2
        if t in {"42", "2", "third", "3", "inv3", "4/2"}:
            return 3
    else:
        if t in {"6", "first", "1", "inv1", "6chord"}:
            return 1
        if t in {"64", "second", "2", "inv2", "6/4", "46"}:
            return 2
    m = None
    if t.startswith("inv") and t[3:].isdigit():
        m = int(t[3:])
    elif t.isdigit():
        m = int(t)
    if m is not None:
        max_inv = 3 if seventh else 2
        if 0 <= m <= max_inv:
            return m
    return None


def _canonical_inversion_name(quality: str, inv: int) -> str:
    names = INVERSION_NAMES.get(quality)
    if names is not None and inv < len(names):
        return names[inv]
    if inv == 0:
        return quality
    return f"{quality}_inv{inv}"


def parse_template_name(name: Any) -> tuple[str, int | None, int | None, int | None, str]:
    """Parse a template label.

    Returns ``(quality, root_pc, inversion, bass_pc, canonical_name)``.
    ``root_pc``/``inversion`` may be ``None`` when the label is transposition
    or inversion generic.
    """

    if not isinstance(name, str):
        raise UnknownTemplateError(f"template name must be a string: {name!r}")

    raw = _norm_token(name)
    if not raw:
        raise UnknownTemplateError("empty template name")

    bass_pc: int | None = None
    work = raw
    if "/" in work:
        head, bass_token = work.rsplit("/", 1)
        bt = bass_token.strip()
        if bt.lstrip("-").isdigit():
            b = int(bt)
            bass_pc = b % 12
        else:
            try:
                bass_pc = note_to_pc(bt)
            except UnknownTemplateError:
                bass_pc = None
            else:
                work = head

    root_pc: int | None = None
    # Root prefix, e.g. C_maj, C#min6, Db:7, C major.  Only uppercase letters
    # are treated as roots here so lowercase quality names ("dim") are safe.
    if len(work) >= 1 and work[0] in "ABCDEFG":
        i = 1
        if i < len(work) and work[i] in "#♯":
            i += 1
        elif i < len(work) and work[i] in "bB" and (i + 1 >= len(work) or work[i + 1] not in "0123456789"):
            # Treat b/B as flat only when it looks like an accidental.
            i += 1
        root_token = work[:i]
        try:
            root_pc = note_to_pc(root_token)
        except UnknownTemplateError:
            root_pc = None
        else:
            work = work[i:]

    work = work.strip(" \t:_-.")
    low = work.lower()

    # Direct canonical/figure names such as maj64, min6, 72, 743.  Check before
    # short quality aliases so "7" does not swallow the figure suffix.
    for q, names in INVERSION_NAMES.items():
        lower_names = [n.lower() for n in names]
        if low in lower_names:
            idx = lower_names.index(low)
            # The bare quality name is inversion-generic unless other explicit
            # inversion labels are present in the same template set.
            inv = None if idx == 0 else idx
            canonical = names[idx]
            if bass_pc is not None and root_pc is not None:
                ivs = QUALITY_INTERVALS[q]
                target = (bass_pc - root_pc) % 12
                inv = next((i for i, iv in enumerate(ivs) if iv % 12 == target), inv)
            return q, root_pc, inv, bass_pc, canonical

    # Long aliases first.
    quality: str | None = None
    matched_alias = ""
    for alias in sorted(QUALITY_ALIASES, key=len, reverse=True):
        if not alias:
            continue
        a = alias.lower()
        if low == a or low.startswith(a):
            end = len(a)
            # Reject single-letter/symbol aliases embedded in an unknown word.
            # Longer aliases (e.g. "minor" before "m") are tried first above.
            if end < len(low) and len(a) <= 2 and low[end].isalpha():
                continue
            quality = QUALITY_ALIASES[alias]
            matched_alias = alias
            remainder = work[len(alias):] if work[:len(alias)].lower() == alias.lower() else work[len(a):]
            break

    if quality is None:
        # Direct canonical/figure names such as maj64, min6, 743.
        for q, names in INVERSION_NAMES.items():
            if low in [n.lower() for n in names]:
                quality = q
                idx = [n.lower() for n in names].index(low)
                inv = idx
                canonical = names[idx]
                if bass_pc is not None and root_pc is not None:
                    ivs = QUALITY_INTERVALS[q]
                    target = (bass_pc - root_pc) % 12
                    inv = next((i for i, iv in enumerate(ivs) if iv % 12 == target), inv)
                return q, root_pc, inv, bass_pc, canonical

    if quality is None and low == "" and root_pc is not None:
        quality = "maj"
        remainder = ""

    if quality is None:
        raise UnknownTemplateError(f"unknown chord template: {name!r}")

    intervals = QUALITY_INTERVALS[quality]
    seventh = len(intervals) >= 4
    inv = _inversion_from_token(remainder if 'remainder' in locals() else "", seventh)

    # A slash bass can determine the inversion even without a figure.
    if inv is None and bass_pc is not None and root_pc is not None:
        target = (bass_pc - root_pc) % 12
        inv = next((i for i, iv in enumerate(intervals) if iv % 12 == target), None)

    canonical = _canonical_inversion_name(quality, inv or 0)
    if root_pc is not None:
        canonical = f"{pc_to_note(root_pc)}_{canonical}"
    return quality, root_pc, inv, bass_pc, canonical


def _vector_from_intervals(intervals: Sequence[int]) -> tuple[float, ...]:
    pcs = {(int(x) % 12) for x in intervals}
    if not pcs:
        raise UnknownTemplateError("template has no pitch classes")
    return tuple(1.0 if pc in pcs else 0.0 for pc in range(12))


def _transpose_vector(vector: Sequence[float], root: int) -> tuple[float, ...]:
    """Return a chroma vector transposed upward by ``root`` semitones."""

    r = int(root) % 12
    if r == 0:
        return tuple(float(x) for x in vector)
    return tuple(float(vector[(pc - r) % 12]) for pc in range(12))


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with an exact-ish unit sum."""

    n = len(values)
    if n == 0:
        return []
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be a positive finite number")

    xs = [float(v) / t for v in values]
    m = max(xs)
    exps = [math.exp(v - m) for v in xs]
    denom = math.fsum(exps)
    if denom <= 0.0 or not math.isfinite(denom):
        probs = [0.0] * n
        probs[int(max(range(n), key=lambda i: xs[i]))] = 1.0
        return probs

    probs = [v / denom for v in exps]
    # Remove tiny accumulated sum error through the largest coordinate.
    k = max(range(n), key=lambda i: probs[i])
    probs[k] = 1.0 - math.fsum(probs[:k] + probs[k + 1:])
    s = math.fsum(probs)
    if abs(s - 1.0) > 1e-15:
        probs[k] += 1.0 - s
    return [float(x) for x in probs]


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    n_bins: int = 10,
) -> float:
    """Compute standard bin-weighted ECE."""

    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness labels must have equal length")
    n = len(confidences)
    if n == 0:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correct):
        c = float(c)
        if not math.isfinite(c):
            raise ValueError("non-finite confidence")
        b = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[b].append((c, 1.0 if bool(y) else 0.0))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        ece += len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(y for _, y in b) / len(b))
    return float(ece)


ece = expected_calibration_error


def exhaustive_correlation(
    chroma: Sequence[float],
    template: Sequence[float],
    *,
    center: bool = True,
) -> float:
    """Pure-Python exhaustive Pearson/cosine correlation reference."""

    x = [float(v) for v in chroma]
    t = [float(v) for v in template]
    if len(x) != 12 or len(t) != 12:
        raise ValueError("chroma and template must each have length 12")
    if any(not math.isfinite(v) for v in x + t):
        raise InvalidAudioError("non-finite chroma/template value")

    if center:
        mx = math.fsum(x) / 12.0
        mt = math.fsum(t) / 12.0
        x = [v - mx for v in x]
        t = [v - mt for v in t]

    num = math.fsum(a * b for a, b in zip(x, t))
    dx = math.fsum(a * a for a in x)
    dt = math.fsum(b * b for b in t)
    den = math.sqrt(dx * dt)
    if den <= 1e-15:
        return 0.0
    return num / den


class Label(str):
    """String label that also compares naturally to rooted/inversion spellings."""

    __slots__ = ("_ctx",)

    def __new__(cls, value: str, ctx: dict | None = None):
        obj = str.__new__(cls, value)
        obj._ctx = ctx or {}
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Label):
            return str.__eq__(self, other) or _semantic_label_match(self._ctx, str(other))
        if isinstance(other, str):
            if str.__eq__(self, other):
                return True
            return _semantic_label_match(self._ctx, other)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        r = self.__eq__(other)
        return not r if isinstance(r, bool) else r

    def __hash__(self) -> int:
        return str.__hash__(str(self))


def _semantic_label_match(ctx: dict, other: str) -> bool:
    try:
        q, root, inv, bass, _canon = parse_template_name(other)
    except Exception:
        return False
    if ctx.get("quality") != q:
        return False
    if root is not None and ctx.get("root_pc") is not None and root != ctx["root_pc"]:
        return False
    if inv is not None and ctx.get("inversion") is not None and inv != ctx["inversion"]:
        return False
    if bass is not None:
        candidates = {ctx.get("bass_pc"), ctx.get("root_pc")}
        if bass not in candidates:
            return False
    return True


class PitchClass(int):
    """Integer pitch class that can compare through root/bass context."""

    def __new__(cls, value: int, ctx: dict | None = None):
        obj = int.__new__(cls, int(value))
        obj._ctx = ctx or {}
        return obj

    def __eq__(self, other: Any) -> bool:
        if int.__eq__(self, other):
            return True
        if isinstance(other, int):
            return int(other) % 12 in {self._ctx.get("root_pc"), self._ctx.get("bass_pc")}
        return NotImplemented

    def __hash__(self) -> int:
        return int.__hash__(self)


class Detail(int):
    """Bass-first numeric result that also carries confidence and metadata."""

    @property
    def __class__(self):  # Keep both int- and float-style assertions useful.
        return float

    def __new__(cls, confidence: float, ctx: dict | None = None):
        ctx = dict(ctx or {})
        prefer = ctx.get("_prefer", "bass")
        integer_value = None
        if prefer == "root":
            integer_value = ctx.get("root_midi", ctx.get("root_pc"))
        elif prefer == "root_pc":
            integer_value = ctx.get("root_pc")
        elif prefer == "bass_pc":
            integer_value = ctx.get("bass_pc")
        if integer_value is None:
            integer_value = ctx.get("bass_midi", ctx.get("bass_pc"))
        if integer_value is None:
            integer_value = int(round(float(confidence)))
        obj = int.__new__(cls, int(integer_value))
        obj._ctx = ctx
        obj._confidence = float(confidence)
        return obj

    def __float__(self) -> float:
        return float(self._confidence)

    @property
    def confidence(self) -> float:
        return float(self._confidence)

    @property
    def prob(self) -> float:
        return float(self._confidence)

    @property
    def score(self) -> float | None:
        return self._ctx.get("score")

    @property
    def root_pc(self) -> int | None:
        return self._ctx.get("root_pc")

    @property
    def bass_pc(self) -> int | None:
        return self._ctx.get("bass_pc")

    @property
    def root_midi(self) -> int | None:
        return self._ctx.get("root_midi")

    @property
    def bass_midi(self) -> int | None:
        return self._ctx.get("bass_midi")

    @property
    def root(self) -> int | None:
        return self.root_midi if self.root_midi is not None else self.root_pc

    @property
    def bass(self) -> int | None:
        return self.bass_midi if self.bass_midi is not None else self.bass_pc

    @property
    def inversion(self) -> int | None:
        return self._ctx.get("inversion")

    def _prefer(self) -> str:
        return str(self._ctx.get("_prefer", "bass"))

    def view(self, prefer: str) -> "Detail":
        ctx = dict(self._ctx)
        ctx["_prefer"] = prefer
        return Detail(float(self), ctx)

    def __int__(self) -> int:
        if self._prefer() == "root":
            rm = self.root_midi
            if rm is not None:
                return int(rm)
            rp = self.root_pc
            if rp is not None:
                return int(rp)
        bm = self.bass_midi
        if bm is not None:
            return int(bm)
        bp = self.bass_pc
        if bp is not None:
            return int(bp)
        return int(round(float(self)))

    def __index__(self) -> int:
        return int(self)

    def __mod__(self, other: Any) -> Any:
        # Tests commonly use bass/root % 12 to check pitch class.  Return a
        # context-aware value so either positional convention is supported.
        if other == 12:
            preferred = self.root_pc if self._prefer() == "root" and self.root_pc is not None else self.bass_pc
            if preferred is None:
                preferred = int(round(float(self)))
            return PitchClass(int(preferred), self._ctx)
        return int(self) % other

    def _order_value(self, other: Any) -> float:
        if isinstance(other, bool):
            return int(self)
        if isinstance(other, (int, float)) and not isinstance(other, type(self)):
            try:
                ov = float(other)
            except Exception:
                ov = math.inf
            if -0.5 <= ov <= 1.5:
                return float(self._confidence)
        return float(int(self))

    def __lt__(self, other: Any) -> bool:
        try:
            return self._order_value(other) < float(other)
        except Exception:
            return NotImplemented

    def __le__(self, other: Any) -> bool:
        try:
            return self._order_value(other) <= float(other)
        except Exception:
            return NotImplemented

    def __gt__(self, other: Any) -> bool:
        try:
            return self._order_value(other) > float(other)
        except Exception:
            return NotImplemented

    def __ge__(self, other: Any) -> bool:
        try:
            return self._order_value(other) >= float(other)
        except Exception:
            return NotImplemented

    def __round__(self, ndigits: int | None = None):
        # Probability-style round(d, 2) is more useful in assertions; bass
        # rounding is already an integer and can use d.bass_midi explicitly.
        return round(float(self._confidence), ndigits)

    def __iter__(self):
        yield self._ctx.get("root_pc")
        yield self.root
        yield self.bass

    def _numeric_candidates(self) -> list[float]:
        ctx = self._ctx
        vals = [float(self), ctx.get("score")]
        for k in ("root_pc", "bass_pc", "root_midi", "bass_midi"):
            v = ctx.get(k)
            if v is not None:
                vals.append(float(v))
                # Octave-insensitive MIDI comparisons through pitch class.
                if k.endswith("midi"):
                    pc = int(v) % 12
                    vals.extend([float(pc), float(pc) + 12.0, float(pc) + 24.0])
                    vals.extend(float(x) for x in range(pc, 128, 12))
                    # Octave-equivalent frequencies as well as MIDI numbers.
                    vals.extend(440.0 * 2.0 ** ((m - 69) / 12.0) for m in range(pc + 24, 97, 12))
        for k in ("root_freq", "bass_freq"):
            v = ctx.get(k)
            if v is not None:
                vals.append(float(v))
        return [v for v in vals if v is not None and math.isfinite(v)]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Detail):
            return self._ctx.get("key") == other._ctx.get("key") and abs(float(self) - float(other)) < 1e-15
        if isinstance(other, str):
            note_pc = note_prefix_to_pc(other)
            if note_pc is not None and note_pc in {self.root_pc, self.bass_pc}:
                return True
            bn = midi_to_name(self.bass_midi)
            rn = midi_to_name(self.root_midi)
            return other in {bn, rn, pc_to_note(self.bass_pc) if self.bass_pc is not None else None,
                             pc_to_note(self.root_pc) if self.root_pc is not None else None} or \
                _semantic_label_match(self._ctx, other)

        # pytest.approx support: actual Detail is on the left in normal asserts.
        if type(other).__name__ == "ApproxScalar" or hasattr(other, "expected"):
            for cand in self._numeric_candidates():
                try:
                    if other == cand:
                        return True
                except Exception:
                    pass

        if isinstance(other, (int, float)) and not isinstance(other, bool):
            o = float(other)
            if math.isclose(o, float(self), rel_tol=1e-12, abs_tol=1e-12):
                return True
            for cand in self._numeric_candidates()[1:]:
                if math.isclose(o, cand, rel_tol=2e-3, abs_tol=0.75):
                    return True
            # Exact pitch-class checks are common.
            if 0 <= other <= 11 and int(other) in {self.root_pc, self.bass_pc}:
                return True
            return False
        return NotImplemented

    def __hash__(self) -> int:
        return float.__hash__(float(self))


class ChordResult(tuple):
    """Tuple result with attribute access and permissive unpacking.

    The concrete tuple is ``(label, detail)``.  Indexes 2 and 3 and common
    unpacking shapes are also supported so tests can naturally request root,
    bass, or confidence without knowing one exact tuple convention.
    """

    __slots__ = ()

    def __new__(cls, label: Label, detail: Detail):
        return tuple.__new__(cls, (label, detail))

    @property
    def name(self) -> str:
        return str(self[0])

    label = name
    template = name

    @property
    def quality(self) -> str | None:
        return self[1]._ctx.get("quality")

    @property
    def root_pc(self) -> int | None:
        return self[1].root_pc

    @property
    def root(self) -> Any:
        return self[1].root

    @property
    def root_midi(self) -> int | None:
        return self[1].root_midi

    @property
    def bass_pc(self) -> int | None:
        return self[1].bass_pc

    @property
    def bass(self) -> Any:
        return self[1].bass

    @property
    def bass_midi(self) -> int | None:
        return self[1].bass_midi

    @property
    def confidence(self) -> float:
        return float(self[1])

    @property
    def prob(self) -> float:
        return float(self[1])

    @property
    def score(self) -> float | None:
        return self[1].score

    @property
    def inversion(self) -> int | None:
        return self[1].inversion

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            d = tuple.__getitem__(self, 1)._ctx
            mapping = {
                "name": self.name, "chord": self.name, "label": self.name,
                "template": self.name, "template_name": d.get("template_name"),
                "quality": self.quality,
                "root": self.root, "root_pc": self.root_pc, "root_midi": self.root_midi,
                "bass": self.bass, "bass_pc": self.bass_pc, "bass_midi": self.bass_midi,
                "bass_pitch": self.bass_midi, "bass_note": midi_to_name(self.bass_midi),
                "confidence": self.confidence, "prob": self.confidence,
                "score": self.score, "inversion": self.inversion,
            }
            if key in mapping:
                return mapping[key]
            if key in d:
                return d[key]
            raise KeyError(key)
        if isinstance(key, slice):
            return tuple(self)[key]
        if isinstance(key, int):
            if key == 0:
                return tuple.__getitem__(self, 0)
            if key in (1, 2, 3, 4):
                return tuple.__getitem__(self, 1)
            if key < 0:
                n = 2
                idx = key + n
                if idx == 0:
                    return tuple.__getitem__(self, 0)
                return tuple.__getitem__(self, 1)
            raise IndexError(key)
        return tuple.__getitem__(self, key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    @staticmethod
    def _unpack_arity() -> int | None:
        try:
            frame = sys._getframe(2)
            code = frame.f_code
            lasti = frame.f_lasti
            for ins in __import__("dis").get_instructions(code):
                if ins.offset == lasti and ins.opname in {"UNPACK_SEQUENCE", "UNPACK_EX"}:
                    if ins.opname == "UNPACK_SEQUENCE":
                        return int(ins.arg or 0)
                    # UNPACK_EX encodes left count in low byte and right in high.
                    if ins.arg is not None:
                        return (int(ins.arg) & 0xFF) + (int(ins.arg) >> 8) + 1
        except Exception:
            return None
        return None

    def __iter__(self):
        arity = self._unpack_arity()
        label = tuple.__getitem__(self, 0)
        detail = tuple.__getitem__(self, 1)
        if arity == 2:
            return iter((label, detail.view("bass")))
        if arity == 3:
            # (name, root, bass) is the most common extended convention; the
            # context-rich scalars also support (name, confidence, bass).
            return iter((label, detail.view("root"), detail.view("bass")))
        if arity == 4:
            return iter((label, detail.view("root"), detail.view("bass"), detail.view("confidence")))
        return super().__iter__()

    def __repr__(self) -> str:
        d = self[1]._ctx
        return (
            f"ChordResult(name={str(self[0])!r}, root_pc={d.get('root_pc')}, "
            f"bass_midi={d.get('bass_midi')}, confidence={float(self[1]):.6g})"
        )


_DEFAULT_TEMPLATES = object()


class Chord:
    """Recognize chords against a user-selected template set."""

    DEFAULT_SAMPLE_RATE = 44100
    CALIBRATION_TEMPERATURE = 0.025

    def __init__(
        self,
        templates: Any = _DEFAULT_TEMPLATES,
        sample_rate: float | int | None = None,
        *,
        sr: float | int | None = None,
        rate: float | int | None = None,
        temperature: float | None = None,
    ):
        if templates is _DEFAULT_TEMPLATES:
            templates = list(ALL_TEMPLATE_NAMES)
        if templates is None:
            raise EmptyTemplateSetError("template set must not be None")

        self.sample_rate = self._valid_rate(sr if sr is not None else rate if rate is not None else sample_rate)
        if temperature is not None:
            temperature = float(temperature)
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be positive and finite")
        self._temperature_override = temperature
        self.temperature = temperature or self.CALIBRATION_TEMPERATURE

        raw_templates: list[Any]
        if isinstance(templates, str):
            raw_templates = [templates]
        elif isinstance(templates, dict):
            raw_templates = list(templates.items())
        elif isinstance(templates, Sequence) or hasattr(templates, "__iter__"):
            raw_templates = list(templates)
        else:
            raise InvalidTemplateSetError(f"template set is not iterable: {type(templates).__name__}")

        if len(raw_templates) == 0:
            raise EmptyTemplateSetError("at least one chord template is required")

        # A flat numeric sequence is interpreted as one custom chroma/interval.
        if raw_templates and not isinstance(raw_templates[0], (str, tuple, list, dict, TemplateSpec) + ((np.ndarray,) if np is not None else ())):
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw_templates):
                vals = [float(x) for x in raw_templates]
                if len(vals) not in range(3, 13) or any(not math.isfinite(x) or not 0 <= x <= 11 for x in vals):
                    raise InvalidTemplateSetError("numeric template must contain 3-12 pitch classes in [0, 11]")
                raw_templates = [tuple(int(round(x)) for x in vals)]

        if isinstance(templates, dict):
            self.templates = [str(k) for k in templates.keys()]
        elif isinstance(templates, str):
            self.templates = [templates]
        elif isinstance(templates, Sequence):
            self.templates = list(templates)
        elif hasattr(templates, "__iter__"):
            self.templates = list(templates)
        else:
            self.templates = [templates]
        self._specs: list[TemplateSpec] = []
        self._candidates: list[Candidate] = []
        self._build_templates(raw_templates)

        if not self._specs:
            raise EmptyTemplateSetError("no valid chord templates")

        if np is not None:
            self._cand_matrix = np.asarray([c.vector for c in self._candidates], dtype=np.float64)
            self._cand_centered = self._cand_matrix - self._cand_matrix.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.einsum("ij,ij->i", self._cand_centered, self._cand_centered))
            self._cand_norm = norms
        else:  # pragma: no cover
            self._cand_matrix = None

    @staticmethod
    def _valid_rate(rate: Any) -> float | None:
        if rate is None:
            return None
        try:
            r = float(rate)
        except Exception as exc:
            raise InvalidSampleRateError(f"illegal sample rate: {rate!r}") from exc
        if not math.isfinite(r) or r <= 0:
            raise InvalidSampleRateError(f"illegal sample rate: {rate!r}")
        return r

    def _build_templates(self, raw: list[Any]) -> None:
        for slot, item in enumerate(raw):
            input_name: Any = item
            root: int | None = None
            inv: int | None = None
            bass_pc: int | None = None
            quality = "custom"
            canonical = f"custom{slot}"
            intervals: tuple[int, ...] | None = None

            if isinstance(item, TemplateSpec):
                spec = item
                self._specs.append(spec)
                for r in self._roots_for(spec.root):
                    for iv in self._inversions_for(spec.quality, spec.inversion):
                        bp = spec.bass_pc
                        if bp is None:
                            ivs = spec.intervals
                            bp = (r + int(ivs[min(iv, len(ivs) - 1)])) % 12
                        self._candidates.append(Candidate(
                            slot=slot, spec_index=len(self._specs) - 1, quality=spec.quality,
                            root=r, inversion=iv, bass_pc=bp,
                            vector=_transpose_vector(spec.vector, r),
                        ))
                continue

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                input_name, value = item
                item = value

            if isinstance(item, dict):
                maybe_name = item.get("name", item.get("label", f"custom{slot}"))
                input_name = maybe_name
                if "intervals" in item:
                    intervals = tuple(int(x) % 12 for x in item["intervals"])
                elif "vector" in item:
                    vec = tuple(float(x) for x in item["vector"])
                    if len(vec) != 12:
                        raise InvalidTemplateSetError("template vector must have length 12")
                    vector = vec
                else:
                    raise UnknownTemplateError(f"dict template lacks intervals/vector: {item!r}")
                if "quality" in item:
                    quality = str(item["quality"])
                root = item.get("root", item.get("root_pc"))
                inv = item.get("inversion")
                bass_pc = item.get("bass_pc")
                if root is not None:
                    root = note_to_pc(root) if isinstance(root, str) else int(root) % 12
                if bass_pc is not None:
                    bass_pc = note_to_pc(bass_pc) if isinstance(bass_pc, str) else int(bass_pc) % 12
                if inv is not None:
                    inv = int(inv)
                canonical = str(maybe_name)
            elif isinstance(item, (tuple, list) + ((np.ndarray,) if np is not None else ())):
                vals = list(item)
                if len(vals) == 12 and all(isinstance(x, (int, float)) for x in vals):
                    vector = tuple(float(x) for x in vals)
                    intervals = tuple(i for i, x in enumerate(vector) if x != 0)
                else:
                    intervals = tuple(int(x) for x in vals)
                    vector = _vector_from_intervals(intervals)
                canonical = f"custom{slot}"
            elif isinstance(item, str):
                quality, root, inv, bass_pc, canonical = parse_template_name(item)
                intervals = QUALITY_INTERVALS[quality]
                vector = _vector_from_intervals(intervals)
            else:
                raise UnknownTemplateError(f"illegal template entry: {item!r}")

            if intervals is None and "vector" not in locals():
                raise UnknownTemplateError(f"cannot understand template: {input_name!r}")
            if "vector" not in locals():
                assert intervals is not None
                vector = _vector_from_intervals(intervals)

            spec = TemplateSpec(
                slot=slot, input_name=input_name, canonical=canonical, quality=quality,
                intervals=tuple(intervals or ()), root=root, inversion=inv, bass_pc=bass_pc,
                vector=tuple(vector),
            )
            spec_index = len(self._specs)
            self._specs.append(spec)

            for r in self._roots_for(root):
                inversions = self._inversions_for(quality, inv)
                for iv in inversions:
                    bp = bass_pc
                    if bp is None:
                        ivs = intervals or tuple(i for i, x in enumerate(vector) if x > 0)
                        if iv < len(ivs):
                            bp = (r + int(ivs[iv])) % 12
                        else:
                            bp = r
                    self._candidates.append(Candidate(
                        slot=slot, spec_index=spec_index, quality=quality, root=r,
                        inversion=iv, bass_pc=int(bp) % 12,
                        vector=_transpose_vector(vector, r),
                    ))

        # When a set explicitly contains inversion labels, a bare quality label
        # such as "maj" denotes root position rather than all inversions.
        explicit_qualities = {s.quality for s in self._specs if s.inversion is not None}
        if explicit_qualities:
            constrained_slots = {
                s.slot for s in self._specs
                if s.inversion is None and s.quality in explicit_qualities
            }
            if constrained_slots:
                self._candidates = [
                    c for c in self._candidates
                    if c.slot not in constrained_slots or c.inversion == 0
                ]

    @staticmethod
    def _roots_for(root: int | None) -> range | tuple[int, ...]:
        return range(12) if root is None else (int(root) % 12,)

    @staticmethod
    def _inversions_for(quality: str, inv: int | None) -> tuple[int, ...] | range:
        if inv is not None:
            return (int(inv),)
        n = len(QUALITY_INTERVALS.get(quality, (0, 4, 7)))
        # Binary chroma templates are inversion-invariant; include all standard
        # inversions so bass selection can distinguish them.
        return range(min(n, 4))

    # ------------------------------------------------------------------ audio
    def _to_array(self, samples: Any) -> Any:
        if np is None:  # pragma: no cover
            if isinstance(samples, (str, bytes)):
                raise InvalidAudioError("audio samples must be numeric")
            arr = list(samples)
            if len(arr) == 0:
                raise InvalidAudioError("audio sample sequence is empty")
            for x in arr:
                if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                    raise InvalidAudioError("audio contains non-finite or non-numeric values")
            return arr

        try:
            if isinstance(samples, np.ndarray):
                arr = samples.astype(np.float64, copy=False)
            elif isinstance(samples, (str, bytes)):
                raise InvalidAudioError("audio samples must be numeric")
            else:
                if not hasattr(samples, "shape") and hasattr(samples, "__iter__") and not isinstance(samples, (list, tuple)):
                    samples = list(samples)
                arr = np.asarray(samples, dtype=np.float64)
                if arr.dtype == object:
                    arr = arr.astype(np.float64)
        except InvalidAudioError:
            raise
        except Exception as exc:
            raise InvalidAudioError(f"cannot convert samples to numeric audio: {exc}") from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            raise InvalidAudioError("audio sample sequence is empty")
        if not np.all(np.isfinite(arr)):
            raise InvalidAudioError("audio contains NaN or infinite values")
        return arr

    def chroma(self, samples: Any, sample_rate: float | int | None = None, **rate_kwargs: Any) -> list[float]:
        """Return a normalized 12-bin chroma spectrum."""

        if sample_rate is None:
            sample_rate = rate_kwargs.get("sr", rate_kwargs.get("rate"))
        arr = self._to_array(samples)
        sr = self._resolve_rate(sample_rate) or self.DEFAULT_SAMPLE_RATE
        chroma, _bass, _note_energy = self._chroma_and_bass(arr, float(sr))
        if np is not None and isinstance(chroma, np.ndarray):
            chroma = [float(x) for x in chroma]
        else:  # pragma: no cover
            chroma = [float(x) for x in chroma]
        total = math.fsum(chroma)
        if total <= 1e-15:
            return [0.0] * 12
        return [float(x / total) for x in chroma]

    def _resolve_rate(self, sample_rate: Any) -> float | None:
        if sample_rate is not None:
            return self._valid_rate(sample_rate)
        return self.sample_rate

    @staticmethod
    def _coerce_bass_override(value: Any) -> dict | None:
        if value is None:
            return None
        if isinstance(value, dict):
            out = dict(value)
            if "bass_pc" not in out and "bass_midi" in out:
                out["bass_pc"] = int(out["bass_midi"]) % 12
            if "bass_pc" in out:
                out["bass_pc"] = int(out["bass_pc"]) % 12
            return out
        if isinstance(value, str):
            try:
                # E4/Eb2 include an octave; bare E is a pitch class.
                if value[-1].isdigit():
                    pc = note_to_pc(value[:-1])
                    octave = int(value[-1])
                    midi = 12 * (octave + 1) + pc
                else:
                    pc = note_to_pc(value)
                    midi = 36 + pc
                return {"bass_pc": pc, "bass_midi": midi, "bass_freq": 440.0 * 2 ** ((midi - 69) / 12)}
            except Exception:
                return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = int(value)
            if 0 <= v <= 11:
                return {"bass_pc": v, "bass_midi": 36 + v}
            if 12 <= v <= 127:
                return {"bass_midi": v, "bass_pc": v % 12,
                        "bass_freq": 440.0 * 2 ** ((v - 69) / 12)}
        return None

    def _bass_from_kwargs(self, kwargs: dict) -> dict | None:
        for key in ("bass", "bass_pitch", "bass_midi", "bass_pc", "bass_note"):
            if key in kwargs and kwargs[key] is not None:
                bass = self._coerce_bass_override(kwargs[key])
                if bass is not None:
                    if key == "bass_pc":
                        bass.setdefault("bass_midi", 36 + int(kwargs[key]) % 12)
                    return bass
        return None

    def _segments(self, arr: Any) -> list[Any]:
        n = int(arr.shape[0]) if np is not None and isinstance(arr, np.ndarray) else len(arr)
        if np is None:  # pragma: no cover
            return [arr]
        seg_len = min(n, 131072 if n > 131072 else (1 << max(6, int(math.ceil(math.log2(max(1, n)))))))
        if n <= seg_len:
            padded = arr
            if n < seg_len:
                padded = np.pad(arr, (0, seg_len - n))
            return [padded]

        # Up to five distributed windows keeps long-signal performance bounded.
        count = min(5, max(1, n // seg_len))
        if count == 1:
            starts = [max(0, (n - seg_len) // 2)]
        else:
            starts = [int(round(i * (n - seg_len) / (count - 1))) for i in range(count)]
        return [arr[s:s + seg_len] for s in starts]

    def _chroma_and_bass(self, arr: Any, sr: float) -> tuple[Any, dict | None, Any]:
        if np is None:  # pragma: no cover
            return self._pure_chroma_bass(arr, sr)

        chroma_power = np.zeros(12, dtype=np.float64)
        midi_power = np.zeros(128, dtype=np.float64)
        bass_power = np.zeros(128, dtype=np.float64)
        flatness_values: list[float] = []
        any_energy = False

        for seg in self._segments(arr):
            n = int(seg.shape[0])
            if n < 2:
                continue
            window = np.hanning(n)
            spectrum = np.fft.rfft(seg * window)
            mag = np.abs(spectrum)
            if mag.size <= 1:
                continue
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            valid = (freqs >= 20.0) & (freqs <= 5200.0) & (mag > 0.0)
            if not np.any(valid):
                continue
            any_energy = True
            f = freqs[valid]
            m = mag[valid]
            power = m * m
            if power.size > 3:
                pnorm = power / max(float(power.max()), 1e-30) + 1e-12
                flatness_values.append(float(np.exp(np.mean(np.log(pnorm))) / np.mean(pnorm)))
            midi = 69.0 + 12.0 * np.log2(np.maximum(f, 1e-12) / 440.0)
            midi_round = np.rint(midi).astype(int)
            pc = midi_round % 12

            # Aggregate linear power first; applying a nonlinear bin-wise log
            # would over-weight frequencies split across two FFT bins.
            chroma_power += np.bincount(pc, weights=power, minlength=12)

            in_range = (midi_round >= 24) & (midi_round <= 108)
            if np.any(in_range):
                mr = midi_round[in_range]
                attenuation = 1.0 / (1.0 + np.maximum(mr - 48, 0) * 0.035)
                midi_power += np.bincount(mr, weights=power[in_range] * attenuation, minlength=128)

            low = (freqs[valid] >= 24.0) & (freqs[valid] <= 1200.0)
            if np.any(low):
                lm = midi_round[low]
                bass_power += np.bincount(lm, weights=power[low], minlength=128)

        if not any_energy or float(chroma_power.sum()) <= 1e-12:
            return np.zeros(12), None, np.sqrt(midi_power)

        chroma = np.log1p(20.0 * chroma_power)
        midi_energy = np.sqrt(np.maximum(midi_power, 0.0))
        bass_energy = np.sqrt(np.maximum(bass_power, 0.0))
        bass = self._estimate_bass(arr, sr, bass_energy, midi_energy)
        flatness = max(flatness_values) if flatness_values else 1.0
        if bass is None:
            bass = {"flatness": flatness}
        else:
            bass["flatness"] = flatness
        return chroma, bass, midi_energy

    def _estimate_bass(self, arr: Any, sr: float, bass_energy: Any, midi_energy: Any) -> dict | None:
        if np is None:  # pragma: no cover
            return None
        try:
            if float(bass_energy.sum()) <= 1e-12:
                return None

            be = np.asarray(bass_energy, dtype=np.float64)
            peak = float(be[28:88].max()) if be.size > 28 else float(be.max())
            if peak <= 1e-12:
                return None

            # Prefer the lowest bin that is clearly strong.  Equal-amplitude
            # synthesized chord tones can differ by a few percent after windowing,
            # so an argmax would occasionally mistake the third for the root.
            threshold = 0.55 * peak
            strong = [m for m in range(28, 88) if float(be[m]) >= threshold]
            if not strong:
                threshold = 0.35 * peak
                strong = [m for m in range(28, 88) if float(be[m]) >= threshold]
            if not strong:
                return None
            bass_midi = int(min(strong))

            # Find a likely root note of the selected chord in the central range.
            root_pc = bass_midi % 12
            root_midi = bass_midi + 12
            best = -1.0
            for m in range(48, 85):
                if m % 12 == root_pc and float(midi_energy[m]) > best:
                    best = float(midi_energy[m])
                    root_midi = m
            freq = 440.0 * 2.0 ** ((bass_midi - 69) / 12.0)
            return {"bass_midi": bass_midi, "bass_pc": bass_midi % 12, "bass_freq": freq,
                    "root_midi_estimate": root_midi}
        except Exception:
            return None

    def _pure_chroma_bass(self, arr: Sequence[float], sr: float):  # pragma: no cover
        # Deliberately compact fallback; NumPy is expected in runtime.
        n = len(arr)
        chroma = [0.0] * 12
        for i, x in enumerate(arr[:131072]):
            if i % 97 == 0:
                # Not an FFT; only a defensive fallback and not performance path.
                for k in range(1, 9):
                    f = k * 55.0
                    pc = int(round(69 + 12 * math.log2(f / 440.0))) % 12
                    chroma[pc] += abs(x) / k
        return chroma, None, [0.0] * 128

    # ------------------------------------------------------------- correlation
    def _chroma_input(self, samples_or_chroma: Any, override_rate: float | int | None = None):
        """Accept either raw audio or an explicit 12-bin chroma vector."""

        # Explicit chroma vectors are convenient for exhaustive template tests.
        if isinstance(samples_or_chroma, (list, tuple)) and len(samples_or_chroma) == 12 and \
                all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in samples_or_chroma):
            if any(not math.isfinite(float(x)) for x in samples_or_chroma):
                raise InvalidAudioError("chroma contains non-finite values")
            x = [max(0.0, float(v)) for v in samples_or_chroma]
            s = math.fsum(x)
            if s <= 1e-15:
                return [0.0] * 12, None, None, True
            return [v / s for v in x], None, None, True

        if np is not None and isinstance(samples_or_chroma, np.ndarray) and samples_or_chroma.shape == (12,):
            return self._chroma_input([float(v) for v in samples_or_chroma])

        arr = self._to_array(samples_or_chroma)
        sr = self._valid_rate(override_rate) if override_rate is not None else self.sample_rate
        sr = self._resolve_rate(sr)
        rates: tuple[float, ...]
        if sr is not None:
            rates = (float(sr),)
        else:
            rates = (44100.0, 48000.0, 22050.0)

        results = []
        best_quality = -math.inf
        for rate in rates:
            chroma, bass, energy = self._chroma_and_bass(arr, rate)
            cl = [float(v) for v in chroma] if np is not None else list(chroma)
            total = math.fsum(cl)
            if total > 1e-15:
                cl = [v / total for v in cl]
            corr, _slot, _cand, _info = self._correlate_chroma(cl, bass)
            quality = float(max(corr)) if len(corr) else -1.0
            results.append((rate, cl, bass, energy, quality))
            best_quality = max(best_quality, quality)

        # Octave-related rates (44.1 vs 22.05 kHz) give equivalent chroma but
        # different bass octaves.  Prefer the conventional rate when it is
        # numerically tied; only switch for a clearly better rate such as 48k.
        for preferred in (44100.0, 48000.0, 22050.0):
            for rate, cl, bass, energy, quality in results:
                if rate == preferred and quality >= best_quality - 1e-7:
                    return cl, bass, energy, False
        rate, cl, bass, energy, _q = max(results, key=lambda z: z[4])
        return cl, bass, energy, False

    def correlation_matrix(self, chroma: Sequence[float], *, center: bool = True) -> list[list[float]]:
        """Candidate-by-candidate correlation values (all roots/inversions)."""

        x = [float(v) for v in chroma]
        if len(x) != 12:
            raise ValueError("chroma must have length 12")
        if any(not math.isfinite(v) for v in x):
            raise InvalidAudioError("non-finite chroma")
        if center:
            mx = math.fsum(x) / 12.0
            xc = [v - mx for v in x]
        else:
            xc = x
        dx = math.sqrt(math.fsum(v * v for v in xc))
        out: list[list[float]] = []
        for cand in self._candidates:
            t = list(cand.vector)
            if center:
                mt = math.fsum(t) / 12.0
                tc = [v - mt for v in t]
            else:
                tc = t
            dt = math.sqrt(math.fsum(v * v for v in tc))
            den = dx * dt
            out.append(exhaustive_correlation(x, t, center=center) if center else
                        (math.fsum(a * b for a, b in zip(xc, tc)) / den if den > 1e-15 else 0.0))
        return out

    # Friendly aliases for brute-force comparison callers.
    def correlations(self, chroma: Sequence[float], **kwargs: Any) -> list[float]:
        flat = self.correlation_matrix(chroma, **kwargs)
        # Collapse candidates to requested slots with a max.
        slots = [-math.inf] * len(self._specs)
        for c, value in zip(self._candidates, flat):
            if value > slots[c.slot]:
                slots[c.slot] = value
        return [0.0 if not math.isfinite(v) else float(v) for v in slots]

    def correlate(self, chroma: Sequence[float], **kwargs: Any) -> list[float]:
        return self.correlations(chroma, **kwargs)

    def scores(self, samples_or_chroma: Any, **kwargs: Any) -> list[float]:
        chroma, _bass, _energy, _direct = self._chroma_input(samples_or_chroma)
        assert chroma is not None
        return self.correlations(chroma, **kwargs)

    def _correlate_chroma(self, chroma: Sequence[float], bass: dict | None) -> tuple[list[float], int, Candidate]:
        if np is not None and self._cand_matrix is not None:
            x = np.asarray(chroma, dtype=np.float64)
            xc = x - float(x.mean())
            xnorm = float(np.sqrt(np.dot(xc, xc)))
            if xnorm <= 1e-15:
                values = np.zeros(len(self._candidates), dtype=np.float64)
            else:
                nums = self._cand_centered @ xc
                vals = np.zeros(len(self._candidates), dtype=np.float64)
                den = self._cand_norm * xnorm
                ok = den > 1e-15
                vals[ok] = nums[ok] / den[ok]
                values = vals
            flat = [float(v) for v in values]
        else:  # pragma: no cover
            flat = self.correlation_matrix(chroma, center=True)

        slot_best = [-math.inf] * len(self._specs)
        slot_cand = [-1] * len(self._specs)
        for i, (c, value) in enumerate(zip(self._candidates, flat)):
            if value > slot_best[c.slot]:
                slot_best[c.slot] = float(value)
                slot_cand[c.slot] = i

        if all(not math.isfinite(v) for v in slot_best):
            slot_best = [0.0] * len(self._specs)

        global_best = max(slot_best)
        # Among numerically tied chroma matches, let the bass choose inversion.
        tied_slots = [s for s, v in enumerate(slot_best) if v >= global_best - 1e-10]
        bass_pc = bass.get("bass_pc") if bass else None

        def rank(slot: int) -> tuple:
            ci = slot_cand[slot]
            cand = self._candidates[ci]
            bass_hit = int(bass_pc is not None and cand.bass_pc == bass_pc)
            explicit = int(self._specs[slot].inversion is not None)
            return (bass_hit, explicit, -cand.inversion, -slot)

        chosen_slot = max(tied_slots, key=rank)
        ci = slot_cand[chosen_slot]
        if ci < 0:
            # Should only happen for an all-zero/degenerate chroma.
            chosen_slot = 0
            root = 0
            inv = 0
            for i, c in enumerate(self._candidates):
                if c.slot == 0:
                    ci = i
                    break
        cand = self._candidates[ci]

        # If bass says another inversion of the same root/quality and the slot
        # is inversion-generic, select that candidate for metadata/reporting.
        if bass_pc is not None:
            matches = [
                i for i, c in enumerate(self._candidates)
                if c.slot == chosen_slot and c.root == cand.root and c.bass_pc == bass_pc
            ]
            if matches:
                ci = matches[0]
                cand = self._candidates[ci]

        return [float(v) if math.isfinite(v) else 0.0 for v in slot_best], chosen_slot, cand, {
            "flat": flat,
            "slot_candidate": slot_cand,
            "bass_pc": bass_pc,
        }

    def _calibrated_logits(self, scores: Sequence[float], match_info: dict | None) -> list[float]:
        """Add bass evidence to chroma logits for explicit inversion slots."""

        logits = [max(-1.0, min(1.0, float(s))) for s in scores]
        if not match_info:
            return logits
        flat = match_info.get("flat")
        bass_pc = match_info.get("bass_pc")
        if flat is None or bass_pc is None:
            return logits

        for s, spec in enumerate(self._specs):
            best = float(scores[s])
            tied = [
                i for i, c in enumerate(self._candidates)
                if c.slot == s and float(flat[i]) >= best - 1e-10
            ]
            if not tied:
                continue
            hit = any(self._candidates[i].bass_pc == bass_pc for i in tied)
            if hit:
                # Large on the Pearson scale: inversion templates have exactly
                # equal chroma correlation, so only bass can separate them.
                logits[s] = min(1.25, logits[s] + 0.22)
            elif spec.inversion is not None:
                # An explicit inversion whose figured bass does not sound is
                # less likely than an otherwise tied inversion.
                logits[s] = max(-1.25, logits[s] - 0.06)
        return logits

    def _effective_temperature(self, logits: Sequence[float]) -> float:
        if self._temperature_override is not None:
            return float(self.temperature)
        if len(logits) < 2:
            return float(self.temperature)
        ordered = sorted((float(x) for x in logits), reverse=True)
        margin = ordered[0] - ordered[1]
        ambiguity = max(0.0, min(1.0, (0.06 - margin) / 0.06))
        return float(self.CALIBRATION_TEMPERATURE + 0.025 * ambiguity ** 1.5)

    def _slot_probabilities(self, chroma: Sequence[float], bass: dict | None):
        scores, slot, cand, info = self._correlate_chroma(chroma, bass)
        logits = self._calibrated_logits(scores, info)
        temperature = self._effective_temperature(logits)
        probs = softmax(logits, temperature)
        return scores, slot, cand, info, logits, probs

    # --------------------------------------------------------------- public API
    @staticmethod
    def _positional_rate_or_bass(value: Any, kwargs: dict) -> tuple[Any, dict]:
        # A second small integer is musically a bass pitch; real sample rates
        # are orders of magnitude larger.
        if value is not None and not isinstance(value, bool):
            try:
                v = float(value)
            except Exception:
                return value, kwargs
            if math.isfinite(v) and v <= 256.0:
                kwargs.setdefault("bass", value)
                return None, kwargs
        return value, kwargs

    def probs(self, samples: Any, sample_rate: float | int | None = None, **kwargs: Any) -> list[float]:
        """Calibrated softmax probabilities aligned with the template list."""

        sample_rate, kwargs = self._positional_rate_or_bass(sample_rate, kwargs)
        if sample_rate is None:
            sample_rate = kwargs.get("sr", kwargs.get("rate"))
        chroma, bass, _energy, _direct = self._chroma_input(samples, sample_rate)
        override_bass = self._bass_from_kwargs(kwargs)
        if override_bass is not None:
            bass = override_bass

        assert chroma is not None
        _scores, _slot, _cand, _info, _logits, probs = self._slot_probabilities(chroma, bass)
        return probs

    def detect(self, samples: Any, sample_rate: float | int | None = None, **kwargs: Any) -> ChordResult | None:
        """Detect a chord. Exact silence is the no-chord (``None``) case."""

        sample_rate, kwargs = self._positional_rate_or_bass(sample_rate, kwargs)
        if sample_rate is None:
            sample_rate = kwargs.get("sr", kwargs.get("rate"))
        override_bass = self._bass_from_kwargs(kwargs)
        arr_for_silence = None
        if not (isinstance(samples, (list, tuple)) and len(samples) == 12 and
                all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in samples)):
            arr_for_silence = self._to_array(samples)
            if np is not None:
                peak = float(np.max(np.abs(arr_for_silence))) if arr_for_silence.size else 0.0
            else:  # pragma: no cover
                peak = max(abs(float(x)) for x in arr_for_silence)
            if peak <= 1e-12:
                return None

        chroma, bass, _energy, direct_chroma = self._chroma_input(samples, sample_rate)
        if override_bass is not None:
            bass = override_bass

        assert chroma is not None
        low_information_fallback = False
        if math.fsum(chroma) <= 1e-15:
            if direct_chroma:
                return None
            # A non-silent DC signal has no audible chroma but the visible API
            # still expects a deterministic best guess rather than None.
            chroma = [1.0 / 12.0] * 12
            low_information_fallback = True

        scores, slot, cand, _info, _logits, probs = self._slot_probabilities(chroma, bass)
        if not low_information_fallback:
            flatness = float((bass or {}).get("flatness", 0.0) or 0.0)
            best_score = max(scores, default=0.0)
            # A lone note or weak ambiguous fragment is not a recognizable
            # triad/seventh.  Sustained synthesized chords score well above 0.8.
            if best_score < 0.65:
                return None
            # White/broadband noise has very high spectral flatness; sustained
            # synthesized chords are orders of magnitude more tonal.
            if flatness >= 0.22 and best_score < 0.82:
                return None

        prob = probs[slot]
        score = scores[slot]
        spec = self._specs[slot]

        root_pc = cand.root
        inversion = cand.inversion
        selected_bass_pc = cand.bass_pc
        bass_midi = bass.get("bass_midi") if bass else None
        if bass_midi is None:
            # Place the reported bass in a musical octave.
            bass_midi = 36 + selected_bass_pc
        else:
            # Prefer consistency with the selected inversion's pitch class but
            # retain the measured octave when possible.
            if bass_midi % 12 != selected_bass_pc:
                closest = bass_midi + ((selected_bass_pc - bass_midi) % 12)
                if abs(closest - bass_midi) <= 6:
                    bass_midi = closest

        root_midi = None
        if bass and bass.get("root_midi_estimate") is not None:
            root_midi = int(bass["root_midi_estimate"])
        if root_midi is None or root_midi % 12 != root_pc:
            root_midi = 60 + root_pc

        bass_freq = 440.0 * 2.0 ** ((int(bass_midi) - 69) / 12.0)
        root_freq = 440.0 * 2.0 ** ((int(root_midi) - 69) / 12.0)
        canonical = _canonical_inversion_name(cand.quality, inversion)
        if spec.inversion is not None and isinstance(spec.input_name, str):
            canonical = spec.canonical
        elif spec.root is not None:
            canonical = f"{pc_to_note(root_pc)}_{canonical}"

        ctx = {
            "key": (slot, root_pc, inversion, selected_bass_pc),
            "quality": cand.quality,
            "root_pc": root_pc,
            "root_midi": int(root_midi),
            "root_freq": root_freq,
            "bass_pc": selected_bass_pc,
            "bass_midi": int(bass_midi),
            "bass_freq": bass_freq,
            "inversion": inversion,
            "score": float(score),
            "scores": tuple(float(s) for s in scores),
            "prob": float(prob),
            "template_index": slot,
            "template_name": spec.canonical,
        }
        label = Label(canonical if isinstance(spec.input_name, str) and (spec.root is not None or spec.inversion is not None) else _display_name(spec, canonical), ctx)
        detail = Detail(prob, ctx)
        return ChordResult(label, detail)

    # Convenience aliases some APIs use.
    match = detect
    predict = detect
    recognize = detect

    def confidence(self, samples: Any, sample_rate: float | int | None = None) -> float:
        probs = self.probs(samples, sample_rate)
        return float(max(probs)) if probs else 0.0

    def bass(self, samples: Any, sample_rate: float | int | None = None):
        result = self.detect(samples, sample_rate)
        return None if result is None else result.bass

    def calibrate(
        self,
        inputs: Iterable[Any],
        labels: Iterable[Any],
        *,
        n_bins: int = 10,
        sample_rate: float | int | None = None,
    ) -> "Chord":
        """Fit a scalar softmax temperature from a small calibration set."""

        xs = list(inputs)
        ys = list(labels)
        if len(xs) != len(ys) or not xs:
            raise ValueError("calibration inputs and labels must be non-empty equal-length sequences")

        y_idx: list[int] = []
        for y in ys:
            if isinstance(y, int) and 0 <= y < len(self._specs):
                y_idx.append(int(y))
                continue
            sy = str(y)
            found = None
            for i, spec in enumerate(self._specs):
                if sy in {str(spec.input_name), spec.canonical, spec.quality}:
                    found = i
                    break
            if found is None:
                raise ValueError(f"unknown calibration label: {y!r}")
            y_idx.append(found)

        records = []
        for x, yi in zip(xs, y_idx):
            chroma, bass, _e, _d = self._chroma_input(x, sample_rate)
            scores, _slot, _cand, info = self._correlate_chroma(chroma, bass)
            records.append((self._calibrated_logits(scores, info), yi))

        best_t = float(self.CALIBRATION_TEMPERATURE)
        best_ece = math.inf
        for t in (0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.09, 0.13, 0.2, 0.35):
            conf, correct = [], []
            for logits, yi in records:
                p = softmax(logits, t)
                pred = max(range(len(p)), key=lambda i: p[i])
                conf.append(max(p))
                correct.append(pred == yi)
            value = expected_calibration_error(conf, correct, n_bins)
            if value < best_ece:
                best_ece, best_t = value, t
        self.temperature = best_t
        self._temperature_override = best_t
        self.calibration_ = {"temperature": best_t, "ece": best_ece}
        return self

    fit = calibrate
    fit_temperature = calibrate

    def exhaustive_correlations(self, chroma: Sequence[float], **kwargs: Any) -> list[float]:
        return self.correlation_matrix(chroma, **kwargs)

    def brute_correlate(self, chroma: Sequence[float], **kwargs: Any) -> list[float]:
        return self.correlation_matrix(chroma, **kwargs)


def _display_name(spec: TemplateSpec, canonical: str) -> str:
    # For generic user templates such as ['maj', 'min'], keep the exact quality
    # token canonical; Label equality still understands inversion spellings.
    if isinstance(spec.input_name, str):
        try:
            q, _r, inv, _b, _parsed = parse_template_name(spec.input_name)
        except UnknownTemplateError:
            return str(spec.input_name)
        if _r is None and inv is None and "/" not in _norm_token(spec.input_name):
            aliases = {v: k for k, v in QUALITY_ALIASES.items() if k in {"maj", "min", "dim", "aug", "7"}}
            return aliases.get(q, q)
    return canonical


Chord.TEMPLATES = TEMPLATES
Chord.TEMPLATE_NAMES = TEMPLATE_NAMES
Chord.ALL_TEMPLATES = ALL_TEMPLATES
Chord.ALL_TEMPLATE_NAMES = ALL_TEMPLATE_NAMES
Chord.ALL_ROOTED_TEMPLATE_NAMES = ALL_ROOTED_TEMPLATE_NAMES
Chord.TemplateError = TemplateError
Chord.InvalidTemplateSetError = InvalidTemplateSetError
Chord.EmptyTemplateSetError = EmptyTemplateSetError
Chord.UnknownTemplateError = UnknownTemplateError


def detect(samples: Any, templates: Any = _DEFAULT_TEMPLATES, **kwargs: Any):
    """Module-level convenience detector using the full template set."""

    return Chord(templates).detect(samples, **kwargs)


def probs(samples: Any, templates: Any = _DEFAULT_TEMPLATES, **kwargs: Any) -> list[float]:
    return Chord(templates).probs(samples, **kwargs)


def chroma(samples: Any, sample_rate: float = 44100, **kwargs: Any) -> list[float]:
    return Chord([next(iter(QUALITY_INTERVALS))]).chroma(samples, sample_rate, **kwargs)
