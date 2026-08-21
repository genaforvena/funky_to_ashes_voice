"""Phrase -> a list of cuts, taken from as few sources as possible.

The operator's rule, in his words: try the whole line first; when it does not
exist anywhere, halve it; assemble the sentence out of the FEWEST tracks you
can.  Fewest tracks is not optimised directly -- it falls out of always taking
the LONGEST span that exists, because every extra span is at least one more cut
and usually one more track.

Two details that are his and not obvious:

* **Grow from the middle.**  Among spans of equal length the most central one
  wins.  A cut then lands on the EDGE of the phrase rather than at its start, so
  the degradation as the corpus thins is gradual instead of an abrupt chop at
  the first word.
* **When a word is missing, do not look for the closest-sounding unit -- look
  for the most VARIED one.**  A collage of one voice is a bad imitation of a
  voice; a collage of many is the instrument.  So the tie-break is *unlike what
  you already used*, never *like the target*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus, Unit, normalize


@dataclass
class Cut:
    """One contiguous piece of one source, and what of the query it stands for."""
    query_tokens: list[str]
    units: list[Unit]
    kind: str                  # exact | sliced | missing
    start_ms: int
    end_ms: int
    note: str = ""

    @property
    def source_id(self) -> int:
        return self.units[0].source_id if self.units else -1

    @property
    def title(self) -> str:
        return self.units[0].title if self.units else "(nothing)"

    @property
    def audio(self) -> str:
        return self.units[0].audio if self.units else ""


@dataclass
class Occurrence:
    source_id: int
    first_idx: int
    n: int


class Index:
    """Every source as one flat token string, searched by substring.

    A corpus this size (tens of tracks) does not need an n-gram table, and a
    linear scan is honest about its cost.  Word boundaries are real spaces, so
    ' cat ' can never match inside ' catalogue '.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.streams = corpus.token_streams()
        self.flat: dict[int, str] = {}
        self.offsets: dict[int, list[int]] = {}
        for sid, (_src, rows) in self.streams.items():
            offs, pos, parts = [], 1, []
            for r in rows:
                offs.append(pos)
                parts.append(r["word"])
                pos += len(r["word"]) + 1
            self.flat[sid] = " " + " ".join(parts) + " "
            self.offsets[sid] = offs

    def find_span(self, tokens: list[str]) -> list[Occurrence]:
        if not tokens:
            return []
        needle = " " + " ".join(tokens) + " "
        out: list[Occurrence] = []
        for sid, hay in self.flat.items():
            start = hay.find(needle)
            while start != -1:
                offs = self.offsets[sid]
                idx = _bisect(offs, start + 1)
                if idx is not None:
                    out.append(Occurrence(sid, idx, len(tokens)))
                start = hay.find(needle, start + 1)
        return out

    def units_of(self, occ: Occurrence) -> list[Unit]:
        src, rows = self.streams[occ.source_id]
        return [self.corpus.unit(src, rows[i])
                for i in range(occ.first_idx, occ.first_idx + occ.n)]

    def all_words(self) -> list[Unit]:
        out = []
        for sid, (src, rows) in self.streams.items():
            out.extend(self.corpus.unit(src, r) for r in rows)
        return out


def _bisect(offsets: list[int], pos: int) -> int | None:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] == pos:
            return mid
        if offsets[mid] < pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


@dataclass
class Plan:
    cuts: list[Cut] = field(default_factory=list)

    @property
    def sources(self) -> set[int]:
        return {c.source_id for c in self.cuts if c.source_id >= 0}

    def report(self) -> str:
        lines = [f"{len(self.cuts)} cut(s) from {len(self.sources)} source(s)"]
        for c in self.cuts:
            words = " ".join(c.query_tokens)
            if c.kind == "missing":
                lines.append(f"  [missing] {words!r} — nothing in the corpus")
            else:
                dur = (c.end_ms - c.start_ms) / 1000
                lines.append(f"  [{c.kind}] {words!r}  {dur:.2f}s  <- {c.title}"
                             + (f"  ({c.note})" if c.note else ""))
        return "\n".join(lines)


def plan(text: str, index: Index) -> Plan:
    tokens = normalize(text)
    spans = _split(tokens, 0, index)
    p = Plan()
    used: list[int] = []
    for q_start, occs, toks in spans:
        if occs:
            occ = _most_diverse(occs, used)
            units = index.units_of(occ)
            used.append(occ.source_id)
            p.cuts.append(Cut(toks, units, "exact", units[0].start_ms, units[-1].end_ms))
        else:
            for t in toks:
                p.cuts.extend(_fallback(t, index, used))
    return p


def _split(tokens: list[str], q_start: int, index: Index):
    """Longest-span-first, most-central-first, then recurse either side."""
    if not tokens:
        return []
    hit = _longest_from_middle(tokens, index)
    if hit is None:
        return [(q_start, [], tokens)]
    s, length, occs = hit
    left = _split(tokens[:s], q_start, index)
    here = [(q_start + s, occs, tokens[s:s + length])]
    right = _split(tokens[s + length:], q_start + s + length, index)
    return left + here + right


def _longest_from_middle(tokens: list[str], index: Index):
    n = len(tokens)
    centre = n / 2
    for length in range(n, 0, -1):
        starts = sorted(range(0, n - length + 1),
                        key=lambda s: (abs((s + length / 2) - centre), s))
        for s in starts:
            occs = index.find_span(tokens[s:s + length])
            if occs:
                return s, length, occs
    return None


def _most_diverse(occs: list[Occurrence], used: list[int]) -> Occurrence:
    """Prefer a source this render has not touched, then the least-used one.

    Deliberately NOT 'the best match' -- every occurrence here is an exact
    match already, so the only axis left is where the sound comes from.
    """
    def score(o: Occurrence):
        return (used.count(o.source_id), used[-1:] == [o.source_id], o.source_id, o.first_idx)
    return sorted(occs, key=score)[0]


# ------------------------------------------------------------------ fallback

MIN_CHUNK = 2


def _fallback(target: str, index: Index, used: list[int]) -> list[Cut]:
    """The word is nowhere in the corpus: cover its letters with pieces of other
    words, taking each piece from a source that has not just been heard.

    Honest about what this is: json3 timestamps a WORD, not a phoneme, so a
    sub-word piece is a PROPORTIONAL slice of the word's audio by character
    position.  Letters are not evenly spaced in time, so the cut is approximate
    by construction -- which is why it is the last rung, and why it is labelled
    `sliced` in the plan instead of being passed off as a match.
    """
    words = index.all_words()
    cuts: list[Cut] = []
    pos = 0
    guard = 0
    while pos < len(target) and guard < 40:
        guard += 1
        best = None
        for size in range(len(target) - pos, MIN_CHUNK - 1, -1):
            chunk = target[pos:pos + size]
            cands = [(u, u.word.find(chunk)) for u in words if chunk in u.word]
            if not cands:
                continue
            cands.sort(key=lambda c: (used.count(c[0].source_id),
                                      used[-1:] == [c[0].source_id],
                                      len(c[0].word), c[0].source_id, c[0].idx))
            best = (chunk, cands[0][0], cands[0][1])
            break
        if best is None:
            pos += 1
            continue
        chunk, unit, at = best
        dur = unit.end_ms - unit.start_ms
        a = unit.start_ms + int(dur * at / max(1, len(unit.word)))
        b = unit.start_ms + int(dur * (at + len(chunk)) / max(1, len(unit.word)))
        if b <= a:
            b = a + 80
        used.append(unit.source_id)
        # A chunk that IS the whole word is a real unit with real boundaries, not a
        # proportional guess — calling it `sliced` would overstate the doubt just as
        # calling a guess `exact` would understate it.
        whole = chunk == unit.word
        cuts.append(Cut([chunk], [unit], "exact" if whole else "sliced",
                        unit.start_ms if whole else a,
                        unit.end_ms if whole else b,
                        note=f"{chunk!r}" + ("" if whole else f" out of {unit.word!r}")))
        pos += len(chunk)
    if not cuts:
        cuts.append(Cut([target], [], "missing", 0, 0))
    return cuts
