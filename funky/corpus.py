"""The corpus: youtube video -> audio on disk + word-level units in sqlite.

The whole point of this file is that a word is found ONCE, not once per phrase.
Auto-captions carry per-word start offsets (json3 `segs[].tOffsetMs`), so no
transcription model is involved anywhere -- no key, no GPU, no rate limit past
the download itself.

json3 gives the START of a word and never its end.  A unit therefore ends where
the next one starts, which drags the instrumental in with it.  That is the
instrument, not an error: a word cut out of a mix carries its own beat.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOME = Path(os.environ.get("FUNKY_HOME", Path.home() / ".funky"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id         INTEGER PRIMARY KEY,
    video_id   TEXT UNIQUE NOT NULL,
    title      TEXT NOT NULL,
    url        TEXT NOT NULL,
    audio      TEXT NOT NULL,
    lang       TEXT NOT NULL,
    n_words    INTEGER NOT NULL,
    added_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS words (
    source_id  INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    word       TEXT NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    PRIMARY KEY (source_id, idx)
);
CREATE INDEX IF NOT EXISTS words_by_word ON words(word);
"""

WORD_RE = re.compile(r"[a-z0-9']+", re.UNICODE)
_CYR = re.compile(r"[Ѐ-ӿ]")


def normalize(text: str) -> list[str]:
    """Text -> the token alphabet the index is keyed on.

    One alphabet for the corpus AND for the query.  The old implementation
    searched Genius lyrics and cut against a Whisper transcript -- two different
    alphabets, so a phrase that existed in one was silently absent from the other.
    """
    text = text.lower().replace("’", "'")
    if _CYR.search(text):
        return [t for t in re.findall(r"[\w']+", text, re.UNICODE) if t]
    return WORD_RE.findall(text)


@dataclass(frozen=True)
class Unit:
    """One cuttable moment of audio."""
    source_id: int
    video_id: str
    title: str
    audio: str
    idx: int
    word: str
    start_ms: int
    end_ms: int


class Corpus:
    def __init__(self, home: Path | str = DEFAULT_HOME):
        self.home = Path(home)
        self.audio_dir = self.home / "audio"
        self.subs_dir = self.home / "subs"
        for d in (self.home, self.audio_dir, self.subs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.home / "corpus.db")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------------------------------------------------------------- ingest

    def has(self, video_id: str) -> bool:
        cur = self.db.execute("SELECT 1 FROM sources WHERE video_id = ?", (video_id,))
        return cur.fetchone() is not None

    def add(self, url: str, lang: str = "en", quiet: bool = False) -> int | None:
        """Download audio + auto-captions for one video and index its words.

        Returns the number of words indexed, or None if the video yielded no
        word-level captions (which is a real and common outcome -- a video with
        no auto-captions simply cannot join the corpus).
        """
        vid = video_id_of(url)
        if self.has(vid):
            if not quiet:
                print(f"already indexed: {vid}", file=sys.stderr)
            return 0

        out = self.subs_dir / vid
        cmd = [
            _ytdlp(), "--no-progress", "--no-warnings",
            "-f", "bestaudio/best", "-x", "--audio-format", "mp3", "--audio-quality", "5",
            # AUTO captions only.  A human-uploaded track (and, on official music
            # videos, the label-supplied "en" that YouTube files under automatic
            # captions) is LINE-level: correctly spelled, beautifully punctuated,
            # and carrying no word timing at all.  Asking for --write-subs pulls
            # exactly that track and it wins, so the request is narrowed here.
            "--write-auto-subs",
            "--sub-langs", f"{lang}-orig,{lang}", "--sub-format", "json3/vtt",
            "--paths", f"home:{self.audio_dir}", "--paths", f"subtitle:{self.subs_dir}",
            "-o", "%(id)s.%(ext)s",
            "--print-to-file", "%(title)s", str(out.with_suffix(".title")),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and not (self.audio_dir / f"{vid}.mp3").exists():
            print(f"yt-dlp failed for {vid}: {proc.stderr.strip()[-400:]}", file=sys.stderr)
            return None

        audio = self.audio_dir / f"{vid}.mp3"
        if not audio.exists():
            print(f"no audio for {vid}", file=sys.stderr)
            return None

        sub, sub_lang = self._pick_subs(vid, lang)
        if sub is None:
            print(f"no word-level captions for {vid} -- not indexable", file=sys.stderr)
            audio.unlink(missing_ok=True)
            return None

        words = parse_captions(sub)
        if not words:
            print(f"captions for {vid} carry no word offsets", file=sys.stderr)
            audio.unlink(missing_ok=True)
            return None
        moving = _distinct_start_ratio(words)
        if moving < MIN_MOVING:
            # Every word in a caption EVENT sharing one timestamp is what a
            # line-level track looks like after parsing: well-formed rows, real
            # numbers, and a clock that does not move inside a line.  Cutting
            # from it yields whole lines wearing a word's name, so it is refused
            # here rather than silently indexed.
            print(f"{vid}: captions are line-level ({moving:.0%} of words move the "
                  f"clock, need {MIN_MOVING:.0%}) — not word-cuttable", file=sys.stderr)
            audio.unlink(missing_ok=True)
            return None

        title = vid
        tf = out.with_suffix(".title")
        if tf.exists():
            title = tf.read_text().strip().splitlines()[0] or vid

        cur = self.db.execute(
            "INSERT INTO sources(video_id,title,url,audio,lang,n_words) VALUES(?,?,?,?,?,?)",
            (vid, title, url, str(audio), sub_lang, len(words)),
        )
        sid = cur.lastrowid
        self.db.executemany(
            "INSERT INTO words(source_id,idx,word,start_ms,end_ms) VALUES(?,?,?,?,?)",
            [(sid, i, w, s, e) for i, (w, s, e) in enumerate(words)],
        )
        self.db.commit()
        if not quiet:
            print(f"indexed {len(words)} words from {title!r} ({vid})", file=sys.stderr)
        return len(words)

    def _pick_subs(self, vid: str, lang: str) -> tuple[Path | None, str]:
        # `-orig` is the machine transcript of what was actually sung; the plain
        # code can be a translation or the label's lyric sheet.  Order matters.
        for cand in (f"{vid}.{lang}-orig.json3", f"{vid}.{lang}-orig.vtt",
                     f"{vid}.{lang}.json3", f"{vid}.{lang}.vtt"):
            p = self.subs_dir / cand
            if p.exists():
                return p, cand.rsplit(".", 2)[-2]
        for ext in ("json3", "vtt"):
            hits = sorted(self.subs_dir.glob(f"{vid}.*.{ext}"))
            if hits:
                return hits[0], hits[0].name.rsplit(".", 2)[-2]
        return None, ""

    # ---------------------------------------------------------------- lookup

    def sources(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM sources ORDER BY id"))

    def token_streams(self) -> dict[int, tuple[sqlite3.Row, list[sqlite3.Row]]]:
        """source_id -> (source row, its words in order).  Loaded once per run."""
        out: dict[int, tuple[sqlite3.Row, list[sqlite3.Row]]] = {}
        for src in self.sources():
            rows = list(self.db.execute(
                "SELECT * FROM words WHERE source_id = ? ORDER BY idx", (src["id"],)))
            out[src["id"]] = (src, rows)
        return out

    def unit(self, src: sqlite3.Row, row: sqlite3.Row) -> Unit:
        return Unit(src["id"], src["video_id"], src["title"], src["audio"],
                    row["idx"], row["word"], row["start_ms"], row["end_ms"])


# -------------------------------------------------------------------- helpers

def _ytdlp() -> str:
    exe = shutil.which("yt-dlp")
    if not exe:
        raise SystemExit("yt-dlp not on PATH")
    return exe


_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def resolve(target: str) -> str | None:
    """A url, a bare id, or free text -> a watch url.

    Free text goes through `ytsearch1:`.  The search is the only place a track
    is chosen for you; everything downstream works off ids that are already fixed.
    """
    try:
        return "https://www.youtube.com/watch?v=" + video_id_of(target)
    except SystemExit:
        pass
    proc = subprocess.run(
        [_ytdlp(), "--no-warnings", "--skip-download", "--print", "id",
         f"ytsearch1:{target}"],
        capture_output=True, text=True)
    ids = [l.strip() for l in proc.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_-]{11}", l.strip())]
    if not ids:
        print(f"search found nothing for {target!r}", file=sys.stderr)
        return None
    return "https://www.youtube.com/watch?v=" + ids[0]


def video_id_of(url: str) -> str:
    m = _YT_ID.search(url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise SystemExit(f"cannot read a youtube id out of {url!r}")


TAIL_MS = 700          # last word of a caption event has no successor
MAX_UNIT_MS = 2500     # a gap longer than this is silence, not a held word
MIN_MOVING = 0.55      # share of words that must start later than the one before


def _distinct_start_ratio(words: list[tuple[str, int, int]]) -> float:
    """How often the clock actually moves from one word to the next.

    This is the one number that separates a word-level transcript from a
    line-level one, and it cannot be read off the file's format: json3 carries
    per-word `tOffsetMs` only when YouTube generated the track, and the same
    extension holds line-level events whose segs have no offsets at all.
    """
    if len(words) < 2:
        return 0.0
    moved = sum(1 for i in range(1, len(words)) if words[i][1] > words[i - 1][1])
    return moved / (len(words) - 1)


def parse_captions(path: Path) -> list[tuple[str, int, int]]:
    """Dispatch on what the file IS, not on what was requested.

    `--sub-format json3` falls back to vtt without complaining when a track
    publishes no json3, so the extension on disk is the only honest signal.
    """
    path = Path(path)
    if path.suffix == ".vtt":
        return parse_vtt(path)
    return parse_json3(path)


_VTT_TS = re.compile(r"<(\d{2}):(\d{2}):(\d{2})\.(\d{3})>")
_VTT_CUE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})")
# <c>…</c> wraps every auto-captioned word.  Left in, `c` becomes a token of its
# own and the index fills with a word nobody ever said.
_VTT_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def parse_vtt(path: Path) -> list[tuple[str, int, int]]:
    """YouTube auto-caption vtt -> [(word, start_ms, end_ms)].

    An auto-generated cue interleaves inline `<00:00:12.345>` stamps between
    words; a human/label track has none, and then every word in the cue inherits
    the cue start -- which `_distinct_start_ratio` catches upstream.
    """
    raw: list[tuple[str, int]] = []
    cue_start = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _VTT_CUE.match(line.strip())
        if m:
            h, mi, sec, ms = (int(x) for x in m.groups()[:4])
            cue_start = ((h * 60 + mi) * 60 + sec) * 1000 + ms
            continue
        if not line.strip() or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        pos, at = 0, cue_start
        for m in _VTT_TS.finditer(line):
            for t in normalize(_VTT_TAG.sub(" ", line[pos:m.start()])):
                raw.append((t, at))
            h, mi, sec, ms = (int(x) for x in m.groups())
            at = ((h * 60 + mi) * 60 + sec) * 1000 + ms
            pos = m.end()
        for t in normalize(_VTT_TAG.sub(" ", line[pos:])):
            raw.append((t, at))

    # A rolling caption repeats the previous line under the new one; the repeats
    # arrive with the SAME stamps, so drop a (word, start) pair already seen.
    seen, out = set(), []
    for w, t in raw:
        if (w, t) in seen:
            continue
        seen.add((w, t))
        out.append((w, t))
    return _close_ends(out)


def parse_json3(path: Path) -> list[tuple[str, int, int]]:
    """json3 auto-captions -> [(word, start_ms, end_ms)].

    A word's end is the NEXT word's start.  json3 does not carry ends, and
    inventing one from a duration model would be a fabricated number.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: list[tuple[str, int]] = []
    for ev in data.get("events", []):
        base = ev.get("tStartMs")
        if base is None:
            continue
        for seg in ev.get("segs", []) or []:
            text = (seg.get("utf8") or "")
            toks = normalize(text)
            if not toks:
                continue
            start = base + (seg.get("tOffsetMs") or 0)
            # a seg is normally one word; when it is not, the offsets inside it
            # do not exist, so every token in it shares the seg's start.
            for t in toks:
                raw.append((t, start))

    return _close_ends(raw)


def _close_ends(raw: list[tuple[str, int]]) -> list[tuple[str, int, int]]:
    raw = sorted(raw, key=lambda x: x[1])
    out: list[tuple[str, int, int]] = []
    for i, (word, start) in enumerate(raw):
        if i + 1 < len(raw):
            end = raw[i + 1][1]
            if end <= start:
                end = start + 200
            end = min(end, start + MAX_UNIT_MS)
        else:
            end = start + TAIL_MS
        out.append((word, start, end))
    return out
