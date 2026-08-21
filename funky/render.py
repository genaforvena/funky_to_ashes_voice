"""Cuts -> one audio file, via ffmpeg only.

No crossfade between cuts by default.  The seam IS the material: each word
arrives with its own beat, key and room, and smoothing that boundary is an
attempt to make the collage sound like one voice, which is the opposite of the
point.  A short fade (default 8 ms) is applied at each edge purely to stop the
DC click of a hard cut mid-waveform -- short enough to be inaudible as a fade,
long enough that the seam is a seam and not a tick.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .solve import Cut, Plan

SR = 44100


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not on PATH")
    return exe


def render(plan: Plan, out_path: str | Path, fade_ms: int = 8,
           gap_ms: int = 0, quiet: bool = True) -> Path:
    out_path = Path(out_path)
    cuts = [c for c in plan.cuts if c.kind != "missing" and c.audio]
    if not cuts:
        raise SystemExit("nothing to render — every piece of the phrase is missing "
                         "from the corpus (add sources with `funky add`)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for i, cut in enumerate(cuts):
            part = tmp / f"{i:04d}.wav"
            _cut_one(cut, part, fade_ms, quiet)
            if part.exists() and part.stat().st_size > 44:
                parts.append(part)
            if gap_ms and i < len(cuts) - 1:
                sil = tmp / f"{i:04d}_gap.wav"
                _silence(sil, gap_ms, quiet)
                parts.append(sil)
        if not parts:
            raise SystemExit("every cut came back empty — the corpus audio may be missing")

        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
        if out_path.suffix.lower() == ".wav":
            cmd += ["-c:a", "pcm_s16le"]
        else:
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        cmd += [str(out_path)]
        _run(cmd, quiet)
    return out_path


def _cut_one(cut: Cut, dest: Path, fade_ms: int, quiet: bool) -> None:
    dur_ms = max(1, cut.end_ms - cut.start_ms)
    fade = min(fade_ms, dur_ms // 3) / 1000.0
    filters = [f"aformat=sample_fmts=s16:sample_rates={SR}:channel_layouts=stereo"]
    if fade > 0:
        filters.append(f"afade=t=in:st=0:d={fade:.4f}")
        filters.append(f"afade=t=out:st={max(0.0, dur_ms/1000 - fade):.4f}:d={fade:.4f}")
    # -ss BEFORE -i is the fast seek; -accurate_seek keeps it sample-honest, which
    # matters because a word is often shorter than an mp3 frame is forgiving.
    _run([_ffmpeg(), "-y", "-accurate_seek", "-ss", f"{cut.start_ms/1000:.3f}",
          "-t", f"{dur_ms/1000:.3f}", "-i", cut.audio,
          "-af", ",".join(filters), "-c:a", "pcm_s16le", str(dest)], quiet)


def _silence(dest: Path, ms: int, quiet: bool) -> None:
    _run([_ffmpeg(), "-y", "-f", "lavfi", "-i",
          f"anullsrc=r={SR}:cl=stereo", "-t", f"{ms/1000:.3f}",
          "-c:a", "pcm_s16le", str(dest)], quiet)


def _run(cmd: list[str], quiet: bool) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and not quiet:
        print(proc.stderr[-600:])
