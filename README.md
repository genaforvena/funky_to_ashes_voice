# funky to ashes voice

Type a sentence. Get it back as an mp3, spoken by other people's records — each word cut
out of the track that happens to sing it, at the moment it is sung, instrumental and all.

The instrumental bleeding through every word is the **point**, not an artefact. A word
carries its own beat, key and room out of the mix with it; the result is a collage of
strangers, not an imitation of a voice. Nothing here tries to make it sound clean.

## How it finds the words

The hard question is not cutting — it is **which track**. Two different searches answer it:

1. **The whole line, first.** Maybe the sentence already exists as a quote somewhere. The
   solver always tries the longest span before any shorter one, so if one track sings the
   entire input, the output is one cut from one track.
2. **What is left, out of the index.** Every source you add contributes its words to a
   local index (`word → track, millisecond`). Searching it is a substring scan: instant,
   offline, no API key, no rate limit, no model.

When the whole line is not there, the sentence is **halved, and halved again** — always
taking the longest span that exists anywhere, then recursing on what is left of either
side. Fewest tracks is not optimised directly; it falls out of longest-first, because every
extra span costs at least one more cut.

Among spans of equal length, the **most central** one wins. Growing the window outward from
the middle puts the seam on an *edge* of the phrase rather than at its start, so the output
degrades gradually as the corpus thins instead of chopping off the first word.

When a word exists in no track at all, the fallback does **not** look for the
closest-sounding substitute. It covers the word's letters with pieces of *other* words,
preferring a source it has not just used — the tie-break is **variety, not similarity**. A
collage of one voice is a bad voice; a collage of many is the instrument.

Every piece is labelled in the plan (`exact` / `sliced` / `missing`), so nothing approximate
is ever passed off as a match.

## Use it

```bash
python funky.py add "Nas - N.Y. State of Mind"      # url, video id, or free-text search
python funky.py add https://www.youtube.com/watch?v=_JZom_gVfuw

python funky.py find "it was all a dream"          # show the plan, render nothing
python funky.py say  "it was all a dream" -o out.mp3
python funky.py sources
```

```
$ python funky.py find "be like a black hole never giving yourself away"
2 cut(s) from 2 source(s)
  [exact] 'be like a black hole'  1.98s  <- …
  [exact] 'never giving yourself away'  2.31s  <- …
```

Needs `yt-dlp` and `ffmpeg` on PATH. That is all — **no Genius key, no Groq key, no GPU,
no transcription model**.

## Where the timestamps come from

YouTube's own **automatic captions**, pulled once per source as `json3` (or `vtt`), which
carry a start offset per word. No speech model runs anywhere in this project.

Two things that are true and worth knowing before you build a corpus:

* **A word's END is not published** — only its start. A cut therefore runs to the *next*
  word's start and drags the instrumental tail in with it. That is the sound of the thing.
* **Most official music videos cannot be indexed.** Their English track is the label's
  lyric sheet: correctly spelled, beautifully punctuated, and timed *per line*. It looks
  identical in shape to a machine transcript and it has no word timing at all. Measured on
  one video here, the machine track (`en-orig`) moves the clock at **100%** of word
  boundaries and the lyric sheet at **13%** — same file extension, same JSON. So the
  granularity is read off the *timestamps*, never the format, and a line-level source is
  refused rather than indexed into word-shaped rows that are all secretly whole lines.
  Lyric-video uploads and anything speech-heavy index far more reliably than official
  uploads.

## What this replaced

The previous version asked Groq for `verbose_json` **without** `timestamp_granularities`,
so it received whole sung *lines*, stored each line in a field named `word`, and computed
character offsets across the concatenation. The README promised word-level cutting the code
could not produce. It also searched Genius **lyrics** and cut against a Whisper
**transcript** — two different alphabets, so a phrase present in one was silently absent
from the other. And it paid a search, a download and a transcription *per phrase*, which
made a sentence expensive and a second sentence just as expensive again.

Same idea, none of the keys, and the corpus is built once.

## License

[CC0 1.0 Universal](LICENSE) — public domain.
