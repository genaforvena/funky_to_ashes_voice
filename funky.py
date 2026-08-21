#!/usr/bin/env python3
"""funky to ashes voice — say a sentence in other people's tracks.

    funky add "Nas - N.Y. State of Mind"        # put a source in the corpus
    funky say "be like a black hole"            # cut that sentence out of it
    funky sources                               # what the corpus holds
    funky find "black hole"                     # where a phrase lives, no render

The corpus is built ONCE and searched offline.  The old version paid a Genius
lookup, a download and a transcription per phrase; here a word already in the
corpus costs a substring search.
"""

from __future__ import annotations

import argparse
import sys

from funky.corpus import Corpus, resolve
from funky.render import render
from funky.solve import Index, plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="funky", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=None, help="corpus directory (default ~/.funky)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add youtube sources (url, id, or free-text search)")
    a.add_argument("targets", nargs="+")
    a.add_argument("--lang", default="en")

    s = sub.add_parser("say", help="assemble a sentence out of the corpus")
    s.add_argument("text")
    s.add_argument("-o", "--out", default="out.mp3")
    s.add_argument("--fade", type=int, default=8, help="edge fade in ms (anti-click, not a blend)")
    s.add_argument("--gap", type=int, default=0, help="silence between cuts in ms")

    f = sub.add_parser("find", help="show the plan without rendering")
    f.add_argument("text")

    sub.add_parser("sources", help="list what the corpus holds")

    args = ap.parse_args(argv)
    corpus = Corpus(args.home) if args.home else Corpus()

    if args.cmd == "add":
        added = 0
        for t in args.targets:
            url = resolve(t)
            if not url:
                continue
            if corpus.add(url, lang=args.lang):
                added += 1
        print(f"{added} source(s) added")
        return 0

    if args.cmd == "sources":
        rows = corpus.sources()
        if not rows:
            print("corpus is empty — `funky add \"artist - track\"` first")
            return 1
        for r in rows:
            print(f"{r['video_id']}  {r['n_words']:5d} words  {r['title']}")
        print(f"{len(rows)} source(s), {sum(r['n_words'] for r in rows)} words")
        return 0

    index = Index(corpus)
    if not index.flat:
        print("corpus is empty — `funky add \"artist - track\"` first", file=sys.stderr)
        return 1

    p = plan(args.text, index)
    print(p.report())
    if args.cmd == "find":
        return 0

    out = render(p, args.out, fade_ms=args.fade, gap_ms=args.gap)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
