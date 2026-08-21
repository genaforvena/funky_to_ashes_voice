"""The solver, against a hand-built corpus — no network, no audio.

Every case here is one of the operator's stated rules, so a regression in the
assembly strategy fails by name rather than by a worse-sounding mp3.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from funky.corpus import Corpus, parse_vtt, parse_json3, _distinct_start_ratio  # noqa: E402
from funky.solve import Index, plan  # noqa: E402


def build(tmp_path, sources: dict[str, str]) -> Corpus:
    c = Corpus(tmp_path)
    for i, (title, text) in enumerate(sources.items(), start=1):
        words = text.split()
        c.db.execute(
            "INSERT INTO sources(video_id,title,url,audio,lang,n_words) VALUES(?,?,?,?,?,?)",
            (f"vid{i:08d}xx", title, "u", f"/tmp/{title}.mp3", "en", len(words)))
        sid = c.db.execute("SELECT id FROM sources WHERE title=?", (title,)).fetchone()["id"]
        c.db.executemany(
            "INSERT INTO words(source_id,idx,word,start_ms,end_ms) VALUES(?,?,?,?,?)",
            [(sid, j, w, j * 400, j * 400 + 400) for j, w in enumerate(words)])
    c.db.commit()
    return c


def test_a_phrase_that_exists_whole_is_taken_whole(tmp_path):
    """The operator's first case: maybe the entire line is already a quote."""
    c = build(tmp_path, {"A": "it was all a dream i used to read", "B": "all a dream"})
    p = plan("it was all a dream", Index(c))
    assert len(p.cuts) == 1
    assert len(p.sources) == 1
    assert p.cuts[0].kind == "exact"


def test_fewest_sources_falls_out_of_longest_first(tmp_path):
    c = build(tmp_path, {
        "long":  "the quick brown fox jumps",
        "short1": "the quick", "short2": "brown fox jumps",
    })
    p = plan("the quick brown fox jumps", Index(c))
    assert len(p.cuts) == 1, "a whole-phrase source must beat two partial ones"


def test_it_splits_when_no_single_source_has_the_whole_line(tmp_path):
    c = build(tmp_path, {"A": "never giving yourself away", "B": "be like a black hole"})
    p = plan("be like a black hole never giving yourself away", Index(c))
    assert [" ".join(cut.query_tokens) for cut in p.cuts] == [
        "be like a black hole", "never giving yourself away"]
    assert len(p.sources) == 2


def test_equal_length_spans_are_taken_from_the_middle(tmp_path):
    """'наверное, нужно искать где-то с середины начиная' — among spans of the
    same length the central one wins, so the cut lands on an edge of the phrase."""
    c = build(tmp_path, {"A": "aa bb", "B": "bb cc", "C": "cc dd"})
    p = plan("aa bb cc dd", Index(c))
    # Cuts come out in QUERY order; what the middle-first rule decides is WHICH
    # pair stayed whole. A left-to-right greedy solver yields aa bb | cc dd.
    got = [" ".join(cut.query_tokens) for cut in p.cuts]
    assert "bb cc" in got, f"expected the central pair to survive as one cut, got {got}"
    assert got == ["aa", "bb cc", "dd"]


def test_a_repeated_span_is_taken_from_a_source_not_yet_heard(tmp_path):
    """Diversity, not similarity: the same words exist twice, so the second cut
    must come from the other track."""
    c = build(tmp_path, {"A": "hello world stop", "B": "hello world go"})
    idx = Index(c)
    p = plan("hello world stop hello world go", idx)
    assert len(p.sources) == 2


def test_a_word_nowhere_in_the_corpus_is_cut_out_of_other_words(tmp_path):
    c = build(tmp_path, {"A": "stranger things happen"})
    p = plan("strange", Index(c))
    assert p.cuts, "a missing word must still produce something"
    assert all(cut.kind in ("sliced", "exact") for cut in p.cuts)
    assert "".join("".join(cut.query_tokens) for cut in p.cuts).startswith("strang")


def test_a_word_with_no_letters_in_common_is_reported_missing_not_faked(tmp_path):
    c = build(tmp_path, {"A": "aaa"})
    p = plan("zzz", Index(c))
    assert [cut.kind for cut in p.cuts] == ["missing"]
    assert "missing" in p.report()


def test_word_boundaries_are_real(tmp_path):
    c = build(tmp_path, {"A": "catalogue of ships"})
    p = plan("cat", Index(c))
    assert p.cuts[0].kind == "sliced", "'cat' must not match inside 'catalogue' as a whole word"
