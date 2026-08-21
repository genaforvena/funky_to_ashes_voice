"""Caption parsing, and the one gate that separates a word-level track from a
line-level one.

Measured on the live corpus while building this: the SAME video's `en-orig`
json3 moves the clock on 100% of word boundaries while its plain `en` json3
moves it on 13% -- identical extension, identical shape, one is cuttable and one
is a lyric sheet.  Format is not granularity, so the gate reads the timestamps.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from funky.corpus import parse_captions, _distinct_start_ratio, MIN_MOVING  # noqa: E402

AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
<00:00:01.200><c> hello</c><00:00:01.600><c> there</c><00:00:02.100><c> world</c>
"""

LINE_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
♪ Cash rules everything around me ♪
"""

JSON3_WORD = """{"events":[{"tStartMs":1000,"segs":[
 {"utf8":"hello"},{"utf8":" there","tOffsetMs":400},{"utf8":" world","tOffsetMs":900}]}]}"""

JSON3_LINE = """{"events":[{"tStartMs":1000,"segs":[
 {"utf8":"cash rules everything around me"}]}]}"""


def w(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return parse_captions(p)


def test_auto_vtt_word_tags_become_units(tmp_path):
    got = w(tmp_path, "a.en.vtt", AUTO_VTT)
    assert [g[0] for g in got] == ["hello", "there", "world"]
    assert [g[1] for g in got] == [1200, 1600, 2100]
    assert got[0][2] == 1600, "a word ends where the next one starts"
    assert _distinct_start_ratio(got) == 1.0


def test_a_line_level_vtt_is_caught_by_the_gate(tmp_path):
    got = w(tmp_path, "b.en.vtt", LINE_VTT)
    assert got, "it still parses -- the point is that it must not pass the gate"
    assert _distinct_start_ratio(got) < MIN_MOVING


def test_json3_word_offsets(tmp_path):
    got = w(tmp_path, "c.en.json3", JSON3_WORD)
    assert [g[1] for g in got] == [1000, 1400, 1900]
    assert _distinct_start_ratio(got) == 1.0


def test_json3_without_offsets_is_line_level(tmp_path):
    got = w(tmp_path, "d.en.json3", JSON3_LINE)
    assert len(got) == 5
    assert _distinct_start_ratio(got) < MIN_MOVING


def test_the_dispatch_is_on_the_file_not_the_request(tmp_path):
    """`--sub-format json3` hands back vtt without complaining when a track
    publishes no json3, so the extension on disk is the only honest signal."""
    got = w(tmp_path, "e.en.vtt", AUTO_VTT)
    assert [g[0] for g in got] == ["hello", "there", "world"]
