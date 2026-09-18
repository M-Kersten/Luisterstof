from pipeline.audio.chunker import chunk_script, eleven_text, line_chars
from pipeline.models import Line, Overlap, Script, Segment


def _line(i, speaker, n=100, seg=None):
    return Line(id=f"l{i:03d}", speaker=speaker, text=("woord " * (n // 6)).strip())


def test_blocks_respect_limit_and_seams():
    segs = []
    n = 0
    for seg_type in ("cold_open", "body", "body", "quiz"):
        lines = []
        for _ in range(12):
            n += 1
            lines.append(_line(n, "tessa" if n % 2 else "joris", 400))
        segs.append(Segment(type=seg_type, lines=lines))
    script = Script(episode_id="x", segments=segs)
    blocks = chunk_script(script, max_chars=3000)
    assert all(b.chars <= 3000 for b in blocks)
    assert [lid for b in blocks for lid in b.line_ids] == [l.id for l in script.lines()]
    assert not any(b.too_long for b in blocks)
    # a segment boundary closes a block that is at least 40% full
    first_body = next(i for i, b in enumerate(blocks) if "body" in b.segment_types)
    assert "cold_open" not in blocks[first_body].segment_types or blocks[first_body].chars < 3000 * 0.4


def test_giant_line_flagged_and_tags_counted():
    big = Line(id="l001", speaker="tessa", text="a " * 2000, tags=["laughs"], overlap=Overlap(mode="interrupt", target="l000"))
    script = Script(episode_id="x", segments=[Segment(type="body", lines=[big])])
    blocks = chunk_script(script)
    assert blocks[0].too_long
    assert eleven_text(big).startswith("[interrupting] [laughs] ")
    assert line_chars(big) == len(eleven_text(big))
