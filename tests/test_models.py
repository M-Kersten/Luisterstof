import pytest

from pipeline.models import Glossary, LexiconEntry, Line, Overlap, Script, Segment
from pipeline.paths import BookPaths


def _script():
    return Script(episode_id="ch01", segments=[Segment(type="body", lines=[
        Line(id="l001", speaker="tessa", text="Hallo."),
        Line(id="l002", speaker="joris", text="Wacht even—"),
        Line(id="l003", speaker="tessa", text="Ja.", overlap=Overlap(mode="interrupt", target="l002", cut_word="even")),
    ])])


def test_duplicate_line_ids_rejected():
    with pytest.raises(ValueError):
        Script(episode_id="x", segments=[Segment(type="body", lines=[
            Line(id="l001", speaker="a", text="x"), Line(id="l001", speaker="b", text="y")])])


def test_next_id_and_renumber():
    s = _script()
    assert s.next_line_id() == "l004"
    s.segments[0].lines[0].id = "l010"
    s.renumber()
    assert [l.id for l in s.lines()] == ["l001", "l002", "l003"]
    assert s.line("l003").overlap.target == "l002"


def test_fragment_detection_and_estimate():
    s = _script()
    assert s.line("l002").ends_with_fragment
    assert not s.line("l001").ends_with_fragment
    assert s.estimated_seconds(15.0) > 0


def test_glossary_merge_respects_locks():
    g = Glossary(book_id="b")
    g.merge([LexiconEntry(surface="AI", kind="abbreviation", spoken="A I", lock=True)])
    added = g.merge([LexiconEntry(surface="ai", kind="abbreviation", spoken="aai"),
                     LexiconEntry(surface="ML", kind="abbreviation", spoken="M L")])
    assert added == 1
    assert g.find("AI").spoken == "A I"
    assert [e.surface for e in g.entries] == ["AI", "ML"]


def test_paths_reject_traversal(tmp_path):
    p = BookPaths(tmp_path, "book1").ensure()
    assert p.plan("ch03").name == "ch03.plan.json"
    with pytest.raises(ValueError):
        p.resolve_artifact("../../etc/passwd")
    with pytest.raises(ValueError):
        BookPaths(tmp_path, "../x")
    assert p.resolve_artifact("out/ch03.mp3") == (p.root / "out" / "ch03.mp3").resolve()
