from pipeline.models import ContinuityEntry
from pipeline.script.cast import append_continuity, cast_bible_text, continuity_text, load_continuity


def test_append_and_dedupe(cast_dir, cast):
    for i in range(3):
        append_continuity(cast_dir, ContinuityEntry(episode=f"ch0{i}", callbacks=[f"cb{i}"]))
    append_continuity(cast_dir, ContinuityEntry(episode="ch01", callbacks=["cb1-nieuw"]))
    entries = load_continuity(cast_dir)
    assert [e.episode for e in entries] == ["ch00", "ch02", "ch01"]
    assert entries[-1].callbacks == ["cb1-nieuw"]
    assert [e.episode for e in load_continuity(cast_dir, last_n=2)] == ["ch02", "ch01"]
    text = continuity_text(entries)
    assert "Aflevering ch02" in text and "cb1-nieuw" in text
    assert "eerste aflevering" in continuity_text([])
    bible = cast_bible_text(cast, cast.guests[0])
    assert "tessa" in bible and "Gast: Hanna Vos" in bible and "oké dus" in bible
