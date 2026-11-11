"""A book run survives a failed chapter and, run again, only does what is missing."""

import os

import pytest

from pipeline.models import Script
from pipeline.runner import StageError


def _tasks(p, since=0):
    return [c.task for c in p.llm.calls[since:]]


def test_a_second_run_skips_everything_that_is_done_and_current(ingested):
    p = ingested
    p.run_book("demo", upto="render", chapters=["ch01"])
    calls = len(p.llm.calls)
    events = len(p.events)
    p.run_book("demo", upto="render", chapters=["ch01"])
    assert _tasks(p, calls) == []  # nothing written, nothing audited, nothing rendered again
    skipped = {s for s, st, _ in p.events[events:] if st == "skipped"}
    assert skipped == {"plan", "glossary", "script", "approve", "render"}


def test_a_reingested_book_makes_the_old_plan_stale(ingested):
    p = ingested
    p.run_book("demo", upto="glossary", chapters=["ch01"])
    book_json = p.paths("demo").book_json
    later = os.stat(p.paths("demo").plan("ch01")).st_mtime + 5
    os.utime(book_json, (later, later))  # as if the PDF was read in again
    calls = len(p.llm.calls)
    p.run_book("demo", upto="glossary", chapters=["ch01"])
    assert "plan_section" in _tasks(p, calls) and "lexicon" in _tasks(p, calls)


def test_a_script_edited_by_hand_is_audited_again_never_rewritten(ingested):
    p = ingested
    p.run_book("demo", upto="script", chapters=["ch01"])
    script = Script.load(p.paths("demo").script("ch01"))
    script.segments[0].lines[0].text = "Dit heb ik zelf herschreven."
    edited = p.save_script("demo", "ch01", script)  # the web app's save: new revision, the audit is now stale
    calls = len(p.llm.calls)
    out = p.run_book("demo", upto="script", chapters=["ch01"])
    assert "script_segment" not in _tasks(p, calls) and "support" in _tasks(p, calls)
    assert Script.load(p.paths("demo").script("ch01")).segments[0].lines[0].text == "Dit heb ik zelf herschreven."
    assert out["ch01"]["audit"].script_revision == edited.revision


def test_a_failing_chapter_does_not_stop_the_others(ingested):
    p = ingested
    original = p.llm.handlers["plan_section"]

    def broken_for_ch02(req):
        if "Hoofdstuk ch02" in "".join(b["text"] for b in req.system):
            raise RuntimeError("model fell over")
        return original(req)

    p.llm.on("plan_section", broken_for_ch02)
    with pytest.raises(StageError, match=r"1 of 3 chapters failed \(ch02: model fell over\)"):
        p.run_book("demo", upto="plan")
    paths = p.paths("demo")
    assert paths.plan("ch01").is_file() and paths.plan("ch03").is_file() and not paths.plan("ch02").is_file()
    assert ("book", "chapter_failed") in {(s, st) for s, st, _ in p.events}

    p.llm.on("plan_section", original)  # the model is back: a rerun only makes ch02
    calls = len(p.llm.calls)
    p.run_book("demo", upto="plan")
    assert paths.plan("ch02").is_file()
    assert all("Hoofdstuk ch02" in "".join(b["text"] for b in c.system) for c in p.llm.calls[calls:] if c.task == "plan_section")


def test_fresh_redoes_finished_work(ingested):
    p = ingested
    p.run_book("demo", upto="plan", chapters=["ch01"])
    calls = len(p.llm.calls)
    p.run_book("demo", upto="plan", chapters=["ch01"], resume=False)
    assert "plan_section" in _tasks(p, calls)
