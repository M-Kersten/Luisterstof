from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from pipeline.config import Settings
from pipeline.fake_handlers import default_fake_llm
from pipeline.ingest.run import ingest_book
from pipeline.models import Book, Chapter, ContentPlan, Script, Section
from pipeline.plan.content_plan import resolve_span
from pipeline.runner import Pipeline
from pipeline.sample_book import build
from pipeline.script.cast import load_banned, load_cast

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "sample.pdf")


@pytest.fixture(scope="session")
def sample_pdf_notoc(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "sample_notoc.pdf", with_toc=False)


@pytest.fixture
def cast_dir(tmp_path) -> Path:
    target = tmp_path / "cast"
    # The scratch reaction pool and the voice refs are machine state, not cast fixtures: a copy
    # of someone's local _candidates would decide how many clips `reactions generate` adds.
    shutil.copytree(REPO / "cast", target,
                    ignore=shutil.ignore_patterns("continuity.jsonl", "refs", "stings", "_candidates"))
    (target / "refs").mkdir()
    return target


@pytest.fixture
def settings(tmp_path, cast_dir) -> Settings:
    return Settings(data_dir=tmp_path / "data", cast_dir=cast_dir, target_minutes=25)


@pytest.fixture
def fake_llm():
    return default_fake_llm()


@pytest.fixture
def cast(cast_dir):
    return load_cast(cast_dir)


@pytest.fixture
def banned(cast_dir):
    return load_banned(cast_dir)


@pytest.fixture
def book(sample_pdf, fake_llm):
    return ingest_book(sample_pdf, "demo", llm=fake_llm)


@pytest.fixture
def pipeline(settings, fake_llm) -> Pipeline:
    events = []
    p = Pipeline(settings, llm=fake_llm, fake_audio=True, on_event=lambda s, st, d: events.append((s, st, d)))
    p.events = events  # type: ignore[attr-defined]
    return p


@pytest.fixture
def ingested(pipeline, sample_pdf):
    pipeline.ingest(sample_pdf, "demo")
    return pipeline


class Hoogtevrees:
    """Chapter 1 of the hoogtevrees book, its content plan, and the first-draft script that was reviewed by
    hand: formal words, private references nobody set up, one finding recited four times, questions left
    unanswered. The checks that should have caught it are tested against it."""

    def __init__(self) -> None:
        md = (REPO / "boeken" / "hoogtevrees" / "hoogtevrees.md").read_text(encoding="utf-8")
        chapter_md = md[md.index("# Hoofdstuk 1"):md.index("# Hoofdstuk 2")]
        parts = re.split(r"^## (.+)$", chapter_md, flags=re.MULTILINE)
        sections = [Section(id=f"ch01.{n}", title=title.strip(), text=body.strip())
                    for n, (title, body) in enumerate(zip(parts[1::2], parts[2::2], strict=True), start=1)
                    if not title.startswith("Bronnen")]
        self.book = Book(book_id="hoogtevrees", title="Hoogtevrees", chapters=[
            Chapter(id="ch01", title="Wat is hoogtevrees?", pages=(1, 8), sections=sections),
            Chapter(id="ch02", title="Het evenwicht en de ogen", pages=(9, 16),
                    sections=[Section(id="ch02.1", title="Rechtop staan is een kunststuk", text="Rechtop staan is een kunststuk.")]),
        ])
        self.plan = ContentPlan.load(FIXTURES / "hoogtevrees_ch01.plan.json")
        for claim in self.plan.key_claims:
            for section in sections:
                hit = resolve_span(section.text, claim.source_span.quote or "")
                if hit:
                    claim.source_span.section, claim.source_span.start, claim.source_span.end = section.id, hit[0], hit[1]
                    break
        self.script = Script.load(FIXTURES / "hoogtevrees_ch01_rev1.script.json")


@pytest.fixture(scope="session")
def _hoogtevrees() -> Hoogtevrees:
    return Hoogtevrees()


@pytest.fixture
def hoogtevrees(_hoogtevrees) -> Hoogtevrees:
    """A fresh copy per test: plans and scripts get edited."""
    fresh = Hoogtevrees.__new__(Hoogtevrees)
    fresh.book = _hoogtevrees.book
    fresh.plan = _hoogtevrees.plan.model_copy(deep=True)
    fresh.script = _hoogtevrees.script.model_copy(deep=True)
    return fresh
