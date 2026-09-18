from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pipeline.config import Settings
from pipeline.fake_handlers import default_fake_llm
from pipeline.ingest.run import ingest_book
from pipeline.runner import Pipeline
from pipeline.sample_book import build
from pipeline.script.cast import load_banned, load_cast

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "sample.pdf")


@pytest.fixture(scope="session")
def sample_pdf_notoc(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "sample_notoc.pdf", with_toc=False)


@pytest.fixture
def cast_dir(tmp_path) -> Path:
    target = tmp_path / "cast"
    shutil.copytree(REPO / "cast", target, ignore=shutil.ignore_patterns("continuity.jsonl", "refs", "stings"))
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
