"""Draft tier: Piper, flat, seconds per episode. Hear pacing and lexicon problems before any real render."""

from __future__ import annotations

from pathlib import Path

from pipeline.audio.asr import UniformAligner
from pipeline.audio.render import EventFn, render_episode
from pipeline.audio.synth import Synth, make_synth
from pipeline.config import Settings
from pipeline.models import Cast, Glossary, RenderManifest, Script
from pipeline.paths import BookPaths


def render_draft(
    script: Script,
    glossary: Glossary | None,
    cast: Cast,
    settings: Settings,
    paths: BookPaths,
    *,
    synth: Synth | None = None,
    on_event: EventFn | None = None,
) -> tuple[Path, Path, RenderManifest]:
    synth = synth or make_synth("draft", settings)
    return render_episode(script, glossary, cast, settings, paths, tier="draft", synth=synth, transcriber=None,
                          aligner=UniformAligner(), on_event=on_event)
