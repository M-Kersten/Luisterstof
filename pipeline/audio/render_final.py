"""The render: Chatterbox with take selection, WhisperX alignment, timeline assembly."""

from __future__ import annotations

from pathlib import Path

from pipeline.audio.asr import Aligner, Transcriber, make_aligner, make_transcriber
from pipeline.audio.render import EventFn, render_episode
from pipeline.audio.synth import Synth, make_synth
from pipeline.config import Settings
from pipeline.models import Cast, Glossary, RenderManifest, Script
from pipeline.paths import BookPaths


def render_final(
    script: Script,
    glossary: Glossary | None,
    cast: Cast,
    settings: Settings,
    paths: BookPaths,
    *,
    synth: Synth | None = None,
    transcriber: Transcriber | None = None,
    aligner: Aligner | None = None,
    fake: bool = False,
    on_event: EventFn | None = None,
) -> tuple[Path, Path, RenderManifest]:
    synth = synth or make_synth("null" if fake else "final", settings)
    transcriber = transcriber or make_transcriber(settings, fake=fake)
    aligner = aligner or make_aligner(settings, fake=fake)
    try:
        return render_episode(script, glossary, cast, settings, paths, synth=synth,
                              transcriber=transcriber, aligner=aligner, on_event=on_event)
    finally:
        close = getattr(synth, "close", None)
        if callable(close):
            close()
