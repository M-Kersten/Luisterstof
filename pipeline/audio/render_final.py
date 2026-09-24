"""Final tier: Chatterbox with take selection, WhisperX alignment, timeline assembly.

Also the ElevenLabs accent tier: selected blocks are rendered as real
dialogue and spliced in through ``RenderManifest.block_overrides``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from pipeline.audio.asr import Aligner, Transcriber, make_aligner, make_transcriber
from pipeline.audio.chunker import chunk_script, eleven_inputs
from pipeline.audio.render import EventFn, render_episode
from pipeline.audio.synth import ElevenLabsDialogue, Synth, make_synth
from pipeline.config import Settings
from pipeline.cues import speakable
from pipeline.models import Cast, Glossary, RenderManifest, Script
from pipeline.paths import BookPaths
from pipeline.plan.glossary import apply_lexicon


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
        return render_episode(script, glossary, cast, settings, paths, tier="final", synth=synth,
                              transcriber=transcriber, aligner=aligner, on_event=on_event)
    finally:
        close = getattr(synth, "close", None)
        if callable(close):
            close()


def render_eleven_blocks(
    script: Script,
    glossary: Glossary | None,
    cast: Cast,
    settings: Settings,
    paths: BookPaths,
    block_ids: list[str],
    *,
    client: ElevenLabsDialogue | None = None,
    on_event: EventFn | None = None,
    render_fn: Callable[[list[dict[str, str]]], object] | None = None,
) -> RenderManifest:
    """Render the given blocks with ElevenLabs v3 Text-to-Dialogue and record them as overrides."""
    manifest = RenderManifest.load_or_none(paths.manifest(script.episode_id, "final")) or RenderManifest(
        episode_id=script.episode_id, tier="final", synth="chatterbox")
    voice_ids = {s.id: (s.voice_id_final or "") for s in list(cast.hosts) + list(cast.guests)}
    blocks = {b.id: b for b in chunk_script(script)}
    lines_by_id = {line.id: line for line in script.lines()}
    if render_fn is None:
        from pipeline.offline import block_if_offline

        block_if_offline(settings, "ElevenLabs")
    renderer = render_fn or (client or ElevenLabsDialogue(settings)).render_dialogue
    out_dir = paths.blocks_dir(script.episode_id) / "elevenlabs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for block_id in block_ids:
        block = blocks.get(block_id)
        if block is None:
            raise KeyError(f"unknown block {block_id}")
        lines = [lines_by_id[lid] for lid in block.line_ids]
        missing = [l.speaker for l in lines if not voice_ids.get(l.speaker)]
        if missing:
            raise RuntimeError(f"no ElevenLabs voice id for speaker(s): {sorted(set(missing))}")
        spoken = []
        for line in lines:
            copy = line.model_copy()
            text = speakable(line.text, line.reaction)
            copy.text = apply_lexicon(text, glossary) if glossary else text
            spoken.append(copy)
        inputs = eleven_inputs(spoken, voice_ids)
        total = sum(len(i["text"]) for i in inputs)
        if total > ElevenLabsDialogue.MAX_CHARS:
            raise ValueError(f"block {block_id} has {total} characters after tag injection")
        digest = hashlib.sha256(repr(inputs).encode("utf-8")).hexdigest()[:12]
        path = out_dir / f"{block_id}-{digest}.wav"
        if not path.is_file():
            clip = renderer(inputs)
            clip.write(path)  # type: ignore[attr-defined]
        manifest.block_overrides[block_id] = str(path)
        if on_event:
            on_event("eleven_block", {"block": block_id, "chars": total, "path": str(path)})
    manifest.save(paths.manifest(script.episode_id, "final"))
    return manifest
