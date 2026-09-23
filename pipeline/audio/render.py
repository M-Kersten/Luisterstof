"""Shared render flow for both tiers: turns -> audio -> timeline -> mix -> files."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from pipeline.audio.aligner_cache import load_alignment, save_alignment
from pipeline.audio.asr import Aligner, Transcriber, UniformAligner, WordTiming
from pipeline.audio.cache import RenderCache
from pipeline.audio.chunker import chunk_script
from pipeline.audio.mixer import build_transcript, export_blocks, load_sting, mix
from pipeline.audio.synth import AudioClip, Synth, VoiceSpec, voice_spec_for
from pipeline.audio.takes import render_with_takes
from pipeline.audio.timeline import assemble
from pipeline.audio.turns import Turn, group_turns
from pipeline.config import Settings
from pipeline.models import Cast, Glossary, RenderManifest, Script, TurnRender
from pipeline.paths import BookPaths
from pipeline.tags import exaggeration_for

log = logging.getLogger(__name__)

EventFn = Callable[[str, dict], None]

EMOTION_CARRY = 0.6  # share of the previous turn's exaggeration carried into the next (same speaker)


def _emit(on_event: EventFn | None, kind: str, **data) -> None:
    if on_event:
        on_event(kind, data)


def _voices(cast: Cast, settings: Settings) -> dict[str, VoiceSpec]:
    return {s.id: voice_spec_for(s, settings.cast_dir) for s in list(cast.hosts) + list(cast.guests)}


def render_episode(
    script: Script,
    glossary: Glossary | None,
    cast: Cast,
    settings: Settings,
    paths: BookPaths,
    *,
    tier: str,
    synth: Synth,
    transcriber: Transcriber | None,
    aligner: Aligner | None,
    on_event: EventFn | None = None,
    only_lines: set[str] | None = None,
    seed: int = 7,
) -> tuple[Path, Path, RenderManifest]:
    cache = RenderCache(paths.cache_dir)
    voices = _voices(cast, settings)
    turns = group_turns(script, glossary)
    aligner = aligner or UniformAligner()
    manifest = RenderManifest(episode_id=script.episode_id, tier=tier, synth=synth.name)  # type: ignore[arg-type]
    previous_manifest = RenderManifest.load_or_none(paths.manifest(script.episode_id, tier))
    if previous_manifest is not None:
        manifest.block_overrides = dict(previous_manifest.block_overrides)

    rendered: dict[str, tuple[AudioClip, list[WordTiming]]] = {}
    state: dict[str, float] = {}
    _emit(on_event, "render_start", tier=tier, turns=len(turns))
    for i, turn in enumerate(turns):
        voice = voices.get(turn.speaker)
        if voice is None:
            manifest.turns.append(TurnRender(turn_id=turn.turn_id, speaker=turn.speaker, line_ids=turn.line_ids,
                                             text_spoken=turn.text, flagged=True, flag_reason="onbekende spreker"))
            continue
        target = exaggeration_for(turn.tags, voice.base_exaggeration)
        carried = state.get(turn.speaker, target)
        exaggeration = round(EMOTION_CARRY * carried + (1 - EMOTION_CARRY) * target, 3)
        state[turn.speaker] = exaggeration
        result = render_with_takes(
            turn.text, voice, synth, cache,
            transcriber=transcriber, exaggeration=exaggeration, n_takes=1 if tier == "draft" else settings.takes_per_line,
            wer_threshold=settings.wer_threshold, chars_per_second=settings.chars_per_second, glossary=glossary,
            seed_base=1000 + i * 10,
            on_take=lambda rec, t=turn: _emit(on_event, "take", turn=t.turn_id, wer=rec.wer, accepted=rec.accepted),
        )
        tr = TurnRender(turn_id=turn.turn_id, speaker=turn.speaker, line_ids=turn.line_ids, text_spoken=turn.text,
                        takes=result.takes, chosen=result.chosen, from_cache=result.from_cache, flagged=result.flagged,
                        flag_reason=result.flag_reason)
        manifest.turns.append(tr)
        if result.clip is not None:
            cached = load_alignment(cache, result.take_key) if result.take_key else None
            if cached is not None:
                words, used_fallback = cached
            else:
                words = aligner.align(result.clip, turn.text)
                used_fallback = bool(getattr(aligner, "last_used_fallback", False))
                if result.take_key:
                    save_alignment(cache, result.take_key, words, fallback=used_fallback)
            tr.aligned_with_fallback = used_fallback
            rendered[turn.turn_id] = (result.clip, words)
        _emit(on_event, "turn", index=i + 1, total=len(turns), turn=turn.turn_id, speaker=turn.speaker,
              cached=result.from_cache, flagged=result.flagged)

    timeline = assemble(turns, rendered, sr=synth.sample_rate, seed=seed, max_unwritten_gap=settings.max_unwritten_gap_s)
    _apply_block_overrides(timeline, turns, manifest, cache)
    sr = settings.sample_rate if tier == "final" else synth.sample_rate
    intro = load_sting(settings.cast_dir / "stings" / "intro.wav")
    outro = load_sting(settings.cast_dir / "stings" / "outro.wav")
    mixed, offset = mix(timeline, sr=sr, target_lufs=settings.target_lufs, line_lufs=settings.line_lufs, intro=intro, outro=outro)

    audio_path = paths.out_audio(script.episode_id, tier)
    mixed.write(audio_path)
    blocks = chunk_script(script)
    transcript = build_transcript(timeline, script, tier=tier, offset=offset, duration_s=mixed.duration_s, blocks=blocks)
    if tier == "final":
        export_blocks(mixed, transcript, paths.blocks_dir(script.episode_id))
    transcript_path = paths.transcript(script.episode_id, tier)
    transcript.save(transcript_path)
    manifest.save(paths.manifest(script.episode_id, tier))
    verified, verifiable = manifest.verification_coverage()
    fallback_count = manifest.alignment_fallback_count()
    _emit(on_event, "render_done", tier=tier, audio=str(audio_path), duration_s=round(mixed.duration_s, 1),
          flagged=len(manifest.flagged_turns()), qa=list(timeline.qa),
          verified_turns=verified, verifiable_turns=verifiable, alignment_fallback_turns=fallback_count)
    if tier == "final" and not synth.deterministic:
        if verifiable and verified < verifiable:
            log.warning("WER verification ran for only %d/%d turns; the rest were accepted unverified "
                       "(a broken transcriber degrades this way rather than crashing - see the transcribe() "
                       "log line above for the cause). Fix it and re-render before trusting a clean flagged-turns count.",
                       verified, verifiable)
        if fallback_count:
            log.warning("%d/%d turns used estimated (uniform) word timing instead of real forced alignment; "
                       "interrupt cut points and backchannel placement on those turns are approximate and can "
                       "land mid-word.", fallback_count, len(manifest.turns))
    return audio_path, transcript_path, manifest


def _apply_block_overrides(timeline, turns: list[Turn], manifest: RenderManifest, cache: RenderCache) -> None:
    """Replace the turns of an accent-tier block with the block's own audio."""
    if not manifest.block_overrides:
        return
    from pipeline.audio.timeline import Placement

    blocks = {b.id: b for b in chunk_script_from_turns(turns)}
    for block_id, path in list(manifest.block_overrides.items()):
        block = blocks.get(block_id)
        if block is None or not Path(path).is_file():
            manifest.block_overrides.pop(block_id, None)
            continue
        clip = AudioClip.read(path)
        members = [p for p in timeline.placements if set(p.line_ids) & set(block.line_ids)]
        if not members:
            continue
        start = min(p.start for p in members)
        for p in members:
            timeline.placements.remove(p)
        merged = Turn(turn_id=f"block-{block_id}", speaker=members[0].turn.speaker,
                      lines=[l for p in members for l in p.turn.lines], segment_type=members[0].turn.segment_type,
                      segment_index=members[0].turn.segment_index,
                      spoken_lines=[s for p in members for s in p.turn.spoken_lines])
        words = UniformAligner().align(clip, merged.text)
        timeline.placements.append(Placement(merged, clip, start, [w.shifted(start) for w in words]))
        shift = (start + clip.duration_s) - max(p.end for p in members)
        for p in timeline.placements:
            if p.start > start and p.turn.turn_id != merged.turn_id:
                p.start += shift
                p.words = [w.shifted(shift) for w in p.words]
        timeline.placements.sort(key=lambda p: p.start)


def chunk_script_from_turns(turns: list[Turn]):
    from pipeline.models import Script as _Script
    from pipeline.models import Segment

    segments: dict[int, Segment] = {}
    for t in turns:
        seg = segments.setdefault(t.segment_index, Segment(type=t.segment_type))
        seg.lines.extend(t.lines)
    pseudo = _Script(episode_id="tmp", segments=[segments[k] for k in sorted(segments)])
    return chunk_script(pseudo)
