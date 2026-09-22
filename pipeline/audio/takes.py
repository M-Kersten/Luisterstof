"""The take-selection loop. Per turn, not per block.

1. Cache lookup on sha256(text + voice_ref + exaggeration + seed). Hit means skip.
2. Render N takes with varied seed and a small exaggeration jitter.
3. Transcribe each take.
4. Normalise and compute WER against the script text. Reject above the threshold.
5. Among survivors, pick the duration closest to the expected duration.
6. If all fail, retry once with lower exaggeration, then flag the turn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pipeline.audio.asr import Transcriber
from pipeline.audio.cache import RenderCache
from pipeline.audio.synth import AudioClip, Synth, VoiceSpec
from pipeline.audio.wer import word_error_rate
from pipeline.models import Glossary, TakeRecord

log = logging.getLogger(__name__)

JITTER = (0.0, -0.05, 0.05, -0.1, 0.1)
RETRY_EXAGGERATION_DROP = 0.15


@dataclass
class TakeResult:
    clip: AudioClip | None
    takes: list[TakeRecord]
    chosen: int | None
    from_cache: bool
    flagged: bool
    flag_reason: str | None
    take_key: str | None


def expected_duration(text: str, chars_per_second: float) -> float:
    return len(text) / max(1.0, chars_per_second)


def render_with_takes(
    text: str,
    voice: VoiceSpec,
    synth: Synth,
    cache: RenderCache,
    *,
    transcriber: Transcriber | None,
    exaggeration: float,
    n_takes: int = 3,
    wer_threshold: float = 0.05,
    chars_per_second: float = 15.0,
    glossary: Glossary | None = None,
    seed_base: int = 1000,
    on_take: Callable[[TakeRecord], None] | None = None,
) -> TakeResult:
    cps = voice.chars_per_second or chars_per_second
    verify = transcriber is not None and not synth.deterministic
    if synth.deterministic:
        n_takes = 1
    params = f"n={n_takes};wer={wer_threshold};cps={cps:.1f};verify={verify}"
    sel_key = cache.selection_key(text, voice.ref_hash, exaggeration, synth.name, params)
    cached = cache.get_selection(sel_key)
    if cached:
        hit = cache.get(cached["take_key"])
        if hit:
            clip, meta = hit
            record = TakeRecord(seed=int(cached.get("seed", 0)), exaggeration=float(cached.get("exaggeration", exaggeration)),
                                path=str(cache.path(cached["take_key"])), duration_s=clip.duration_s,
                                wer=cached.get("wer"), transcript=cached.get("transcript"), accepted=True, reason="cache")
            return TakeResult(clip, [record], 0, True, bool(cached.get("flagged")), cached.get("flag_reason"), cached["take_key"])

    target = expected_duration(text, cps)
    takes: list[TakeRecord] = []
    clips: list[AudioClip] = []
    keys: list[str] = []

    def attempt(seed: int, exag: float) -> None:
        nonlocal verify
        exag = max(0.1, min(0.95, exag))
        key = cache.take_key(text, voice.ref_hash, exag, seed, synth.name)
        hit = cache.get(key)
        if hit:
            clip, meta = hit
            transcript = meta.get("transcript")
        else:
            clip = synth.render(text, voice, exaggeration=exag, seed=seed)
            transcript = None
            cache.put(key, clip, {"text": text, "speaker": voice.speaker_id, "seed": seed, "exaggeration": exag, "synth": synth.name})
        if verify and transcript is None:
            # A transcriber that raises (rather than returning None on a known failure, as
            # FasterWhisperTranscriber/MlxWhisperTranscriber now do) must still not take the
            # whole render down: fall back to unverified for the rest of this turn.
            try:
                transcript = transcriber.transcribe(clip)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                log.error("transcriber raised (%s: %s); verification disabled for this turn", type(exc).__name__, exc)
                verify = False
                transcript = None
            else:
                extra = {"asr_words": clip.meta["asr_words"]} if clip.meta.get("asr_words") else {}
                cache.update_meta(key, transcript=transcript, **extra)
        rate = word_error_rate(text, transcript, glossary) if verify and transcript is not None else None
        accepted = True if rate is None else rate <= wer_threshold
        record = TakeRecord(seed=seed, exaggeration=exag, path=str(cache.path(key)), duration_s=clip.duration_s,
                            wer=None if rate is None else round(rate, 4), transcript=transcript, accepted=accepted,
                            reason=None if accepted else f"wer {rate:.3f} > {wer_threshold}")
        takes.append(record)
        clips.append(clip)
        keys.append(key)
        if on_take:
            on_take(record)

    for i in range(n_takes):
        attempt(seed_base + i, exaggeration + JITTER[i % len(JITTER)])

    def pick() -> int | None:
        survivors = [i for i, t in enumerate(takes) if t.accepted]
        if not survivors:
            return None
        return min(survivors, key=lambda i: abs(takes[i].duration_s - target))

    chosen = pick()
    flagged = False
    reason = None
    if chosen is None and not synth.deterministic:
        attempt(seed_base + 100, exaggeration - RETRY_EXAGGERATION_DROP)
        chosen = pick()
    if chosen is None:
        flagged = True
        reason = "alle takes afgekeurd op WER"
        chosen = min(range(len(takes)), key=lambda i: (takes[i].wer if takes[i].wer is not None else 1.0))
        log.warning("turn flagged (%s): %s", reason, text[:60])

    record = takes[chosen]
    cache.put_selection(sel_key, {
        "take_key": keys[chosen], "seed": record.seed, "exaggeration": record.exaggeration, "wer": record.wer,
        "transcript": record.transcript, "flagged": flagged, "flag_reason": reason, "text": text,
    })
    return TakeResult(clips[chosen], takes, chosen, False, flagged, reason, keys[chosen])
