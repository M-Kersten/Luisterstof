"""M0: voice audition. Clone the references and render one real paragraph before building anything else."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.audio.asr import Transcriber
from pipeline.audio.synth import AudioClip, Synth, VoiceSpec
from pipeline.audio.wer import word_error_rate

log = logging.getLogger(__name__)


def audition(
    text: str,
    refs: dict[str, Path],
    out_dir: Path,
    synth: Synth,
    *,
    exaggerations: tuple[float, ...] = (0.3, 0.5, 0.7),
    transcriber: Transcriber | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ref in refs.items():
        voice = VoiceSpec(speaker_id=name, ref_path=ref, base_exaggeration=0.5)
        for exag in exaggerations:
            clip: AudioClip = synth.render(text, voice, exaggeration=exag, seed=1)
            path = out_dir / f"{name}_exag{exag:.2f}.wav"
            clip.write(path)
            row = {"speaker": name, "exaggeration": exag, "path": str(path), "duration_s": round(clip.duration_s, 2)}
            if transcriber is not None:
                transcript = transcriber.transcribe(clip)
                row["wer"] = round(word_error_rate(text, transcript), 4)
                row["transcript"] = transcript
            results.append(row)
            log.info("audition %s", row)
    return results
