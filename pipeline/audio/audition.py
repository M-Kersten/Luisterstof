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
    cfg_weights: tuple[float, ...] = (0.5,),
    speech_rates: tuple[float, ...] = (1.0,),
    temperatures: tuple[float, ...] = (0.8,),
    transcriber: Transcriber | None = None,
) -> list[dict]:
    """Render every combination of exaggeration x cfg_weight x speech_rate x temperature per reference voice.

    cfg_weight is Chatterbox's own generation-side lever most often reported to
    affect pacing (lower tends to slow delivery, as a side effect of what it's
    actually for). speech_rate is a deterministic post-render time-stretch,
    always gets to the requested pace regardless of what the model does.
    temperature is the model's sampling temperature (default 0.8): higher gives
    more delivery variation between takes at the cost of consistency, lower is
    flatter but more predictable. Sweep cfg_weight first; only reach for
    speech_rate if cfg_weight alone doesn't land where you want, it's the more
    surgical of the two. temperature is worth sweeping separately if deliveries
    sound flat even at a high exaggeration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ref in refs.items():
        for exag in exaggerations:
            for cfg in cfg_weights:
                for rate in speech_rates:
                    for temp in temperatures:
                        voice = VoiceSpec(speaker_id=name, ref_path=ref, base_exaggeration=exag, cfg_weight=cfg,
                                          speech_rate=rate, temperature=temp)
                        clip: AudioClip = synth.render(text, voice, exaggeration=exag, seed=1)
                        path = out_dir / f"{name}_exag{exag:.2f}_cfg{cfg:.2f}_rate{rate:.2f}_temp{temp:.2f}.wav"
                        clip.write(path)
                        row = {"speaker": name, "exaggeration": exag, "cfg_weight": cfg, "speech_rate": rate,
                              "temperature": temp, "path": str(path), "duration_s": round(clip.duration_s, 2)}
                        if transcriber is not None:
                            transcript = transcriber.transcribe(clip)
                            row["wer"] = None if transcript is None else round(word_error_rate(text, transcript), 4)
                            row["transcript"] = transcript
                        results.append(row)
                        log.info("audition %s", row)
    return results
