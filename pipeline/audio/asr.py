"""Speech recognition (verification) and forced alignment (word timestamps).

faster-whisper stays resident for the take-selection loop; WhisperX gives the
word onsets the timeline needs for interrupts and backchannels. Both have
fakes so the timeline math is testable without models.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pipeline.audio.synth import AudioClip, resample
from pipeline.config import Settings

log = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float

    def shifted(self, offset: float) -> WordTiming:
        return WordTiming(self.word, self.start + offset, self.end + offset)


class Transcriber(Protocol):
    def transcribe(self, clip: AudioClip, language: str = "nl") -> str: ...


class Aligner(Protocol):
    def align(self, clip: AudioClip, text: str, language: str = "nl") -> list[WordTiming]: ...


def to_16k(clip: AudioClip) -> np.ndarray:
    return resample(clip.samples, clip.sr, 16000).astype(np.float32)


# ---------------------------------------------------------------------------
# Real models (lazy imports)
# ---------------------------------------------------------------------------

class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore

            device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
            self._model = WhisperModel(self.settings.whisper_model, device=device, compute_type=self.settings.whisper_compute_type)
        return self._model

    def transcribe(self, clip: AudioClip, language: str = "nl") -> str:
        model = self._load()
        segments, _info = model.transcribe(to_16k(clip), language=language, beam_size=5, vad_filter=False,
                                           condition_on_previous_text=False)
        return " ".join(seg.text.strip() for seg in segments).strip()


class WhisperXAligner:
    def __init__(self, settings: Settings, fallback: Aligner | None = None):
        self.settings = settings
        self._model = None
        self._metadata = None
        self.fallback = fallback or UniformAligner()

    def _load(self):
        if self._model is None:
            import whisperx  # type: ignore

            self._model, self._metadata = whisperx.load_align_model(language_code="nl", device=self.settings.device)
        return self._model, self._metadata

    def align(self, clip: AudioClip, text: str, language: str = "nl") -> list[WordTiming]:
        try:
            import whisperx  # type: ignore

            model, metadata = self._load()
            audio = to_16k(clip)
            segments = [{"text": text, "start": 0.0, "end": clip.duration_s}]
            result = whisperx.align(segments, model, metadata, audio, self.settings.device, return_char_alignments=False)
            words = []
            for w in result.get("word_segments", []):
                if "start" in w and "end" in w:
                    words.append(WordTiming(str(w.get("word", "")), float(w["start"]), float(w["end"])))
            if words:
                return _fill_missing(words, text, clip.duration_s)
        except Exception as exc:
            log.warning("whisperx alignment failed, using uniform fallback: %s", exc)
        return self.fallback.align(clip, text, language)


def _fill_missing(words: list[WordTiming], text: str, duration: float) -> list[WordTiming]:
    """WhisperX skips words it cannot align (numbers, symbols). Interpolate them."""
    tokens = tokenize(text)
    if len(words) >= len(tokens):
        return words
    aligned = {w.word.casefold().strip(".,;:!?"): w for w in words}
    out: list[WordTiming] = []
    last_end = 0.0
    for i, tok in enumerate(tokens):
        w = aligned.get(tok.casefold().strip(".,;:!?"))
        if w is not None and w.start >= last_end - 0.05:
            out.append(w)
            last_end = w.end
        else:
            remaining = len(tokens) - i
            nxt = next((x for x in words if x.start > last_end), None)
            horizon = nxt.start if nxt else duration
            step = max(0.05, (horizon - last_end) / max(1, remaining))
            out.append(WordTiming(tok, last_end, min(duration, last_end + step)))
            last_end = min(duration, last_end + step)
    return out


# ---------------------------------------------------------------------------
# Fakes and fallbacks
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def speech_bounds(clip: AudioClip, threshold_db: float = -45.0, frame_s: float = 0.01) -> tuple[float, float]:
    """First and last moment with energy above the threshold."""
    frame = max(1, int(clip.sr * frame_s))
    n = len(clip.samples) // frame
    if n == 0:
        return 0.0, clip.duration_s
    frames = clip.samples[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    loud = np.where(db > threshold_db)[0]
    if len(loud) == 0:
        return 0.0, clip.duration_s
    return float(loud[0] * frame / clip.sr), float(min(len(clip.samples), (loud[-1] + 1) * frame) / clip.sr)


class UniformAligner:
    """Spreads words evenly over the voiced part of the clip, weighted by word length."""

    def align(self, clip: AudioClip, text: str, language: str = "nl") -> list[WordTiming]:
        tokens = tokenize(text)
        if not tokens:
            return []
        start, end = speech_bounds(clip)
        span = max(0.05, end - start)
        weights = np.array([len(t) + 1.0 for t in tokens], dtype=np.float64)
        weights /= weights.sum()
        out = []
        cursor = start
        for tok, w in zip(tokens, weights, strict=True):
            width = span * w
            out.append(WordTiming(tok, cursor, cursor + width * 0.9))
            cursor += width
        return out


class EchoTranscriber:
    """Returns the text the NullSynth embedded in the clip, optionally corrupted per seed."""

    def __init__(self, corrupt: Callable[[str, int], str] | None = None):
        self.corrupt = corrupt

    def transcribe(self, clip: AudioClip, language: str = "nl") -> str:
        text = str(clip.meta.get("text", ""))
        seed = int(clip.meta.get("seed", 0))
        return self.corrupt(text, seed) if self.corrupt else text


def make_transcriber(settings: Settings, fake: bool = False) -> Transcriber:
    return EchoTranscriber() if fake else FasterWhisperTranscriber(settings)


def make_aligner(settings: Settings, fake: bool = False) -> Aligner:
    return UniformAligner() if fake else WhisperXAligner(settings)
