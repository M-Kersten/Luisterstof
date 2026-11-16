"""Speech recognition (verification) and forced alignment (word timestamps).

One backend: faster-whisper stays resident for the take-selection loop;
WhisperX forced alignment gives the word onsets the timeline needs.

Both have fakes so the timeline math is testable without models.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import sys
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
    def transcribe(self, clip: AudioClip, language: str = "nl") -> str | None: ...


class Aligner(Protocol):
    def align(self, clip: AudioClip, text: str, language: str = "nl") -> list[WordTiming]: ...


def to_16k(clip: AudioClip) -> np.ndarray:
    return resample(clip.samples, clip.sr, 16000).astype(np.float32)


# ---------------------------------------------------------------------------
# Real models (lazy imports)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# faster-whisper on a Windows GPU: CTranslate2 loads cuDNN itself, by file name. CTranslate2 4.4 and older want
# cuDNN 8, 4.5 and newer cuDNN 9; torch 2.6 (Chatterbox) brings cuDNN 9. When the cuDNN that CTranslate2 wants is
# not there, cuDNN prints "Could not locate cudnn_ops_infer64_8.dll" and the transcription hangs or takes the
# process down, an exception never comes. So the DLLs are looked for before Whisper goes near the GPU, and Whisper
# runs on the processor when they are missing.
# ---------------------------------------------------------------------------
CUDNN_FIX = 'uv pip install --python .venv "whisperx==3.3.2" "ctranslate2>=4.5"'


def _cuda_dll_dirs() -> list[str]:
    """Folders with CUDA/cuDNN DLLs from pip packages: torch's own lib folder and the nvidia-* wheels."""
    import importlib.util

    dirs = []
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        dirs.append(os.path.join(os.path.dirname(spec.origin), "lib"))
    for path in sys.path:
        nvidia = os.path.join(path, "nvidia")
        if os.path.isdir(nvidia):
            dirs += [os.path.join(nvidia, d, "bin") for d in os.listdir(nvidia)]
    return [d for d in dirs if os.path.isdir(d)]


_dll_dirs_added = False


def prepare_cuda_dlls() -> None:
    """Make those DLLs findable for CTranslate2 (which loads cuDNN by name, from PATH)."""
    global _dll_dirs_added
    if sys.platform != "win32" or _dll_dirs_added:
        return
    _dll_dirs_added = True
    dirs = _cuda_dll_dirs()
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


@dataclass
class CudnnCheck:
    ok: bool
    ctranslate2: str = ""
    cudnn: int = 0
    missing: str = ""

    def reason_nl(self) -> str:
        return (f"de GPU-versie van Whisper (CTranslate2 {self.ctranslate2}) zoekt cuDNN {self.cudnn} "
                f"({self.missing}) en dat staat niet op deze computer")


def whisper_gpu_check() -> CudnnCheck:
    """On Windows: can the cuDNN this CTranslate2 needs be loaded? Elsewhere the packages bring it along."""
    if sys.platform != "win32":
        return CudnnCheck(True)
    try:
        import ctranslate2  # type: ignore
    except Exception:  # noqa: BLE001 - faster-whisper itself will say what is missing
        return CudnnCheck(True)
    version = getattr(ctranslate2, "__version__", "0")
    parts = [int(x) for x in re.findall(r"\d+", version)[:2]] + [0, 0]
    cudnn = 9 if tuple(parts[:2]) >= (4, 5) else 8
    names = ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"] if cudnn == 9 else ["cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll"]
    prepare_cuda_dlls()
    import ctypes

    for name in names:
        try:
            ctypes.WinDLL(name)
        except OSError:
            return CudnnCheck(False, version, cudnn, name)
    return CudnnCheck(True, version, cudnn)


class FasterWhisperTranscriber:
    """faster-whisper is a separate C++ engine (CTranslate2) with its own runtime
    requirements (a matching CUDA/cuDNN pair, findable DLLs on Windows). A broken
    environment there must not take the whole render down: the take-selection
    loop already treats a missing transcript as "unverified, accept the take"
    (see render_with_takes), so this degrades to unverified takes with a single
    loud log line instead of crashing, and stops retrying a load that already
    failed once."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._broken: str | None = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore

            device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
            if device == "cuda":
                check = whisper_gpu_check()
                if not check.ok:
                    log.warning("faster-whisper runs on the CPU: ctranslate2 %s needs cuDNN %d (%s), which is not on "
                                "this machine. Fix: %s", check.ctranslate2, check.cudnn, check.missing, CUDNN_FIX)
                    device = "cpu"
            self.device = device
            self._model = WhisperModel(self.settings.whisper_model, device=device, compute_type=self.settings.whisper_compute_type)
        return self._model

    def transcribe(self, clip: AudioClip, language: str = "nl") -> str | None:
        if self._broken:
            return None
        try:
            model = self._load()
            segments, _info = model.transcribe(to_16k(clip), language=language, beam_size=5, vad_filter=False,
                                               condition_on_previous_text=False)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:  # noqa: BLE001
            self._broken = f"{type(exc).__name__}: {exc}"
            log.error(
                "faster-whisper is unavailable (%s). Verification is disabled for the rest of this render; "
                "takes will be accepted without a WER check. A missing cudnn_ops*_8.dll or cudnn*_9.dll "
                "usually means ctranslate2's cuDNN version doesn't match what's on PATH; with torch 2.6 run "
                "'" + CUDNN_FIX + "' to move it onto the cuDNN 9 torch ships.",
                self._broken,
            )
            return None


class WhisperXAligner:
    def __init__(self, settings: Settings, fallback: Aligner | None = None):
        self.settings = settings
        self._model = None
        self._metadata = None
        self.fallback = fallback or UniformAligner()
        self.last_used_fallback = False  # set on every align() call; check it right after calling

    @property
    def device(self) -> str:
        # wav2vec2 alignment is cheap; only CUDA is worth the trouble, everything else runs on cpu.
        return "cuda" if self.settings.device.startswith("cuda") else "cpu"

    def _load(self):
        if self._model is None:
            import whisperx  # type: ignore

            self._model, self._metadata = whisperx.load_align_model(language_code="nl", device=self.device)
        return self._model, self._metadata

    def align(self, clip: AudioClip, text: str, language: str = "nl") -> list[WordTiming]:
        self.last_used_fallback = False
        try:
            import whisperx  # type: ignore

            model, metadata = self._load()
            audio = to_16k(clip)
            segments = [{"text": text, "start": 0.0, "end": clip.duration_s}]
            result = whisperx.align(segments, model, metadata, audio, self.device, return_char_alignments=False)
            words = []
            for w in result.get("word_segments", []):
                if "start" in w and "end" in w:
                    words.append(WordTiming(str(w.get("word", "")), float(w["start"]), float(w["end"])))
            if words:
                return match_words_to_text(tokenize(text), words, clip.duration_s)
        except Exception as exc:
            log.warning("whisperx alignment failed, using uniform fallback: %s", exc)
        self.last_used_fallback = True
        return self.fallback.align(clip, text, language)


def _norm_token(word: str) -> str:
    return re.sub(r"[^\w]", "", word.casefold())


def match_words_to_text(tokens: list[str], asr_words: list[WordTiming], duration: float) -> list[WordTiming]:
    """Map script tokens onto ASR word timings.

    Matched tokens take the recogniser's times; substituted runs of equal length
    are assumed one-to-one (a misheard word still has an onset); anything else
    is interpolated between the nearest anchors, weighted by token length.
    """
    if not tokens:
        return []
    if not asr_words:
        return UniformAligner().align(AudioClip.silence(duration, 100), " ".join(tokens))
    a = [_norm_token(t) for t in tokens]
    b = [_norm_token(w.word) for w in asr_words]
    matched: list[WordTiming | None] = [None] * len(tokens)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                src = asr_words[j1 + k]
                matched[i1 + k] = WordTiming(tokens[i1 + k], src.start, src.end)
    # Enforce monotonic anchors; drop any that run backwards.
    last_end = -1.0
    for i, w in enumerate(matched):
        if w is not None:
            if w.start < last_end - 0.05:
                matched[i] = None
            else:
                last_end = w.end
    out: list[WordTiming] = []
    i = 0
    n = len(tokens)
    while i < n:
        if matched[i] is not None:
            out.append(matched[i])  # type: ignore[arg-type]
            i += 1
            continue
        j = i
        while j < n and matched[j] is None:
            j += 1
        start = out[-1].end if out else (asr_words[0].start if i == 0 else 0.0)
        end = matched[j].start if j < n else max(start + 0.05 * (j - i), min(duration, asr_words[-1].end))  # type: ignore[union-attr]
        end = max(end, start + 0.05 * (j - i))
        weights = [len(tokens[k]) + 1.0 for k in range(i, j)]
        total = sum(weights)
        cursor = start
        for k, w in zip(range(i, j), weights, strict=True):
            width = (end - start) * w / total
            out.append(WordTiming(tokens[k], cursor, cursor + width * 0.9))
            cursor += width
        i = j
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

    last_used_fallback = True  # always estimated timing, by definition; read via getattr like the other aligners

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
