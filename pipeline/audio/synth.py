"""Text-to-speech adapters behind one small protocol.

* ``ChatterboxSynth``: local zero-shot cloning from a frozen Dutch reference
  (Multilingual V3 weights, hardcoded: one model per project, ever).
* ``NullSynth``: shaped noise with plausible durations, for tests and dry runs.

Heavy imports happen inside the classes so the rest of the pipeline stays
importable on machines without torch.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from pipeline.config import Settings
from pipeline.models import Guest, Host

log = logging.getLogger(__name__)


@dataclass
class AudioClip:
    samples: np.ndarray  # float32 mono, -1..1
    sr: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / self.sr if self.sr else 0.0

    @classmethod
    def silence(cls, seconds: float, sr: int) -> AudioClip:
        return cls(np.zeros(int(round(seconds * sr)), dtype=np.float32), sr)

    @classmethod
    def read(cls, path: Path | str) -> AudioClip:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return cls(np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(sr))

    @classmethod
    def from_bytes(cls, payload: bytes) -> AudioClip:
        data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        return cls(np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(sr))

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = "MP3" if path.suffix.lower() == ".mp3" else None
        sf.write(str(path), self.samples, self.sr, format=fmt)
        return path


def resample(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    try:
        import soxr  # type: ignore

        return np.asarray(soxr.resample(samples, sr, target_sr), dtype=np.float32)
    except Exception:
        pass
    try:
        from scipy.signal import resample_poly  # type: ignore

        g = math.gcd(sr, target_sr)
        return np.asarray(resample_poly(samples, target_sr // g, sr // g), dtype=np.float32)
    except Exception:
        pass
    n_out = int(round(len(samples) * target_sr / sr))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


@dataclass
class VoiceSpec:
    speaker_id: str
    ref_path: Path | None = None
    base_exaggeration: float = 0.5
    cfg_weight: float = 0.5
    speech_rate: float = 1.0
    temperature: float = 0.8
    chars_per_second: float | None = None
    _ref_hash: str | None = None

    @property
    def ref_hash(self) -> str:
        """Identity of the frozen reference clip. Re-cloning changes the voice, so the hash is part of every cache key."""
        if self._ref_hash is None:
            if self.ref_path and Path(self.ref_path).is_file():
                self._ref_hash = hashlib.sha256(Path(self.ref_path).read_bytes()).hexdigest()[:16]
            else:
                self._ref_hash = "noref:" + self.speaker_id
        return self._ref_hash


def voice_spec_for(speaker: Host | Guest, cast_dir: Path | str) -> VoiceSpec:
    ref = Path(cast_dir) / speaker.voice_ref if speaker.voice_ref else None
    return VoiceSpec(
        speaker_id=speaker.id,
        ref_path=ref,
        base_exaggeration=speaker.exaggeration,
        cfg_weight=speaker.cfg_weight,
        speech_rate=speaker.speech_rate,
        temperature=speaker.temperature,
        chars_per_second=speaker.chars_per_second,
    )


class Synth(Protocol):
    """One TTS engine.

    A synth may also carry a `cache_tag`: takes.py folds it into the render cache key, which is
    how the Chatterbox synths keep one T3 checkpoint's takes out of another checkpoint's cache.
    Synths without one (null) are keyed as before.
    """

    name: str
    sample_rate: int
    deterministic: bool

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip: ...


# ---------------------------------------------------------------------------
# Null synth: shaped noise, plausible durations, seed-dependent jitter
# ---------------------------------------------------------------------------

class NullSynth:
    name = "null"
    deterministic = False

    def __init__(self, sample_rate: int = 24000, chars_per_second: float = 15.0, jitter: float = 0.12):
        self.sample_rate = sample_rate
        self.cps = chars_per_second
        self.jitter = jitter

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        rng = np.random.default_rng(abs(hash((text, voice.speaker_id, seed))) % (2**32))
        factor = 1.0 + rng.uniform(-self.jitter, self.jitter) * (1.0 + exaggeration)
        seconds = max(0.3, len(text) / self.cps * factor)
        n = int(seconds * self.sample_rate)
        noise = rng.standard_normal(n).astype(np.float32) * 0.05
        # word-like amplitude envelope so silence detection has something to see
        words = max(1, len(text.split()))
        t = np.linspace(0, words * math.pi, n, dtype=np.float32)
        envelope = 0.35 + 0.65 * np.abs(np.sin(t))
        pad = int(0.08 * self.sample_rate)
        samples = np.concatenate([np.zeros(pad, np.float32), noise * envelope, np.zeros(pad, np.float32)])
        return AudioClip(samples, self.sample_rate, meta={"text": text, "seed": seed, "exaggeration": exaggeration,
                                                          "speaker": voice.speaker_id})


# ---------------------------------------------------------------------------
# Chatterbox Multilingual
# ---------------------------------------------------------------------------

def resolve_torch_device(requested: str) -> str:
    """Pick cuda, mps or cpu from the request and what torch can actually see.

    On Apple Silicon this also enables the CPU fallback for the few ops Metal
    lacks and makes checkpoints saved on CUDA load onto the local device.
    """
    if requested.startswith("mps"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch  # type: ignore

    device = requested
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to cpu")
        device = "cpu"
    if requested.startswith("mps") and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        log.warning("MPS requested but not available; falling back to cpu")
        device = "cpu"
    if not device.startswith("cuda") and not getattr(torch.load, "_studiepodcast_patched", False):
        original = torch.load
        target = torch.device(device)

        def patched_load(*args, **kwargs):
            kwargs.setdefault("map_location", target)
            return original(*args, **kwargs)

        patched_load._studiepodcast_patched = True  # type: ignore[attr-defined]
        torch.load = patched_load
    return device


def apply_speech_rate(clip: AudioClip, rate: float) -> AudioClip:
    """Pitch-preserving time-stretch (phase vocoder via librosa).

    Chatterbox has no speed/rate parameter of its own; pacing is an emergent
    property of the model. cfg_weight (below) is the generation-side lever
    that tends to influence it, but it is a side effect, not a dial - this is
    the deterministic fallback that always gets to the requested pace.
    rate < 1.0 slows down, > 1.0 speeds up, 1.0 is a no-op (same convention
    librosa itself uses, passed straight through).
    """
    if abs(rate - 1.0) < 1e-6 or len(clip.samples) == 0:
        return clip
    import librosa  # lazy: only the real synth needs this, already a transitive dependency there

    stretched = librosa.effects.time_stretch(clip.samples, rate=rate)
    return AudioClip(np.asarray(stretched, dtype=np.float32), clip.sr, meta=dict(clip.meta))


# The one checkpoint this pipeline renders with: Multilingual V3 (June 2026), the same 0.5B
# architecture as V2 with less hallucination and better speaker similarity. The chatterbox-tts
# package on PyPI (0.1.7) still hardcodes V2 and Resemble AI has retired the repo, so
# load_multilingual_model below loads V3 itself. Switching this file switches every voice in
# every episode, which is exactly why it is a constant and why it is part of the render cache key.
T3_CHECKPOINT = "t3_mtl23ls_v3.safetensors"
CACHE_TAG = f"t3={T3_CHECKPOINT}"


def chatterbox_required() -> list[str]:
    """The files a load really opens. conds.pt (a built-in voice) and the Chinese dictionary
    come along with the download but are only read for some generations."""
    return ["ve.pt", T3_CHECKPOINT, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"]


def chatterbox_files() -> list[str]:
    """Everything the checkpoint needs from ResembleAI/chatterbox (the tokenizer, encoder and
    vocoder are shared between model versions; only the T3 weights differ)."""
    return chatterbox_required() + ["conds.pt", "Cangjie5_TC.json"]


def chatterbox_checkpoint_dir() -> "Path":
    """Where the weights live: CHATTERBOX_CKPT_DIR if set, else the Hugging Face cache (offline-safe)."""
    from chatterbox.mtl_tts import REPO_ID, snapshot_download  # type: ignore

    local = os.environ.get("CHATTERBOX_CKPT_DIR")
    if local:
        return Path(local).expanduser()
    return Path(snapshot_download(repo_id=REPO_ID, repo_type="model", revision="main",
                                  allow_patterns=chatterbox_files(), token=os.environ.get("HF_TOKEN") or None))


def _multilingual_from_local(ckpt_dir, device: str):
    """ChatterboxMultilingualTTS.from_local with our T3 file instead of the packaged V2."""
    import torch  # type: ignore
    from chatterbox.mtl_tts import (ChatterboxMultilingualTTS, Conditionals, MTLTokenizer, S3Gen, T3,  # type: ignore
                                    T3Config, VoiceEncoder, load_safetensors)

    ckpt_dir = Path(ckpt_dir)
    map_location = torch.device("cpu") if device in ("cpu", "mps") else None

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True))
    ve.to(device).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(str(ckpt_dir / T3_CHECKPOINT))
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True))
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    conds = None
    if (builtin_voice := ckpt_dir / "conds.pt").exists():
        conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


def load_multilingual_model(device: str):
    """A ChatterboxMultilingualTTS with the V3 T3 weights.

    Every pool worker loads the same checkpoint, so one model in the episode always sounds the
    same; a failed V3 load is an error, not a silent fallback to older weights (mixing the two
    in one render would give the same host two voices).
    """
    import inspect

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore

    if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
        return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=T3_CHECKPOINT)  # newer chatterbox
    ckpt_dir = chatterbox_checkpoint_dir()
    try:
        return _multilingual_from_local(ckpt_dir, device)
    except Exception as exc:
        raise RuntimeError(f"could not load Chatterbox {T3_CHECKPOINT} from {ckpt_dir} ({type(exc).__name__}: {exc}). "
                           "Run once with STUDIEPODCAST_OFFLINE=0 to download it, or put the files in "
                           "CHATTERBOX_CKPT_DIR.") from exc


class ChatterboxSynth:
    name = "chatterbox"
    deterministic = False
    cache_tag = CACHE_TAG  # into the render cache key, so old takes never survive a checkpoint switch

    def __init__(self, settings: Settings, language_id: str = "nl"):
        self.settings = settings
        self.language_id = language_id
        self._model = None
        self.sample_rate = 24000
        self.device = settings.device

    def _load(self):
        if self._model is None:
            self.device = resolve_torch_device(self.settings.device)
            self._model = load_multilingual_model(self.device)
            self.sample_rate = int(getattr(self._model, "sr", 24000))
        return self._model

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        model = self._load()
        import torch  # type: ignore

        if voice.ref_path is None or not Path(voice.ref_path).is_file():
            raise FileNotFoundError(f"reference clip missing for {voice.speaker_id}: {voice.ref_path}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        wav = model.generate(
            text,
            language_id=self.language_id,
            audio_prompt_path=str(voice.ref_path),
            exaggeration=float(exaggeration),
            cfg_weight=float(voice.cfg_weight),
            temperature=float(voice.temperature),
        )
        samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        clip = AudioClip(samples, self.sample_rate, meta={"text": text, "seed": seed, "exaggeration": exaggeration,
                                                          "speaker": voice.speaker_id})
        return apply_speech_rate(clip, voice.speech_rate)


# ---------------------------------------------------------------------------
# Worker pool: N synth instances in parallel processes (each holds one model)
# ---------------------------------------------------------------------------

_WORKER_SYNTH: Synth | None = None


def _pool_init(factory, kwargs):
    global _WORKER_SYNTH
    _WORKER_SYNTH = factory(**kwargs)


def _pool_render(text: str, voice: VoiceSpec, exaggeration: float, seed: int):
    assert _WORKER_SYNTH is not None
    clip = _WORKER_SYNTH.render(text, voice, exaggeration=exaggeration, seed=seed)
    return clip.samples, clip.sr, clip.meta


class SynthPool:
    """Runs renders on a process pool; the caller keeps the Synth protocol."""

    def __init__(self, factory, workers: int, name: str, sample_rate: int, deterministic: bool = False, **kwargs):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.name = name
        self.sample_rate = sample_rate
        self.deterministic = deterministic
        self.cache_tag = getattr(factory, "cache_tag", "")  # workers load this factory's checkpoint
        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=_pool_init,
                                         initargs=(factory, kwargs))

    def submit(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int):
        return self._pool.submit(_pool_render, text, voice, exaggeration, seed)

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        samples, sr, meta = self.submit(text, voice, exaggeration=exaggeration, seed=seed).result()
        return AudioClip(np.asarray(samples, dtype=np.float32), int(sr), meta)

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def make_synth(kind: str, settings: Settings, *, pool: bool = True) -> Synth:
    """``"final"`` is the real Chatterbox render; ``"null"`` is the test stand-in."""
    if kind == "null":
        return NullSynth(sample_rate=settings.sample_rate, chars_per_second=settings.chars_per_second)
    if kind == "final":
        if pool and settings.chatterbox_workers > 1 and os.environ.get("STUDIEPODCAST_NO_POOL") != "1":
            return SynthPool(ChatterboxSynth, settings.chatterbox_workers, "chatterbox", 24000, settings=settings)
        return ChatterboxSynth(settings)
    raise ValueError(f"unknown synth kind {kind}")
