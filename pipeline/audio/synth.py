"""Text-to-speech adapters behind one small protocol.

* ``PiperSynth``: local, deterministic, flat delivery, seconds per episode.
* ``ChatterboxSynth``: local zero-shot cloning from a frozen Dutch reference.
* ``ElevenLabsDialogue``: cloud accent tier, per block, real overlapping speech.
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
import shutil
import subprocess
import tempfile
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
    draft_voice: str | None = None
    eleven_voice_id: str | None = None
    base_exaggeration: float = 0.5
    chars_per_second: float | None = None
    _ref_hash: str | None = None

    @property
    def ref_hash(self) -> str:
        """Identity of the frozen reference clip. Re-cloning changes the voice, so the hash is part of every cache key."""
        if self._ref_hash is None:
            if self.ref_path and Path(self.ref_path).is_file():
                self._ref_hash = hashlib.sha256(Path(self.ref_path).read_bytes()).hexdigest()[:16]
            else:
                self._ref_hash = "noref:" + (self.draft_voice or self.speaker_id)
        return self._ref_hash


def voice_spec_for(speaker: Host | Guest, cast_dir: Path | str) -> VoiceSpec:
    ref = Path(cast_dir) / speaker.voice_ref if speaker.voice_ref else None
    return VoiceSpec(
        speaker_id=speaker.id,
        ref_path=ref,
        draft_voice=speaker.voice_id_draft or None,
        eleven_voice_id=speaker.voice_id_final or None,
        base_exaggeration=speaker.exaggeration,
        chars_per_second=speaker.chars_per_second,
    )


class Synth(Protocol):
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
# Piper (draft tier)
# ---------------------------------------------------------------------------

class PiperSynth:
    name = "piper"
    deterministic = True

    def __init__(self, settings: Settings, sample_rate: int = 22050):
        self.settings = settings
        self.sample_rate = sample_rate
        self._voices: dict[str, Any] = {}

    def model_path(self, voice: VoiceSpec) -> Path:
        name = voice.draft_voice
        if not name:
            raise RuntimeError(f"speaker {voice.speaker_id} has no voice_id_draft")
        candidates = [Path(name), self.settings.piper_voices_dir / f"{name}.onnx", self.settings.piper_voices_dir / name / f"{name}.onnx"]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(f"Piper voice {name} not found under {self.settings.piper_voices_dir}")

    def _load(self, voice: VoiceSpec):
        path = self.model_path(voice)
        key = str(path)
        if key not in self._voices:
            from piper import PiperVoice  # type: ignore

            self._voices[key] = PiperVoice.load(str(path))
        return self._voices[key]

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            try:
                model = self._load(voice)
                import wave

                with wave.open(str(out), "wb") as wav_file:
                    if hasattr(model, "synthesize_wav"):
                        model.synthesize_wav(text, wav_file)
                    else:  # piper-tts < 1.3
                        model.synthesize(text, wav_file)
            except ImportError:
                self._render_cli(text, voice, out)
            clip = AudioClip.read(out)
        clip.meta = {"text": text, "speaker": voice.speaker_id}
        return clip

    def _render_cli(self, text: str, voice: VoiceSpec, out: Path) -> None:
        binary = shutil.which(self.settings.piper_bin) or self.settings.piper_bin
        cmd = [binary, "--model", str(self.model_path(voice)), "--output_file", str(out)]
        proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=600)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"piper failed: {proc.stderr.decode('utf-8', 'ignore')[:500]}")


# ---------------------------------------------------------------------------
# Chatterbox Multilingual (final tier)
# ---------------------------------------------------------------------------

class ChatterboxSynth:
    name = "chatterbox"
    deterministic = False

    def __init__(self, settings: Settings, language_id: str = "nl", cfg_weight: float = 0.5):
        self.settings = settings
        self.language_id = language_id
        self.cfg_weight = cfg_weight
        self._model = None
        self.sample_rate = 24000

    def _load(self):
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore

            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.settings.device)
            self.sample_rate = int(getattr(self._model, "sr", 24000))
        return self._model

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        import torch  # type: ignore

        model = self._load()
        if voice.ref_path is None or not Path(voice.ref_path).is_file():
            raise FileNotFoundError(f"reference clip missing for {voice.speaker_id}: {voice.ref_path}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wav = model.generate(
            text,
            language_id=self.language_id,
            audio_prompt_path=str(voice.ref_path),
            exaggeration=float(exaggeration),
            cfg_weight=self.cfg_weight,
        )
        samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return AudioClip(samples, self.sample_rate, meta={"text": text, "seed": seed, "exaggeration": exaggeration,
                                                          "speaker": voice.speaker_id})


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


# ---------------------------------------------------------------------------
# ElevenLabs v3 Text-to-Dialogue (accent tier, per block)
# ---------------------------------------------------------------------------

class ElevenLabsDialogue:
    name = "elevenlabs"
    deterministic = False
    BASE_URL = "https://api.elevenlabs.io/v1"
    MAX_CHARS = 3000

    def __init__(self, settings: Settings, output_format: str = "mp3_44100_128"):
        if not settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        self.settings = settings
        self.output_format = output_format
        self.sample_rate = 44100

    def render_dialogue(self, inputs: list[dict[str, str]]) -> AudioClip:
        """inputs: [{"text": "...", "voice_id": "..."}, ...]; text may contain [audio tags]."""
        import httpx

        total = sum(len(i["text"]) for i in inputs)
        if total > self.MAX_CHARS:
            raise ValueError(f"block has {total} characters including tags; the limit is {self.MAX_CHARS}")
        response = httpx.post(
            f"{self.BASE_URL}/text-to-dialogue",
            params={"output_format": self.output_format},
            headers={"xi-api-key": self.settings.elevenlabs_api_key or "", "Content-Type": "application/json"},
            json={"inputs": inputs, "model_id": self.settings.elevenlabs_model},
            timeout=300.0,
        )
        response.raise_for_status()
        clip = AudioClip.from_bytes(response.content)
        clip.meta = {"chars": total, "inputs": len(inputs)}
        return clip

    def render(self, text: str, voice: VoiceSpec, *, exaggeration: float, seed: int) -> AudioClip:
        if not voice.eleven_voice_id:
            raise RuntimeError(f"speaker {voice.speaker_id} has no voice_id_final")
        return self.render_dialogue([{"text": text, "voice_id": voice.eleven_voice_id}])


def make_synth(tier: str, settings: Settings, *, pool: bool = True) -> Synth:
    if tier == "null":
        return NullSynth(sample_rate=settings.sample_rate, chars_per_second=settings.chars_per_second)
    if tier == "draft":
        return PiperSynth(settings)
    if tier == "final":
        if pool and settings.chatterbox_workers > 1 and os.environ.get("STUDIEPODCAST_NO_POOL") != "1":
            return SynthPool(ChatterboxSynth, settings.chatterbox_workers, "chatterbox", 24000, settings=settings)
        return ChatterboxSynth(settings)
    if tier == "elevenlabs":
        return ElevenLabsDialogue(settings)
    raise ValueError(f"unknown synth tier {tier}")
