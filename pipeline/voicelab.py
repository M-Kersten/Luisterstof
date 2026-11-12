"""Voice lab: one sentence, one voice, the synthesis knobs by hand.

The audition (M0) sweeps a grid from the command line; this is the same thing one take at a time, for the web
app's Stemmen page. Pick a host or guest that has a reference clip, type a sentence, set exaggeration,
cfg_weight, temperature and speech rate, and hear the take next to the ones before it. Every take is kept with
the settings and seed that made it, and the faster-whisper check the render uses (word error rate against the
sentence). When a take sounds right, its settings are written into ``cast/hosts.yaml`` or ``cast/guests.yaml``
for that voice, the comments in those files left as they are.

The model stays loaded between takes, so only the first one waits for Chatterbox to load. Takes run as jobs on
the app's single worker, so they never share the GPU with a render; a render releases the lab's model first.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.audio.synth import AudioClip, VoiceSpec, make_synth, voice_spec_for
from pipeline.config import Settings
from pipeline.cues import speakable
from pipeline.models import Cast, Guest, Host

# The knobs, with the range the page offers. Chatterbox accepts more, but outside these a voice falls apart.
PARAMS: dict[str, dict[str, float]] = {
    "exaggeration": {"min": 0.0, "max": 1.5, "step": 0.05, "default": 0.5},
    "cfg_weight": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
    "temperature": {"min": 0.3, "max": 1.4, "step": 0.05, "default": 0.8},
    "speech_rate": {"min": 0.75, "max": 1.25, "step": 0.01, "default": 1.0},
}
MAX_TEXT = 600  # one take; a longer text belongs in an episode
MIN_WORDS = 3  # Chatterbox Multilingual crashes on texts of about five tokens or fewer
TAKE_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9]+$")


def speakers(cast: Cast) -> list[tuple[str, Host | Guest]]:
    return [("host", h) for h in cast.hosts] + [("guest", g) for g in cast.guests]


def find_speaker(cast: Cast, speaker_id: str) -> tuple[str, Host | Guest]:
    for kind, sp in speakers(cast):
        if sp.id == speaker_id:
            return kind, sp
    raise KeyError(speaker_id)


def current_params(sp: Host | Guest) -> dict[str, float]:
    return {k: float(getattr(sp, k)) for k in PARAMS}


def clean_params(raw: dict[str, Any]) -> dict[str, float]:
    """The four knobs, each a number inside its range."""
    out = {}
    for key, spec in PARAMS.items():
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key} ontbreekt of is geen getal") from exc
        if not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{key} moet tussen {spec['min']} en {spec['max']} liggen")
        out[key] = round(value, 3)
    return out


def voice_info(cast: Cast, cast_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for kind, sp in speakers(cast):
        ref = cast_dir / sp.voice_ref if sp.voice_ref else None
        rows.append({
            "id": sp.id, "name": sp.name, "kind": kind,
            "role": getattr(sp, "role", None),
            "about": (getattr(sp, "background", None) or getattr(sp, "persona", None) or "").strip(),
            "voice_ref": sp.voice_ref, "has_ref": bool(ref and ref.is_file()),
            "ref_hash": voice_spec_for(sp, cast_dir).ref_hash if ref and ref.is_file() else None,
            "params": current_params(sp),
        })
    return rows


class VoiceLab:
    """Renders single takes and keeps them in ``data/voicelab/<speaker>/``."""

    def __init__(self, settings: Settings, *, fake_audio: bool = False):
        self.settings = settings
        self.fake_audio = fake_audio
        self.root = settings.data_dir / "voicelab"
        self._lock = threading.Lock()
        self._synth = None
        self._transcriber = None

    # -------------------------------------------------------------- model
    def loaded(self) -> bool:
        return self._synth is not None

    def release(self) -> None:
        """Give the GPU back, before a render or when the lab is done."""
        with self._lock:
            had = self._synth is not None
            self._synth = None
            self._transcriber = None
        if had and not self.fake_audio:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to free
                pass

    def _ensure(self):
        if self._synth is None:
            from pipeline.audio.asr import make_transcriber

            self._synth = make_synth("null" if self.fake_audio else "final", self.settings, pool=False)
            try:
                self._transcriber = make_transcriber(self.settings, fake=self.fake_audio)
            except Exception:  # noqa: BLE001 - a take without a check is still worth hearing
                self._transcriber = None
        return self._synth, self._transcriber

    # -------------------------------------------------------------- takes
    def generate(self, sp: Host | Guest, text: str, params: dict[str, float], seed: int | None = None,
                 on_event=None) -> dict[str, Any]:
        text = " ".join(text.split())
        spoken = speakable(text)
        if len(spoken) > MAX_TEXT:
            raise ValueError(f"Hooguit {MAX_TEXT} tekens per take.")
        if len(spoken.split()) < MIN_WORDS:
            raise ValueError(f"Typ minstens {MIN_WORDS} woorden: heel korte zinnen laten Chatterbox vastlopen.")
        voice: VoiceSpec = voice_spec_for(sp, self.settings.cast_dir)
        if not self.fake_audio and not (voice.ref_path and Path(voice.ref_path).is_file()):
            raise FileNotFoundError(f"{sp.name} heeft nog geen referentieclip ({sp.voice_ref}).")
        voice.base_exaggeration = params["exaggeration"]
        voice.cfg_weight = params["cfg_weight"]
        voice.temperature = params["temperature"]
        voice.speech_rate = params["speech_rate"]
        seed = int(seed) if seed is not None else random.randrange(1, 1_000_000)
        with self._lock:
            if on_event and self._synth is None:
                on_event("voice", "load", {"message": "het stemmodel wordt geladen; de eerste take duurt langer"})
            synth, transcriber = self._ensure()
            if on_event:
                on_event("voice", "start", {"speaker": sp.id, "label": f"{sp.name}: take maken"})
            clip: AudioClip = synth.render(spoken, voice, exaggeration=params["exaggeration"], seed=seed)
            if getattr(synth, "name", "") == "null":  # the real synth stretches inside render()
                clip = _stretch_plain(clip, voice.speech_rate)
            transcript = None
            if transcriber is not None:
                if on_event:
                    on_event("voice", "check", {"speaker": sp.id, "label": f"{sp.name}: take terugluisteren"})
                try:
                    transcript = transcriber.transcribe(clip)
                except Exception:  # noqa: BLE001
                    transcript = None
        from pipeline.audio.wer import word_error_rate

        take_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{seed}"
        folder = self.root / sp.id
        clip.write(folder / f"{take_id}.wav")
        record = {
            "id": take_id, "speaker": sp.id, "text": text, "spoken": spoken, "params": params, "seed": seed,
            "duration_s": round(clip.duration_s, 2), "transcript": transcript,
            "wer": None if transcript is None else round(word_error_rate(spoken, transcript), 4),
            "ref_hash": voice.ref_hash, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "synth": getattr(synth, "name", ""),
        }
        (folder / f"{take_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record

    def takes(self, speaker_id: str, limit: int = 60) -> list[dict[str, Any]]:
        folder = self.root / speaker_id
        out = []
        for meta in sorted(folder.glob("*.json"), reverse=True)[:limit]:
            try:
                record = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (folder / f"{record.get('id')}.wav").is_file():
                out.append(record)
        return out

    def take_path(self, speaker_id: str, take_id: str) -> Path:
        if not TAKE_ID.match(take_id):
            raise KeyError(take_id)
        path = self.root / speaker_id / f"{take_id}.wav"
        if not path.is_file():
            raise KeyError(take_id)
        return path

    def delete(self, speaker_id: str, take_id: str) -> None:
        path = self.take_path(speaker_id, take_id)
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)


def _stretch_plain(clip: AudioClip, rate: float) -> AudioClip:
    """Test mode only: the length a speech rate gives, without librosa (which comes with the real voices)."""
    if abs(rate - 1.0) < 1e-6 or not len(clip.samples):
        return clip
    import numpy as np

    n = max(1, int(round(len(clip.samples) / rate)))
    x = np.linspace(0.0, 1.0, num=n, endpoint=False)
    xp = np.linspace(0.0, 1.0, num=len(clip.samples), endpoint=False)
    return AudioClip(np.interp(x, xp, clip.samples).astype(np.float32), clip.sr, meta=dict(clip.meta))


# ------------------------------------------------------------------ settings
def _fmt(value: float) -> str:
    return f"{round(value, 3):g}"


def save_params(cast_dir: Path, speaker_id: str, params: dict[str, float]) -> Path:
    """Write the four knobs into the speaker's block in hosts.yaml or guests.yaml.

    A line-level edit: the comments and the layout of the file stay as they are, only the four values change (or
    are added under ``exaggeration`` when the file did not have them). The file is read back as a cast before it
    is kept; anything that does not load puts the old file back."""
    from pipeline.script.cast import load_cast

    for name in ("hosts.yaml", "guests.yaml"):
        path = cast_dir / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        start = next((i for i, ln in enumerate(lines)
                      if re.match(rf"^(\s*)-\s+id:\s*['\"]?{re.escape(speaker_id)}['\"]?\s*(#.*)?$", ln)), None)
        if start is None:
            continue
        dash = len(lines[start]) - len(lines[start].lstrip())
        indent = " " * (dash + 2)
        end = len(lines)
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            lead = len(lines[i]) - len(lines[i].lstrip())
            if lead <= dash:
                end = i
                break
        block = lines[start:end]
        newline = "\n" if lines[start].endswith("\n") else ""
        anchor = None
        for key in PARAMS:
            pattern = re.compile(rf"^{indent}{key}:\s*[^#\n]*?(\s*#.*)?$")
            hit = next((j for j, ln in enumerate(block) if pattern.match(ln.rstrip("\n"))), None)
            if hit is not None:
                comment = pattern.match(block[hit].rstrip("\n")).group(1) or ""
                block[hit] = f"{indent}{key}: {_fmt(params[key])}{comment}{newline or chr(10)}"
                if key == "exaggeration":
                    anchor = hit
        for key in ("cfg_weight", "temperature", "speech_rate"):
            pattern = re.compile(rf"^{indent}{key}:")
            if any(pattern.match(ln) for ln in block):
                continue
            new = f"{indent}{key}: {_fmt(params[key])}\n"
            if anchor is not None:
                anchor += 1
                block.insert(anchor, new)
            else:
                last = max(j for j, ln in enumerate(block) if ln.strip())
                block.insert(last + 1, new)
        if not any(re.match(rf"^{indent}exaggeration:", ln) for ln in block):
            last = max(j for j, ln in enumerate(block) if ln.strip())
            block.insert(last + 1, f"{indent}exaggeration: {_fmt(params['exaggeration'])}\n")
        backup = path.with_suffix(".yaml.bak")
        shutil.copyfile(path, backup)
        path.write_text("".join(lines[:start] + block + lines[end:]), encoding="utf-8")
        try:
            _, sp = find_speaker(load_cast(cast_dir), speaker_id)
            if any(abs(getattr(sp, k) - params[k]) > 1e-6 for k in PARAMS):
                raise ValueError("de waarden kwamen niet terug na het opslaan")
        except Exception:
            shutil.copyfile(backup, path)
            backup.unlink(missing_ok=True)
            raise
        backup.unlink(missing_ok=True)
        return path
    raise KeyError(speaker_id)
