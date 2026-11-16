"""Voice lab: one sentence, one voice, the synthesis knobs by hand.

The audition (M0) sweeps a grid from the command line; this is the same thing one take at a time, for the web
app's Stemmen page. Pick a host or guest that has a reference clip, type a sentence, set exaggeration,
cfg_weight, temperature and speech rate, and hear the take next to the ones before it. Every take is kept with
the settings and seed that made it, and the faster-whisper check the render uses (word error rate against the
sentence). When a take sounds right, its settings are written into ``cast/hosts.yaml`` or ``cast/guests.yaml``
for that voice, the comments in those files left as they are.

The model stays loaded between takes, so only the first one waits for Chatterbox to load. Takes run as jobs on
the app's single worker, so they never share the GPU with a render; a render releases the lab's model first.

The Whisper check after a take is the part that can hang: a first use that would download the model, a GPU
already full with Chatterbox and the writing model (Windows then spills into shared memory and crawls), a
CUDA/cuDNN mix-up between torch and CTranslate2. So the take is saved before the check, the check never downloads,
runs on the processor when the GPU is short of memory, has a time limit, and stops when the job is stopped. A
check that hung once is not tried again until the model is released (the button on the page, or a restart).
"""

from __future__ import annotations

import json
import random
import re
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.audio.synth import AudioClip, VoiceSpec, make_synth, voice_spec_for
from pipeline.cancel import Cancelled
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
CHECK_LOAD_S = 180  # loading Whisper from disk; a download is never done inside a take
CHECK_RUN_S = 90  # listening back to one take (a few seconds on a GPU, under a minute on a laptop processor)
CHECK_GPU_GB = 2.5  # free GPU memory the check wants before it goes on the GPU


class CheckStalled(RuntimeError):
    pass


def _within(fn, seconds: float, should_stop=None):
    """Run ``fn`` on a helper thread and wait at most ``seconds``. A model call cannot be interrupted, so a call
    that runs over is left behind (the thread is a daemon) and CheckStalled is raised; a stop request ends the
    wait at once."""
    box: dict[str, Any] = {}

    def target():
        try:
            box["out"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            box["err"] = exc

    t = threading.Thread(target=target, daemon=True, name="voicelab-check")
    t.start()
    deadline = time.monotonic() + seconds
    while t.is_alive():
        t.join(0.25)
        if should_stop and should_stop():
            raise Cancelled()
        if t.is_alive() and time.monotonic() > deadline:
            raise CheckStalled(f"na {int(seconds)} s geen antwoord")
    if "err" in box:
        raise box["err"]
    return box.get("out")


def whisper_status(name: str) -> str:
    """"ok" when faster-whisper finds this model without the network (a folder, or the Hugging Face cache),
    "missing" when it would have to download it, "no-package" when faster-whisper is not installed."""
    try:
        from faster_whisper.utils import download_model  # type: ignore
    except Exception:  # noqa: BLE001
        return "no-package"
    if Path(name).is_dir():
        return "ok"
    try:
        download_model(name, local_files_only=True)
        return "ok"
    except Exception:  # noqa: BLE001 - not in the cache
        return "missing"


def free_gpu_gb() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
    except Exception:  # noqa: BLE001
        pass
    return None


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
        self._check_device: str | None = None
        self._check_broken: str | None = None  # why the check is off until release(), after it hung

    # -------------------------------------------------------------- model
    def loaded(self) -> bool:
        return self._synth is not None

    def release(self) -> None:
        """Give the GPU back, before a render or when the lab is done."""
        with self._lock:
            had = self._synth is not None
            self._synth = None
            self._transcriber = None
            self._check_device = None
            self._check_broken = None
        if had and not self.fake_audio:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to free
                pass

    def _ensure(self):
        if self._synth is None:
            self._synth = make_synth("null" if self.fake_audio else "final", self.settings, pool=False)
        return self._synth

    def _checker(self, say) -> tuple[Any, str]:
        """The Whisper transcriber for the check and where it runs, or (None, why not)."""
        if self.fake_audio:
            from pipeline.audio.asr import make_transcriber

            return make_transcriber(self.settings, fake=True), "test"
        if self._check_broken:
            return None, self._check_broken
        if self._transcriber is not None:
            return self._transcriber, self._check_device or ""
        name = self.settings.whisper_model
        status = whisper_status(name)
        if status == "no-package":
            return None, "faster-whisper is niet geïnstalleerd, dus takes worden niet teruggeluisterd ('studiepodcast doctor')."
        if status == "missing":
            return None, (f"Whisper ({name}) staat nog niet op deze computer. Haal het één keer op met "
                          "'studiepodcast fetch-whisper'; daarna worden takes teruggeluisterd.")
        device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
        free = free_gpu_gb() if device == "cuda" else None
        if device == "cuda" and free is not None and free < CHECK_GPU_GB:
            device = "cpu"
            say("voice", "check_device", {"device": "cpu", "free_gb": round(free, 1)})
        from pipeline.audio.asr import FasterWhisperTranscriber

        self._transcriber = FasterWhisperTranscriber(self.settings.with_(device=device))
        self._check_device = device
        return self._transcriber, device

    # -------------------------------------------------------------- takes
    def generate(self, sp: Host | Guest, text: str, params: dict[str, float], seed: int | None = None,
                 on_event=None, should_stop=None) -> dict[str, Any]:
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

        def say(stage: str, status: str, data: dict[str, Any]) -> None:
            if on_event:
                on_event(stage, status, data)

        with self._lock:
            synth = self._ensure()
            load = getattr(synth, "_load", None)
            if callable(load) and getattr(synth, "_model", True) is None:  # the first take loads Chatterbox
                say("voice", "load", {"speaker": sp.id})
                started = time.monotonic()
                load()
                say("voice", "loaded", {"seconds": round(time.monotonic() - started, 1),
                                        "device": getattr(synth, "device", "")})
            say("voice", "start", {"speaker": sp.id, "name": sp.name})
            started = time.monotonic()
            clip: AudioClip = synth.render(spoken, voice, exaggeration=params["exaggeration"], seed=seed)
            if getattr(synth, "name", "") == "null":  # the real synth stretches inside render()
                clip = _stretch_plain(clip, voice.speech_rate)
            say("voice", "made", {"seconds": round(time.monotonic() - started, 1), "audio_s": round(clip.duration_s, 1)})

            # Saved before the check: a check that hangs or fails never costs the take itself.
            take_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{seed}"
            folder = self.root / sp.id
            clip.write(folder / f"{take_id}.wav")
            record = {
                "id": take_id, "speaker": sp.id, "text": text, "spoken": spoken, "params": params, "seed": seed,
                "duration_s": round(clip.duration_s, 2), "transcript": None, "wer": None,
                "check": "pending", "check_note": "",
                "ref_hash": voice.ref_hash, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "synth": getattr(synth, "name", ""),
            }
            meta = folder / f"{take_id}.json"
            meta.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            try:
                record.update(self._check(clip, spoken, say, should_stop))
            finally:
                if record["check"] == "pending":  # stopped during the check
                    record.update(check="skipped", check_note="Gestopt tijdens het terugluisteren.")
                meta.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record

    def _check(self, clip: AudioClip, spoken: str, say, should_stop) -> dict[str, Any]:
        """Whisper listens back: the transcript and the word error rate, or why there is none."""
        from pipeline.audio.wer import word_error_rate

        transcriber, where = self._checker(say)
        if transcriber is None:
            say("voice", "check_skipped", {"reason": where})
            return {"check": "skipped", "check_note": where}
        loading = getattr(transcriber, "_model", True) is None and callable(getattr(transcriber, "_load", None))
        say("voice", "check", {"device": where, "loading": loading, "model": self.settings.whisper_model})
        started = time.monotonic()
        phase = "laden" if loading else "terugluisteren"
        try:
            if loading:
                _within(transcriber._load, CHECK_LOAD_S, should_stop)
                phase = "terugluisteren"
            transcript = _within(lambda: transcriber.transcribe(clip), CHECK_RUN_S, should_stop)
        except CheckStalled as exc:
            reason = (f"Whisper liep vast bij het {phase} ({exc}). De volgende takes worden niet teruggeluisterd "
                      "tot je het model vrijgeeft of de app opnieuw start.")
            return self._give_up(reason, "timeout", say)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken Whisper install: the take is still worth hearing
            return self._give_up(f"Whisper kon niet starten: {str(exc)[:200]}", "error", say)
        if transcript is None:
            note = "Whisper gaf geen tekst terug; de fout staat in het venster van de app."
            say("voice", "check_skipped", {"reason": note})
            return {"check": "error", "check_note": note}
        wer = round(word_error_rate(spoken, transcript), 4)
        say("voice", "checked", {"wer": wer, "seconds": round(time.monotonic() - started, 1)})
        return {"check": "ok", "check_note": "", "transcript": transcript, "wer": wer}

    def _give_up(self, reason: str, kind: str, say) -> dict[str, Any]:
        """No check for this take, nor for the next ones until release(): a Whisper that hung or broke once
        does it again, and each try would cost the time limit."""
        self._transcriber = None
        self._check_broken = reason
        say("voice", "check_skipped", {"reason": reason})
        return {"check": kind, "check_note": reason}

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
