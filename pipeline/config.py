"""Runtime settings, read from the environment (and an optional .env file).

Nothing here touches the network or the GPU. Every stage receives a Settings
instance instead of reading os.environ itself, so tests can override freely.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, replace
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    """Load KEY=VALUE lines from a .env file into os.environ (existing keys win)."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.strip().casefold() in ("1", "true", "yes", "on", "ja")


# Quality over speed: what an unhurried run uses. Nothing here is about speed — the point of the mode
# is that a chapter gets read, checked against the source and rewritten more times, so one run can
# legitimately take hours and the per-call ceiling has to be comfortable for a slow local model.
# max means "at least this many", min means "read no more than this many characters per call".
QUALITY_PASSES: dict[str, tuple[int | float, str]] = {
    "plan_review_rounds": (3, "max"),
    "plan_part_chars": (4000, "min"),
    "script_revise_rounds": (3, "max"),
    "opencode_timeout_s": (3600.0, "max"),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    cast_dir: Path = Path("cast")

    # Script side: "opencode" (a running OpenCode server on this machine, the only real backend)
    # or "fake" (canned answers, the same thing --fake-llm does).
    llm_backend: str = "opencode"

    # The OpenCode app writes with whatever model its own opencode.json configures; the pipeline
    # only hands over prompts. Leave url/password empty to find the local desktop server via
    # ~/.config/opencode/service.json and its logs. Model is "provider/model", optionally "#variant".
    opencode_url: str | None = None
    opencode_password: str | None = None
    opencode_model: str = "strata/strata"
    opencode_timeout_s: float = 1800.0
    # A local model server that OpenCode starts on the first prompt (Strata, a llama.cpp launcher plugin) is
    # started the same way by the pipeline, then given this long to answer. OPENCODE_MODEL_START is an
    # optional command that starts it directly, for setups where a prompt in OpenCode does not.
    opencode_wake_timeout_s: float = 300.0
    opencode_model_start: str | None = None

    # Content plan and lexicon: per section, quotes checked against the source, then a chapter pass
    # and review rounds (pipeline/plan/thorough.py).
    plan_review_rounds: int = 1
    plan_part_chars: int = 8000  # small sections are read together up to this size
    # How many times a script the audit blocks gets rewritten before it is accepted as it is.
    script_revise_rounds: int = 1
    script_polish: bool = True  # after the facts pass: one dialogue edit per segment (Writer.polish)

    # Quality over speed. On, the four passes above go to the values in QUALITY_PASSES and a script
    # that merely passes the audit but speaks flat gets rewritten too. See Settings.with_quality().
    quality_mode: bool = False

    # Nothing leaves this machine: Hugging Face libraries (Chatterbox, Whisper weights) run from
    # their local cache only and send no telemetry.
    offline: bool = False

    # Local synthesis: Chatterbox on the CUDA box, faster-whisper for verification and WhisperX for
    # word timings. Run `studiepodcast doctor` to see what is actually installed.
    device: str = "cuda"
    chatterbox_workers: int = 3
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"

    # Episode shape.
    target_minutes: int = 25
    chars_per_second: float = 15.0  # Dutch conversational speech, used for duration estimates
    target_lufs: float = -16.0
    line_lufs: float = -18.0
    wer_threshold: float = 0.05
    takes_per_line: int = 3
    max_interrupts_per_10min: int = 4
    min_connective_per_10min: float = 2.0  # interrupts + backchannels; below this a script tends to read as two monologues
    min_emotional_reaction_rate: float = 0.3  # share of tagged lines the other host must pick up on (mirror or counter) in the next line
    max_unwritten_gap_s: float = 1.2
    sample_rate: int = 24000

    @classmethod
    def from_env(cls, dotenv: Path | str | None = ".env", **overrides) -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        quality = _env_bool("STUDIEPODCAST_QUALITY_MODE", False) or bool(overrides.pop("quality_mode", False))

        def passes(env: str, name: str) -> int | float:
            """How much work a writing stage may spend. A value you set on the environment always
            wins, so quality mode only fills in the passes you left alone."""
            raw = _env_str(env)
            if raw is not None:
                return float(raw) if isinstance(getattr(Settings, name), float) else int(raw)
            return QUALITY_PASSES[name][0] if quality else getattr(Settings, name)

        base = cls(
            plan_review_rounds=passes("STUDIEPODCAST_PLAN_REVIEW_ROUNDS", "plan_review_rounds"),
            plan_part_chars=max(2000, int(passes("STUDIEPODCAST_PLAN_PART_CHARS", "plan_part_chars"))),
            script_revise_rounds=max(0, int(passes("STUDIEPODCAST_REVISE_ROUNDS", "script_revise_rounds"))),
            quality_mode=quality,
            offline=_env_bool("STUDIEPODCAST_OFFLINE", False),
            script_polish=_env_bool("STUDIEPODCAST_POLISH", True),
            llm_backend=_env_str("STUDIEPODCAST_LLM_BACKEND", "opencode").strip().casefold(),
            opencode_url=_env_str("OPENCODE_URL"),
            opencode_password=_env_str("OPENCODE_PASSWORD"),
            opencode_model=_env_str("OPENCODE_MODEL", "strata/strata"),
            opencode_timeout_s=float(passes("OPENCODE_TIMEOUT", "opencode_timeout_s")),
            opencode_wake_timeout_s=_env_float("OPENCODE_WAKE_TIMEOUT", 300.0),
            opencode_model_start=_env_str("OPENCODE_MODEL_START"),
            data_dir=Path(_env_str("STUDIEPODCAST_DATA_DIR", "data")),
            cast_dir=Path(_env_str("STUDIEPODCAST_CAST_DIR", "cast")),
            device=_env_str("STUDIEPODCAST_DEVICE", "cuda"),
            chatterbox_workers=_env_int("CHATTERBOX_WORKERS", 3),
            whisper_model=_env_str("WHISPER_MODEL", "large-v3"),
            whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "int8"),
            target_minutes=_env_int("STUDIEPODCAST_TARGET_MINUTES", 25),
            chars_per_second=_env_float("STUDIEPODCAST_CHARS_PER_SECOND", 15.0),
            takes_per_line=_env_int("STUDIEPODCAST_TAKES", 3),
            wer_threshold=_env_float("STUDIEPODCAST_WER_THRESHOLD", 0.05),
        )
        return replace(base, **overrides) if overrides else base

    def __post_init__(self) -> None:
        if self.llm_backend not in ("opencode", "fake"):
            raise ValueError(f"STUDIEPODCAST_LLM_BACKEND must be 'opencode' or 'fake', not {self.llm_backend!r}")

    def with_(self, **overrides) -> Settings:
        quality = overrides.pop("quality_mode", None)
        out = self.with_quality(bool(quality)) if quality is not None else self
        return replace(out, **overrides) if overrides else out

    def with_quality(self, on: bool = True) -> Settings:
        """The settings an unhurried run should use.

        More review rounds over the plan and lexicon, a smaller piece of source per call so it is
        really read, more rewrite rounds over the script, and a per-call ceiling a thinking local model
        will not bump into. The dialogue-shape warnings (interruptions, reactions, cut words) count as
        work to fix in that mode rather than as a note in the audit file. It only ever moves a number
        towards the values in QUALITY_PASSES, so a run you already made more thorough stays as it is.
        """
        changes: dict[str, object] = {}
        for name, (value, direction) in QUALITY_PASSES.items():
            current = getattr(self, name)
            wants_more = value > current if direction == "max" else value < current
            if wants_more:
                changes[name] = value
        return replace(self, quality_mode=on, **changes)

    def profile(self) -> dict[str, object]:
        return {
            "platform": f"{platform.system()} {platform.machine()}",
            "device": self.device,
            "chatterbox_workers": self.chatterbox_workers,
            "whisper_model": self.whisper_model,
            "takes_per_line": self.takes_per_line,
            "wer_threshold": self.wer_threshold,
            "llm_backend": self.llm_backend,
            "llm_model": self.opencode_model,
            "quality_mode": self.quality_mode,
            "offline": self.offline,
        }

    def writing_passes(self) -> dict[str, object]:
        """How much work the text side may spend on one chapter, for doctor and the API health page.
        Quality mode changes these numbers and nothing else: it is the difference between a chapter
        that gets read once and one that gets read, checked against the source, merged, reviewed and
        rewritten until the audit is quiet."""
        return {
            "plan_review_rounds": self.plan_review_rounds,
            "plan_part_chars": self.plan_part_chars,
            "script_revise_rounds": self.script_revise_rounds,
            "script_polish": self.script_polish,
            "seconds_per_llm_call": self.opencode_timeout_s,
        }
