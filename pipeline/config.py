"""Runtime settings, read from the environment (and an optional .env file).

Nothing here touches the network or the GPU. Every stage receives a Settings
instance instead of reading os.environ itself, so tests can override freely.
"""

from __future__ import annotations

import os
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


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    cast_dir: Path = Path("cast")

    # Script side: frontier API.
    llm_model: str = "claude-opus-5"
    llm_effort: str = "high"
    llm_max_tokens: int = 32000
    anthropic_api_key: str | None = None

    # Accent tier.
    elevenlabs_api_key: str | None = None
    elevenlabs_model: str = "eleven_v3"

    # Local synthesis.
    piper_bin: str = "piper"
    piper_voices_dir: Path = Path("~/.local/share/piper/voices")
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
    max_unwritten_gap_s: float = 1.2
    sample_rate: int = 24000

    @classmethod
    def from_env(cls, dotenv: Path | str | None = ".env", **overrides) -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        base = cls(
            data_dir=Path(_env_str("STUDIEPODCAST_DATA_DIR", "data")),
            cast_dir=Path(_env_str("STUDIEPODCAST_CAST_DIR", "cast")),
            llm_model=_env_str("STUDIEPODCAST_LLM_MODEL", "claude-opus-5"),
            llm_effort=_env_str("STUDIEPODCAST_LLM_EFFORT", "high"),
            llm_max_tokens=_env_int("STUDIEPODCAST_LLM_MAX_TOKENS", 32000),
            anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
            elevenlabs_api_key=_env_str("ELEVENLABS_API_KEY"),
            elevenlabs_model=_env_str("ELEVENLABS_MODEL", "eleven_v3"),
            piper_bin=_env_str("PIPER_BIN", "piper"),
            piper_voices_dir=Path(_env_str("PIPER_VOICES_DIR", "~/.local/share/piper/voices")).expanduser(),
            device=_env_str("STUDIEPODCAST_DEVICE", "cuda"),
            chatterbox_workers=_env_int("CHATTERBOX_WORKERS", 3),
            whisper_model=_env_str("WHISPER_MODEL", "large-v3"),
            whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "int8"),
            target_minutes=_env_int("STUDIEPODCAST_TARGET_MINUTES", 25),
            chars_per_second=_env_float("STUDIEPODCAST_CHARS_PER_SECOND", 15.0),
        )
        return replace(base, **overrides) if overrides else base

    def with_(self, **overrides) -> Settings:
        return replace(self, **overrides)
