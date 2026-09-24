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


def apple_silicon() -> bool:
    """True on an M-series Mac: Metal for Chatterbox, MLX for Whisper, one worker."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def default_device() -> str:
    return "mps" if apple_silicon() else "cuda"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    cast_dir: Path = Path("cast")

    # Script side: "anthropic" (Claude API) or "local" (a model server on your own network).
    llm_backend: str = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_effort: str = "high"
    llm_max_tokens: int = 32000
    anthropic_api_key: str | None = None

    # Local LLM. api=ollama uses Ollama's native /api/chat, which lets every request set the
    # context window (Ollama's OpenAI endpoint can't, and silently truncates long prompts);
    # api=openai speaks /v1/chat/completions for llama.cpp, vLLM, LM Studio and the like.
    local_llm_url: str = "http://localhost:11434"
    local_llm_api: str = "ollama"
    local_llm_model: str = "gemma3:27b"
    local_llm_context: int = 32768
    local_llm_max_tokens: int = 8192
    local_llm_timeout_s: float = 1800.0
    local_llm_vision: bool = True  # False: figure captions are skipped instead of sent to a model that can't see
    local_llm_api_key: str | None = None  # only for servers started with a key (vLLM --api-key, ...)

    # Nothing leaves the local network: Claude API and ElevenLabs refuse to start, the local LLM
    # must resolve to a private address, Hugging Face libraries run from their cache only.
    offline: bool = False

    # Accent tier.
    elevenlabs_api_key: str | None = None
    elevenlabs_model: str = "eleven_v3"

    # Local synthesis. Defaults follow the platform: CUDA with three workers on a
    # 24GB GPU box, Metal with one worker and MLX Whisper on an M-series Mac.
    piper_bin: str = "piper"
    piper_voices_dir: Path = Path("~/.local/share/piper/voices")
    device: str = "cuda"
    chatterbox_workers: int = 3
    whisper_backend: str = "faster"  # faster (CTranslate2) | mlx (Apple Silicon)
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"
    mlx_whisper_model: str = "mlx-community/whisper-large-v3-turbo"

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
        offline = _env_bool("STUDIEPODCAST_OFFLINE", False)
        base = cls(
            offline=offline,
            llm_backend=_env_str("STUDIEPODCAST_LLM_BACKEND", "local" if offline else "anthropic").strip().casefold(),
            local_llm_url=_env_str("LOCAL_LLM_URL", "http://localhost:11434"),
            local_llm_api=_env_str("LOCAL_LLM_API", "ollama").strip().casefold(),
            local_llm_model=_env_str("LOCAL_LLM_MODEL", "gemma3:27b"),
            local_llm_context=_env_int("LOCAL_LLM_CONTEXT", 32768),
            local_llm_max_tokens=_env_int("LOCAL_LLM_MAX_TOKENS", 8192),
            local_llm_timeout_s=_env_float("LOCAL_LLM_TIMEOUT", 1800.0),
            local_llm_vision=_env_bool("LOCAL_LLM_VISION", True),
            local_llm_api_key=_env_str("LOCAL_LLM_API_KEY"),
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
            device=_env_str("STUDIEPODCAST_DEVICE", default_device()),
            chatterbox_workers=_env_int("CHATTERBOX_WORKERS", 1 if apple_silicon() else 3),
            whisper_backend=_env_str("WHISPER_BACKEND", "mlx" if apple_silicon() else "faster"),
            whisper_model=_env_str("WHISPER_MODEL", "large-v3"),
            whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "int8"),
            mlx_whisper_model=_env_str("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"),
            target_minutes=_env_int("STUDIEPODCAST_TARGET_MINUTES", 25),
            chars_per_second=_env_float("STUDIEPODCAST_CHARS_PER_SECOND", 15.0),
            takes_per_line=_env_int("STUDIEPODCAST_TAKES", 3),
            wer_threshold=_env_float("STUDIEPODCAST_WER_THRESHOLD", 0.05),
        )
        return replace(base, **overrides) if overrides else base

    def with_(self, **overrides) -> Settings:
        return replace(self, **overrides)

    def profile(self) -> dict[str, object]:
        return {
            "platform": f"{platform.system()} {platform.machine()}",
            "apple_silicon": apple_silicon(),
            "device": self.device,
            "chatterbox_workers": self.chatterbox_workers,
            "whisper_backend": self.whisper_backend,
            "whisper_model": self.mlx_whisper_model if self.whisper_backend == "mlx" else self.whisper_model,
            "takes_per_line": self.takes_per_line,
            "wer_threshold": self.wer_threshold,
            "llm_backend": self.llm_backend,
            "llm_model": self.local_llm_model if self.llm_backend == "local" else self.llm_model,
            "offline": self.offline,
        }
