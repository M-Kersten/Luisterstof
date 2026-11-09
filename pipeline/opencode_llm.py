"""Your own model, through the OpenCode server instead of a model server.

This is the pipeline's only writing backend. It asks a running OpenCode server
to generate on its behalf: the model, its context window and its quantisation
live in OpenCode's own configuration (``opencode.json``), and the pipeline just
hands over a prompt. Two endpoints do all the work:

* ``POST /api/experimental/generate/text`` with ``{prompt, model}`` answers
  with ``{data: {text}}``, stateless, in OpenCode's Basic-auth HTTP API.
* ``GET /api/model`` lists what the server can reach (used by ``doctor``).

The JSON schema is written into the prompt and every answer is validated; an
invalid one goes back once together with the error. Temperature, thinking and
the token budget are not sent: they belong to the model's configuration in
OpenCode, not to the caller. The endpoint is text only, so prompts are plain
text: ``pipeline.llm.content_text`` joins the text blocks and nothing sends a
picture.

The server is discovered in this order: ``OPENCODE_URL``, the ``url`` field of
``~/.config/opencode/service.json``, the URL the OpenCode desktop app last
logged (``%APPDATA%/ai.opencode.desktop/logs/*/main.log``), then the
``opencode serve`` default ``http://127.0.0.1:4096``. The Basic-auth password
comes from ``OPENCODE_PASSWORD`` or that same service.json.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from pydantic import BaseModel

from pipeline.cancel import Cancelled
from pipeline.config import Settings
from pipeline.llm import (
    LLMError,
    LLMRequest,
    LLMTruncated,
    SCHEMA_INSTRUCTION,
    Usage,
    clean_answer,
    content_text,
    estimate_tokens,
    inline_schema,
    looping,
    parse_json_into,
)
from pipeline.offline import check_local_url

log = logging.getLogger(__name__)

RETRIES = 1
FALLBACK_URL = "http://127.0.0.1:4096"  # what `opencode serve` listens on by default
AUTH_USERNAME = "opencode"  # the server only checks the password
LOOP_RETRY_NOTE = ("\n\nLet op: een eerder antwoord op deze vraag bleef hetzelfde herhalen tot het werd afgekapt. "
                   "Houd het beknopt, herhaal niets en sluit elke lijst zodra de inhoud op is.")

MODEL_URL = re.compile(r"url: '(https?://[^']+)'")


# ---------------------------------------------------------------------------
# server discovery (the desktop app picks its own port; the password file is the anchor)
# ---------------------------------------------------------------------------

def _service_file() -> dict[str, Any]:
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    for candidate in (Path(config_home) / "opencode" / "service.json", Path.home() / ".config" / "opencode" / "service.json"):
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _desktop_log_url() -> str | None:
    """The URL the OpenCode desktop app last bound, read from its newest main.log."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    logs = Path(appdata) / "ai.opencode.desktop" / "logs"
    try:
        mains = sorted(logs.glob("*/main.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for main in mains:
            found = MODEL_URL.findall(main.read_text(encoding="utf-8", errors="replace"))
            if found:
                return found[-1]
    except OSError:
        pass
    return None


def discover(settings: Settings) -> tuple[str, str | None]:
    """(base_url, password) for the OpenCode server, from settings, its service file and the desktop logs."""
    service = _service_file()
    url = settings.opencode_url or service.get("url") or _desktop_log_url() or FALLBACK_URL
    password = settings.opencode_password if settings.opencode_password is not None else service.get("password")
    return url.rstrip("/"), password


def split_model(ref: str) -> tuple[str, str, str | None]:
    """``provider/model`` (``#variant`` allowed) -> (providerID, id, variant). Without a slash, both are the same."""
    ref, _, variant = ref.partition("#")
    provider, sep, model = ref.partition("/")
    if not sep:
        return provider, provider, variant or None
    return provider, model, variant or None


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

class OpenCodeLLM:
    """The one LLM backend: a request with a pydantic schema in, a validated instance out."""

    name = "opencode"

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.usage = Usage()
        self.base_url, password = discover(settings)
        if settings.offline:
            check_local_url(self.base_url, label="OPENCODE_URL")
        self.provider_id, self.model_id, self.variant = split_model(settings.opencode_model)
        self._abort = threading.Event()
        self.on_tokens = None  # set by the runner: (task, answer tokens, seconds), for the progress display
        headers = {}
        if password:
            token = base64.b64encode(f"{AUTH_USERNAME}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        # trust_env=False: the local server is never reached through an HTTP(S)_PROXY from the environment.
        self.client = httpx.Client(base_url=self.base_url, headers=headers, transport=transport, trust_env=False,
                                   timeout=httpx.Timeout(settings.opencode_timeout_s, connect=10.0))

    @property
    def model_ref(self) -> str:
        ref = f"{self.provider_id}/{self.model_id}"
        return f"{ref}#{self.variant}" if self.variant else ref

    # ------------------------------------------------------------------ request
    def _prompt(self, request: LLMRequest, schema: dict) -> str:
        system_text = content_text(request.system)
        user_text = content_text(request.user)
        return "\n\n".join(part for part in (
            system_text + SCHEMA_INSTRUCTION + json.dumps(schema, ensure_ascii=False), user_text) if part)

    def _post(self, prompt: str) -> str:
        body: dict[str, Any] = {"prompt": prompt, "model": {"providerID": self.provider_id, "id": self.model_id}}
        if self.variant:
            body["model"]["variant"] = self.variant
        try:
            response = self.client.post("/api/experimental/generate/text", json=body)
        except httpx.HTTPError as exc:
            if self._abort.is_set():
                raise Cancelled() from exc
            raise LLMError(f"OpenCode at {self.base_url} not reachable ({type(exc).__name__}: {exc}). "
                           "Is the OpenCode app running? `studiepodcast doctor` checks it.") from exc
        if self._abort.is_set():
            raise Cancelled()
        if response.status_code == 401:
            raise LLMError(f"OpenCode at {self.base_url} rejected the password (401). The desktop app regenerates "
                           "~/.config/opencode/service.json on updates: rerun `studiepodcast doctor`, or set "
                           "OPENCODE_URL and OPENCODE_PASSWORD yourself.")
        if response.status_code in (404, 422):
            raise LLMError(f"OpenCode does not know model {self.model_ref} ({response.status_code}): pick it in the "
                           "OpenCode app or opencode.json, or point OPENCODE_MODEL at one it lists.")
        if response.status_code >= 400:
            raise LLMError(f"OpenCode at {self.base_url} answered {response.status_code}: {response.text[:500]}")
        return ((response.json() or {}).get("data") or {}).get("text") or ""

    def generate(self, request: LLMRequest) -> BaseModel:
        if self._abort.is_set():
            raise Cancelled()
        schema = inline_schema(request.schema.model_json_schema())
        prompt = self._prompt(request, schema)
        error: LLMError | None = None
        for attempt in range(RETRIES + 1):
            started = time.monotonic()
            log.info("llm %s task=%s model=%s attempt=%d", self.name, request.task, self.model_ref, attempt + 1)
            if self._abort.is_set():
                raise Cancelled()
            answer = clean_answer(self._post(prompt))
            seconds = time.monotonic() - started
            out_tokens = estimate_tokens(answer)
            self.usage.add(SimpleNamespace(input_tokens=estimate_tokens(prompt), output_tokens=out_tokens), seconds)
            if self.on_tokens:
                self.on_tokens(request.task, out_tokens, seconds)
            log.info("llm %s task=%s: ~%d prompt + ~%d answer tokens in %.0f s", self.name, request.task,
                     estimate_tokens(prompt), out_tokens, seconds)
            if looping(answer):
                if attempt < RETRIES:
                    log.warning("model repeated itself on %s until the limit; retrying once", request.task)
                    prompt = prompt + LOOP_RETRY_NOTE
                    continue
                raise LLMTruncated(f"task {request.task}: {self.model_ref} got stuck repeating itself, also on the "
                                   "retry. That is the model, not the pipeline: use a stronger model in OpenCode "
                                   "(OPENCODE_MODEL).")
            try:
                return parse_json_into(answer, request.schema)
            except LLMError as exc:
                error = exc
                log.warning("OpenCode answer for %s did not match %s (attempt %d): %s",
                            request.task, request.schema.__name__, attempt + 1, str(exc)[:300])
                prompt = (prompt + "\n\nEerdere poging:\n" + answer[:6000]
                          + f"\n\nDat antwoord is ongeldig: {str(exc)[:800]}\n"
                            "Geef opnieuw uitsluitend één geldig JSON-object volgens het schema.")
        raise LLMError(f"OpenCode model {self.model_ref} gave no valid {request.schema.__name__} for {request.task}: {error}")

    def abort(self) -> None:
        """Refuse new calls and end the one in flight at its next answer (the endpoint is not streaming, so a
        call that is already running is finished by the server first)."""
        self._abort.set()

    # ------------------------------------------------------------------ housekeeping
    def release(self) -> None:
        """Nothing to unload: the model belongs to OpenCode and may be busy with someone else's session."""

    def check(self) -> tuple[bool, str]:
        """(usable, explanation) for `studiepodcast doctor`: server reachable and the model listed."""
        try:
            response = self.client.get("/api/model", timeout=15.0)
            response.raise_for_status()
            models = response.json().get("data", [])
            wanted = f"{self.provider_id}/{self.model_id}"
            for m in models:
                if m.get("providerID") == self.provider_id and m.get("modelID") in (self.model_id, wanted) or m.get("id") == wanted:
                    return True, f"reachable, {self.model_ref} listed ({len(models)} models)"
            listed = sorted({f"{m.get('providerID')}/{m.get('modelID')}" for m in models})
            if not listed:
                return False, f"reachable, but it lists no models at all: configure one in OpenCode (opencode.json)"
            return False, (f"reachable, but {self.model_ref} is not listed. Present: {', '.join(listed[:12])}"
                           + (" ..." if len(listed) > 12 else "") + f". Pick one or set OPENCODE_MODEL=provider/model.")
        except httpx.HTTPError as exc:
            return False, f"not reachable at {self.base_url} ({type(exc).__name__}: {exc}). Is the OpenCode app running?"
