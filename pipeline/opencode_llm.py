"""Your own model, through the OpenCode server instead of a model server.

This is the pipeline's only writing backend. It asks a running OpenCode server
to generate on its behalf: the model, its context window and its quantisation
live in OpenCode's own configuration (``opencode.json``), and the pipeline just
hands over a prompt. Two endpoints do all the work:

* ``POST /api/experimental/generate/text`` with ``{prompt, model}`` answers
  with ``{data: {text}}``, stateless, in OpenCode's Basic-auth HTTP API.
* ``GET /api/model`` lists what the server can reach (used by ``doctor``).

The JSON schema is written into the prompt and every answer is validated; an
invalid one goes back once together with the error. A provider that stumbles —
the local model server dropping a stream, a 5xx, the server restarting under
the call — is retried on the spot with growing waits: a run of hours does not
die on one hiccup. Temperature, thinking and the token budget are not sent:
they belong to the model's configuration in OpenCode, not to the caller. The
endpoint is text only, so prompts are plain text: ``pipeline.llm.content_text``
joins the text blocks and nothing sends a picture.

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
    SCHEMA_INSTRUCTION,
    LLMError,
    LLMRequest,
    LLMTruncated,
    LLMUnavailable,
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
TRANSIENT_RETRIES = 5  # provider hiccups get their own ladder, on top of the schema repair round
RETRY_WAIT_S = 5.0  # 5, 10, 20, 40, 60 s: a dropped stream comes back, a dead server is noticed
FALLBACK_URL = "http://127.0.0.1:4096"  # what `opencode serve` listens on by default
AUTH_USERNAME = "opencode"  # the server only checks the password
TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
ABORT_POLL_S = 0.25  # how often a waiting call looks at the Stop button
LOOP_RETRY_NOTE = ("\n\nLet op: een eerder antwoord op deze vraag bleef hetzelfde herhalen tot het werd afgekapt. "
                   "Houd het beknopt, herhaal niets en sluit elke lijst zodra de inhoud op is.")

MODEL_URL = re.compile(r"url: '(https?://[^']+)'")


class LLMTransient(LLMError):
    """The server or its model stumbled this time — a plain retry can fix it: a dropped stream, a
    5xx, a server that was restarting while the call was in flight."""


# The model server behind OpenCode is not there: nothing listens on its port, its host does not resolve.
UNREACHABLE = re.compile(r"ECONNREFUSED|ENOTFOUND|EHOSTUNREACH|ETIMEDOUT|Unable to connect|fetch failed|connect ECONN", re.I)
PING_PROMPT = "Antwoord met precies één woord: ok"
WAKE_POLL_S = 10.0  # how often a waking model is asked whether it answers yet
WAKE_COOLDOWN_S = 600.0  # one wake per ten minutes: a model that will not start is reported, not hammered
WAKE_TITLE = "Luisterstof: model starten"
# Where OpenCode takes a first prompt: the v2 HttpApi this pipeline talks to, then the v1 server routes.
SESSION_ROUTES = (("/api/session", "/api/session/{id}/prompt", "/api/session/{id}"),
                  ("/session", "/session/{id}/prompt_async", "/session/{id}"))


def _reason(text: str) -> str:
    """The provider's own words for the failure ("ECONNREFUSED: Unable to connect"), without the JSON around them."""
    match = re.search(r'"message"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else text[:200]


def _session_id(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    return data.get("id") or data.get("sessionID") or inner.get("id")


def _stumbled(status: int, text: str) -> bool:
    """Is this error answer worth another try? OpenCode wraps its provider's failures in a
    ProviderRequestError (``stream ended without finish_reason`` is the classic), which says the
    server itself was fine and the model underneath it dropped the ball."""
    return status in TRANSIENT_STATUSES or "ProviderRequestError" in text or "stream ended" in text


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


def _strip_jsonc(text: str) -> str:
    """opencode.jsonc allows comments and trailing commas; json does not. Strings (URLs!) are left alone."""
    out, i, in_str = [], 0, False
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def provider_url(provider_id: str) -> str | None:
    """The base URL OpenCode's own config gives this provider, if a global opencode.json(c) says so."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    for folder in (Path(config_home) / "opencode", Path.home() / ".config" / "opencode"):
        for name in ("opencode.json", "opencode.jsonc"):
            path = folder / name
            try:
                if path.is_file():
                    config = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
                    options = ((config.get("provider") or {}).get(provider_id) or {}).get("options") or {}
                    url = options.get("baseURL") or options.get("baseUrl")
                    if url:
                        return str(url)
            except (OSError, ValueError, AttributeError):
                continue
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
        self.on_status = None  # set by the runner: a sentence for the log ("model wordt gestart")
        self._woken_at: float | None = None
        self._start_command_run = False
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
            raise LLMTransient(f"OpenCode at {self.base_url} not reachable ({type(exc).__name__}: {exc}). "
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
            message = f"OpenCode at {self.base_url} answered {response.status_code}: {response.text[:500]}"
            raise LLMTransient(message) if _stumbled(response.status_code, response.text) else LLMError(message)
        return ((response.json() or {}).get("data") or {}).get("text") or ""

    def _post_abortable(self, prompt: str) -> str:
        """``_post`` on a helper thread, so a Stop ends the wait at once. The endpoint does not stream and a
        blocked read cannot be interrupted from outside, so the call is left to finish on its own: OpenCode
        completes that answer in the background and nobody reads it."""
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["text"] = self._post(prompt)
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread as is
                box["error"] = exc

        worker = threading.Thread(target=call, name="opencode-call", daemon=True)
        worker.start()
        while worker.is_alive():
            if self._abort.wait(ABORT_POLL_S):
                raise Cancelled()
        if "error" in box:
            raise box["error"]
        return box["text"]

    def _post_resilient(self, prompt: str, task: str) -> str:
        """``_post`` with a ladder for stumbles that are not this prompt's fault. The wait is an
        abort-wait, so a Stop interrupts the backoff immediately rather than after it. A model server
        that is not running is started the way the OpenCode app starts it (see ``wake``) before the
        ladder goes on."""
        woke = False
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                return self._post_abortable(prompt)
            except LLMTransient as exc:
                if self._abort.is_set():
                    raise Cancelled() from exc
                if self._provider_down(exc) and not woke:
                    woke = True
                    if self.wake():
                        continue
                if attempt == TRANSIENT_RETRIES:
                    if self._unreachable(exc):
                        raise LLMUnavailable(self._unavailable_message(exc, TRANSIENT_RETRIES)) from exc
                    raise LLMError(f"{exc} (also after {TRANSIENT_RETRIES} retries)") from exc
                wait = min(RETRY_WAIT_S * 2 ** attempt, 60.0)
                log.warning("OpenCode stumbled on %s (attempt %d/%d): %s — retrying in %.0f s",
                            task, attempt + 1, TRANSIENT_RETRIES + 1, str(exc)[:200], wait)
                if self._abort.wait(wait):
                    raise Cancelled() from exc
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ the model behind OpenCode
    def _unreachable(self, exc: Exception) -> bool:
        return isinstance(exc.__cause__, httpx.HTTPError) or bool(UNREACHABLE.search(str(exc)))

    def _provider_down(self, exc: Exception) -> bool:
        """OpenCode answered, but the model server behind it is not there."""
        return not isinstance(exc.__cause__, httpx.HTTPError) and bool(UNREACHABLE.search(str(exc)))

    def _status(self, message: str) -> None:
        log.info(message)
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:  # noqa: BLE001 - a log line must never end the call
                pass

    def _unavailable_message(self, exc: Exception, retries: int) -> str:
        if isinstance(exc.__cause__, httpx.HTTPError):
            return (f"OpenCode at {self.base_url} is not reachable (also after {retries} retries). Is the OpenCode app "
                    "running? Start it or `opencode serve`, then run again; 'Hele boek doorlopen' carries on where it "
                    "stopped.")
        url = provider_url(self.provider_id)
        where = f" at {url} (from opencode.json)" if url else ""
        return (f"OpenCode runs, but the model server behind provider '{self.provider_id}'{where} does not answer: "
                f"{_reason(str(exc))}. Starting it through a first prompt in OpenCode did not help within "
                f"{self.settings.opencode_wake_timeout_s:.0f} s. Start that model server yourself (or set "
                "OPENCODE_MODEL_START to the command that does), then run again; 'Hele boek doorlopen' carries on "
                "where it stopped.")

    def _first_prompt(self) -> str | None:
        """Send one short prompt in a new OpenCode session, without waiting for the answer: that is what the
        OpenCode app does on your first prompt, and what makes OpenCode (or a plugin in it) start a local model
        server. The stateless generate endpoint the pipeline uses does not. Returns the route that deletes the
        session again, or None when neither route set worked."""
        body = {"parts": [{"type": "text", "text": PING_PROMPT}],
                "model": {"providerID": self.provider_id, "modelID": self.model_id}}
        for create, prompt, delete in SESSION_ROUTES:
            try:
                created = self.client.post(create, json={"title": WAKE_TITLE}, timeout=30.0)
                session = _session_id(created.json()) if created.status_code < 400 else None
                if not session:
                    continue
                sent = self.client.post(prompt.format(id=session), json=body, timeout=60.0)
                if sent.status_code < 400:
                    return delete.format(id=session)
                self.client.delete(delete.format(id=session), timeout=15.0)
            except (httpx.HTTPError, ValueError):
                continue
        return None

    def _run_start_command(self) -> None:
        command = self.settings.opencode_model_start
        if not command or self._start_command_run:
            return
        self._start_command_run = True
        import subprocess
        import sys

        flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        try:
            subprocess.Popen(command, shell=True, creationflags=flags, stdin=subprocess.DEVNULL,  # noqa: S602
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._status(f"OPENCODE_MODEL_START gestart: {command}")
        except OSError as exc:
            self._status(f"OPENCODE_MODEL_START kon niet starten: {exc}")

    def wake(self) -> bool:
        """Start the model behind OpenCode and wait until it answers. True when it does, within
        OPENCODE_WAKE_TIMEOUT. At most once per WAKE_COOLDOWN_S: a model that will not come is reported."""
        now = time.monotonic()
        if self._woken_at is not None and now - self._woken_at < WAKE_COOLDOWN_S:
            return False
        self._woken_at = now
        self._status(f"het model achter '{self.provider_id}' draait niet: het wordt gestart via een eerste prompt "
                     "in OpenCode (dat kan een minuut duren)")
        cleanup = self._first_prompt()
        self._run_start_command()
        deadline = now + self.settings.opencode_wake_timeout_s
        try:
            while True:
                try:
                    self._post_abortable(PING_PROMPT)
                    self._status(f"het model achter '{self.provider_id}' antwoordt na {time.monotonic() - now:.0f} s")
                    return True
                except LLMTransient:
                    if time.monotonic() >= deadline:
                        return False
                if self._abort.wait(WAKE_POLL_S):
                    raise Cancelled()
        finally:
            if cleanup:
                try:
                    self.client.delete(cleanup, timeout=15.0)
                except httpx.HTTPError:
                    pass

    def ping(self) -> float:
        """One tiny answer from the model before a job that will need hundreds: seconds it took. A model server
        that is not running is started first (``wake``). Raises LLMUnavailable when OpenCode or the model
        cannot be reached, so a night's run fails in seconds with a clear message instead of an hour in."""
        started = time.monotonic()
        if self._abort.is_set():
            raise Cancelled()
        try:
            self._post_abortable(PING_PROMPT)
            return time.monotonic() - started
        except LLMTransient as exc:
            if self._provider_down(exc) and self.wake():
                return time.monotonic() - started
            if self._unreachable(exc):
                raise LLMUnavailable(self._unavailable_message(exc, 0)) from exc
            raise LLMUnavailable(f"The model behind OpenCode does not answer: {str(exc)[:300]}") from exc

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
            answer = clean_answer(self._post_resilient(prompt, request.task))
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
        """Refuse new calls and stop waiting for the one in flight. The endpoint does not stream, so OpenCode
        still finishes that answer in the background; the job ends right away."""
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
                return False, "reachable, but it lists no models at all: configure one in OpenCode (opencode.json)"
            return False, (f"reachable, but {self.model_ref} is not listed. Present: {', '.join(listed[:12])}"
                           + (" ..." if len(listed) > 12 else "") + ". Pick one or set OPENCODE_MODEL=provider/model.")
        except httpx.HTTPError as exc:
            return False, f"not reachable at {self.base_url} ({type(exc).__name__}: {exc}). Is the OpenCode app running?"


def startup_check() -> int:
    """For start-app.bat: 0 when OpenCode runs and its model answers (starting it when needed), 2 when OpenCode
    runs but the model does not answer (the web app can still start), 1 when OpenCode is not reachable."""
    llm = OpenCodeLLM(Settings.from_env())
    ok, message = llm.check()
    print(("  ok: " if ok else "  FAIL: ") + message)
    if not ok:
        return 1
    llm.on_status = lambda status: print("  " + status)
    try:
        print(f"  model: answers in {llm.ping():.1f} s")
    except LLMError as exc:
        print(f"  WARNING: {exc}")
        return 2
    return 0
