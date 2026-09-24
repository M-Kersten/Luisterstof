"""Offline mode: book content never leaves the local network.

Enforced rather than assumed. With STUDIEPODCAST_OFFLINE=1:

* the Claude API and ElevenLabs refuse to start;
* the local LLM URL must resolve to a loopback or private address;
* Hugging Face libraries (Chatterbox, Whisper, WhisperX weights) run from
  their local cache only and send no telemetry. The variables are set before
  any of those libraries is imported, because they read them at import time.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

HF_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class OfflineViolation(RuntimeError):
    pass


def apply_env(settings) -> None:
    """Switch Hugging Face to cache-only. Call before chatterbox/transformers/whisper are imported."""
    if settings.offline:
        os.environ.update(HF_OFFLINE_ENV)


def block_if_offline(settings, what: str) -> None:
    if settings.offline:
        raise OfflineViolation(f"{what} is blocked: STUDIEPODCAST_OFFLINE=1 keeps all data on the local network")


def _is_local(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone id
    return ip.is_loopback or ip.is_private or ip.is_link_local


def resolve_local(host: str) -> tuple[bool, list[str]]:
    """(every address is loopback/private, the addresses). A host that doesn't resolve is not local."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False, []
    addresses = sorted({info[4][0] for info in infos})
    return bool(addresses) and all(_is_local(a) for a in addresses), addresses


def check_local_url(url: str) -> list[str]:
    """Raise unless the URL points at this machine or the private network. Returns its addresses."""
    host = urlparse(url).hostname
    if not host:
        raise OfflineViolation(f"LOCAL_LLM_URL {url!r} has no host")
    local, addresses = resolve_local(host)
    if not local:
        if not addresses:
            where = f"{host} does not resolve"
        elif addresses == [host]:
            where = host
        else:
            where = f"{host} is {', '.join(addresses)}"
        raise OfflineViolation(
            f"LOCAL_LLM_URL {url!r} is not on the local network ({where}); in offline mode the model "
            "server must be localhost or a private address (10.x, 172.16-31.x, 192.168.x, fc00::/7)")
    return addresses
