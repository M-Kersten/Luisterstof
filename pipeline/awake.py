"""Keep the machine awake while a job runs.

A Windows laptop that goes to sleep at one in the morning stops the night's work, the OpenCode model
included. While a job (or a long CLI command) runs, this tells Windows the system is in use, so it does
not sleep; the screen may still switch off. Closing the lid follows the power settings ("when I close the
lid: do nothing" keeps it going). Elsewhere this does nothing.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake() -> Iterator[bool]:
    """True inside the block when the system was asked to stay awake. Holds for the calling thread, which is
    the one doing the work."""
    kernel32 = None
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                kernel32 = None
        except Exception as exc:  # noqa: BLE001 - never stop a job over this
            log.info("could not keep the system awake: %s", exc)
            kernel32 = None
    try:
        yield kernel32 is not None
    finally:
        if kernel32 is not None:
            try:
                kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # back to normal: sleep is allowed again
            except Exception:  # noqa: BLE001
                pass
