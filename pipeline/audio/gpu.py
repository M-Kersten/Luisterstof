"""How full the GPU is before a render starts.

The writing model belongs to OpenCode, so the pipeline cannot unload it. When
that model runs on the same laptop, it can still hold most of the card when the
Chatterbox workers start. This reads the card's memory from ``nvidia-smi``
(which reports totals on Windows too, where per-process numbers are hidden), so
the render can say so before it crawls or runs out of memory.
"""

from __future__ import annotations

import shutil
import subprocess

BUSY_MIB = 4096  # more than this in use before Chatterbox loads: something else holds the card


def gpu_memory() -> tuple[int, int] | None:
    """(used MiB, total MiB) of the first NVIDIA GPU; None without nvidia-smi or when it fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        used, total = (int(float(x)) for x in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return used, total


def busy_warning(workers: int) -> str | None:
    """A sentence for the log when the card is already well filled; None when it looks free or is unknown."""
    memory = gpu_memory()
    if memory is None or memory[0] <= BUSY_MIB:
        return None
    used, total = memory
    return (f"{used / 1024:.1f} of {total / 1024:.1f} GB of GPU memory is in use before Chatterbox starts. If OpenCode's "
            f"writing model runs on this machine, unload it first (close the model or its server): otherwise the "
            f"{workers} Chatterbox worker(s) share the card with it and the render slows down or runs out of memory.")
