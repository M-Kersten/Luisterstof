"""The GPU check before a render reads nvidia-smi and stays quiet when it can't."""

import subprocess

from pipeline.audio import gpu


def _smi(monkeypatch, stdout=None, fail=False):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "nvidia-smi")

    def run(*args, **kwargs):
        if fail:
            raise subprocess.CalledProcessError(9, "nvidia-smi")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", run)


def test_a_busy_card_is_reported_with_its_numbers(monkeypatch):
    _smi(monkeypatch, "19456, 24576\n")
    assert gpu.gpu_memory() == (19456, 24576)
    warning = gpu.busy_warning(3)
    assert warning.startswith("19.0 of 24.0 GB of GPU memory is in use") and "3 Chatterbox worker(s)" in warning


def test_a_free_card_and_a_missing_or_failing_nvidia_smi_say_nothing(monkeypatch):
    _smi(monkeypatch, "900, 24576\n")
    assert gpu.busy_warning(3) is None
    _smi(monkeypatch, fail=True)
    assert gpu.gpu_memory() is None and gpu.busy_warning(3) is None
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    assert gpu.gpu_memory() is None
