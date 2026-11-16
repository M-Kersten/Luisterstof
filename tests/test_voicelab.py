"""The voice lab: a take per sentence with the knobs set by hand, and the knobs saved into the cast files."""

import time

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from pipeline.script.cast import load_cast
from pipeline.voicelab import PARAMS, VoiceLab, clean_params, find_speaker, save_params

KNOBS = {"exaggeration": 0.7, "cfg_weight": 0.35, "temperature": 0.75, "speech_rate": 0.92}


def test_saving_a_voice_keeps_the_comments_and_the_other_hosts(cast_dir):
    before = (cast_dir / "hosts.yaml").read_text(encoding="utf-8")
    save_params(cast_dir, "joris", KNOBS)
    after = (cast_dir / "hosts.yaml").read_text(encoding="utf-8")
    cast = load_cast(cast_dir)
    joris, tessa = find_speaker(cast, "joris")[1], find_speaker(cast, "tessa")[1]
    assert (joris.exaggeration, joris.cfg_weight, joris.temperature, joris.speech_rate) == (0.7, 0.35, 0.75, 0.92)
    assert tessa.exaggeration == 0.55 and tessa.cfg_weight == 0.5  # untouched
    comments = [ln for ln in before.splitlines() if ln.lstrip().startswith("#")]
    assert comments == [ln for ln in after.splitlines() if ln.lstrip().startswith("#")]
    assert "explain: {rate: 0.92, exaggeration: -0.08}" in after  # the performance table is not a knob
    # saving again changes the values in place, it does not add lines
    save_params(cast_dir, "joris", {**KNOBS, "exaggeration": 0.4})
    again = (cast_dir / "hosts.yaml").read_text(encoding="utf-8")
    assert len(again.splitlines()) == len(after.splitlines())
    assert find_speaker(load_cast(cast_dir), "joris")[1].exaggeration == 0.4


def test_a_guest_is_saved_in_its_own_file(cast_dir):
    save_params(cast_dir, "hanna_vos", KNOBS)
    assert find_speaker(load_cast(cast_dir), "hanna_vos")[1].cfg_weight == 0.35
    assert "hanna_vos" not in (cast_dir / "hosts.yaml").read_text(encoding="utf-8")


def test_knobs_outside_their_range_are_refused():
    with pytest.raises(ValueError):
        clean_params({**KNOBS, "speech_rate": 2.0})
    with pytest.raises(ValueError):
        clean_params({k: v for k, v in KNOBS.items() if k != "temperature"})
    assert all(spec["min"] < spec["default"] < spec["max"] for spec in PARAMS.values())


def test_a_take_is_kept_with_its_settings_and_its_check(settings, cast):
    lab = VoiceLab(settings, fake_audio=True)
    sp = find_speaker(cast, "tessa")[1]
    take = lab.generate(sp, "Stel je voor dat je op een glazen vloer staat, honderd meter boven de stad.", KNOBS, seed=7)
    assert take["params"] == KNOBS and take["seed"] == 7 and take["wer"] == 0.0
    assert [t["id"] for t in lab.takes("tessa")] == [take["id"]]
    assert lab.take_path("tessa", take["id"]).is_file()
    with pytest.raises(ValueError):
        lab.generate(sp, "Ja.", KNOBS)  # too short for Chatterbox
    lab.delete("tessa", take["id"])
    assert lab.takes("tessa") == []
    with pytest.raises(KeyError):
        lab.take_path("tessa", "../../hosts")


def test_the_voices_page_end_to_end(settings):
    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        voices = client.get("/api/voices").json()
        assert {v["id"] for v in voices["voices"]} >= {"tessa", "joris", "hanna_vos"}
        bad = client.post("/api/voices/tessa/try", json={"text": "Een zin van een paar woorden.", **KNOBS, "exaggeration": 9})
        assert bad.status_code == 400
        job = client.post("/api/voices/tessa/try", json={"text": "Een zin van een paar woorden lang.", **KNOBS, "seed": 3}).json()
        deadline = time.time() + 60
        while client.get(f"/api/jobs/{job['id']}").json()["status"] not in ("done", "failed"):
            assert time.time() < deadline
            time.sleep(0.1)
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"
        takes = client.get("/api/voices/tessa/takes").json()
        assert len(takes) == 1 and takes[0]["seed"] == 3
        assert client.get(takes[0]["audio"]).status_code == 200
        saved = client.put("/api/voices/tessa/params", json=KNOBS)
        assert saved.status_code == 200, saved.text
        tessa = next(v for v in client.get("/api/voices").json()["voices"] if v["id"] == "tessa")
        assert tessa["params"] == KNOBS  # the cast was read again
        assert client.delete(f"/api/voices/tessa/takes/{takes[0]['id']}").status_code == 200
        assert client.get("/api/voices/nobody/takes").status_code == 404


# ------------------------------------------------------------------ the check that listens back
class _Hangs:
    """A Whisper that never answers: a first download, a GPU spilling into shared memory, a DLL mix-up."""

    def __init__(self):
        self._model = object()

    def transcribe(self, clip):
        time.sleep(30)
        return "te laat"


def _clip():
    import numpy as np

    from pipeline.audio.synth import AudioClip

    return AudioClip(np.zeros(24000, dtype=np.float32), 24000)


def test_a_check_that_hangs_is_given_up_and_not_tried_again(settings, monkeypatch):
    import pipeline.voicelab as vl

    monkeypatch.setattr(vl, "CHECK_RUN_S", 0.5)
    lab = VoiceLab(settings, fake_audio=False)
    said = []
    say = lambda stage, status, data: said.append((status, data))  # noqa: E731
    monkeypatch.setattr(lab, "_checker", lambda say_: (_Hangs(), "cuda"))
    started = time.monotonic()
    out = lab._check(_clip(), "een zin van vijf woorden", say, None)
    assert time.monotonic() - started < 5
    assert out["check"] == "timeout" and "vast" in out["check_note"]
    assert said[-1][0] == "check_skipped"
    # the next take does not wait for it again, until the model is released
    monkeypatch.undo()
    transcriber, why = lab._checker(say)
    assert transcriber is None and "vast" in why
    lab.release()
    assert lab._check_broken is None


def test_stop_ends_the_check_at_once(settings, monkeypatch):
    from pipeline.cancel import Cancelled

    lab = VoiceLab(settings, fake_audio=False)
    monkeypatch.setattr(lab, "_checker", lambda say_: (_Hangs(), "cuda"))
    started = time.monotonic()
    with pytest.raises(Cancelled):
        lab._check(_clip(), "een zin van vijf woorden", lambda *a: None, lambda: True)
    assert time.monotonic() - started < 3


def test_no_download_inside_a_take_and_cpu_when_the_gpu_is_full(settings, monkeypatch):
    import pipeline.voicelab as vl

    lab = VoiceLab(settings.with_(device="cuda"), fake_audio=False)
    said = []
    say = lambda stage, status, data: said.append((status, data))  # noqa: E731
    monkeypatch.setattr(vl, "whisper_status", lambda name: "missing")
    transcriber, why = lab._checker(say)
    assert transcriber is None and "fetch-whisper" in why

    monkeypatch.setattr(vl, "whisper_status", lambda name: "ok")
    monkeypatch.setattr(vl, "free_gpu_gb", lambda: 1.1)
    transcriber, where = lab._checker(say)
    assert where == "cpu" and transcriber.settings.device == "cpu"
    assert said[-1] == ("check_device", {"device": "cpu", "free_gb": 1.1})


def test_a_take_is_saved_before_it_is_checked(settings, cast, monkeypatch):
    lab = VoiceLab(settings, fake_audio=True)
    seen = {}

    def check(clip, spoken, say, should_stop):
        seen["files"] = sorted(p.suffix for p in (settings.data_dir / "voicelab" / "joris").iterdir())
        return {"check": "timeout", "check_note": "Whisper liep vast."}

    monkeypatch.setattr(lab, "_check", check)
    sp = find_speaker(cast, "joris")[1]
    take = lab.generate(sp, "Dit is een zin om te proberen.", clean_params(KNOBS), 7)
    assert seen["files"] == [".json", ".wav"]  # on disk while the check ran
    assert take["check"] == "timeout" and take["wer"] is None
    kept = lab.takes("joris")[0]
    assert kept["check_note"] == "Whisper liep vast."


# ------------------------------------------------------------------ Whisper on a Windows GPU without its cuDNN
def _fake_windows(monkeypatch, ct2_version, present):
    """A Windows machine with a given CTranslate2 and a given set of loadable DLLs."""
    import ctypes
    import sys
    import types

    import pipeline.audio.asr as asr

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(__version__=ct2_version))
    monkeypatch.setattr(asr, "prepare_cuda_dlls", lambda: None)

    def windll(name):
        if name not in present:
            raise OSError(f"{name} not found")
        return object()

    monkeypatch.setattr(ctypes, "WinDLL", windll, raising=False)


def test_ctranslate2_44_without_cudnn_8_is_caught_before_it_hangs(monkeypatch):
    from pipeline.audio.asr import whisper_gpu_check

    _fake_windows(monkeypatch, "4.4.0", present={"cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"})  # torch 2.6's cuDNN 9
    check = whisper_gpu_check()
    assert not check.ok and check.cudnn == 8 and check.missing == "cudnn_ops_infer64_8.dll"
    assert "cuDNN 8" in check.reason_nl()

    _fake_windows(monkeypatch, "4.9.0", present={"cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"})
    assert whisper_gpu_check().ok and whisper_gpu_check().cudnn == 9


def test_whisper_moves_to_the_processor_when_its_cudnn_is_missing(settings, monkeypatch):
    import sys
    import types

    import pipeline.audio.asr as asr
    import pipeline.voicelab as vl

    _fake_windows(monkeypatch, "4.4.0", present=set())
    monkeypatch.setattr(vl, "whisper_status", lambda name: "ok")
    monkeypatch.setattr(vl, "free_gpu_gb", lambda: 20.0)
    lab = VoiceLab(settings.with_(device="cuda"), fake_audio=False)
    said = []
    transcriber, where = lab._checker(lambda stage, status, data: said.append((status, data)))
    assert where == "cpu" and said[0][0] == "check_device" and "cuDNN 8" in said[0][1]["reason"]

    # the render's own transcriber makes the same choice
    made = {}
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(
        WhisperModel=lambda name, device, compute_type: made.update(device=device) or object()))
    t = asr.FasterWhisperTranscriber(settings.with_(device="cuda"))
    t._load()
    assert made["device"] == "cpu" and t.device == "cpu"


# ------------------------------------------------------------------ the voices on the processor
def _fake_torch(monkeypatch, *, cuda_build, sees_gpu, version):
    import sys
    import types

    torch = types.SimpleNamespace(
        __version__=version,
        version=types.SimpleNamespace(cuda="12.4" if cuda_build else None),
        cuda=types.SimpleNamespace(is_available=lambda: sees_gpu),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_why_the_voices_run_on_the_processor(monkeypatch):
    from pipeline.audio.synth import torch_gpu_status

    _fake_torch(monkeypatch, cuda_build=False, sees_gpu=False, version="2.6.0")  # PyPI's Windows wheel
    ok, why = torch_gpu_status()
    assert not ok and "versie zonder GPU-ondersteuning" in why and "2.6.0" in why

    _fake_torch(monkeypatch, cuda_build=True, sees_gpu=False, version="2.6.0+cu124")
    ok, why = torch_gpu_status()
    assert not ok and "CUDA 12.4" in why and "stuurprogramma" in why

    _fake_torch(monkeypatch, cuda_build=True, sees_gpu=True, version="2.6.0+cu124")
    assert torch_gpu_status() == (True, "")


def test_a_take_on_the_processor_says_why_and_where_to_look():
    from pipeline.narrate import say

    cpu = say("voice", "loaded", {"seconds": 70, "device": "cpu",
                                  "why": "de geïnstalleerde torch (2.6.0) is de versie zonder GPU-ondersteuning"})
    assert cpu["kind"] == "problem" and "op de processor" in cpu["text"] and "zonder GPU" in cpu["text"]
    assert "studiepodcast doctor" in cpu["text"]
    gpu = say("voice", "loaded", {"seconds": 40, "device": "cuda"})
    assert gpu == {"text": "Stemmodel geladen op de GPU in 40 s.", "kind": "result"}


def test_doctor_names_the_cpu_only_torch_and_the_fix(monkeypatch):
    from typer.testing import CliRunner

    from pipeline.cli import app

    _fake_torch(monkeypatch, cuda_build=False, sees_gpu=False, version="2.6.0")
    out = CliRunner().invoke(app, ["doctor"]).output
    assert "CPU-only build of torch" in out and "torch==2.6.0+cu124" in out and "download.pytorch.org/whl/cu124" in out
