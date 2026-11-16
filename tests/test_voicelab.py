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
CU124 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU128 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def _fake_torch(monkeypatch, *, cuda_build, sees_gpu, version, card=("NVIDIA GeForce RTX 4060 Laptop GPU", (8, 9)),
                archs=CU124):
    import sys
    import types

    torch = types.SimpleNamespace(
        __version__=version,
        load=lambda *a, **k: None,  # resolve_torch_device maps checkpoints onto the processor when it falls back
        device=lambda name: name,
        version=types.SimpleNamespace(cuda="12.4" if cuda_build else None),
        cuda=types.SimpleNamespace(is_available=lambda: sees_gpu, get_device_name=lambda i=0: card[0],
                                   get_device_capability=lambda i=0: card[1], get_arch_list=lambda: archs),
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
    assert "CPU-only build of torch" in out and "torch==2.7.1+cu128" in out and "download.pytorch.org/whl/cu128" in out


def test_an_rtx_50_card_on_torch_26_is_named_and_gets_the_cuda_128_build(monkeypatch):
    from typer.testing import CliRunner

    from pipeline.audio.synth import (
        TORCH_CUDA_FIX,
        TORCH_CUDA_FIX_OLDER,
        resolve_torch_device,
        torch_gpu_status,
    )
    from pipeline.cli import app

    blackwell = ("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0))
    _fake_torch(monkeypatch, cuda_build=True, sees_gpu=True, version="2.6.0+cu124", card=blackwell, archs=CU124)
    ok, why = torch_gpu_status()
    assert not ok and "RTX 5070" in why and "sm_120" in why and "nieuwer" in why and "CUDA 12.8" in why
    assert resolve_torch_device("cuda") == "cpu"  # no "no kernel image" crash: the processor, with a reason
    out = CliRunner().invoke(app, ["doctor"]).output
    assert "sm_120" in out and "no code for this GPU" in out and TORCH_CUDA_FIX in out

    _fake_torch(monkeypatch, cuda_build=True, sees_gpu=True, version="2.7.1+cu128", card=blackwell, archs=CU128)
    assert torch_gpu_status() == (True, "")

    pascal = ("NVIDIA GeForce GTX 1060", (6, 1))  # too old for the CUDA 12.8 build
    _fake_torch(monkeypatch, cuda_build=True, sees_gpu=True, version="2.7.1+cu128", card=pascal, archs=CU128)
    ok, why = torch_gpu_status()
    assert not ok and "ouder" in why
    import sys

    from pipeline.audio.synth import gpu_fit
    assert gpu_fit(sys.modules["torch"])[1] == TORCH_CUDA_FIX_OLDER


def test_a_gpu_error_while_loading_does_not_send_you_to_download_the_weights(monkeypatch):
    import sys
    import types

    import pipeline.audio.synth as synth

    class Model:
        @classmethod
        def from_local(cls, ckpt_dir, device, t3_model=None):
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", types.SimpleNamespace(ChatterboxMultilingualTTS=Model))
    monkeypatch.setattr(synth, "chatterbox_checkpoint_dir", lambda: "C:/cache/chatterbox")
    with pytest.raises(RuntimeError) as err:
        synth.load_multilingual_model("cuda")
    assert "could not start on the GPU" in str(err.value) and "doctor" in str(err.value)
    assert "download" not in str(err.value).lower()
