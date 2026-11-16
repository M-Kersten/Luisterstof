"""The sound of the show: what it does to a voice, what it leaves alone, and how the voice lab lets you hear it."""

import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from scipy.signal import butter, sosfiltfilt

from app.api import create_app
from pipeline.audio import finish
from pipeline.audio.mixer import measure_lufs

SR = 24000


def _voice_with_an_s():
    """Three vowels, each followed by an s: harmonics up to 4 kHz, then noise above 4 kHz."""
    rng = np.random.default_rng(1)
    t = np.arange(int(SR * 3)) / SR
    x = np.zeros_like(t, dtype=np.float32)
    hiss = sosfiltfilt(butter(4, 4000, btype="high", fs=SR, output="sos"), rng.standard_normal(len(t))).astype(np.float32)
    for start in (0.2, 1.2, 2.1):
        vowel = (t >= start) & (t < start + 0.6)
        x[vowel] += sum(np.sin(2 * np.pi * 140 * k * t[vowel]) / k for k in range(1, 30)) * 0.2
        s = (t >= start + 0.6) & (t < start + 0.75)
        x[s] += hiss[s] * 0.25
    return t, x


def _db(a):
    return 20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-12)


def test_dry_changes_nothing():
    _, x = _voice_with_an_s()
    dry = finish.Finish.from_preset("droog")
    assert dry.dry
    assert np.array_equal(finish.take(x, SR, dry), x)
    assert np.array_equal(finish.voice(x, SR, dry), x) and np.array_equal(finish.bus(x, SR, dry), x)


@pytest.mark.parametrize("name", ["studio", "huiskamer", "radio"])
def test_a_treated_take_is_as_loud_as_the_dry_one(name):
    _, x = _voice_with_an_s()
    y = finish.take(x, SR, finish.Finish.from_preset(name))
    assert abs(measure_lufs(y, SR) - measure_lufs(x, SR)) < 0.3  # the louder of two always sounds better
    assert len(y) > len(x)  # the room rings out after the last word
    assert float(np.max(np.abs(y))) <= 10 ** (-1 / 20) + 1e-3


def test_sharp_s_sounds_go_down_and_vowels_stay():
    t, x = _voice_with_an_s()
    y = finish._deess(x, SR, 0.5)
    s, vowel = (t >= 0.82) & (t < 0.93), (t >= 0.3) & (t < 0.7)
    assert _db(y[s]) - _db(x[s]) < -3.0
    assert abs(_db(y[vowel]) - _db(x[vowel])) < 0.1


def test_the_limiter_only_turns_peaks_down():
    _, x = _voice_with_an_s()
    loud = x * 3
    y = finish.limit(loud, SR, finish.Finish())
    assert float(np.max(np.abs(y))) <= 10 ** (-1 / 20) + 1e-3
    quiet = x * 0.1
    assert np.array_equal(finish.limit(quiet, SR, finish.Finish()), quiet)  # under the ceiling: untouched, no gain


def test_one_room_for_everyone():
    a, b = finish.room_ir(SR, 0.45), finish.room_ir(SR, 0.45)
    assert np.array_equal(a, b) and abs(float(np.sum(a**2)) - 1.0) < 1e-3
    assert len(finish.room_ir(SR, 1.0)) > len(finish.room_ir(SR, 0.0))  # a bigger room rings longer


def test_settings_are_kept_in_the_cast_folder(tmp_path):
    assert finish.load(tmp_path).preset == finish.DEFAULT_PRESET  # no file: the default sound
    mine = finish.Finish(preset="eigen", room=0.4, size=0.3, warmth=0.2, deess=0.6, compression=0.1)
    path = finish.save(tmp_path, mine)
    assert path.read_text(encoding="utf-8").startswith("# De klank van de show")
    assert finish.load(tmp_path) == mine
    with pytest.raises(ValueError):
        finish.Finish(room=1.5)
    with pytest.raises(ValueError):
        finish.Finish.from_preset("kathedraal")


def test_the_mix_is_unchanged_when_dry(monkeypatch):
    from pipeline.audio import mixer

    seen = []
    monkeypatch.setattr(finish, "bus", lambda x, sr, f: seen.append("bus") or x)
    dry = finish.Finish.from_preset("droog")
    timeline = type("T", (), {"placements": [], "duration": 1.0, "qa": []})()
    a, _ = mixer.mix(timeline, sr=SR, finish=dry, room_tone_db=None)
    b, _ = mixer.mix(timeline, sr=SR, room_tone_db=None)
    assert np.array_equal(a.samples, b.samples) and not seen


def _wait(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_the_voice_lab_plays_a_take_dry_or_with_the_show_sound(settings):
    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        k = client.get("/api/klank").json()
        assert set(k["presets"]) == {"droog", "studio", "huiskamer", "radio"}
        job = client.post("/api/voices/tessa/try", json={"text": "Stel je voor dat je op een glazen vloer staat.",
                                                          "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
                                                          "speech_rate": 1.0, "seed": 3}).json()
        assert _wait(client, job["id"])["status"] == "done"
        take = client.get("/api/voices/tessa/takes").json()[0]
        dry = client.get(take["audio"]).content
        treated = client.get(take["audio"], params={"klank": 1, "room": 0.4, "size": 0.3, "warmth": 0.3,
                                                    "deess": 0.5, "compression": 0.3})
        assert treated.status_code == 200 and treated.headers["content-type"] == "audio/wav"
        d, sr = sf.read(io.BytesIO(dry))
        t, _ = sf.read(io.BytesIO(treated.content))
        assert len(t) > len(d) and not np.allclose(t[: len(d)], d)
        assert client.get(take["audio"]).content == dry  # the file on disk stays dry
        assert client.get(take["audio"], params={"klank": 1, "room": 3}).status_code == 400

        saved = client.put("/api/klank", json={"preset": "radio", **finish.PRESETS["radio"]}).json()
        assert saved["settings"]["preset"] == "radio"
        assert finish.load(settings.cast_dir).preset == "radio"
        assert client.put("/api/klank", json={"preset": "studio", "room": -1}).status_code == 400
