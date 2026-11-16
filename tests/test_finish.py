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


def _mono(f):
    return f.model_copy(update={"width": 0.0})


def test_dry_changes_nothing():
    _, x = _voice_with_an_s()
    dry = finish.Finish.from_preset("droog")
    assert dry.dry and not dry.stereo
    assert np.array_equal(finish.take(x, SR, dry, breath=np.ones(4000, dtype=np.float32)), x)
    assert np.array_equal(finish.voice(x, SR, dry), x) and np.array_equal(finish.bus(x, SR, dry), x)


@pytest.mark.parametrize("name", ["studio", "huiskamer", "radio"])
@pytest.mark.parametrize("width", [True, False])
def test_a_treated_take_is_as_loud_as_the_dry_one(name, width):
    _, x = _voice_with_an_s()
    f = finish.Finish.from_preset(name)
    y = finish.take(x, SR, f if width else _mono(f), pan=-0.3)
    assert y.ndim == (2 if width else 1)
    assert abs(measure_lufs(y, SR) - measure_lufs(x, SR)) < 0.3  # the louder of two always sounds better
    assert len(y) > len(x)  # the room rings out after the last word
    assert float(np.max(np.abs(y))) <= 10 ** (-1 / 20) + 1e-3


def test_sharp_s_sounds_go_down_and_vowels_stay():
    t, x = _voice_with_an_s()
    y = finish._deess(x, SR, 0.5)
    s, vowel = (t >= 0.82) & (t < 0.93), (t >= 0.3) & (t < 0.7)
    assert _db(y[s]) - _db(x[s]) < -3.0
    assert abs(_db(y[vowel]) - _db(x[vowel])) < 0.1


def test_hosts_sit_left_and_right_and_a_guest_in_the_middle():
    from pipeline.models import Cast, Guest, Host

    cast = Cast(hosts=[Host(id="tessa", name="Tessa", role="explainer"), Host(id="joris", name="Joris", role="skeptic")],
                guests=[Guest(id="sam", name="Sam")])
    f = finish.Finish.from_preset("studio")
    pans = finish.pans(cast, f)
    assert pans["tessa"] < 0 < pans["joris"] and pans["sam"] == 0 and abs(pans["tessa"]) <= 0.5
    assert finish.pans(cast, _mono(f)) == {"tessa": 0.0, "joris": 0.0, "sam": 0.0}
    wider = finish.pans(cast, f.model_copy(update={"width": 0.9}))
    assert wider["joris"] > pans["joris"]
    left, right = finish.pan_gains(pans["tessa"])
    assert left > right and abs(left**2 + right**2 - 1) < 1e-6  # as loud on the left as it would be in the middle


def test_a_stereo_take_leans_to_its_side_and_the_room_is_wide():
    _, x = _voice_with_an_s()
    f = finish.Finish.from_preset("huiskamer")
    y = finish.take(x, SR, f, pan=-0.4)
    assert _db(y[:, 0]) - _db(y[:, 1]) > 3  # the explainer is heard on the left
    tail = y[len(x) + int(0.02 * SR):]  # after the last word only the room is left
    corr = np.corrcoef(tail[:, 0], tail[:, 1])[0, 1]
    assert abs(corr) < 0.5  # the room answers differently on both sides, so it sounds around you


def test_compression_moves_both_channels_together():
    _, x = _voice_with_an_s()
    stereo = finish.place(x * 2, -0.5)
    y = finish._compress(stereo, SR, 0.8)
    loud = np.abs(stereo[:, 0]) > 0.05
    ratio_before = stereo[loud, 0] / stereo[loud, 1]
    ratio_after = y[loud, 0] / y[loud, 1]
    assert np.allclose(ratio_before, ratio_after, rtol=1e-4)  # the voice stays where it sits
    assert _db(y) < _db(stereo) - 1


def test_a_take_with_breath_starts_with_a_soft_inhale():
    from pipeline.audio import breaths

    _, x = _voice_with_an_s()
    f = _mono(finish.Finish.from_preset("studio"))
    inhale = breaths.synthetic(SR)
    y = finish.take(x, SR, f, breath=inhale)
    plain = finish.take(x, SR, f)
    n = len(inhale)
    assert len(y) == len(plain) + n + int(finish.BREATH_LEAD_S * SR)
    assert -32 < _db(y[:n]) - _db(y[n:n + len(x)]) < -14  # heard, well under the voice


def test_the_limiter_only_turns_peaks_down():
    _, x = _voice_with_an_s()
    loud = x * 3
    y = finish.limit(loud, SR, finish.Finish())
    assert float(np.max(np.abs(y))) <= 10 ** (-1 / 20) + 1e-3
    quiet = x * 0.1
    assert np.array_equal(finish.limit(quiet, SR, finish.Finish()), quiet)  # under the ceiling: untouched, no gain
    both = finish.place(loud, 0.4)
    z = finish.limit(both, SR, finish.Finish())
    assert float(np.max(np.abs(z))) <= 10 ** (-1 / 20) + 1e-3
    hot = np.abs(both[:, 1]) > 0.5
    assert np.allclose(z[hot, 0] / z[hot, 1], both[hot, 0] / both[hot, 1], rtol=1e-4)  # both sides down together


def test_one_room_for_everyone():
    a, b = finish.room_ir(SR, 0.45), finish.room_ir(SR, 0.45)
    assert np.array_equal(a, b) and abs(float(np.sum(a**2)) - 1.0) < 1e-3
    assert len(finish.room_ir(SR, 1.0)) > len(finish.room_ir(SR, 0.0))  # a bigger room rings longer


def test_settings_are_kept_in_the_cast_folder(tmp_path):
    assert finish.load(tmp_path).preset == finish.DEFAULT_PRESET  # no file: the default sound
    mine = finish.Finish(preset="eigen", room=0.4, size=0.3, warmth=0.2, deess=0.6, compression=0.1, breath=0.2,
                         width=0.6)
    path = finish.save(tmp_path, mine)
    assert path.read_text(encoding="utf-8").startswith("# De klank van de show")
    assert finish.load(tmp_path) == mine
    with pytest.raises(ValueError):
        finish.Finish(room=1.5)
    with pytest.raises(ValueError):
        finish.Finish.from_preset("kathedraal")


def test_an_older_settings_file_keeps_its_sound(tmp_path):
    """Written before breath and width existed: the new knobs come from its own preset, so droog stays droog."""
    path = tmp_path / finish.FILE
    path.write_text("preset: droog\nroom: 0\nsize: 0\nwarmth: 0\ndeess: 0\ncompression: 0\n", encoding="utf-8")
    assert finish.load(tmp_path).dry
    path.write_text("preset: radio\nroom: 0.1\nsize: 0.15\nwarmth: 0.35\ndeess: 0.6\ncompression: 0.7\n",
                    encoding="utf-8")
    old = finish.load(tmp_path)
    assert old.breath == finish.PRESETS["radio"]["breath"] and old.width == finish.PRESETS["radio"]["width"]


def test_the_mix_is_unchanged_when_dry(monkeypatch):
    from pipeline.audio import mixer

    seen = []
    monkeypatch.setattr(finish, "bus", lambda x, sr, f: seen.append("bus") or x)
    dry = finish.Finish.from_preset("droog")
    timeline = type("T", (), {"placements": [], "duration": 1.0, "qa": []})()
    a, _ = mixer.mix(timeline, sr=SR, finish=dry, room_tone_db=None)
    b, _ = mixer.mix(timeline, sr=SR, room_tone_db=None)
    assert np.array_equal(a.samples, b.samples) and not seen and a.samples.ndim == 1


def _conversation(lines):
    """A timeline of (speaker, start, seconds) lines: vowels at -18 dBFS-ish, nothing else."""
    from pipeline.audio.synth import AudioClip
    from pipeline.audio.timeline import Placement, Timeline
    from pipeline.models import Line

    placements = []
    for i, (speaker, start, seconds) in enumerate(lines):
        t = np.arange(int(seconds * SR)) / SR
        f0 = 140 if speaker == "tessa" else 110
        voice = (sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20)) * 0.1).astype(np.float32)
        turn = type("Turn", (), {"speaker": speaker, "turn_id": f"t{i:03d}",
                                 "first": Line(id=f"l{i}", speaker=speaker, text="Een zin."), "line_ids": [f"l{i}"]})()
        placements.append(Placement(turn=turn, clip=AudioClip(voice, SR), start=start, words=[]))
    return Timeline(placements=placements, sr=SR)


def test_the_mix_puts_each_host_on_their_side():
    from pipeline.audio import mixer

    timeline = _conversation([("tessa", 0.0, 2.0), ("joris", 2.4, 2.0)])
    f = finish.Finish.from_preset("studio")
    mixed, offset = mixer.mix(timeline, sr=SR, finish=f, pans={"tessa": -0.3, "joris": 0.3}, room_tone_db=-60)
    y = mixed.samples
    assert y.shape[1] == 2
    a = int((offset + 0.5) * SR), int((offset + 1.5) * SR)
    b = int((offset + 2.9) * SR), int((offset + 3.9) * SR)
    assert _db(y[a[0]:a[1], 0]) > _db(y[a[0]:a[1], 1]) + 2  # Tessa on the left
    assert _db(y[b[0]:b[1], 1]) > _db(y[b[0]:b[1], 0]) + 2  # Joris on the right
    assert abs(measure_lufs(y, SR) - (-16.0)) < 0.5
    mono, _ = mixer.mix(timeline, sr=SR, finish=_mono(f), room_tone_db=-60)
    assert mono.samples.ndim == 1


def test_breaths_come_before_longer_lines_with_room_for_them():
    from pipeline.audio import breaths, mixer

    lines = [("tessa", 0.0, 5.0)] + [("joris" if i % 2 else "tessa", 5.6 + i * 6.6, 6.0) for i in range(12)]
    lines += [("joris", 85.0, 1.0)]  # short: never a breath
    timeline = _conversation(lines)
    f = _mono(finish.Finish.from_preset("huiskamer"))
    bank = {s: breaths.Breaths([breaths.synthetic(SR)], SR, "nagemaakt") for s in ("tessa", "joris")}
    found = list(mixer._breaths(timeline, bank, f, SR, -18.0, seed=7))
    assert 4 <= len(found) <= 12  # most long lines, not all
    starts = {p.start: p for p in timeline.placements}
    for at, inhale, speaker in found:
        line = next(p for p in timeline.placements if abs(p.start - (at + len(inhale) / SR + finish.BREATH_LEAD_S)) < 1e-3)
        assert line.turn.speaker == speaker and line.end - line.start >= mixer.BREATH_MIN_LINE_S
        assert at >= max(p.end for p in starts.values() if p.start < line.start) + mixer.BREATH_AFTER_PREVIOUS_S - 1e-3
        assert 0.12 <= len(inhale) / SR <= 0.7
    again = list(mixer._breaths(timeline, bank, f, SR, -18.0, seed=7))  # same script, same breaths
    assert [(a, s) for a, _, s in again] == [(a, s) for a, _, s in found]
    tight = _conversation([("tessa", 0.0, 3.0), ("joris", 3.1, 6.0)])  # 0.1 s gap: no room to breathe
    assert not list(mixer._breaths(tight, bank, f, SR, -18.0, seed=7))


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
                                                    "deess": 0.5, "compression": 0.3, "breath": 0, "width": 0})
        assert treated.status_code == 200 and treated.headers["content-type"] == "audio/wav"
        d, sr = sf.read(io.BytesIO(dry))
        t, _ = sf.read(io.BytesIO(treated.content))
        assert t.ndim == 1 and len(t) > len(d) and not np.allclose(t[: len(d)], d)
        wide = client.get(take["audio"], params={"klank": 1, "breath": 0.6, "width": 0.5})
        w, _ = sf.read(io.BytesIO(wide.content))
        assert w.ndim == 2 and w.shape[1] == 2  # where Tessa sits in an episode
        assert len(w) > len(t)  # the breath comes first
        assert client.get(take["audio"]).content == dry  # the file on disk stays dry
        assert client.get(take["audio"], params={"klank": 1, "room": 3}).status_code == 400

        saved = client.put("/api/klank", json={"preset": "radio", **finish.PRESETS["radio"]}).json()
        assert saved["settings"]["preset"] == "radio"
        assert finish.load(settings.cast_dir).preset == "radio"
        assert client.put("/api/klank", json={"preset": "studio", "room": -1}).status_code == 400
