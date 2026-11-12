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
