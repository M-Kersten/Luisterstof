"""Characters edited from the web app: only the changed fields are written, comments stay, a broken file goes back."""

import numpy as np
import pytest
import soundfile as sf

from pipeline.characters import create_guest, describe, install_clip, update
from pipeline.script.cast import load_cast


def _comments(text):
    return [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("#")]


def test_editing_a_host_changes_only_those_fields(cast_dir):
    path = cast_dir / "hosts.yaml"
    before = path.read_text(encoding="utf-8")
    cast = load_cast(cast_dir)
    written = update(cast_dir, cast, "tessa", {
        "background": "Studeerde sterrenkunde in Leiden.\nGeeft nu les op een middelbare school.",
        "stories": "De nacht dat de telescoop bevroor.\n\n  Haar eerste klas, die alles wist.  ",
        "wants": "Dat jij het snapt.",
        "role": "skeptic",  # not editable
    })
    assert set(written) == {"background", "stories", "wants"}
    after = path.read_text(encoding="utf-8")
    assert _comments(after) == _comments(before)
    tessa = describe(load_cast(cast_dir), "tessa")["fields"]
    assert tessa["background"] == "Studeerde sterrenkunde in Leiden.\nGeeft nu les op een middelbare school."
    assert tessa["stories"] == ["De nacht dat de telescoop bevroor.", "Haar eerste klas, die alles wist."]
    assert tessa["voice"] and tessa["tic_markers"] == ["oké dus"]  # untouched
    joris_before = before[before.index("- id: joris"):]
    assert after[after.index("- id: joris"):] == joris_before  # the other host byte for byte
    assert "explain: {rate: 0.95, exaggeration: -0.03}" in after
    assert load_cast(cast_dir).hosts[0].role == "explainer"


def test_a_new_guest_lands_in_guests_yaml(cast_dir):
    cast = load_cast(cast_dir)
    new_id = create_guest(cast_dir, cast, {"name": "Dr. Eva Ånström", "domains": "psychologie\nneurowetenschap",
                                           "persona": "Klinisch psycholoog, nuchter en precies."})
    assert new_id == "dr_eva_anstrom"
    guest = describe(load_cast(cast_dir), new_id)
    assert guest["kind"] == "guest" and guest["fields"]["domains"] == ["psychologie", "neurowetenschap"]
    assert guest["voice_ref"] == "refs/dr_eva_anstrom.wav"
    again = create_guest(cast_dir, load_cast(cast_dir), {"name": "Dr Eva Anstrom"})
    assert again == "dr_eva_anstrom_2"
    update(cast_dir, load_cast(cast_dir), new_id, {"persona": "Psycholoog die alles met een proef laat zien."})
    assert describe(load_cast(cast_dir), new_id)["fields"]["persona"].startswith("Psycholoog")
    with pytest.raises(ValueError):
        create_guest(cast_dir, load_cast(cast_dir), {"name": "   "})


def test_a_clip_is_installed_and_the_old_one_kept(cast_dir, tmp_path):
    cast = load_cast(cast_dir)
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(2400, dtype=np.float32), 24000)
    target = install_clip(cast_dir, cast, "joris", clip)
    assert target == cast_dir / "refs" / "joris.wav" and target.is_file()
    install_clip(cast_dir, cast, "joris", clip)
    assert len(list((cast_dir / "refs" / "backup").glob("joris-*.wav"))) == 1


def test_a_change_that_breaks_the_file_is_put_back(cast_dir):
    path = cast_dir / "hosts.yaml"
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        update(cast_dir, load_cast(cast_dir), "tessa", {"name": ""})
    assert path.read_text(encoding="utf-8") == before


def _lively(seconds, sr=24000, seed=5):
    rng = np.random.default_rng(seed)
    out = [np.zeros(int(1.0 * sr))]
    t = 0.0
    while t < seconds:
        n = int(0.35 * sr)
        f0 = np.linspace(rng.uniform(95, 140), rng.uniform(180, 250), n)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        out.append(0.2 * sum(np.sin(k * phase) / k for k in range(1, 6)) * np.sin(np.linspace(0, np.pi, n)) ** 0.5)
        out.append(rng.normal(0, 0.0005, int(0.08 * sr)))
        t += 0.43
    return np.concatenate(out)


def test_the_setup_page_end_to_end(settings, tmp_path):
    from fastapi.testclient import TestClient

    from app.api import create_app

    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        made = client.post("/api/characters", json={"name": "Lotte Kramer", "domains": "neurowetenschap",
                                                    "persona": "Hersenonderzoeker."})
        assert made.status_code == 200, made.text
        guest = made.json()
        assert guest["kind"] == "guest" and guest["id"] == "lotte_kramer" and not guest["has_ref"]
        assert client.post("/api/characters", json={"name": ""}).status_code == 400

        edited = client.put("/api/characters/lotte_kramer", json={"persona": "Hersenonderzoeker in Nijmegen.",
                                                                  "tic_markers": "kijk, het zit zo"}).json()
        assert edited["fields"]["persona"] == "Hersenonderzoeker in Nijmegen."
        assert edited["fields"]["tic_markers"] == ["kijk, het zit zo"]
        assert any(v["id"] == "lotte_kramer" for v in client.get("/api/voices").json()["voices"])

        raw = tmp_path / "opname.wav"
        x = _lively(36)
        sf.write(raw, np.stack([x, x], axis=1), 24000)
        with open(raw, "rb") as fh:
            up = client.post("/api/characters/lotte_kramer/clips", files={"file": ("opname.wav", fh, "audio/wav")})
        assert up.status_code == 200, up.text
        clip = up.json()
        assert clip["report"]["after"]["lufs"] < -19 and clip["name"] == "opname.wav"
        assert client.get(clip["clip"]).status_code == 200 and client.get(clip["original"]).status_code == 200
        with open(raw, "rb") as fh:
            bad = client.post("/api/characters/lotte_kramer/clips", files={"file": ("x.wav", b"geen audio", "audio/wav")})
        assert bad.status_code == 400

        used = client.post(f"/api/characters/lotte_kramer/clips/{clip['id']}/use").json()
        assert used["has_ref"] and used["installed"] == "lotte_kramer.wav"
        assert (settings.cast_dir / "refs" / "lotte_kramer.wav").is_file()
        assert client.get("/api/characters/nobody").status_code == 404
        assert client.get(f"/api/characters/lotte_kramer/clips/{clip['id']}/../../x").status_code == 404
