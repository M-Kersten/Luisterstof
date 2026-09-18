import numpy as np

from pipeline.audio.cache import RenderCache
from pipeline.audio.synth import AudioClip


def test_keys_stable_and_roundtrip(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    k1 = cache.take_key("hallo", "ref1", 0.5, 1, "null")
    assert k1 == cache.take_key("hallo", "ref1", 0.5, 1, "null")
    assert k1 != cache.take_key("hallo", "ref2", 0.5, 1, "null") != cache.take_key("hallo", "ref1", 0.5, 2, "null")
    assert cache.get(k1) is None and not cache.has(k1)
    clip = AudioClip(np.zeros(2400, dtype=np.float32), 24000, meta={"text": "hallo", "seed": 1})
    cache.put(k1, clip, {"speaker": "tessa"})
    got, meta = cache.get(k1)
    assert got.sr == 24000 and len(got.samples) == 2400 and meta["speaker"] == "tessa" and meta["text"] == "hallo"
    cache.update_meta(k1, transcript="hallo")
    assert cache.get(k1)[1]["transcript"] == "hallo"


def test_selection_requires_take(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    sel = cache.selection_key("t", "r", 0.5, "null", "n=3")
    cache.put_selection(sel, {"take_key": "missing"})
    assert cache.get_selection(sel) is None
    k = cache.take_key("t", "r", 0.5, 1, "null")
    cache.put(k, AudioClip(np.zeros(100, dtype=np.float32), 24000))
    cache.put_selection(sel, {"take_key": k, "wer": 0.0})
    assert cache.get_selection(sel)["take_key"] == k
    assert cache.stats() == {"takes": 1, "selections": 1}
