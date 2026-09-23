"""Platform profile, MLX backend plumbing and token-to-timing matching."""

import numpy as np

from pipeline import config
from pipeline.audio.asr import (
    EchoTranscriber,
    MlxWhisperAligner,
    MlxWhisperTranscriber,
    UniformAligner,
    WhisperXAligner,
    WordTiming,
    asr_words_from_meta,
    make_aligner,
    make_transcriber,
    match_words_to_text,
    tokenize,
)
from pipeline.audio.cache import RenderCache
from pipeline.audio.synth import AudioClip, NullSynth, VoiceSpec
from pipeline.audio.takes import render_with_takes
from pipeline.config import Settings


def test_defaults_follow_platform(monkeypatch):
    monkeypatch.setattr(config, "apple_silicon", lambda: True)
    for var in ("STUDIEPODCAST_DEVICE", "CHATTERBOX_WORKERS", "WHISPER_BACKEND", "STUDIEPODCAST_TAKES"):
        monkeypatch.delenv(var, raising=False)
    mac = Settings.from_env(dotenv=None)
    assert (mac.device, mac.chatterbox_workers, mac.whisper_backend) == ("mps", 1, "mlx")
    assert mac.profile()["whisper_model"] == "mlx-community/whisper-large-v3-turbo"
    monkeypatch.setattr(config, "apple_silicon", lambda: False)
    linux = Settings.from_env(dotenv=None)
    assert (linux.device, linux.chatterbox_workers, linux.whisper_backend) == ("cuda", 3, "faster")
    monkeypatch.setenv("STUDIEPODCAST_TAKES", "2")
    monkeypatch.setenv("STUDIEPODCAST_DEVICE", "cpu")
    custom = Settings.from_env(dotenv=None)
    assert custom.takes_per_line == 2 and custom.device == "cpu"


def test_backend_selection():
    mac = Settings(whisper_backend="mlx", device="mps")
    assert isinstance(make_transcriber(mac), MlxWhisperTranscriber) and isinstance(make_aligner(mac), MlxWhisperAligner)
    cuda = Settings(whisper_backend="faster", device="cuda")
    assert isinstance(make_aligner(cuda), WhisperXAligner) and make_aligner(cuda).device == "cuda"
    assert WhisperXAligner(Settings(device="mps")).device == "cpu"
    assert isinstance(make_transcriber(mac, fake=True), EchoTranscriber) and isinstance(make_aligner(mac, fake=True), UniformAligner)


def test_match_words_to_text_uses_asr_times_and_interpolates_the_rest():
    tokens = tokenize("De kans is 0,5 en dat is precies de definitie.")
    asr = [WordTiming(" De", 0.10, 0.20), WordTiming(" kans", 0.25, 0.50), WordTiming(" is", 0.55, 0.60),
           WordTiming(" nul", 0.65, 0.80), WordTiming(" komma", 0.82, 1.0), WordTiming(" vijf", 1.02, 1.2),
           WordTiming(" en", 1.3, 1.35), WordTiming(" dat", 1.4, 1.5), WordTiming(" is", 1.55, 1.6),
           WordTiming(" precies", 1.65, 1.9), WordTiming(" de", 1.95, 2.0), WordTiming(" definitie.", 2.05, 2.5)]
    words = match_words_to_text(tokens, asr, 2.7)
    assert [w.word for w in words] == tokens
    assert words[0].start == 0.10 and words[1].start == 0.25
    # "0,5" was spoken as three words: interpolated between "is" (0.60) and "en" (1.3)
    assert 0.60 <= words[3].start < words[3].end <= 1.3
    assert words[-1].start == 2.05
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    # misheard word of equal length keeps its slot
    words2 = match_words_to_text(tokenize("wacht even hele"), [WordTiming("wacht", 0, .2), WordTiming("evenn", .3, .5), WordTiming("hele", .6, .8)], 1.0)
    assert words2[1].start == 0.3 and words2[2].start == 0.6
    assert match_words_to_text([], asr, 1.0) == []
    assert len(match_words_to_text(tokens, [], 2.0)) == len(tokens)


def test_mlx_aligner_reads_meta_and_take_loop_persists_it(tmp_path):
    clip = AudioClip(np.zeros(24000, dtype=np.float32), 24000, meta={"asr_words": [["hallo", 0.1, 0.4], ["wereld", 0.5, 0.9]]})
    words = MlxWhisperAligner(Settings(whisper_backend="mlx")).align(clip, "Hallo wereld")
    assert [(w.word, w.start) for w in words] == [("Hallo", 0.1), ("wereld", 0.5)]
    assert asr_words_from_meta("nope") is None and asr_words_from_meta([["x", "a", 1]]) is None

    class MetaTranscriber(EchoTranscriber):
        def transcribe(self, clip, language="nl"):
            text = super().transcribe(clip, language)
            clip.meta["asr_words"] = [[w, i * 0.3, i * 0.3 + 0.2] for i, w in enumerate(text.split())]
            return text

    cache = RenderCache(tmp_path / "cache")
    voice = VoiceSpec(speaker_id="tessa")
    result = render_with_takes("dit is een test", voice, NullSynth(), cache, transcriber=MetaTranscriber(), exaggeration=0.5, n_takes=1)
    cached, meta = cache.get(result.take_key)
    assert meta["asr_words"][1][0] == "is" and cached.meta["asr_words"] == meta["asr_words"]
    words = MlxWhisperAligner(Settings(whisper_backend="mlx")).align(cached, "dit is een test")
    assert [round(w.start, 6) for w in words] == [0.0, 0.3, 0.6, 0.9]
    # a clip without meta and without mlx installed falls back to uniform spacing
    bare = AudioClip(np.zeros(24000, dtype=np.float32), 24000)
    assert len(MlxWhisperAligner(Settings(whisper_backend="mlx")).align(bare, "een twee drie")) == 3


def test_last_used_fallback_is_tracked_per_call():
    """The render loop reads this flag right after align() to report real-vs-estimated timing on the manifest."""
    assert UniformAligner.last_used_fallback is True  # by definition: always an estimate

    mlx_aligner = MlxWhisperAligner(Settings(whisper_backend="mlx"))
    with_meta = AudioClip(np.zeros(24000, dtype=np.float32), 24000, meta={"asr_words": [["hallo", 0.1, 0.4]]})
    mlx_aligner.align(with_meta, "hallo")
    assert mlx_aligner.last_used_fallback is False
    without_meta = AudioClip(np.zeros(24000, dtype=np.float32), 24000)  # no mlx installed -> falls back
    mlx_aligner.align(without_meta, "hallo")
    assert mlx_aligner.last_used_fallback is True

    whisperx_aligner = WhisperXAligner(Settings(device="cpu"))  # whisperx not installed here -> falls back
    whisperx_aligner.align(with_meta, "hallo")
    assert whisperx_aligner.last_used_fallback is True


def test_aligner_cache_round_trips_fallback_flag_and_reads_old_format(tmp_path):
    from pipeline.audio.aligner_cache import _path, load_alignment, save_alignment

    cache = RenderCache(tmp_path / "cache")
    words = [WordTiming("hallo", 0.0, 0.3), WordTiming("wereld", 0.3, 0.7)]

    save_alignment(cache, "keyA", words, fallback=True)
    loaded_words, loaded_fallback = load_alignment(cache, "keyA")
    assert loaded_fallback is True
    assert [(w.word, w.start, w.end) for w in loaded_words] == [(w.word, w.start, w.end) for w in words]

    save_alignment(cache, "keyB", words, fallback=False)
    assert load_alignment(cache, "keyB")[1] is False

    # a cache entry written before fallback tracking existed (a bare list) still loads, as non-fallback
    import json

    p = _path(cache, "keyOld")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"word": "oud", "start": 0.0, "end": 0.2}]))
    old_words, old_fallback = load_alignment(cache, "keyOld")
    assert old_fallback is False and old_words[0].word == "oud"

    assert load_alignment(cache, "keyMissing") is None
