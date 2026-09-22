from pipeline.audio.asr import EchoTranscriber
from pipeline.audio.cache import RenderCache
from pipeline.audio.synth import NullSynth, VoiceSpec
from pipeline.audio.takes import render_with_takes

TEXT = "Een kans is een getal tussen nul en een dat uitdrukt hoe waarschijnlijk een gebeurtenis is."


def _run(tmp_path, corrupt, n_takes=3, synth=None, transcriber=None, text=TEXT, **kw):
    cache = RenderCache(tmp_path / "cache")
    voice = VoiceSpec(speaker_id="tessa", base_exaggeration=0.5)
    synth = synth or NullSynth()
    transcriber = transcriber or EchoTranscriber(corrupt)
    return render_with_takes(text, voice, synth, cache, transcriber=transcriber, exaggeration=0.5, n_takes=n_takes,
                             wer_threshold=0.05, chars_per_second=15.0, seed_base=1000, **kw), cache


def test_rejects_corrupted_takes_and_picks_closest_duration(tmp_path):
    def corrupt(text, seed):
        words = text.split()
        return " ".join(words[:-2]) if seed == 1000 else text

    result, cache = _run(tmp_path, corrupt)
    assert len(result.takes) == 3 and not result.flagged and not result.from_cache
    assert result.takes[0].accepted is False and result.takes[0].wer > 0.05
    assert result.chosen in (1, 2)
    survivors = [t for t in result.takes if t.accepted]
    assert result.takes[result.chosen].duration_s == min(survivors, key=lambda t: abs(t.duration_s - len(TEXT) / 15)).duration_s
    again = render_with_takes(TEXT, VoiceSpec(speaker_id="tessa"), NullSynth(), cache, transcriber=EchoTranscriber(corrupt),
                              exaggeration=0.5, n_takes=3, wer_threshold=0.05, chars_per_second=15.0, seed_base=1000)
    assert again.from_cache and again.chosen == 0 and again.takes[0].reason == "cache"


def test_all_fail_retries_then_flags(tmp_path):
    result, _ = _run(tmp_path, lambda text, seed: "helemaal iets anders")
    assert result.flagged and result.flag_reason and len(result.takes) == 4  # 3 + 1 retry
    assert result.takes[3].exaggeration < result.takes[0].exaggeration
    assert result.clip is not None


def test_deterministic_synth_renders_once(tmp_path):
    synth = NullSynth()
    synth.deterministic = True
    result, _ = _run(tmp_path, None, synth=synth)
    assert len(result.takes) == 1 and result.takes[0].wer is None and result.takes[0].accepted


def test_no_transcriber_skips_verification(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    result = render_with_takes(TEXT, VoiceSpec(speaker_id="joris"), NullSynth(), cache, transcriber=None, exaggeration=0.4, n_takes=2)
    assert len(result.takes) == 2 and all(t.wer is None for t in result.takes)


class _RaisingTranscriber:
    """Simulates a broken faster-whisper: every call fails, e.g. a missing cuDNN DLL."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, clip, language="nl"):
        self.calls += 1
        raise RuntimeError("Could not locate cudnn_ops_infer64_8.dll")


def test_broken_transcriber_degrades_to_unverified_not_a_crash(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    broken = _RaisingTranscriber()
    result = render_with_takes(TEXT, VoiceSpec(speaker_id="joris"), NullSynth(), cache, transcriber=broken,
                               exaggeration=0.4, n_takes=2, seed_base=5000)
    assert broken.calls == 1  # never retried after the first failure
    assert len(result.takes) == 2 and all(t.wer is None and t.accepted for t in result.takes)
    assert not result.flagged
