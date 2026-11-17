"""How an episode opens: a teaser picked from the recording, the welcome, and how both sit on the music."""

from dataclasses import replace

from pipeline.audio.asr import UniformAligner
from pipeline.audio.opening import (
    LAST_BAR_S,
    TEASER_MAX_S,
    TEASER_MIN_S,
    lay_out,
    pick_teaser,
    teaser_room,
    welcome_text,
)
from pipeline.audio.render import render_episode
from pipeline.audio.synth import NullSynth, VoiceSpec
from pipeline.audio.turns import group_turns
from pipeline.models import EpisodeOutline, EpisodeTranscript, Line, Overlap, Script, Segment
from pipeline.paths import BookPaths

SR = 24000


def _line(n, speaker, text, **kw):
    return Line(id=f"l{n:03d}", speaker=speaker, text=text, **kw)


def _episode() -> Script:
    """A calm block, then one where it happens (from l008), then a quiz."""
    return Script(episode_id="ch01", title="3. Angst en aandacht",
                  outline=EpisodeOutline(question="Waarom bonst je hart als je op een ladder staat?"), segments=[
        Segment(type="cold_open", lines=[
            _line(1, "tessa", "Stel je voor dat je op een ladder staat om de dakgoot schoon te maken."),
            _line(2, "joris", "Dan kijk ik dus vooral niet naar beneden.", tags=["excited"]),
        ]),
        Segment(type="body", lines=[
            _line(3, "tessa", "Het lichaam houdt een aantal waarden in de gaten en stuurt bij als er een afwijkt."),
            _line(4, "joris", "Dat klinkt als een thermostaat die de hele dag aan staat en nooit pauze neemt."),
            _line(5, "tessa", "Het lijkt erop, alleen verschuift de ingestelde waarde mee met wat je aan het doen bent."),
            _line(6, "joris", "Dus de thermostaat verzet zichzelf zodra hij merkt dat ik de trap op ren."),
            _line(7, "tessa", "Ja, en dat verzetten kost je op de lange duur meer dan het oplevert."),
        ]),
        Segment(type="body", lines=[
            _line(8, "joris", "Hoe weet mijn lichaam het verschil tussen een ladder en een leeuw?", tags=["question"]),
            _line(9, "tessa", "Dat weet het dus niet! Het hart gaat al sneller voordat jij iets hebt besloten.", tags=["excited"]),
            _line(10, "joris", "Wacht, serieus? Dus ik ben bang voordat ik weet waarvoor?", tags=["surprised"],
                  overlap=Overlap(mode="interrupt", target="l009", cut_word="besloten")),
            _line(11, "tessa", "Precies dat, en je hoofd verzint de reden er pas achteraf bij.", tags=["laughs"]),
            _line(12, "joris", "Maar wie heeft er dan eigenlijk de leiding daarboven?", tags=["surprised"]),
            _line(13, "tessa", "Daar zijn onderzoekers in 1988 een begrip voor gaan gebruiken, en dat heet allostase."),
        ]),
        Segment(type="quiz", lines=[
            _line(14, "joris", "Vraag een! Wat gaat er het eerst omhoog?", tags=["excited"], pause_after_ms=3000),
            _line(15, "tessa", "De hartslag!", tags=["excited"]),
        ]),
    ])


def _rendered(script):
    turns = group_turns(script)
    synth, aligner = NullSynth(sample_rate=SR, jitter=0.0), UniformAligner()
    rendered = {}
    for t in turns:
        clip = synth.render(t.text, VoiceSpec(speaker_id=t.speaker), exaggeration=0.5, seed=1)
        rendered[t.turn_id] = (clip, aligner.align(clip, t.text))
    return turns, rendered


def test_welcome_names_the_question_of_the_day():
    script = _episode()
    assert welcome_text(script) == "Welkom bij een nieuwe Luisterstof, vandaag: waarom bonst je hart als je op een ladder staat?"
    assert welcome_text(replace_outline(script, None)) == "Welkom bij een nieuwe Luisterstof, vandaag: Angst en aandacht."
    assert welcome_text(Script(episode_id="ch01")) == "Welkom bij een nieuwe Luisterstof."


def replace_outline(script, outline):
    return script.model_copy(update={"outline": outline})


def test_the_teaser_is_where_it_happens_and_stands_on_its_own():
    turns, rendered = _rendered(_episode())
    teaser = pick_teaser(turns, rendered, sr=SR)
    assert teaser is not None
    assert set(teaser.line_ids) <= {"l008", "l009", "l010", "l011", "l012"}  # not the calm block, the quiz or the opening
    assert {"l009", "l010"} <= set(teaser.line_ids)  # the exclamation and the interruption are in it
    assert teaser.line_ids[0] == "l008"  # starts on the question, not on an answer to something unheard
    assert teaser.line_ids[-1] == "l010" and "eindigt op een vraag" in teaser.why  # and stops on an open one
    assert TEASER_MIN_S - 1.0 <= teaser.duration <= TEASER_MAX_S
    assert len({t.speaker for t in teaser.turns}) == 2
    # how the takes came out counts too: push the calm block's voices up and it can win
    lifts = {t.turn_id: 0.3 for t in turns if set(t.line_ids) & {"l003", "l004", "l005", "l006"}}
    assert pick_teaser(turns, rendered, lifts=lifts, sr=SR).score > 0
    only_opening = Script(episode_id="ch01", segments=[_episode().segments[0]])
    assert pick_teaser(*_rendered(only_opening), sr=SR) is None


def test_the_opening_is_laid_out_on_the_music():
    # 14 s of teaser and 5.5 s of welcome under 30 s of music: the welcome is over before the last bar begins
    lay = lay_out(14.0, 5.5, 30.0)
    assert 30.0 - LAST_BAR_S - 2.6 <= lay.welcome_at + 5.5 <= 30.0 - LAST_BAR_S + 1e-6 and lay.music_until is None
    assert 2.4 <= lay.teaser_at <= 3.4 + 1e-6  # the music first, but not for long
    assert 1.6 <= lay.welcome_at - (lay.teaser_at + 14.0) <= 3.0 + 1e-6  # and it comes up between the two
    assert abs(lay.main_at - 29.4) < 1e-6 and lay.voices == [(lay.teaser_at, lay.teaser_at + 14.0), (lay.welcome_at, lay.welcome_at + 5.5)]
    # a short opening does not wait for long music: it comes up once more and fades
    short = lay_out(0.0, 5.5, 30.0)
    assert short.teaser_at is None and short.music_until is not None and short.main_at < 15.0
    # voices that outlast the music let it finish before the script starts
    long = lay_out(22.0, 5.5, 30.0)
    assert long.main_at >= long.welcome_at + 5.5 + 0.9 and long.music_until is None
    # no music: teaser, a breath, the welcome, the script
    dry = lay_out(14.0, 5.5, None)
    assert dry.teaser_at == 0.0 and dry.welcome_at == 14.7 and abs(dry.main_at - 21.1) < 1e-6 and dry.voices == []
    room, prefer = teaser_room(5.5, 30.0)
    assert room == 17.0 and prefer == (12.0, 17.0) and teaser_room(5.5, None) == (TEASER_MAX_S, None)


def test_an_episode_opens_with_teaser_and_welcome(settings, cast):
    script = _episode()
    paths = BookPaths(settings.data_dir, "demo").ensure()
    _, transcript_path, manifest = render_episode(script, None, cast, settings, paths, synth=NullSynth(jitter=0.0),
                                                  transcriber=None, aligner=None)
    opening = manifest.opening
    assert opening["welcome"].startswith("Welkom bij een nieuwe Luisterstof, vandaag: waarom bonst je hart")
    assert opening["teaser"]["line_ids"][0] == "l008" and opening["teaser"]["why"]
    assert manifest.turns[-1].turn_id == "welkom" and manifest.turns[-1].speaker == "tessa"  # the explainer hosts
    lines = EpisodeTranscript.load(transcript_path).lines
    ids = [line.line_id for line in lines]
    hook = [i for i in ids if i.startswith("hook:")]
    assert hook == ids[: len(hook)] and hook[-1] == "hook:welkom"  # teaser, then the welcome, then the script
    assert hook[:-1] == [f"hook:{i}" for i in opening["teaser"]["line_ids"]]
    assert ids[len(hook)] == "l001" and all(line.segment is None for line in lines[: len(hook)])
    first = lines[len(hook)]
    assert first.segment == "cold_open" and abs(first.start - (opening["script_at"] + 0.3)) < 0.1
    assert lines[0].start >= 2.4 + 0.3  # the music is heard before the first voice
    assert ids.count("l009") == 1 and ids.count("hook:l009") == 1  # the same take, heard twice

    quiet = replace(settings, opening=False, hook_music=False)
    _, transcript_path, manifest = render_episode(script, None, cast, quiet, BookPaths(quiet.data_dir, "quiet").ensure(),
                                                  synth=NullSynth(jitter=0.0), transcriber=None, aligner=None)
    lines = EpisodeTranscript.load(transcript_path).lines
    assert manifest.opening == {} and lines[0].line_id == "l001" and abs(lines[0].start - 0.3) < 0.1


def test_a_picked_teaser_is_used_and_a_lost_one_says_so():
    from pipeline.audio.opening import build_opening, chosen_teaser, welcome_turn
    from pipeline.audio.timeline import assemble
    from pipeline.script.cast import load_cast

    script = _episode()
    turns, rendered = _rendered(script)
    teaser = chosen_teaser(turns, rendered, "l004", "l006", sr=SR)
    assert teaser.line_ids == ["l004", "l005", "l006"] and teaser.why == ["door jou gekozen"]
    assert chosen_teaser(turns, rendered, "l006", "l004", sr=SR) is None
    assert chosen_teaser(turns, rendered, "l004", "l404", sr=SR) is None

    cast = load_cast("cast")
    welcome = welcome_turn(script, cast)
    synth, aligner = NullSynth(sample_rate=SR, jitter=0.0), UniformAligner()
    clip = synth.render(welcome.text, VoiceSpec(speaker_id=welcome.speaker), exaggeration=0.5, seed=1)
    rendered[welcome.turn_id] = (clip, aligner.align(clip, welcome.text))
    main = assemble(turns, rendered, sr=SR)
    picked = build_opening(main, turns, rendered, welcome, music_s=30.0, choice=("l004", "l006"), sr=SR)
    record = picked.record()
    assert record["teaser"]["line_ids"] == ["l004", "l005", "l006"] and record["teaser"]["chosen"]
    assert "choice_lost" not in record
    lost = build_opening(main, turns, rendered, welcome, music_s=30.0, choice=("l004", "l404"), sr=SR).record()
    assert not lost["teaser"]["chosen"] and lost["teaser"]["line_ids"][0] == "l008"  # picked here, as without a choice
    assert "l404" in lost["choice_lost"]
