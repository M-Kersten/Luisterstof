"""Stage cues must never reach a TTS voice as words."""

from pipeline.audio.turns import group_turns, spoken_text
from pipeline.cues import cue_reaction, speakable, strip_cues
from pipeline.fake_handlers import default_fake_llm
from pipeline.llm import FakeLLM
from pipeline.models import ContentPlan, Line, Script, Segment
from pipeline.script.writer import SegmentBrief, Writer


def test_whole_line_cues_become_the_sound_and_the_reaction():
    for text, label, sound in [("chuckle", "chuckle", "Hehe."), ("Chuckles.", "chuckle", "Hehe."),
                               ("[chuckle]", "chuckle", "Hehe."), ("(lacht)", "laugh", "Haha."),
                               ("*zucht*", "sigh", "Pff."), ("grinnikt zacht", "chuckle", "Hehe."),
                               ("(laughs)", "laugh", "Haha.")]:
        assert cue_reaction(text) == label, text
        assert speakable(text) == sound, text


def test_real_speech_is_left_alone():
    for text in ["Haha, nee.", "Hm.", "Ja.", "Precies, dat bedoel ik.", "Hij lacht altijd om zijn eigen grappen."]:
        assert cue_reaction(text) is None, text
        assert speakable(text) == text


def test_inline_cues_are_stripped_from_speech():
    assert strip_cues("Dat is (lacht) echt onzin.") == "Dat is echt onzin."
    assert strip_cues("Nou [chuckle], vooruit dan.") == "Nou, vooruit dan."
    line = Line(id="l001", speaker="tessa", text="Oké dus *grinnikt* dat klopt niet.")
    assert spoken_text(line, None) == "Oké dus dat klopt niet."


def test_existing_scripts_are_cleaned_at_render_time():
    script = Script(episode_id="ch01", segments=[Segment(type="body", lines=[
        Line(id="l001", speaker="tessa", text="En toen viel hij om."),
        Line(id="l002", speaker="joris", text="chuckle", overlap={"mode": "backchannel", "target": "l001"}),
    ])])
    turns = group_turns(script)
    assert turns[1].text == "Hehe."


def test_writer_cleans_cues_and_labels_the_reaction(cast, settings):
    def handler(req):
        return {"lines": [
            {"speaker": "tessa", "text": "Dit is (lacht) echt zo.", "tags": [], "covers": [], "overlap": "none",
             "pause_after_ms": 0, "delivery": "explain", "mood": "", "timing": "", "phrases": [], "reaction": ""},
            {"speaker": "joris", "text": "chuckle", "tags": [], "covers": [], "overlap": "backchannel",
             "pause_after_ms": 0, "delivery": "", "mood": "", "timing": "", "phrases": [], "reaction": ""},
        ]}

    writer = Writer(FakeLLM({"script_scene": handler}), cast, settings, performance=True)
    brief = SegmentBrief(type="body", title="t", covers=[], target_chars=300, instructions="x", speakers=["tessa", "joris"])
    lines = list(writer.write_scene(ContentPlan(chapter_id="ch01"), None, brief).lines())
    assert lines[0].text == "Dit is echt zo."
    assert (lines[1].text, lines[1].reaction) == ("Hehe.", "chuckle")
    plain = Writer(default_fake_llm(), cast, settings)
    assert "letterlijk uitgesproken" in plain._system(ContentPlan(chapter_id="c"), None, None, [])[0]["text"]
