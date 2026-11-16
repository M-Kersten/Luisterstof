"""How an episode opens: its own most exciting moment, then the welcome.

Before the script starts the listener hears a teaser: a few turns from further on in the episode, the very takes
that come back later. Then the main host says what show this is and what today's question is, and the episode
begins. With the jingle under it this is the hook.

The teaser is found in the recording, after every turn has been rendered, so it costs no writing and a new render
of an approved script picks one too. Every stretch of consecutive turns that fits is scored on how much happens in
it: what the script marks (an excited or surprised line, a laugh, an interruption, a question or an exclamation),
how the takes came out (how far the voice was pushed, how fast it speaks), whether two people are going back and
forth, whether it can be followed without what came before, and whether it ends on something left open.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from pipeline.audio.asr import WordTiming
from pipeline.audio.mixer import HOOK_HOLD_S, HOOK_LEAD_S, HOOK_OUT_S, HOOK_RISE_S
from pipeline.audio.synth import AudioClip
from pipeline.audio.timeline import Placement, Timeline, assemble
from pipeline.audio.turns import Turn, spoken_text
from pipeline.cues import pure_reaction
from pipeline.models import Cast, Glossary, Line, Script

WELCOME = "Welkom bij een nieuwe Luisterstof, vandaag: {topic}"
WELCOME_BARE = "Welkom bij een nieuwe Luisterstof."
WELCOME_ID = "welkom"
WELCOME_LIFT = 0.1  # a little brighter than the host's everyday voice

TEASER_SEGMENTS = ("body", "guest", "reexplain")  # not the opening itself, the recap of another episode, quiz or goodbye
TEASER_MIN_S = 7.0
TEASER_MAX_S = 17.0
TEASER_MAX_TURNS = 9

# With music: the teaser, the music coming up for a moment, the welcome, and the last bar of the music on its own.
SWELL_S = 1.6  # music between the teaser and the welcome
SWELL_MORE_S = 1.4  # room that is left over goes here first,
LEAD_MORE_S = 1.0  # then before the first voice, and what is still left is heard after the welcome
LAST_BAR_S = 3.5  # the end of the music is heard without a voice over it
SLACK_MAX_S = 5.0  # more room than this and the music is not waited for: it fades after the welcome
AFTER_MUSIC_S = 0.6  # the script starts this long before the file ends (the jingle is silent by then)
AFTER_WELCOME_S = 0.9
# Without music: a breath between the teaser and the welcome.
BETWEEN_S = 0.7

TAG_HEAT = {"excited": 3.0, "surprised": 3.0, "laughs": 3.0, "laughing": 3.0, "annoyed": 2.0, "fast": 1.0,
            "question": 1.0, "skeptical": 1.0, "deadpan": 1.0}
DELIVERY_HEAT = {"excite": 3.0, "realize": 3.0, "punchline": 3.0, "disagree": 2.5, "interrupt": 2.0, "react": 1.0,
                 "setup": 0.5}
MOOD_HEAT = {"surprised": 2.0, "amused": 2.0, "challenged": 1.5}
REACTION_HEAT = {"laugh": 2.0, "chuckle": 1.5, "wacht": 2.0, "oh": 1.5}
SPARKS = re.compile(r"\b(wacht|serieus|echt|hoezo|onmogelijk|nooit|bizar|gek|raar|omgekeerde|klopt niet|"
                    r"geloof|meen je|wat\?)", re.IGNORECASE)
# A first line that answers something the listener did not hear.
LEANS_BACK = re.compile(r"^(ja|nee|maar|en|dus|precies|oké|ok|klopt|inderdaad|dat|die|daarom|daardoor|want|of|"
                        r"juist|nou|ook|dan|toch|hè|o|oh|ah|aha|goed|zeker|eens)\b", re.IGNORECASE)
NUMBER = re.compile(r"\d")


@dataclass
class Teaser:
    turns: list[Turn]
    timeline: Timeline  # the turns on their own clock, from zero
    score: float
    why: list[str] = field(default_factory=list)

    @property
    def line_ids(self) -> list[str]:
        return [lid for t in self.turns for lid in t.line_ids]

    @property
    def duration(self) -> float:
        return self.timeline.duration


def _line_heat(line: Line) -> tuple[float, list[str]]:
    heat, why = 0.0, []
    for tag in line.tags:
        if tag in TAG_HEAT:
            heat += TAG_HEAT[tag]
            why.append(tag)
    for value, table in ((line.delivery, DELIVERY_HEAT), (line.mood, MOOD_HEAT), (line.reaction, REACTION_HEAT)):
        if value in table:
            heat += table[value]
            why.append(str(value))
    if line.overlap.mode == "interrupt":
        heat += 2.5
        why.append("valt in de rede")
    elif line.overlap.mode == "backchannel":
        heat += 0.5
    text = line.text
    marks = min(2, text.count("!")) + min(2, text.count("?"))
    if marks:
        heat += marks
        why.append("uitroep of vraag")
    sparks = min(2, len(SPARKS.findall(text)))
    if sparks:
        heat += sparks
        why.append("verbazing in de woorden")
    return heat, why


def _advances(turn: Turn) -> bool:
    return turn.first.overlap.mode != "backchannel"


def pick_teaser(
    turns: list[Turn],
    rendered: dict[str, tuple[AudioClip, list[WordTiming]]],
    *,
    lifts: dict[str, float] | None = None,
    sr: int,
    seed: int = 7,
    min_s: float = TEASER_MIN_S,
    max_s: float = TEASER_MAX_S,
    prefer_s: tuple[float, float] | None = None,
) -> Teaser | None:
    """The most exciting stretch of the episode that can stand on its own, or None when nothing fits.

    ``lifts`` says per turn how far above the speaker's everyday voice its take was rendered. ``prefer_s`` is a
    length that suits the music; a stretch of that length wins from one that is as good but leaves the music idle.
    """
    lifts = lifts or {}
    pace: dict[str, list[float]] = {}
    for t in turns:
        if t.turn_id in rendered and rendered[t.turn_id][0].duration_s > 0.5:
            pace.setdefault(t.speaker, []).append(len(t.text) / rendered[t.turn_id][0].duration_s)
    usual = {speaker: statistics.median(rates) for speaker, rates in pace.items()}

    def turn_heat(t: Turn) -> tuple[float, list[str]]:
        heat, why = 0.0, []
        for line in t.lines:
            h, w = _line_heat(line)
            heat += h
            why += w
        lift = lifts.get(t.turn_id, 0.0)
        if lift > 0.02:
            heat += min(3.0, lift * 10)
            why.append("stem opgezweept")
        clip = rendered[t.turn_id][0]
        if clip.duration_s > 0.5 and usual.get(t.speaker):
            quick = (len(t.text) / clip.duration_s) / usual[t.speaker] - 1.0
            if quick > 0.05:
                heat += min(2.0, quick * 8)
                why.append("praat sneller")
        return heat, why

    candidates: list[tuple[float, int, int, list[str]]] = []
    for i, first in enumerate(turns):
        if (first.segment_type not in TEASER_SEGMENTS or first.turn_id not in rendered
                or first.first.overlap.mode != "none" or pure_reaction(first.first)):
            continue
        heat, why, speakers, length, advancing, handoffs, before = 0.0, [], set(), 0.0, 0, 0, None
        for j in range(i, min(len(turns), i + TEASER_MAX_TURNS)):
            t = turns[j]
            if t.segment_index != first.segment_index or t.turn_id not in rendered or t.pause_after_ms > 1500:
                break
            h, w = turn_heat(t)
            heat += h
            why += w
            if _advances(t):
                if before is not None and t.speaker != before:
                    handoffs += 1
                before = t.speaker
                length += rendered[t.turn_id][0].duration_s + (0.25 if advancing else 0.0)
                advancing += 1
            speakers.add(t.speaker)
            if length > max_s + 1.0:
                break
            last = t.last
            if length < min_s or len(speakers) < 2 or not _advances(t) or last.ends_with_fragment:
                continue
            score = heat * (10.0 / length) ** 0.6 + min(4, handoffs)
            notes = list(dict.fromkeys(why))
            ending = last.text.rstrip()
            if ending.endswith("?"):
                score += 3.0
                notes.append("eindigt op een vraag")
            elif ending.endswith("!"):
                score += 1.5
            if LEANS_BACK.match(first.first.text.strip()):
                score -= 4.0
            elif first.first.text.rstrip().endswith("?"):
                score += 1.0
            score -= min(2.0, 0.5 * sum(1 for x in turns[i: j + 1] for line in x.lines if NUMBER.search(line.text)))
            if prefer_s and prefer_s[0] <= length <= prefer_s[1]:
                score += 3.0
            candidates.append((score, i, j, notes))

    for score, i, j, notes in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        chosen = turns[i: j + 1]
        timeline = assemble(chosen, rendered, sr=sr, seed=seed)
        if min_s - 1.0 <= timeline.duration <= max_s:
            return Teaser(chosen, timeline, round(score, 2), notes)
    return None


def topic_of(script: Script) -> str:
    """What the episode is about, as the host can say it: the question the outline settled on, or the chapter title."""
    topic = (script.outline.question if script.outline else "") or script.title or ""
    topic = " ".join(topic.split())
    return re.sub(r"^(hoofdstuk\s+)?\d+\s*[.:)\-–]\s*", "", topic, flags=re.IGNORECASE)


def welcome_text(script: Script) -> str:
    topic = topic_of(script)
    if not topic:
        return WELCOME_BARE
    if topic.endswith("?") and topic[0].isupper() and not topic[1:2].isupper():
        topic = topic[0].lower() + topic[1:]  # a question carries on the sentence; a title keeps its capital
    if topic[-1] not in ".?!":
        topic += "."
    return WELCOME.format(topic=topic)


def welcome_turn(script: Script, cast: Cast, glossary: Glossary | None = None) -> Turn:
    """The welcome as a turn of the main host (the explainer), rendered like any other turn but in no segment."""
    host = next(h for h in cast.hosts if h.role == "explainer")
    line = Line(id=WELCOME_ID, speaker=host.id, text=welcome_text(script))
    return Turn(turn_id=WELCOME_ID, speaker=host.id, lines=[line], segment_type="cold_open", segment_index=-1,
                spoken_lines=[spoken_text(line, glossary)])


@dataclass
class Layout:
    """Where everything starts, in seconds from the start of the music (or of the episode without music)."""
    teaser_at: float | None
    welcome_at: float
    main_at: float
    voices: list[tuple[float, float]] = field(default_factory=list)  # where the music stays under
    music_until: float | None = None  # where the music has faded out, when the opening is over long before it is


def teaser_room(welcome_s: float, music_s: float | None) -> tuple[float, tuple[float, float] | None]:
    """The longest teaser that still lets the music finish as written, and the lengths that fill the music best."""
    if music_s is None:
        return TEASER_MAX_S, None
    room = music_s - LAST_BAR_S - welcome_s - SWELL_S - HOOK_LEAD_S
    if room < TEASER_MIN_S:
        return TEASER_MAX_S, None  # short music: the voices simply outlast it
    return min(room, TEASER_MAX_S), (max(TEASER_MIN_S, room - SLACK_MAX_S), room)


def lay_out(teaser_s: float, welcome_s: float, music_s: float | None) -> Layout:
    if music_s is None:
        welcome_at = teaser_s + BETWEEN_S if teaser_s else 0.0
        return Layout(0.0 if teaser_s else None, welcome_at, welcome_at + welcome_s + AFTER_WELCOME_S)
    lead, swell = HOOK_LEAD_S, SWELL_S if teaser_s else 0.0
    slack = (music_s - LAST_BAR_S) - (lead + teaser_s + swell + welcome_s)
    until = None
    if 0.0 <= slack <= SLACK_MAX_S:  # the welcome is over before the music's last bar, and the script waits for it
        more = min(slack, SWELL_MORE_S) if teaser_s else 0.0
        swell += more
        lead += min(slack - more, LEAD_MORE_S)
    welcome_at = lead + teaser_s + swell
    end = welcome_at + welcome_s
    if 0.0 <= slack <= SLACK_MAX_S:
        main_at = music_s - AFTER_MUSIC_S
    elif slack < 0.0:  # the voices outlast the music, or nearly: let it finish first
        main_at = max(end + AFTER_WELCOME_S, music_s - AFTER_MUSIC_S)
    else:  # a short opening under long music: it comes up once more and leaves
        until = end + HOOK_HOLD_S + HOOK_RISE_S + 1.0 + HOOK_OUT_S
        main_at = until - 0.8
    voices = ([(lead, lead + teaser_s)] if teaser_s else []) + [(welcome_at, end)]
    return Layout(lead if teaser_s else None, welcome_at, main_at, voices, until)


@dataclass
class Opening:
    teaser: Teaser | None
    welcome: Turn
    layout: Layout
    timeline: Timeline  # the whole episode: teaser, welcome, then the script, on one clock

    def record(self) -> dict:
        """For the manifest: what was picked and why, so a choice you do not like can be traced."""
        out: dict = {"welcome": self.welcome.text, "welcome_at": round(float(self.layout.welcome_at), 2),
                     "script_at": round(float(self.layout.main_at), 2)}
        if self.teaser is not None:
            out["teaser"] = {"line_ids": self.teaser.line_ids, "at": round(float(self.layout.teaser_at or 0.0), 2),
                             "duration_s": round(float(self.teaser.duration), 2), "score": float(self.teaser.score),
                             "why": self.teaser.why}
        return out


def build_opening(
    main: Timeline,
    turns: list[Turn],
    rendered: dict[str, tuple[AudioClip, list[WordTiming]]],
    welcome: Turn,
    *,
    lifts: dict[str, float] | None = None,
    music_s: float | None = None,
    sr: int,
    seed: int = 7,
) -> Opening:
    """The episode with its opening in front: the script's own timeline moves back to make room."""
    clip, words = rendered[welcome.turn_id]
    max_s, prefer_s = teaser_room(clip.duration_s, music_s)
    teaser = pick_teaser(turns, rendered, lifts=lifts, sr=sr, seed=seed, max_s=max_s, prefer_s=prefer_s)
    layout = lay_out(teaser.duration if teaser else 0.0, clip.duration_s, music_s)
    placements: list[Placement] = []
    if teaser is not None and layout.teaser_at is not None:
        placements += [p.moved(layout.teaser_at, opening=True) for p in teaser.timeline.placements]
    placements.append(Placement(welcome, clip, layout.welcome_at, [w.shifted(layout.welcome_at) for w in words],
                                opening=True))
    placements += [p.moved(layout.main_at) for p in main.placements]
    return Opening(teaser, welcome, layout, Timeline(placements=placements, sr=main.sr, qa=list(main.qa)))
