"""Speaker turns: the unit of rendering.

Consecutive lines by one speaker are rendered in a single call for prosody
continuity. A turn also ends at a deliberate beat and at a segment boundary,
so pauses and segment gaps can be inserted between renders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pipeline.cues import speakable
from pipeline.models import Glossary, Line, Script, SegmentType
from pipeline.plan.glossary import apply_lexicon


@dataclass
class Turn:
    turn_id: str
    speaker: str
    lines: list[Line]
    segment_type: SegmentType
    segment_index: int
    spoken_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.spoken_lines)

    @property
    def line_ids(self) -> list[str]:
        return [line.id for line in self.lines]

    @property
    def first(self) -> Line:
        return self.lines[0]

    @property
    def last(self) -> Line:
        return self.lines[-1]

    @property
    def tags(self) -> list[str]:
        seen: list[str] = []
        for line in self.lines:
            for t in line.tags:
                if t not in seen:
                    seen.append(t)
        return seen

    @property
    def pause_after_ms(self) -> int:
        return self.last.pause_after_ms

    def word_counts(self) -> list[int]:
        return [len(s.split()) for s in self.spoken_lines]


def spoken_text(line: Line, glossary: Glossary | None) -> str:
    source = speakable(line.text, line.reaction)  # stage cues like "(lacht)" or "chuckle" are never read aloud
    # A trailing fragment marker is meant to be buried under the interruption, never spoken as a dash.
    text = source.rstrip("—…- ").strip()
    if not text:
        text = source
    return apply_lexicon(text, glossary) if glossary else text


def group_turns(script: Script, glossary: Glossary | None = None,
                split_between: Callable[[Line, Line], bool] | None = None) -> list[Turn]:
    """``split_between(prev, line)`` can force a new turn at a line boundary (performance prototype)."""
    turns: list[Turn] = []
    counter = 0
    for si, seg in enumerate(script.segments):
        current: Turn | None = None
        for line in seg.lines:
            starts_new = (
                current is None
                or line.speaker != current.speaker
                or line.overlap.mode != "none"
                or current.last.pause_after_ms > 0
                or current.last.overlap.mode != "none"
                or (split_between is not None and split_between(current.last, line))
            )
            if starts_new:
                counter += 1
                current = Turn(turn_id=f"t{counter:03d}", speaker=line.speaker, lines=[], segment_type=seg.type, segment_index=si)
                turns.append(current)
            assert current is not None
            current.lines.append(line)
            current.spoken_lines.append(spoken_text(line, glossary))
    return turns
