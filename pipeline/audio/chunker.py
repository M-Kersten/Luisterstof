"""Split a script into blocks of at most 3000 characters at scene seams.

Used for the ElevenLabs accent tier (its request limit, counted after tag
injection) and for the per-block artifacts under render/<chapter>/blocks.
Seam preference: segment boundary, then speaker-turn boundary, then any
line boundary. A line is never split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.models import Line, Script
from pipeline.tags import eleven_tags

MAX_BLOCK_CHARS = 3000


@dataclass
class Block:
    id: str
    line_ids: list[str]
    chars: int
    segment_types: list[str] = field(default_factory=list)
    too_long: bool = False


def eleven_text(line: Line) -> str:
    tags = eleven_tags(line.tags, line.overlap.mode)
    return (" ".join(tags) + " " + line.text).strip() if tags else line.text


def eleven_inputs(lines: list[Line], voice_ids: dict[str, str]) -> list[dict[str, str]]:
    return [{"text": eleven_text(line), "voice_id": voice_ids[line.speaker]} for line in lines]


def line_chars(line: Line) -> int:
    return len(eleven_text(line))


def chunk_script(script: Script, max_chars: int = MAX_BLOCK_CHARS, min_fill: float = 0.4) -> list[Block]:
    entries: list[tuple[Line, str, bool, bool]] = []  # line, segment type, segment start, turn start
    prev_speaker = None
    for seg in script.segments:
        for i, line in enumerate(seg.lines):
            entries.append((line, seg.type, i == 0, line.speaker != prev_speaker))
            prev_speaker = line.speaker

    blocks: list[Block] = []
    current: list[tuple[Line, str]] = []
    current_chars = 0
    last_turn_seam = 0  # index in current where the last speaker turn started

    def close(upto: int | None = None) -> None:
        nonlocal current, current_chars, last_turn_seam
        chunk = current if upto is None else current[:upto]
        rest = [] if upto is None else current[upto:]
        if chunk:
            chars = sum(line_chars(l) for l, _ in chunk)
            blocks.append(Block(id=f"b{len(blocks) + 1:03d}", line_ids=[l.id for l, _ in chunk], chars=chars,
                                segment_types=sorted({t for _, t in chunk}), too_long=chars > max_chars))
        current = rest
        current_chars = sum(line_chars(l) for l, _ in current)
        last_turn_seam = 0

    for line, seg_type, seg_start, turn_start in entries:
        n = line_chars(line)
        if seg_start and current and current_chars >= max_chars * min_fill:
            close()
        if current and current_chars + n > max_chars:
            if last_turn_seam > 0 and sum(line_chars(l) for l, _ in current[:last_turn_seam]) >= max_chars * 0.5:
                close(last_turn_seam)
                if current_chars + n > max_chars:
                    close()
            else:
                close()
        if turn_start:
            last_turn_seam = len(current)
        current.append((line, seg_type))
        current_chars += n
    close()
    return blocks
