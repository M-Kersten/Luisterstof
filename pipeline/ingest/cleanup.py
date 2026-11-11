"""What a study book prints but a podcast must not read.

Exercises, fill-in-the-blank copies of the text and footnote markers are made
for a reader with a pen. Left in, the plan quotes them ("is het niet ______
met") and the script pads itself with them. ``build_book`` uses these checks;
the PDF stage already drops superscript footnote numbers and picture captions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

# A line that is nothing but an exercise heading: "Oefening 3", "Opdrachten", "Vragen bij 2.4", "Test jezelf".
EXERCISE_HEADING = re.compile(
    r"^(oefening(en)?|opdracht(en)?|opgave(n)?|vragen|vraag|toets jezelf|test jezelf|zelftoets|"
    r"exercises?|questions|practice)(\s+(bij\s+)?[\w.‘’'\-]+){0,6}\s*[:.]?\s*$",
    re.IGNORECASE,
)
# Something a reader writes in: a blank, a dotted line, a box to tick.
FILL_IN = re.compile(r"_{4,}|\.{8,}|…{3,}|\[\s*\]")
# A footnote body ("* Wil je het Wilhelmus horen? ...") and its marker glued to a word ("Wilhelmus’*").
FOOTNOTE_BODY = re.compile(r"^\s*[*†‡]\s+\S")
FOOTNOTE_MARK = re.compile(r"(?<=[^\W\d_]|[’'”»])[*†‡]+")
REPEAT_MIN_CHARS = 60
REPEAT_MIN_SCORE = 88.0


def is_exercise_heading(text: str) -> bool:
    return len(text) <= 80 and bool(EXERCISE_HEADING.match(text.strip()))


@dataclass
class CleanupReport:
    exercise_chars: int = 0
    fill_ins: int = 0
    repeats: int = 0
    footnotes: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str | None:
        parts = []
        if self.exercise_chars:
            parts.append(f"{self.exercise_chars} characters of exercises")
        if self.fill_ins:
            parts.append(f"{self.fill_ins} fill-in paragraph(s)")
        if self.repeats:
            parts.append(f"{self.repeats} repeated paragraph(s)")
        if self.footnotes:
            parts.append(f"{self.footnotes} footnote(s)")
        return ("left out of the text: " + ", ".join(parts)) if parts else None


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", FILL_IN.sub(" ", text)).strip().casefold()


def rejoin_broken(paragraphs: list[str]) -> list[str]:
    """A sentence cut by a page break ("Op 10" | "juli 1584 hebben ze...") becomes one paragraph again."""
    out: list[str] = []
    for para in paragraphs:
        if out and not out[-1].startswith("[") and not para.startswith("[") \
                and not re.search(r"[.!?:;’'”\])]\s*$", out[-1]) and re.match(r"\s*[a-zà-ÿ]", para):
            out[-1] = out[-1].rstrip() + " " + para.lstrip()
        else:
            out.append(para)
    return out


def clean_paragraphs(paragraphs: list[str], seen: list[str], report: CleanupReport) -> list[str]:
    """Drop fill-ins, footnote bodies and paragraphs that repeat an earlier one (``seen`` holds what was kept
    before, in this chapter). Footnote markers glued to words are removed; sentences cut by a page break are
    joined again."""
    kept: list[str] = []
    for para in rejoin_broken(paragraphs):
        if FILL_IN.search(para):
            report.fill_ins += 1
            continue
        if FOOTNOTE_BODY.match(para):
            report.footnotes += 1
            continue
        para = FOOTNOTE_MARK.sub("", para)
        norm = _normalised(para)
        if len(norm) >= REPEAT_MIN_CHARS and any(
                fuzz.ratio(norm, earlier) >= REPEAT_MIN_SCORE for earlier in seen if abs(len(earlier) - len(norm)) < 0.3 * len(norm)):
            report.repeats += 1
            continue
        seen.append(norm)
        kept.append(para)
    return kept
