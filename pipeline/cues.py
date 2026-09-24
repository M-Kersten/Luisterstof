"""Stage cues in script text ("(lacht)", "[chuckle]", "*zucht*", a bare "chuckle").

A TTS model reads every character it gets, so a cue in the text is spoken aloud as
the word. Cues are stripped from spoken text everywhere; a line that is nothing
but a cue becomes the sound itself ("Hehe.") and counts as that reaction.
"""

from __future__ import annotations

import re

SPOKEN = {"laugh": "Haha.", "chuckle": "Hehe.", "sigh": "Pff.", "hm": "Hm.", "ja": "Ja.", "oh": "Oh.",
          "wacht": "Wacht.", "precies": "Precies."}

# Description words, never speech. Speakable sounds ("haha", "hm", "ja") are deliberately absent.
CUE_WORDS: dict[str, str] = {}
for _label, _words in {
    "laugh": "laugh laughs laughing laughter lacht lachen lach lachend schatert gelach",
    "chuckle": "chuckle chuckles chuckling giggle giggles grinnik grinnikt grinniken giechelt gniffelt",
    "sigh": "sigh sighs sighing zucht zuchten zuchtend",
    "hm": "hums mumbles mompelt bromt",
}.items():
    for _w in _words.split():
        CUE_WORDS[_w] = _label

_BRACKETED = re.compile(r"[\(\[\*]\s*([^()\[\]*]{1,40}?)\s*[\)\]\*]")
_WORD = re.compile(r"[^\W\d_]+")


def _cue_label(fragment: str) -> str | None:
    words = [w.casefold() for w in _WORD.findall(fragment)]
    for w in words:
        if w in CUE_WORDS:
            return CUE_WORDS[w]
    return None


def _is_bare_cue(text: str) -> bool:
    words = [w.casefold() for w in _WORD.findall(text)]
    return 0 < len(words) <= 2 and all(w in CUE_WORDS or w in ("zacht", "even", "kort", "softly") for w in words) \
        and any(w in CUE_WORDS for w in words)


def cue_reaction(text: str) -> str | None:
    """The reaction a line stands for when its whole text is a stage cue, else None."""
    stripped = text.strip()
    if not stripped:
        return None
    if not _BRACKETED.sub("", stripped).strip(" .,!?…-—"):
        return next((label for m in _BRACKETED.finditer(stripped) if (label := _cue_label(m.group(1)))), None)
    if _is_bare_cue(stripped):
        return _cue_label(stripped)
    return None


def strip_cues(text: str) -> str:
    """Remove bracketed/starred stage cues from inside a line; tidy the spacing they leave behind."""
    out = _BRACKETED.sub(" ", text)
    out = re.sub(r"\s+([,.!?…])", r"\1", out)
    return " ".join(out.split()).strip(" ,")


def speakable(text: str, reaction: str | None = None) -> str:
    """What a voice should actually say for this text."""
    whole = cue_reaction(text)
    if whole or (reaction and not strip_cues(text).strip(" .,!?…-—")):
        return SPOKEN[whole or reaction]  # type: ignore[index]
    cleaned = strip_cues(text)
    return cleaned or text.strip()


# Words a pure reaction consists of: laughs, hums and one-word acknowledgements.
_REACTION_WORD = re.compile(
    r"^(h?a(ha)+h?|he(he)+|ha+|he|h+m+|m+h?m+|ja+|jawel|o+h*|oké|ok|okay|wacht|even|precies|pf+|zucht|nou|tja|a+h+|hè|hé)$")


def pure_reaction(line) -> str | None:
    """The reaction a line is, when it is nothing but a reaction sound ("Haha.", "Hm-hm, ja.", "(lacht)").

    Only such a line may be swapped for a reaction-bank clip. A full sentence that carries a reaction
    label ("Haha, dat is goed gevonden!") keeps its words: the label is ignored for it.
    """
    cue = cue_reaction(line.text)
    if cue:
        return cue
    label = getattr(line, "reaction", None)
    if not label:
        return None
    words = [w.casefold() for w in _WORD.findall(strip_cues(line.text))]
    if 0 < len(words) <= 4 and all(_REACTION_WORD.match(w) for w in words):
        return label
    return None
