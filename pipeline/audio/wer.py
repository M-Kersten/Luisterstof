"""Word error rate between a script line and its transcript, after normalisation."""

from __future__ import annotations

import re
import unicodedata

from pipeline.models import Glossary
from pipeline.plan.glossary import canonicalise

_NUM_WORDS = {
    "nul": "0", "een": "1", "één": "1", "twee": "2", "drie": "3", "vier": "4", "vijf": "5", "zes": "6",
    "zeven": "7", "acht": "8", "negen": "9", "tien": "10", "elf": "11", "twaalf": "12", "twintig": "20",
    "dertig": "30", "veertig": "40", "vijftig": "50", "honderd": "100", "duizend": "1000",
}


def normalise(text: str, glossary: Glossary | None = None) -> list[str]:
    if glossary is not None:
        text = canonicalise(text, glossary)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = text.replace("—", " ").replace("…", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return [_NUM_WORDS.get(t, t) for t in tokens if t not in ("eh", "uh", "hm", "mm")]


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference_text: str, hypothesis_text: str, glossary: Glossary | None = None) -> float:
    return wer(normalise(reference_text, glossary), normalise(hypothesis_text, glossary))
