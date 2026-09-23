"""Reaction bank: curated short clips (laughs, sighs, hm, ja, ...) per speaker.

Chatterbox Multilingual has no paralinguistic tags, and one-word generations are
unstable and can't be WER-verified, so reactions come from a folder you curate
once: ``cast/reactions/<speaker>/<label>/*.wav``. ``generate_candidates`` fills a
separate ``_candidates`` folder to pick from; real recorded clips can be dropped
into the bank directly.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.audio.synth import AudioClip, Synth, VoiceSpec
from pipeline.models import Line
from pipeline.performance import REACTIONS, seeded_fraction

SPOKEN = {"laugh": "Haha.", "chuckle": "Hehe.", "sigh": "Pff.", "hm": "Hm.", "ja": "Ja.", "oh": "Oh.",
          "wacht": "Wacht.", "precies": "Precies."}
# Chatterbox Multilingual crashes on texts of about five tokens or fewer (its alignment analyzer
# slices off the last five text positions), so every candidate is long enough to be safe.
CANDIDATE_TEXTS = {
    "laugh": ["Hahaha, haha!", "Haha, ha ha ha.", "Hahahaha, nee."],
    "chuckle": ["Hehe, hehe.", "Hm-hm, hehe."],
    "sigh": ["Pfff... nou ja.", "Haaah, oké dan."],
    "hm": ["Hmm, hmm hmm.", "Hmm... tja, hm."],
    "ja": ["Ja, ja ja.", "Jaa, dat klopt."],
    "oh": ["Oh! Oh, oké.", "Ooh, oh ja."],
    "wacht": ["Wacht even.", "Wacht, wacht even."],
    "precies": ["Precies, ja!", "Ja, precies dat."],
}
CANDIDATE_EXAGGERATIONS = (0.4, 0.55, 0.7)


def speakable_text(line: Line) -> str:
    """A reaction written as a stage cue ("(lacht)", "[zucht]") becomes something a voice can say."""
    text = line.text.strip()
    if line.reaction and text[:1] in "([*":
        return SPOKEN[line.reaction]
    return text


class ReactionBank:
    def __init__(self, cast_dir: Path | str):
        self.root = Path(cast_dir) / "reactions"

    def clips(self, speaker: str, label: str) -> list[Path]:
        folder = self.root / speaker / label
        return sorted(folder.glob("*.wav")) if folder.is_dir() else []

    def pick(self, speaker: str, label: str, *seed_parts: object) -> Path | None:
        options = self.clips(speaker, label)
        if not options:
            return None
        return options[int(seeded_fraction(speaker, label, *seed_parts) * len(options)) % len(options)]

    def coverage(self, speakers: list[str]) -> dict[str, dict[str, int]]:
        return {s: {label: len(self.clips(s, label)) for label in REACTIONS} for s in speakers}

    def candidates_dir(self, speaker: str, label: str) -> Path:
        return self.root / "_candidates" / speaker / label


def generate_candidates(synth: Synth, voice: VoiceSpec, label: str, out_dir: Path, n: int = 12) -> tuple[list[Path], list[str]]:
    """Render ``n`` candidates over text variants x exaggerations x seeds for you to listen to and curate."""
    if label not in CANDIDATE_TEXTS:
        raise ValueError(f"unknown reaction {label}; choose from {', '.join(REACTIONS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = CANDIDATE_TEXTS[label]
    paths: list[Path] = []
    failures: list[str] = []
    for i in range(n):
        text = texts[i % len(texts)]
        exag = CANDIDATE_EXAGGERATIONS[(i // len(texts)) % len(CANDIDATE_EXAGGERATIONS)]
        try:
            clip: AudioClip = synth.render(text, voice, exaggeration=exag, seed=100 + i)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not end the batch
            failures.append(f"{text!r} (seed {100 + i}): {type(exc).__name__}: {exc}")
            continue
        path = out_dir / f"{voice.speaker_id}_{label}_{i:02d}_exag{exag:.2f}.wav"
        clip.write(path)
        paths.append(path)
    return paths, failures
