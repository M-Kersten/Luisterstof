"""After an episode: extract what the next episode can call back to."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.llm import LLM, LLMRequest
from pipeline.models import Cast, ContinuityEntry, Script


class ContinuityOut(BaseModel):
    callbacks: list[str] = Field(description="Concrete momenten (analogieën, formuleringen, ruzies) die een volgende aflevering kan terughalen. Elk één zin met de namen erin.")
    running_jokes: list[str] = Field(description="Grappen die herhaalbaar zijn omdat ze aan de stof of aan een host hangen.")
    mistakes: list[str] = Field(description="Fouten die een host maakte en de correctie, één zin per fout.")
    open_threads: list[str] = Field(description="Beloften of vragen die bewust open bleven.")


CONTINUITY_SYSTEM = """Je houdt het continuïteitslogboek bij van een Nederlandse studiepodcast met een vaste cast.
Je krijgt het script van een aflevering. Noteer wat een volgende aflevering kan terughalen, zo concreet dat een schrijver die het script niet heeft gelezen het kan gebruiken. Per categorie maximaal zes items, korter is beter. Alles in het Nederlands."""


def script_as_text(script: Script) -> str:
    lines = []
    for seg in script.segments:
        lines.append(f"## {seg.type}")
        for line in seg.lines:
            lines.append(f"[{line.id}] {line.speaker}: {line.text}")
    return "\n".join(lines)


def extract_continuity(script: Script, cast: Cast, llm: LLM, previous: list[ContinuityEntry] | None = None) -> ContinuityEntry:
    names = ", ".join(f"{h.id} = {h.name}" for h in cast.hosts)
    prev = ""
    if previous:
        prev = "\n\nAl bekende running jokes (niet opnieuw noteren, wel noteren als ze terugkwamen):\n" + "\n".join(
            f"- {j}" for e in previous for j in e.running_jokes
        )
    request = LLMRequest(
        task="continuity",
        system=CONTINUITY_SYSTEM + f"\nSprekers: {names}." + prev,
        user=script_as_text(script),
        schema=ContinuityOut,
        effort="medium",
        max_tokens=4000,
        cache_system=False,
    )
    out = llm.generate(request)
    assert isinstance(out, ContinuityOut)
    return ContinuityEntry(
        episode=script.episode_id,
        callbacks=out.callbacks[:6],
        running_jokes=out.running_jokes[:6],
        mistakes=out.mistakes[:6],
        open_threads=out.open_threads[:6],
        guest=script.guest_id,
    )
