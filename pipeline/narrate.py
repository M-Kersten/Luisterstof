"""What the pipeline is doing, in plain Dutch.

Every event the pipeline emits (stage, status, data) gets a sentence here, plus a kind that says how much it
matters to someone watching:

- ``step``: something new starts (a chapter, the plan, writing, a check).
- ``result``: something finished and this is what came out of it (15 claims, 60 lines, 4 problems).
- ``problem``: something went wrong or needs you.
- ``detail``: progress inside a step (block 3 of 7, a question to the model). Shown live, folded away later.

The web app shows these in the progress banner and on the Meekijken page; the CLI prints them. The module is
pure: no files, no clock, so the same event always reads the same, now or the morning after.
"""

from __future__ import annotations

from typing import Any

# What the writing model is working on, by request task.
TASKS = {
    "structure": "de hoofdstukken in het boek herkennen",
    "plan_section": "een stuk van het hoofdstuk lezen en noteren wat erin staat",
    "plan_merge": "de gevonden beweringen kiezen en samenvoegen",
    "plan_review": "het plan nalopen tegen het boek",
    "plan_quotes": "bij elke bewering de zin uit het boek zoeken",
    "lexicon": "de uitspraak van namen en vaktermen bepalen",
    "script_outline": "de rode draad van de aflevering bepalen",
    "script_segment": "een blok van het gesprek schrijven",
    "script_scene": "een losse scène schrijven",
    "script_polish": "een blok herschrijven zodat het klinkt als een gesprek",
    "support": "nagaan of een regel klopt met het boek",
    "lint": "de regels van het gesprek nalopen",
    "listener": "luisteren alsof je het voor het eerst hoort",
    "landing": "nagaan of de moeilijkste punten echt zijn uitgelegd",
    "continuity": "onthouden wat er in deze aflevering gebeurde",
}

SKIPPED_STEPS = {
    "plan": "Het plan", "glossary": "De uitspraaklijst", "script": "Het script",
    "approve": "De goedkeuring", "render": "De opname",
}


def duration(seconds: float | int | None) -> str:
    """3 s, 2 min 05 s, 1 u 12 min."""
    s = max(0, int(round(seconds or 0)))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h} u {m:02d} min"
    if m:
        return f"{m} min {sec:02d} s"
    return f"{sec} s"


def _number(value: float) -> str:
    """1,9 and 15: Dutch decimals, no trailing zero."""
    return f"{value:g}".replace(".", ",")


def task_text(task: str | None) -> str:
    return TASKS.get(task or "", "een vraag beantwoorden")


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _chapter_label(data: dict[str, Any]) -> str:
    label = str(data.get("label") or "")
    chapter = str(data.get("chapter") or "")
    # The book loop labels a chapter "ch03 De titel"; the title alone reads better.
    return label[len(chapter):].strip() if chapter and label.startswith(chapter) else label


def say(stage: str, status: str, data: dict[str, Any] | None = None) -> dict[str, str] | None:
    """The sentence for one event, or None for events nobody needs to read."""
    d = data or {}
    text, kind = _say(stage, status, d)
    if not text:
        return None
    return {"text": text, "kind": kind}


def _say(stage: str, status: str, d: dict[str, Any]) -> tuple[str, str]:  # noqa: C901, PLR0911, PLR0912 - one table
    idx, total, label = d.get("index"), d.get("total"), d.get("label") or ""
    of = f"{idx} van {total}" if idx and total else ""

    # ------------------------------------------------------------ the run itself
    if stage == "book" and status == "chapter":
        title = _chapter_label(d)
        return f"Hoofdstuk {of}: {title}." if of else f"Hoofdstuk {title}.", "step"
    if stage == "book" and status == "chapter_failed":
        error = str(d.get("error") or "onbekende fout")[:300]
        return f"Dit hoofdstuk liep vast ({error}). De run gaat door met het volgende hoofdstuk.", "problem"
    if stage == "run" and status == "step":
        return "", ""  # the stage's own start event says it better
    if stage == "job" and status == "stopping":
        return "Stoppen gevraagd. De run stopt bij het volgende controlepunt; wat klaar is, blijft bewaard.", "problem"
    if status == "skipped":
        what = SKIPPED_STEPS.get(stage, stage.capitalize())
        return f"{what} was al klaar, dus dat slaat Luisterstof over.", "detail"

    # ------------------------------------------------------------ the writing model
    if stage == "llm":
        if status == "check":
            return "Kijken of het schrijfmodel antwoordt.", "detail"
        if status == "ready":
            return f"Het schrijfmodel antwoordt (na {duration(d.get('seconds'))}).", "detail"
        if status == "wake":
            return str(d.get("message") or "Het schrijfmodel wordt gestart."), "detail"
        if status == "call":
            typical = d.get("typical")
            usual = f" Dat duurt meestal ongeveer {duration(typical)}." if typical else ""
            return f"Het schrijfmodel is bezig: {d.get('what') or task_text(d.get('task'))}.{usual}", "detail"
        if status == "tokens":
            return (f"Antwoord binnen na {duration(d.get('seconds'))}: ongeveer {d.get('tokens', 0)} tokens "
                    f"({task_text(d.get('task'))})."), "detail"
        return "", ""

    # ------------------------------------------------------------ reading the book
    if stage == "ingest":
        if status == "start":
            return "Het boek inlezen: de tekst uit de PDF halen en de hoofdstukken vinden.", "step"
        if status == "done":
            warnings = d.get("warnings") or []
            extra = f" {_n(len(warnings), 'opmerking', 'opmerkingen')} over wat er is weggelaten." if warnings else ""
            return f"Ingelezen: {_n(int(d.get('chapters') or 0), 'hoofdstuk', 'hoofdstukken')} gevonden.{extra}", "result"
        return "", ""

    # ------------------------------------------------------------ the plan
    if stage == "plan":
        if status == "start":
            return "Het plan maken: Luisterstof leest het hoofdstuk stuk voor stuk en noteert wat erin moet.", "step"
        if status == "section":
            return f"Lezen, deel {of}{': ' + label if label else ''}.", "detail"
        if status == "merge":
            return (f"Uit {_n(int(d.get('candidates') or 0), 'gevonden bewering', 'gevonden beweringen')} kiezen "
                    "wat in de aflevering komt."), "detail"
        if status == "review":
            return (f"Het plan nalopen tegen het boek, ronde {d.get('round')}: "
                    f"{_n(int(d.get('claims') or 0), 'bewering', 'beweringen')}."), "detail"
        if status == "done":
            claims = int(d.get("claims") or 0)
            guest = " Er komt een gast bij die er verstand van heeft." if d.get("needs_expert") else ""
            warnings = d.get("warnings") or []
            note = f" {_n(len(warnings), 'opmerking', 'opmerkingen')} bij het plan." if warnings else ""
            return f"Plan klaar: {_n(claims, 'bewering', 'beweringen')} om uit te leggen.{guest}{note}", "result"
        return "", ""

    # ------------------------------------------------------------ pronunciation
    if stage == "glossary":
        if status == "start":
            return "De uitspraaklijst bijwerken: namen, afkortingen en vaktermen.", "step"
        if status == "lexicon_section":
            return f"Uitspraak zoeken, deel {of}.", "detail"
        if status == "done":
            added, total_n = int(d.get("added") or 0), int(d.get("total") or 0)
            return (f"Uitspraaklijst bijgewerkt: {added} nieuw, {_n(total_n, 'woord', 'woorden')} in totaal."), "result"
        return "", ""

    # ------------------------------------------------------------ the script
    if stage == "script":
        if status == "start":
            guest = ", met een gast erbij" if d.get("guest") else ""
            return f"Het gesprek schrijven{guest}, blok voor blok.", "step"
        if status == "segment":
            if label.startswith("tafellezing: "):
                return f"Tafellezing, blok {of}: {label[len('tafellezing: '):]}.", "detail"
            return f"Blok {of} schrijven: {label}.", "detail"
        if status == "written":
            how = "herschreven" if d.get("polish") else "staat er"
            return f"Blok {of} {how}: {_n(int(d.get('lines') or 0), 'regel', 'regels')}.", "detail"
        if status == "done":
            minutes = d.get("estimated_minutes")
            length = f", ongeveer {_number(minutes)} minuten" if isinstance(minutes, (int, float)) else ""
            return f"Eerste versie van het script klaar: {_n(int(d.get('lines') or 0), 'regel', 'regels')}{length}.", "result"
        if status == "revise":
            blocking, shape = int(d.get("blocking") or 0), int(d.get("naturalness") or 0)
            what = [p for p in (_n(blocking, "probleem", "problemen") if blocking else "",
                                _n(shape, "opmerking over hoe het klinkt", "opmerkingen over hoe het klinkt") if shape else "") if p]
            return (f"Herschrijfronde {d.get('round')}: {' en '.join(what) or 'de opmerkingen'} gaan terug naar de "
                    "schrijver."), "step"
        if status == "polish":
            return "Tafellezing: elk blok nog een keer herschrijven, zodat het klinkt als een echt gesprek.", "step"
        if status == "polished":
            left = int(d.get("blocking") or 0)
            return (f"Tafellezing klaar, {_n(left, 'probleem', 'problemen')} over." if left
                    else "Tafellezing klaar, zonder problemen."), "result"
        if status == "kept_best":
            return (f"De laatste herschrijving was niet beter ({d.get('last_blocking')} problemen), dus de versie met "
                    f"{_n(int(d.get('blocking') or 0), 'probleem', 'problemen')} blijft staan."), "result"
        return "", ""

    # ------------------------------------------------------------ the check against the book
    if stage == "audit":
        if status == "start":
            revision = int(d.get("revision") or 0)
            if revision > 1:
                return f"Versie {revision} van het script controleren tegen het boek.", "step"
            return "Het script controleren tegen het boek: klopt elke bewering, komt alles aan bod?", "step"
        if status == "segment":
            kind, _, title = label.partition(": ")
            if title:  # "bronnencontrole: Koude opening"
                what = {"bronnencontrole": "bronnen"}.get(kind, kind)
                return f"Controle ({what}), blok {of}: {title}.", "detail"
            return f"Controle, blok {of}{': ' + label if label else ''}.", "detail"
        if status == "resume":
            return "Het script is veranderd sinds de laatste controle, dus het wordt opnieuw gecontroleerd.", "step"
        if status == "done":
            blocking, warnings = int(d.get("blocking") or 0), int(d.get("warnings") or 0)
            missing = d.get("coverage_missing") or []
            gaps = (f" {_n(len(missing), 'bewering', 'beweringen')} uit het plan {'komt' if len(missing) == 1 else 'komen'} "
                    "nog niet aan bod.") if missing else ""
            tips = f" en {_n(warnings, 'tip', 'tips')}" if warnings else ""
            if d.get("passed"):
                return f"Controle geslaagd{tips}.{gaps}", "result"
            return f"Controle vond {_n(blocking, 'probleem', 'problemen')}{tips}.{gaps}", "problem"
        return "", ""

    # ------------------------------------------------------------ a take in the voice lab
    if stage == "voice":
        if status == "load":
            return "Het stemmodel laden. Dat gebeurt bij de eerste take en duurt een minuut of twee.", "step"
        if status == "loaded":
            if str(d.get("device", "")).startswith("cpu"):
                why = f": {d['why']}" if d.get("why") else ""
                return (f"Stemmodel geladen in {duration(d.get('seconds'))}, maar op de processor{why}. Een take duurt "
                        "zo vele malen langer dan op de GPU; 'studiepodcast doctor' zegt hoe je dat oplost."), "problem"
            return f"Stemmodel geladen op de GPU in {duration(d.get('seconds'))}.", "result"
        if status == "start":
            return f"{d.get('name') or 'De stem'} spreekt de zin in.", "step"
        if status == "made":
            return f"Take gemaakt in {duration(d.get('seconds'))}: {duration(d.get('audio_s'))} audio. Hij is bewaard.", "result"
        if status == "check_device":
            if d.get("reason"):
                return (f"Whisper luistert op de processor terug: {d['reason']}. Dat duurt langer; "
                        "'studiepodcast doctor' zegt hoe je het oplost."), "problem"
            return (f"De GPU heeft nog maar {_number(float(d.get('free_gb') or 0))} GB vrij, dus Whisper luistert "
                    "op de processor terug. Dat duurt langer."), "detail"
        if status == "check":
            where = {"cpu": " op de processor", "cuda": " op de GPU"}.get(str(d.get("device")), "")
            what = "Whisper laden en terugluisteren" if d.get("loading") else "Terugluisteren"
            return f"{what}{where}: verstaat Whisper wat er gezegd moest worden?", "step"
        if status == "checked":
            wer = float(d.get("wer") or 0)
            verdict = "goed verstaan" if wer <= 0.05 else "te veel verkeerd verstaan voor een opname"
            return f"Teruggeluisterd in {duration(d.get('seconds'))}: {round(wer * 100)}% verkeerd verstaan, {verdict}.", "result"
        if status == "check_skipped":
            return f"Niet teruggeluisterd. {d.get('reason') or ''}".strip(), "problem"
        return "", ""

    if stage == "continuity" and status == "done":
        callbacks = int(d.get("callbacks") or 0)
        return (f"Onthouden voor de volgende afleveringen: {_n(callbacks, 'terugverwijzing', 'terugverwijzingen')}."
                if callbacks else "Onthouden wat er in deze aflevering gebeurde."), "detail"
    if stage == "approve" and status == "done":
        return ("Goedgekeurd, ook al vond de controle nog problemen." if d.get("forced") else "Goedgekeurd."), "result"

    # ------------------------------------------------------------ recording
    if stage == "render":
        if status == "start":
            return "Opnemen: de stemmen spreken het script in, beurt voor beurt.", "step"
        if status == "note":
            return str(d.get("message") or ""), "problem"
        if status == "render_start":
            return f"{_n(int(d.get('turns') or 0), 'beurt', 'beurten')} om op te nemen.", "detail"
        if status == "turn":
            who = f" ({d.get('speaker')})" if d.get("speaker") else ""
            how = ", uit de cache" if d.get("cached") else ", opnieuw te beluisteren" if d.get("flagged") else ""
            return f"Beurt {of} opgenomen{who}{how}.", "detail"
        if status == "take":
            wer = d.get("wer")
            heard = f"{round(float(wer) * 100)}% verkeerd verstaan" if isinstance(wer, (int, float)) else "teruggeluisterd"
            verdict = "goed" if d.get("accepted") else "nog een poging"
            return f"Take van beurt {d.get('turn')}: {heard}, {verdict}.", "detail"
        if status == "render_done":
            flagged = int(d.get("flagged") or 0)
            check = f" {_n(flagged, 'beurt', 'beurten')} om na te luisteren." if flagged else ""
            return f"Opname klaar: {duration(d.get('duration_s'))}.{check}", "result"
        return "", ""

    # ------------------------------------------------------------ how long it took
    if stage == "timing":
        if d.get("pass_name") != "chapter":
            return "", ""  # each pass already has its own result sentence
        if status == "done":
            return f"Hoofdstuk klaar na {duration(d.get('seconds'))}.", "result"
        if status == "failed":
            return f"Hoofdstuk mislukt na {duration(d.get('seconds'))}.", "problem"
        if status == "stopped":
            return f"Gestopt na {duration(d.get('seconds'))}.", "problem"
        return "", ""

    return "", ""


def entry(event: dict[str, Any]) -> dict[str, Any] | None:
    """A stored job event as a line for the Meekijken page, or None when there is nothing to say."""
    data = event.get("data") or {}
    said = say(event.get("stage", ""), event.get("status", ""), data)
    if not said:
        return None
    out = {"seq": event.get("seq"), "ts": event.get("ts"), "stage": event.get("stage"), "status": event.get("status"),
           "chapter": data.get("chapter"), **said}
    if data.get("preview"):
        out["preview"] = data["preview"]
    for key in ("index", "total", "task", "what", "typical"):
        if data.get(key) is not None:
            out[key] = data[key]
    return out
