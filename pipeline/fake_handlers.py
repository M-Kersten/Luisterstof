"""A FakeLLM that produces plausible, deterministic output for every task.

Used by tests and by ``--fake-llm`` so the whole pipeline can be exercised
without an API key: the text is dull, the structure is right, and every
claim's quote is verbatim so spans resolve and the support check passes.
"""

from __future__ import annotations

import re

from pipeline.llm import FakeLLM, LLMRequest

_SECTION_RE = re.compile(r"^### Sectie (?P<id>\S+): (?P<title>.*)$", re.MULTILINE)
_CLAIM_RE = re.compile(r"\[(?P<id>c\d+)\] \(moeilijkheid (?P<d>\d), tentamen (?P<e>\d)\) (?P<text>.+)")
_PAGE_RE = re.compile(r"=== pagina (\d+) ===")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def fake_structure(req: LLMRequest) -> dict:
    text = _text_of(req.user)
    chapters = []
    page = 1
    for line in text.splitlines():
        m = _PAGE_RE.search(line)
        if m:
            page = int(m.group(1))
            continue
        h = re.match(r"^(\d{1,2})\s+([A-Z].{2,60})$", line.strip())
        if h:
            chapters.append({"title": h.group(2).strip(), "number": h.group(1), "printed_page": page})
    return {"has_printed_toc": False, "chapters": chapters}


def fake_figure_caption(req: LLMRequest) -> dict:
    return {"carries_meaning": True, "description": "De figuur toont een grafiek met een stijgende lijn."}


def fake_plan(req: LLMRequest) -> dict:
    system = _text_of(req.system)
    sections = list(_SECTION_RE.finditer(system))
    claims, definitions, formula_dense = [], [], []
    for i, m in enumerate(sections):
        start = m.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(system)
        body = system[start:end].strip()
        sents = _sentences(body)
        for j, sent in enumerate(sents[:3]):
            quote = sent[:200]
            claims.append({
                "claim": sent,
                "section": m.group("id"),
                "source_quote": quote,
                "difficulty": 2 + (j % 3),
                "exam_relevance": 5 - (j % 3),
            })
        if sents:
            definitions.append({"term": m.group("title"), "definition": sents[0], "section": m.group("id"),
                                "source_quote": sents[0][:120]})
        if "=" in body:
            formula_dense.append(m.group("id"))
    return {
        "summary": "Dit hoofdstuk behandelt de kernbegrippen uit de secties, met definities en voorbeelden uit de bron.",
        "learning_objectives": ["De student kan de kernbegrippen uitleggen.", "De student kan de voorbeelden narekenen.", "De student herkent de veelgemaakte fouten."],
        "key_claims": claims,
        "definitions": definitions,
        "misconceptions": [
            {"wrong": "Het begrip geldt altijd, ongeacht de voorwaarden.", "right": "Het begrip geldt alleen onder de voorwaarden uit de bron.", "why_tempting": "De voorwaarden staan in een bijzin."},
            {"wrong": "Twee verwante begrippen zijn hetzelfde.", "right": "Ze verschillen in richting en betekenis.", "why_tempting": "Ze lijken op elkaar in notatie."},
        ],
        "worked_example": {"setup": "Neem het voorbeeld uit de bron.", "steps": ["Schrijf de gegevens op.", "Pas de regel toe."], "answer": "Het antwoord volgt uit de bron."},
        "expert_domain": None,
        "expert_reason": None,
        "formula_dense_sections": formula_dense,
    }


def fake_lexicon(req: LLMRequest) -> dict:
    user = _text_of(req.user)
    entries = []
    m = re.search(r"acronyms: (.+)", user)
    if m:
        for acro in m.group(1).split(", ")[:5]:
            acro = acro.strip()
            if acro:
                entries.append({"surface": acro, "kind": "abbreviation", "spoken": " ".join(acro), "note": None})
    m = re.search(r"notation: (.+)", user)
    if m:
        for item in m.group(1).split(", ")[:8]:
            item = item.strip()
            if "/" in item and re.fullmatch(r"\d+/\d+", item):
                a, b = item.split("/")
                entries.append({"surface": item, "kind": "notation", "spoken": f"{a} gedeeld door {b}", "note": None})
            elif re.fullmatch(r"[A-Za-z]\([A-Za-z|,\s]+\)", item):
                inner = item[2:-1].replace("|", " gegeven ")
                entries.append({"surface": item, "kind": "notation", "spoken": f"{item[0]} van {inner}", "note": None})
    return {"entries": entries}


def fake_script_segment(req: LLMRequest) -> dict:
    user = _text_of(req.user)
    seg_type = req.metadata.get("segment", "body")
    speakers_m = re.search(r"Toegestane sprekers: (.+?)\.", user)
    speakers = [s.strip() for s in speakers_m.group(1).split(",")] if speakers_m else ["tessa", "joris"]
    explainer, skeptic = speakers[0], speakers[1] if len(speakers) > 1 else speakers[0]
    guest = speakers[2] if len(speakers) > 2 else None
    claims = [(m.group("id"), m.group("text").strip()) for m in _CLAIM_RE.finditer(user)]
    prev_claim = re.search(r"Het kernpunt: (.+)", user)
    lines: list[dict] = []

    def line(speaker, text, covers=None, tags=None, overlap="none", pause=0):
        lines.append({"speaker": speaker, "text": text, "tags": tags or [], "covers": covers or [],
                      "overlap": overlap, "pause_after_ms": pause})

    if seg_type == "cold_open":
        line(skeptic, "Nee, dat accepteer ik niet. Je zegt het alsof het vanzelf spreekt, en dat doet het dus niet.", tags=["skeptical"])
        line(explainer, "Het spreekt ook vanzelf, als je eenmaal ziet hoe de stukjes in elkaar—", tags=["excited"])
        line(skeptic, "Wacht even. Welke stukjes?", overlap="interrupt")
        if claims:
            line(explainer, claims[0][1], covers=[claims[0][0]])
        line(skeptic, "Oké. Bewijs het.", tags=["deadpan"])
    elif seg_type == "recap":
        line(skeptic, "Eerst jij, luisteraar. Wat was vorige keer het punt waar Tessa zo van in haar sas was?", pause=2500)
        if prev_claim:
            cov = re.search(r"covers: \['([^']+)'\]", user)
            line(explainer, prev_claim.group(1).strip(), covers=[cov.group(1)] if cov else [])
        line(skeptic, "Ja. Dat.", tags=["deadpan"])
    elif seg_type == "quiz":
        for i, (cid, text) in enumerate(claims[:4]):
            line(skeptic, f"Vraag {i + 1}. Wat zegt het boek hierover: {text.split(',')[0].rstrip('.')}?", pause=3000)
            if i == 1:
                line(explainer, "Volgens mij het omgekeerde.", tags=["thinking"])
                line(skeptic, f"Fout. {text}", covers=[cid])
            else:
                line(explainer, text, covers=[cid])
                line(skeptic, "Klopt.", tags=["deadpan"])
    elif seg_type == "outro":
        line(explainer, "En volgende keer ga ik je een beeld geven waar je niet meer vanaf komt.")
        line(skeptic, "Dat zei je vorige keer ook.", tags=["deadpan"])
    elif seg_type == "reexplain":
        line(skeptic, "Nee. Opnieuw. Andere route, zonder dat beeld.")
        for cid, text in claims[:1]:
            line(explainer, "Dan andersom, vanuit het voorbeeld.", tags=["thinking"])
            line(explainer, text, covers=[cid])
        line(skeptic, "Zo had je moeten beginnen.")
    else:  # body, guest
        for i, (cid, text) in enumerate(claims):
            asker = skeptic
            answerer = guest if (guest and i % 2 == 0) else explainer
            line(asker, f"En hoe zit dat dan met {text.split(' ')[0].lower()} {text.split(' ')[1] if len(text.split(' ')) > 1 else ''}?".replace("  ", " "))
            line(answerer, text, covers=[cid])
            if i == 0:
                line(answerer, "Dat is de kern ervan, en de rest volgt daaruit.", tags=["warm"])
                line(skeptic, "Ja, precies.", overlap="backchannel")
        if "misvatting" in user.casefold():
            line(skeptic, "Dus dat geldt altijd?", tags=["skeptical"])
            line(explainer, "Nee, alleen onder de voorwaarden die het boek noemt. Daar trapte ik zelf ook in.", tags=["laughs"])
    return {"lines": lines}


def fake_script_scene(req: LLMRequest) -> dict:
    """A prototype scene that exercises every performance moment the coverage check asks for."""
    user = _text_of(req.user)
    speakers_m = re.search(r"Toegestane sprekers: (.+?)\.", user)
    speakers = [s.strip() for s in speakers_m.group(1).split(",")] if speakers_m else ["tessa", "joris"]
    t, j = speakers[0], speakers[1] if len(speakers) > 1 else speakers[0]
    lines: list[dict] = []

    def line(speaker, text, delivery="", mood="", timing="", overlap="none", phrases=None, reaction="", tags=None):
        lines.append({"speaker": speaker, "text": text, "tags": tags or [], "covers": [], "overlap": overlap,
                      "pause_after_ms": 0, "delivery": delivery, "mood": mood, "timing": timing,
                      "phrases": phrases or [], "reaction": reaction})

    line(t, "Kijk, een kans is gewoon een getal tussen nul en een.", "explain", "confident")
    line(j, "Nee, dat is te makkelijk. Een getal waarvan dan?", "disagree", "challenged", "immediate", tags=["skeptical"])
    line(t, "Van hoe vaak iets gebeurt, als je het maar vaak genoeg herhaalt, dan kruipt het naar één vaste—", "explain",
         "confident", "hesitate", phrases=[
             {"text": "Van hoe vaak iets gebeurt,", "delivery": "explain", "pause_after": "short"},
             {"text": "als je het maar vaak genoeg herhaalt, dan kruipt het naar één vaste—", "delivery": "excite",
              "pause_after": "none"}])
    line(j, "Wacht even, kruipt?", "interrupt", "challenged", overlap="interrupt")
    line(t, "Hm, ja. Het komt steeds dichter bij de kans zelf.", "think", "thoughtful", "search")
    line(j, "Ja.", reaction="ja", overlap="backchannel")
    line(j, "Oh. Dus het is geen belofte over één keer gooien.", "realize", "surprised", "deliberate", phrases=[
        {"text": "Oh.", "delivery": "realize", "pause_after": "beat"},
        {"text": "Dus het is geen belofte over één keer gooien.", "delivery": "realize", "pause_after": "none"}])
    line(t, "Precies. En daarom wint het casino altijd, ook als jij die ene avond geluk hebt.", "setup", "amused",
         "immediate")
    line(j, "(lacht)", reaction="laugh", overlap="backchannel")
    line(j, "Dan ga ik voortaan gewoon één keer.", "punchline", "amused", "immediate", tags=["deadpan"])
    return {"lines": lines}


def fake_support(req: LLMRequest) -> dict:
    ids = re.findall(r"### Regel (l\d+)", _text_of(req.user))
    return {"verdicts": [{"line_id": i, "supported": True, "problem": None} for i in ids]}


def fake_lint(req: LLMRequest) -> dict:
    return {"hits": []}


def fake_continuity(req: LLMRequest) -> dict:
    return {
        "callbacks": ["Joris eiste een tweede route voor het moeilijkste begrip en kreeg die."],
        "running_jokes": ["Tessa belooft elke aflevering een beeld waar je niet meer vanaf komt."],
        "mistakes": ["Tessa had in de quiz één antwoord omgekeerd, Joris corrigeerde."],
        "open_threads": ["Joris wil volgende keer weten of het beeld van Tessa ook bij het volgende hoofdstuk werkt."],
    }


def default_fake_llm() -> FakeLLM:
    return FakeLLM({
        "structure": fake_structure,
        "figure_caption": fake_figure_caption,
        "plan": fake_plan,
        "lexicon": fake_lexicon,
        "script_segment": fake_script_segment,
        "script_scene": fake_script_scene,
        "support": fake_support,
        "lint": fake_lint,
        "continuity": fake_continuity,
    })
