"""Side-by-side report of two content plans for one chapter (``studiepodcast plan --compare``)."""

from __future__ import annotations

from pipeline.models import Chapter, ContentPlan


def _quotes(plan: ContentPlan) -> tuple[int, int, int]:
    spans = [c.source_span for c in plan.key_claims] + [d.source_span for d in plan.definitions]
    exact = sum(1 for s in spans if (s.match_score or 0) >= 99)
    missing = sum(1 for s in spans if (s.match_score or 0) == 0)
    return exact, len(spans) - exact - missing, missing


def compare_report(chapter: Chapter, plans: dict[str, tuple[ContentPlan, float]]) -> str:
    modes = list(plans)
    out = [f"# Plans for {chapter.id}: {chapter.title}", "",
           "| | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]

    def row(label: str, values: list[object]) -> None:
        out.append(f"| {label} | " + " | ".join(str(v) for v in values) + " |")

    content = [s for s in chapter.sections if len(s.text.strip()) >= 400]
    row("time", [f"{plans[m][1] / 60:.1f} min" for m in modes])
    row("claims", [len(plans[m][0].key_claims) for m in modes])
    row("sections covered (of those with content)",
        [f"{len({c.source_span.section for c in plans[m][0].key_claims} & {s.id for s in content})}/{len(content)}" for m in modes])
    row("quotes exact / fuzzy / not in source", [" / ".join(str(n) for n in _quotes(plans[m][0])) for m in modes])
    row("definitions", [len(plans[m][0].definitions) for m in modes])
    row("misconceptions", [len(plans[m][0].misconceptions) for m in modes])
    row("worked example", ["yes" if plans[m][0].worked_example else "no" for m in modes])
    row("hardest claim (difficulty)", [max((c.difficulty for c in plans[m][0].key_claims), default="-") for m in modes])
    row("warnings", [len(plans[m][0].warnings) for m in modes])

    out += ["", "## Claims per section", "", "| section | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for s in chapter.sections:
        counts = [sum(1 for c in plans[m][0].key_claims if c.source_span.section == s.id) for m in modes]
        out.append(f"| {s.id} {s.title[:40]} | " + " | ".join(str(n) for n in counts) + " |")

    for m in modes:
        plan = plans[m][0]
        out += ["", f"## {m}", "", plan.summary, "", "Learning objectives:"]
        out += [f"- {o}" for o in plan.learning_objectives]
        out += ["", "Claims:"]
        out += [f"- `{c.source_span.section}` (d{c.difficulty}, e{c.exam_relevance}"
                + ("" if (c.source_span.match_score or 0) else ", quote not in source") + f") {c.claim}"
                for c in plan.key_claims]
        if plan.warnings:
            out += ["", "Warnings:"] + [f"- {w}" for w in plan.warnings]
    return "\n".join(out) + "\n"
