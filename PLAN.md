# PLAN.md — Studiepodcast Generator

Build brief. Hand this to Claude Code as the project spec.

## 0. Goal and constraints

Turn a dense study book (PDF) into a Dutch podcast series: one episode per chapter, 20 to 30 minutes, hosted by a stable recurring cast with real banter, humour and occasional expert guests.

Hard constraints:

- **Output language: Dutch.** Source books are Dutch, so no translation layer.
- **Accuracy is non-negotiable.** This is study material. A funny episode that hallucinates is worse than no episode.
- **Synthesis is local-first.** Target hardware is a single 24GB GPU. Unlimited free retakes matter more than peak voice quality, because humour needs many listens before it lands.
- Stack: Python pipeline + FastAPI + small web frontend. Same interaction model as the webrecorder project (upload, watch stages stream, inspect artifacts per stage).

## 1. Repository structure

```
studiepodcast/
  app/
    api.py                 # FastAPI: upload, job status (SSE), artifact browsing, approve-script
    jobs.py                # job queue + state machine
    static/                # frontend
  pipeline/
    ingest/
      pdf.py               # extraction -> book.json
      structure.py         # chapter/section detection
      figures.py           # figure + table captioning
    plan/
      content_plan.py      # stage 2: outline, claims, misconceptions
      glossary.py          # NL/EN terminology decisions
    script/
      cast.py              # loads cast bible + continuity log
      writer.py            # stage 3: dialogue generation
      audit.py             # fact-check + coverage + lint
    audio/
      render_draft.py      # local Piper, free
      render_final.py      # ElevenLabs v3 Text-to-Dialogue
      chunker.py           # 3000-char block splitting at scene seams
      mixer.py             # stitching, ducking, loudness, intro/outro
      cache.py             # content-hash cache so unchanged blocks never re-render
  cast/
    hosts.yaml
    guests.yaml
    continuity.jsonl
  data/
    books/<book_id>/
      source.pdf
      book.json
      glossary.json
      plans/ch03.plan.json
      scripts/ch03.script.json
      audits/ch03.audit.json
      render/ch03/blocks/*.mp3
      out/ch03.mp3
      out/ch03.transcript.json
  tests/
```

## 2. Pipeline stages and data contracts

Every stage writes a file. Every file is inspectable in the web UI. No stage reaches into another stage's internals.

### Stage 1: Ingest

`book.json`:
```json
{
  "book_id": "str",
  "title": "str",
  "language": "en|nl",
  "chapters": [
    {"id": "ch03", "title": "str", "level": 1, "pages": [41, 68],
     "sections": [{"id": "ch03.2", "title": "str", "text": "str"}]}
  ]
}
```

Figures and tables are converted to text captions inline at their position. A listener cannot see a diagram, so a diagram that carries meaning must become a sentence.

Use `pymupdf` for extraction. Chapter detection: font-size heuristics first, TOC parse if the PDF has one, LLM fallback on the first pages when both fail.

### Stage 2: Content plan

Runs per chapter. Contains zero jokes and zero personality. This is the accuracy backbone.

`ch03.plan.json`:
```json
{
  "chapter_id": "ch03",
  "learning_objectives": ["str"],
  "key_claims": [
    {"id": "c1", "claim": "str", "source_span": {"section": "ch03.2", "start": 1200, "end": 1480},
     "difficulty": 1-5, "exam_relevance": 1-5}
  ],
  "definitions": [{"term": "str", "definition": "str", "source_span": {}}],
  "misconceptions": [{"wrong": "str", "right": "str", "why_tempting": "str"}],
  "worked_example": {"setup": "str", "steps": ["str"], "answer": "str"},
  "needs_expert": true,
  "expert_domain": "str|null"
}
```

`needs_expert` fires when a section has a mean difficulty above 4 or introduces a domain neither host is written to carry credibly.

### Stage 2b: Lexicon

With a Dutch source book there is no translation layer. What remains is pronunciation, which still breaks audio if you skip it.

```json
{"surface": "gradient descent", "kind": "loanword_en",
 "spoken": "greedient discent", "lock": true},
{"surface": "ca. 1,5 x 10^3", "kind": "notation",
 "spoken": "ongeveer anderhalf keer tien tot de derde"},
{"surface": "o.a.", "kind": "abbreviation", "spoken": "onder andere"}
```

Three categories, all resolved before render:

- **English loanwords inside Dutch sentences.** Dutch academic writing is full of them. Zero-shot TTS handles them inconsistently: sometimes Dutch phonology, sometimes English, sometimes both within one episode. Lock a spelling per term and reuse it everywhere.
- **Notation.** Formulas, units, exponents, ranges. Written out longhand or the model guesses.
- **Abbreviations.** `bijv.`, `m.b.t.`, `o.a.`, and any field-specific ones.

The lexicon is per book and applies to every episode of that book, so the same term never sounds different in chapter 9 than it did in chapter 2.

### Stage 3: Script

Input: the content plan, the glossary, `cast/hosts.yaml`, and the last 10 entries of `continuity.jsonl`.

`ch03.script.json`:
```json
{
  "episode_id": "ch03",
  "target_minutes": 25,
  "segments": [
    {"type": "cold_open|recap|body|guest|reexplain|quiz|outro",
     "covers": ["c1", "c4"],
     "lines": [
       {"id": "l012", "speaker": "tessa", "text": "str",
        "tags": ["excited"], "covers": ["c1"],
        "overlap": {"mode": "interrupt|backchannel|none", "target": "l011"}}
     ]}
  ]
}
```

### Stage 3b: Audit

Three checks, all blocking:

1. **Support.** Every line marked with `covers` is verified against its claim's `source_span`. Unsupported factual assertions are flagged with line id.
2. **Coverage.** Every `key_claim` with `exam_relevance >= 3` appears in at least one line. Missing claims are listed.
3. **Lint.** Banned-phrase list, zero tolerance. Includes AI-podcast filler: "dat is echt fascinerend", "goede vraag", "laten we eens duiken in", "aan het eind van de dag", any three-part list used as a rhetorical flourish, any sentence that summarises the segment that just happened. Also flags a host using another host's verbal tics.

`ch03.audit.json` holds the results. The UI shows them next to the script. The script cannot go to premium render with open blocking issues.

## 3. Cast

`cast/hosts.yaml`. Two hosts with different **epistemic** roles, not just different vibes. The roles are the teaching engine and the comedy engine at the same time.

```yaml
hosts:
  - id: tessa
    role: explainer
    voice_id_final: "<elevenlabs id>"
    voice_id_draft: "nl_NL-mls-medium"
    background: "..."
    strength: "analogies, connecting a concept to something physical"
    weakness: "oversells, pushes an analogy two steps past where it works"
    tics: ["begint antwoorden met 'oké dus', gebruikt 'eigenlijk' te vaak"]
    humour: "enthousiast, lacht om haar eigen grappen"
    opinions: ["vindt formele notatie onnodig ingewikkeld gemaakt"]
  - id: joris
    role: skeptic
    voice_id_final: "<elevenlabs id>"
    voice_id_draft: "nl_NL-pim-medium"
    strength: "weigert een hand-wave te accepteren, stelt de vraag die de luisteraar heeft"
    weakness: "blijft hangen op details, mist soms het grote punt"
    tics: ["'wacht even', droge understatement"]
    humour: "droog, deadpan, plaagt Tessa"
```

`guests.yaml` holds a small pool with pinned voice IDs so a recurring expert stays recognisable across episodes.

`continuity.jsonl`, appended after every episode:
```json
{"episode": "ch03", "callbacks": ["Tessa noemde entropie 'een rommelige la', Joris kwam daar 3x op terug"],
 "running_jokes": ["..."], "mistakes": ["Tessa verwarde X met Y, Joris corrigeerde"],
 "open_threads": ["Joris beloofde het in hoofdstuk 4 nog een keer te vragen"]}
```

This file is what makes it a show instead of two voices. Episode 12 referencing something from episode 3 is the whole trick.

## 4. Episode format

| Segment | Length | What happens |
|---|---|---|
| Cold open | 45-60s | Hosts already mid-argument about something from the chapter. No intro, no welcome. |
| Recap | 90s | One key point from the previous episode, posed as a question to the listener before it is answered. Spaced repetition, free. |
| Body | 3-5 blocks | Each block anchored on one claim cluster. Joris pushes, Tessa explains, a misconception from the plan gets walked into deliberately and then corrected. |
| Guest | 5-8 min, conditional | Fires when `needs_expert` is true. Three-speaker block. |
| Wacht, opnieuw | 3 min | The highest-difficulty concept in the chapter gets explained a second time, by a different route. Joris triggers it by refusing the first explanation. |
| Quiz | 3-4 min | Joris quizzes Tessa, four questions from `exam_relevance >= 4`. Pause after each question for the listener. Tessa gets one wrong. |
| Outro | 30s | Hook into the next chapter. |

## 5. Humour and overlap rules

These go into the writer prompt as hard constraints, and into the linter.

**Humour must attach to the material.** Permitted sources: an analogy pushed until it breaks, a host misapplying the concept, a callback from the continuity log, the two hosts disagreeing about how hard something is. Banned: topical jokes, jokes that would work in any episode, jokes that require the listener to see something.

**Overlap rules:**

- Never over a definition, a number, a formula, or a glossary term. Content the listener needs must be clean.
- Overlap only in: reactions, transitions, a host getting excited, a joke landing.
- Maximum 4 interruptions per 10 minutes. More than that stops reading as energy and starts reading as chaos.
- `interrupt`: the interrupted line must be written with a trailing fragment designed to be buried (`"...en dan verschuift dus de hele—"`). The writer must produce that fragment, not the renderer.
- `backchannel`: short ("ja", "precies", lach). Layered at -8 dB, does not advance the timeline.

## 6. Audio

Target hardware: single 24GB GPU.

### Model choice

**Chatterbox Multilingual V3** (Resemble AI, MIT, 0.5B) is the backbone. 23 languages including Dutch, zero-shot voice cloning from a short reference clip, emotion exaggeration control. Roughly 8GB per worker.

**Piper** is the draft tier. Dutch voices, flat delivery, a full episode in seconds. Used to hear pacing and catch lexicon problems before any real render.

**ElevenLabs v3 Text-to-Dialogue** stays available as an accent tier, called per block rather than per episode. It is the only option that generates genuine overlapping speech from the script itself. $0.10 per 1,000 characters, so a handful of blocks per episode costs cents. Its tags are English words even inside Dutch text (`[interrupting]`, `[overlapping]`, `[laughs]`), and it caps at 3,000 characters per request including tags.

**VibeVoice is not the backbone.** The long-form model that does 90 minutes with four speakers is English and Chinese only. Dutch exists only in the experimental Realtime-0.5B speaker set, and the original Microsoft TTS code was pulled from the repo, so you would be on community forks. Worth an afternoon of experimentation, not worth building on.

### VRAM budget

| Component | VRAM |
|---|---|
| Chatterbox worker x3 (parallel) | ~18GB |
| faster-whisper large-v3, int8, resident | ~3GB |
| Headroom | ~3GB |

Three parallel workers render a 25-minute episode in minutes, which is what makes the take-selection loop below affordable.

The script stage stays on a frontier API. Do not put a local LLM in the humour path; script quality is the entire product and a 24GB-class model is not close.

### Reference voices

30 to 60 seconds of clean Dutch audio per host, mono, no music, no room echo. Stored in `cast/refs/`. Use your own voice and someone who has consented. Reference audio must be Dutch, because cross-language reference leaks accent into the clone.

Once cloned, the reference files are frozen. Re-cloning changes the voice subtly and the cast stops being the cast.

### Take selection loop

Per line, not per block. This loop is the reason local wins.

1. Cache lookup on `sha256(text + voice_ref + exaggeration + seed)`. Hit means skip.
2. Render N=3 takes with varied seed and a small exaggeration jitter around the value the line's tags imply.
3. Transcribe each take with faster-whisper.
4. Normalise both strings and compute WER against the script line. Reject anything above 5%.
5. Among survivors, pick the take whose duration is closest to the expected duration for that character count. Outliers are usually the model rushing or dragging.
6. If all three fail, retry once with a lower exaggeration, then flag the line in the UI.

Zero-shot TTS dropping or duplicating words is the dominant local failure mode, and it is silent unless you check. This loop catches it automatically and costs nothing but GPU time.

### Timeline assembly

Run WhisperX forced alignment on each accepted take to get word-level timestamps. This is the piece that cloud dialogue APIs do not give you, and it makes local overlap more precise than model-generated overlap, not less.

- **Normal turn.** Gap sampled from 180-420ms rather than a constant. A fixed inter-speaker gap is the single clearest tell that dialogue is synthetic.
- **`interrupt`.** The writer marks the cut word in line A. Find its onset in A's alignment, start line B 250ms before it, duck A by 12dB with an 80ms fade, let A's buried fragment run 400-600ms underneath, then fade out.
- **`backchannel`.** Place at the nearest word boundary past 60% of A's duration, mix at -8dB, do not advance the timeline.
- **Laughter over speech.** Same treatment as backchannel, slightly earlier placement.

### Prosody continuity

Per-line rendering makes each line sound isolated. Two mitigations: render an entire speaker turn (all consecutive sentences by one speaker) in a single call rather than sentence by sentence, and carry the emotional state forward so exaggeration moves gradually across a segment instead of resetting per line.

### Mixing

Normalise every line to the same LUFS before placement, light room tone under the whole episode so silence does not read as a dropout, intro and outro sting, final normalise to -16 LUFS mono. Reject any gap above 1.2 seconds that was not written as a deliberate beat.

## 7. App and job flow

FastAPI + a background worker. SQLite for job state is sufficient.

```
upload PDF
  -> ingest            [auto]
  -> content plan      [auto]
  -> glossary          [auto, user can edit decisions in UI]
  -> script            [auto]
  -> audit             [auto, blocking issues surface in UI]
  -> piper draft       [auto, seconds]
  === APPROVAL GATE ===
  -> chatterbox render [manual, minutes, free]
  -> mix + transcript
  -> elevenlabs blocks [optional, per block, cents]
```

The approval gate exists for your attention, not your wallet. You listen to the Piper draft, edit the script in the UI, re-audit, then commit the GPU minutes. Edited lines re-render; untouched lines come from cache.

Line-level editing in the UI is worth building properly, because the loop it enables (change one joke, re-render one line, listen) is the whole advantage of running locally.

Stream stage events over SSE. Every artifact is browsable and downloadable at every stage.

## 8. Milestones

- **M0** Voice audition. Before any pipeline work: clone two Dutch references with Chatterbox and render one paragraph of the real book. If this does not sound acceptable, everything downstream changes. Half a day, and it derisks the entire project.
- **M1** PDF to `book.json` with correct chapter boundaries on one real study book.
- **M2** Content plan + lexicon for one chapter. Verify by hand that the claims are right.
- **M3** Script generation + audit. Target: coverage 100% on `exam_relevance >= 3`, zero lint hits, zero unsupported claims.
- **M4** Piper draft render end to end. First listen. This is where you find out whether the cast is actually funny. Expect to rewrite `hosts.yaml` here, possibly twice.
- **M5** Chatterbox render with take selection, alignment and timeline assembly. One complete episode.
- **M6** Continuity log + multi-episode run across a full book. Verify that episode 5 references episode 2.

Do not build the web UI before M3. Run the pipeline from the CLI until the output is good.

## 9. Known pitfalls

- **Dutch is weaker than English in every open model.** Audition Chatterbox on a real paragraph of the actual book before building anything around it. If Dutch quality is unacceptable, the whole tier strategy inverts and ElevenLabs becomes the backbone at roughly $3-5 per finished episode.
- **Silent word drops.** Zero-shot TTS omits or repeats words without any audible artifact. Without the Whisper verification loop you will ship episodes with missing words and not notice.
- **Reference audio quality sets the ceiling.** A clone is never better than its reference. Background noise or room echo in the 30-second clip shows up in every line of every episode, permanently.
- **Frozen references.** Re-cloning a host drifts the voice. Freeze the reference files and the cloning parameters together.
- **Character counting on the accent tier.** The 3,000-char v3 limit includes audio tags. Count after injection.
- **Formulas.** Anything heavily notational does not survive audio. The content plan should mark formula-dense sections and the script should describe the shape and meaning rather than reading symbols aloud.
- **One-pass generation.** If content and comedy are generated together the model drops material to make room for jokes. Keep stages 2 and 3 separate. This is the most important rule in this document.
