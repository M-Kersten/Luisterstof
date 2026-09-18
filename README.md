# Studiepodcast

Turn a dense Dutch study book (PDF) into a podcast series: one episode per chapter, 20 to 30 minutes, hosted by a stable recurring cast with real banter and occasional expert guests. Output language is Dutch. Accuracy is non-negotiable: every factual line traces back to a source span and is audited before anything is rendered.

The build brief this implements is in `PLAN.md`.

## What the pipeline does

```
upload PDF
  -> ingest            book.json          chapters, sections, figures and tables as sentences
  -> content plan      chXX.plan.json     claims with source spans, definitions, misconceptions
  -> lexicon           glossary.json      loanwords, notation, abbreviations, resolved before render
  -> script            chXX.script.json   cold open, recap, body blocks, guest, "wacht opnieuw", quiz, outro
  -> audit             chXX.audit.json    support (LLM), coverage, lint; blocking issues gate the render
  -> piper draft       out/chXX.draft.mp3 seconds, free, for a first listen
  === APPROVAL GATE ===
  -> chatterbox render out/chXX.mp3       take selection, WhisperX alignment, timeline assembly, mix
  -> elevenlabs blocks (optional)         per block, cents, real overlapping speech
```

Every stage writes a file under `data/books/<book_id>/`, every file is inspectable in the web UI, and no stage reaches into another stage's internals. Content (stage 2) and comedy (stage 3) are generated separately on purpose.

## Setup

Python 3.11 or newer.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # fill in ANTHROPIC_API_KEY; the rest is optional
```

Local synthesis is a separate install because torch pulls a CUDA build:

```bash
uv pip install -e ".[local]"   # chatterbox-tts, faster-whisper, whisperx, piper-tts
```

Piper voices go under `PIPER_VOICES_DIR` (default `~/.local/share/piper/voices`) as `<name>.onnx` plus `<name>.onnx.json`. The draft voices named in `cast/hosts.yaml` are `nl_NL-mls-medium` and `nl_NL-pim-medium`.

### Dry run without any keys or models

The whole pipeline runs with canned model output and shaped noise, which is how the tests work and a quick way to see every artifact:

```bash
studiepodcast --fake-llm --fake-audio sample-pdf --out data/sample_book.pdf
studiepodcast --fake-llm --fake-audio run demo --pdf data/sample_book.pdf --upto render
studiepodcast status demo
```

## Milestones, in order

The brief says not to build around a voice you have not heard, and not to build the UI before the script is good. The CLI follows that order.

**M0 Voice audition.** Put 30 to 60 seconds of clean Dutch audio per host in `cast/refs/` (mono, no music, no room echo; the reference must be Dutch or the accent leaks into the clone). Then render one paragraph of the real book:

```bash
studiepodcast audition paragraaf.txt --ref tessa=cast/refs/tessa.wav --ref joris=cast/refs/joris.wav
```

It renders three exaggeration levels per voice into `data/auditions/` and prints the WER of a faster-whisper transcript per take. If Dutch quality is unacceptable here, the tier strategy inverts and ElevenLabs becomes the backbone. Once you are happy, freeze the reference files: re-cloning drifts the voice and the cast stops being the cast.

**M1 Ingest.**

```bash
studiepodcast ingest boek.pdf --book-id statistiek1 [--method auto|fonts|toc|llm]
```

Chapter detection tries font-size heuristics, then the PDF's bookmark TOC, then the model on the first pages. Figures are described with the vision model (`--no-captions` to skip), tables are spelled out as sentences. Check the chapter boundaries in `data/books/statistiek1/book.json` before going on.

**M2 Content plan and lexicon** for one chapter, then verify the claims by hand:

```bash
studiepodcast plan statistiek1 ch03
studiepodcast glossary statistiek1 ch03
```

Each claim carries a verbatim quote resolved to a character span (`match_score` 100 means an exact hit; 0 means the quote was not found and the whole section was taken as the source, listed under `warnings`). The lexicon is per book and applies to every episode.

**M3 Script and audit.**

```bash
studiepodcast script statistiek1 ch03     # writes, audits, revises once if blocked, logs continuity
studiepodcast audit statistiek1 ch03      # re-audit after editing the script json by hand
```

Target: coverage 100% on `exam_relevance >= 3`, zero lint hits, zero unsupported claims. Blocking issues are listed with line ids.

**M4 Piper draft.** First listen. Expect to rewrite `cast/hosts.yaml` here.

```bash
studiepodcast draft statistiek1 ch03
```

**M5 Chatterbox render.** Approve the script revision you listened to, then commit the GPU minutes:

```bash
studiepodcast approve statistiek1 ch03
studiepodcast render statistiek1 ch03
```

Per speaker turn: cache lookup, three takes with seed and exaggeration jitter, faster-whisper transcription, WER against the script (reject above 5%), pick the take closest to the expected duration, retry once at lower exaggeration, flag the turn in the manifest if everything fails. Edited turns re-render; untouched turns come from `render/cache`.

**M6 Whole book.**

```bash
studiepodcast run statistiek1 --upto draft        # every chapter up to the approval gate
studiepodcast continuity                           # what the writer sees from earlier episodes
```

`cast/continuity.jsonl` is appended after every episode. Episode 5 referencing episode 2 is the whole trick.

## Web app

```bash
studiepodcast serve            # http://127.0.0.1:8000
```

Upload a PDF, watch the stages stream over SSE, browse and download every artifact, edit the script line by line (text, speaker, tags, covers, overlap, pause), re-audit, approve, render, and pick blocks for the ElevenLabs accent tier. Saving a script bumps its revision and clears the approval, so an edited script is always re-audited before it renders.

API summary: `POST /api/books` (upload), `GET /api/books/{id}`, `GET /api/books/{id}/chapters/{ch}`, `PUT .../script`, `POST .../run` with `{"stage": "plan|glossary|script|audit|draft|render|all"}`, `POST .../approve`, `POST .../eleven` with `{"blocks": ["b001"]}`, `GET /api/jobs/{id}/events` (SSE), `GET /api/books/{id}/artifacts/{path}`.

## The cast

`cast/hosts.yaml` defines two hosts with different epistemic roles: Tessa explains with analogies and pushes them too far, Joris refuses hand-waves and asks the question the listener has. `tic_markers` are phrases only that host may use; the linter blocks a script in which anyone else uses them. `cast/guests.yaml` is a small pool picked by domain when a chapter's plan sets `needs_expert`. `cast/banned_phrases.yaml` is the zero-tolerance list, editable.

## Audio details

- Speaker turns (consecutive lines by one speaker) render in one call for prosody continuity; the emotional state carries forward so exaggeration moves gradually.
- Inter-turn gaps are sampled from 180 to 420 ms. Interrupts start 250 ms before the cut word's onset, duck the interrupted line 12 dB with an 80 ms fade and let the buried fragment run 400 to 600 ms. Backchannels sit at the first word boundary past 60% of the target line at -8 dB and do not advance the timeline.
- Every line is normalised to the same loudness before placement, room tone sits under the whole episode, optional stings live in `cast/stings/intro.wav` and `outro.wav`, and the episode is normalised to -16 LUFS mono. Gaps above 1.2 s that were not written as a beat are reported in the transcript's `qa` list.
- ElevenLabs v3 Text-to-Dialogue is called per block of at most 3000 characters counted after tag injection. Rendered blocks are spliced into the timeline through `block_overrides` in the render manifest.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Frontier model for plan, lexicon, script, audit, continuity |
| `STUDIEPODCAST_LLM_MODEL` | `claude-opus-5` | Model id |
| `STUDIEPODCAST_LLM_EFFORT` | `high` | `low` to `max` |
| `ELEVENLABS_API_KEY` | | Accent tier |
| `PIPER_BIN`, `PIPER_VOICES_DIR` | `piper`, `~/.local/share/piper/voices` | Draft tier |
| `STUDIEPODCAST_DEVICE` | `cuda` | Chatterbox, faster-whisper, WhisperX |
| `CHATTERBOX_WORKERS` | `3` | Parallel model instances (about 6 GB each) |
| `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE` | `large-v3`, `int8` | Verification model |
| `STUDIEPODCAST_DATA_DIR`, `STUDIEPODCAST_CAST_DIR` | `data`, `cast` | Storage |

## Tests

```bash
pytest
```

The suite runs the whole pipeline on a generated four-page study book with a fake model and a null synthesizer, including take rejection, cache reuse, timeline maths, loudness, the approval gate, the accent-tier splice and the HTTP API.

## What has not been exercised here

The Chatterbox, Piper, faster-whisper, WhisperX and ElevenLabs adapters are written against their documented APIs but were not run in this environment (no GPU, no keys). M0 is where they get their first real test, which is also the point of M0.
