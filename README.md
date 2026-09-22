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

Python 3.11, 3.12 or 3.13. Not 3.14 yet: torch commonly lags a new Python release by months and doesn't have wheels for it at the time of writing. If `python3 --version` on your machine already says 3.14, install an older one first (`uv python install 3.12`) and point `uv venv` at it, as below.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # fill in ANTHROPIC_API_KEY; the rest is optional
```

Local synthesis is a separate install because torch is large and platform-specific. Pick one:

```bash
uv pip install -e ".[cuda]" --no-build-isolation-package pkuseg    # 24GB GPU box
uv pip install -e ".[mac]" --no-build-isolation-package pkuseg     # Apple Silicon
studiepodcast doctor           # shows the platform profile and what is installed
```

`--no-build-isolation-package pkuseg` works around a real bug in `pkuseg` (a `chatterbox-tts` dependency, used for Chinese text segmentation): its `setup.py` calls `numpy.get_include()` without declaring numpy as a build dependency, which breaks under the isolated build environment `uv` (and modern `pip`) uses by default. The flag builds that one package against the venv's own packages instead of a throwaway one, so numpy — already installed by the `.[dev]` step above — is visible to it. If you're on plain `pip` instead of `uv`, run `pip install --no-build-isolation "pkuseg==0.0.25"` once first, then the normal `pip install -e ".[mac]"`.

### Troubleshooting: chatterbox crashes with `'NoneType' object is not callable` in `perth`

This means `chatterbox-tts`'s watermarking dependency, `resemble-perth`, failed to import its `PerthImplicitWatermarker` class and silently fell back to `None` instead of raising — `perth/__init__.py` wraps that import in a bare `try/except ImportError`. The real cause is always that one of `torch==2.6.0`, `torchaudio==2.6.0`, `librosa==0.11.0`, `pyyaml` or `scipy` isn't actually importable, even though `pip`/`uv` may have reported no error (a partial install, a version conflict resolved by silently skipping one, or a build that produced a broken wheel). `studiepodcast doctor` (or `python -m pipeline.cli doctor`) now catches this and prints the real underlying exception instead of the generic crash. If you'd rather check by hand:

```bash
python -c "from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker"
```

That raises the actual missing or broken import. Reinstall exactly that package (pin the version `chatterbox-tts` wants if it's `torch` or `torchaudio`, e.g. `uv pip install torch==2.6.0 torchaudio==2.6.0`), then retry.

Piper voices go under `PIPER_VOICES_DIR` (default `~/.local/share/piper/voices`) as `<name>.onnx` plus `<name>.onnx.json`. The draft voices named in `cast/hosts.yaml` are `nl_NL-mls-medium` and `nl_NL-pim-medium`.

### Apple Silicon (M3 Pro, 18 GB)

The defaults switch when the machine is an M-series Mac, so nothing needs to be set by hand. What changes:

- Chatterbox runs on Metal (`STUDIEPODCAST_DEVICE=mps`) with one worker instead of three. The multilingual model takes a few GB of unified memory; three copies would not fit next to Whisper and the OS. Expect roughly real time per take, so a 25-minute episode at three takes per turn is an hour or two of rendering. It is unattended, and the cache means a re-render after editing a few lines takes minutes.
- Verification uses MLX Whisper (`WHISPER_BACKEND=mlx`, model `mlx-community/whisper-large-v3-turbo`) on the GPU. faster-whisper only has a CPU path on Apple Silicon and would take longer than the synthesis it checks. MLX Whisper returns word timestamps in the same pass, so the accepted take's timestamps are matched onto the script and WhisperX is not needed.
- Piper runs on the CPU everywhere and is unchanged.

The first thing to lower on a laptop is the take count: `STUDIEPODCAST_TAKES=2` cuts render time by a third at the cost of a few more flagged turns. If Metal lacks an op, PyTorch falls back to the CPU for that op automatically (`PYTORCH_ENABLE_MPS_FALLBACK` is set by the adapter). `studiepodcast doctor` prints the resolved profile, whether torch sees MPS, which packages import, and whether the reference clips and Piper voices are in place.

An Intel Mac works the same way on the CPU, just slowly; set `STUDIEPODCAST_DEVICE=cpu` and `WHISPER_BACKEND=faster` there.

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

If the voice sounds right but talks too fast, Chatterbox has no direct speed parameter, pacing is an emergent property of the model, not a dial. Two levers, in the order worth trying them:

1. `cfg_weight` (Chatterbox's own generation-side parameter, default 0.5): lowering it tends to slow delivery, as a side effect of what it's actually for. Sweep it the same way as exaggeration:
   ```bash
   studiepodcast audition paragraaf.txt --ref tessa=cast/refs/tessa.wav --cfg-weights 0.3,0.4,0.5
   ```
2. `speech_rate` (a deterministic post-render time-stretch that preserves pitch, default 1.0, 0.85 means 15% slower): the fallback that always gets to the requested pace regardless of what the model does.
   ```bash
   studiepodcast audition paragraaf.txt --ref tessa=cast/refs/tessa.wav --speech-rates 0.85,0.9,1.0
   ```

Both are per-host settings in `cast/hosts.yaml` (and per-guest in `guests.yaml`) once you've picked values, alongside `exaggeration`. Different hosts can want different values, Tessa's excitable delivery might stay brisk while Joris's deadpan reads better slower.

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
- Quick handoffs get their own, much tighter gap. Every line is synthesized in isolation, Chatterbox has no way to hear what the other host just said, so a uniform 180-420 ms silence before every single handoff is what makes a factually correct script still read as two people taking turns reading lines rather than a conversation. When the previous line ends in a question mark, or is short and reactive (six words or fewer), the next line's gap is instead sampled from -60 to 90 ms: sometimes a touch of real overlap, always tight, with the outgoing line's tail ducked a gentle 4 dB (not the 12 dB an interrupt gets) so the handoff blends without burying a word or cutting anyone off. A genuine written beat (`pause_after_ms`) or a segment boundary always overrides this and keeps the deliberate gap.
- Every line is normalised to the same loudness before placement, room tone sits under the whole episode, optional stings live in `cast/stings/intro.wav` and `outro.wav`, and the episode is normalised to -16 LUFS mono. Gaps above 1.2 s that were not written as a beat are reported in the transcript's `qa` list.
- ElevenLabs v3 Text-to-Dialogue is called per block of at most 3000 characters counted after tag injection. Rendered blocks are spliced into the timeline through `block_overrides` in the render manifest.

### If the render sounds like two monologues, not a dialogue

The timeline tightening above is a mixing-level fix and applies on the *next* `render`, it rebuilds the timeline from whatever takes are already cached, so it does not by itself require re-synthesizing anything or a long run. Three things to check, roughly in order of effort:

1. **Re-run `render` on an already-approved episode.** If the audio you're unhappy with predates this fix, this alone often measurably improves it, and finishes in seconds to minutes since every turn is a cache hit.
2. **Check the script has enough connective tissue.** `studiepodcast audit` now warns (`too_few_overlaps`, non-blocking) when a script has fewer interrupt/backchannel moments than `min_connective_per_10min` (default 2 per 10 minutes) calls for. No amount of mixing fixes a script that never has the hosts react to each other; if this warning fires, the fix is in the script, not the audio, edit in a few more short reactions and re-audit.
3. **Push take quality further.** This is the one actually worth an overnight run: raise `STUDIEPODCAST_TAKES` (default 3) to 6-8 and re-render. Since take count is part of the cache key, this forces fresh Chatterbox generation for every turn, giving the take-selection loop a better shot at picking a delivery that already sounds natural at the source, on top of the timeline fix.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Frontier model for plan, lexicon, script, audit, continuity |
| `STUDIEPODCAST_LLM_MODEL` | `claude-opus-5` | Model id |
| `STUDIEPODCAST_LLM_EFFORT` | `high` | `low` to `max` |
| `ELEVENLABS_API_KEY` | | Accent tier |
| `PIPER_BIN`, `PIPER_VOICES_DIR` | `piper`, `~/.local/share/piper/voices` | Draft tier |
| `STUDIEPODCAST_DEVICE` | `cuda`, `mps` on Apple Silicon | Chatterbox device |
| `CHATTERBOX_WORKERS` | `3`, `1` on Apple Silicon | Parallel model instances (a few GB each) |
| `WHISPER_BACKEND` | `faster`, `mlx` on Apple Silicon | Verification and alignment backend |
| `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE` | `large-v3`, `int8` | faster-whisper model |
| `MLX_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | MLX Whisper model |
| `STUDIEPODCAST_TAKES`, `STUDIEPODCAST_WER_THRESHOLD` | `3`, `0.05` | Take loop |
| `STUDIEPODCAST_DATA_DIR`, `STUDIEPODCAST_CAST_DIR` | `data`, `cast` | Storage |

## Tests

```bash
pytest
```

The suite runs the whole pipeline on a generated four-page study book with a fake model and a null synthesizer, including take rejection, cache reuse, timeline maths, loudness, the approval gate, the accent-tier splice and the HTTP API.

## What has not been exercised here

The Chatterbox, Piper, faster-whisper, WhisperX, MLX Whisper and ElevenLabs adapters are written against their documented APIs but were not run in this environment (no GPU, no Apple Silicon, no keys). M0 is where they get their first real test, which is also the point of M0.
