# Studiepodcast

<img src="app/static/logo.svg" alt="Luisterstof" height="62">

Turn a dense Dutch study book (PDF) into a podcast series: one episode per chapter, as long as its material carries (6 to 25 minutes), hosted by a stable recurring cast with real banter and occasional expert guests. Output language is Dutch. Accuracy is non-negotiable: every factual line traces back to a source span and is audited before anything is rendered.

The build brief this implements is in `PLAN.md`.

## What the pipeline does

```
upload PDF
  -> ingest            book.json          chapters, sections, figures and tables as sentences; exercises and source lists out
  -> content plan      chXX.plan.json     claims with source spans, definitions, misconceptions
  -> lexicon           glossary.json      loanwords, notation, abbreviations, resolved before render
  -> script            chXX.script.json   outline (one question, a question per block), then opening,
                                          recap, body blocks, guest*, "wacht opnieuw"*, quiz, outro
  -> audit             chXX.audit.json    support (LLM), coverage, lint; blocking issues gate the render
  === APPROVAL GATE (read the script and the audit) ===
  -> chatterbox render out/chXX.mp3       take selection, WhisperX alignment, timeline assembly, mix
```

(* only when the chapter calls for it, see M3.) Every stage writes a file under `data/books/<book_id>/`, every file is inspectable in the web UI, and no stage reaches into another stage's internals. Content (stage 2) and comedy (stage 3) are generated separately on purpose.

## Setup

Python 3.11, 3.12 or 3.13. Not 3.14 yet: torch commonly lags a new Python release by months and doesn't have wheels for it at the time of writing. If `python3 --version` on your machine already says 3.14, install an older one first (`uv python install 3.12`) and point `uv venv` at it, as below.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # everything has a default; see OpenCode below
```

Local synthesis is a separate install because torch is large and platform-specific:

```bash
uv pip install -e ".[cuda]" --no-build-isolation-package pkuseg    # 24GB GPU box
studiepodcast doctor           # shows the platform profile and what is installed
```

**On Windows** PyPI only has the CPU-only build of torch, so the voices would run on the processor, and torch 2.6 (which chatterbox-tts pins) has no code for RTX 50 cards. `pyproject.toml` therefore gives Windows `torch 2.7.1+cu128` straight from PyTorch's index (about 3 GB; it needs a recent NVIDIA driver): the GTX 16 series and every RTX card up to the RTX 50 series. An install made before that, or with plain `pip`, has the wrong build; `doctor` says so, and this replaces it:

```bat
uv pip install --python .venv "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
```

A GTX 900 or 1000 card is too old for that build; `doctor` then gives the CUDA 12.4 one (`torch==2.6.0+cu124` from `.../whl/cu124`).

`--no-build-isolation-package pkuseg` works around a real bug in `pkuseg` (a `chatterbox-tts` dependency, used for Chinese text segmentation): its `setup.py` calls `numpy.get_include()` without declaring numpy as a build dependency, which breaks under the isolated build environment `uv` (and modern `pip`) uses by default. The flag builds that one package against the venv's own packages instead of a throwaway one, so numpy — already installed by the `.[dev]` step above — is visible to it. If you're on plain `pip` instead of `uv`, run `pip install --no-build-isolation "pkuseg==0.0.25"` once first, then the normal `pip install -e ".[cuda]"`.

### Troubleshooting: chatterbox crashes with `'NoneType' object is not callable` in `perth`

This means `chatterbox-tts`'s watermarking dependency, `resemble-perth`, failed to import its `PerthImplicitWatermarker` class and silently fell back to `None` instead of raising — `perth/__init__.py` wraps that import in a bare `try/except ImportError`. The real cause is always that one of `torch==2.6.0`, `torchaudio==2.6.0`, `librosa==0.11.0`, `pyyaml` or `scipy` isn't actually importable, even though `pip`/`uv` may have reported no error (a partial install, a version conflict resolved by silently skipping one, or a build that produced a broken wheel). `studiepodcast doctor` (or `python -m pipeline.cli doctor`) now catches this and prints the real underlying exception instead of the generic crash. If you'd rather check by hand:

```bash
python -c "from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker"
```

That raises the actual missing or broken import. Reinstall exactly that package (pin the version `chatterbox-tts` wants if it's `torch` or `torchaudio`, e.g. `uv pip install torch==2.6.0 torchaudio==2.6.0`), then retry.

### Dry run without any keys or models

The whole pipeline runs with canned model output and shaped noise, which is how the tests work and a quick way to see every artifact:

```bash
studiepodcast --fake-llm --fake-audio sample-pdf --out data/sample_book.pdf
studiepodcast --fake-llm --fake-audio run demo --pdf data/sample_book.pdf --upto render
studiepodcast status demo
```

## Writing with your own model, through OpenCode

The text side (plan, lexicon, script, audit and continuity) runs on the model you already run
inside [OpenCode](https://opencode.ai) — a quantised local checkpoint, a provider you've configured
in `opencode.json`. Every text stage is handed to the OpenCode server's HTTP API
(`POST /api/experimental/generate/text`, Basic auth) and nothing else; each answer is validated
against its pydantic schema, and an invalid answer goes back to the model once with the error
attached.

```
OPENCODE_MODEL=strata/strata        # provider/id, optionally #variant
```

No key, no port to configure: the pipeline finds the server through `OPENCODE_URL`, the `url` in
`~/.config/opencode/service.json`, the URL the OpenCode desktop app last logged
(`%APPDATA%/ai.opencode.desktop/logs/*/main.log`), and finally `http://127.0.0.1:4096`, the
`opencode serve` default. The Basic-auth password comes from that same `service.json`, or from
`OPENCODE_PASSWORD`. `studiepodcast doctor` prints which URL and model it settled on and checks the
model against the server's own list.

What this means in practice:

- **The model's settings belong to OpenCode.** Temperature, reasoning effort and the token budget
  are not sent per request; whatever that model does in `opencode.json` is what the pipeline gets.
  `OPENCODE_MODEL` takes `provider/id` (as `studiepodcast doctor` lists them) with an optional
  `#variant`.
- **Content plan and lexicon run in thorough mode** by default: per section, quotes checked against
  the source, merged and reviewed. Short sections are read together up to
  `STUDIEPODCAST_PLAN_PART_CHARS` (default 8000 characters, about two and a half pages); a longer
  section is read in pieces of that size. Each part yields candidate claims, definitions and
  misconceptions, each marked with its section, and every quote has to be near-literal in that
  text. A near miss (a changed word, a dropped comma) is replaced by the source passage itself,
  without a model call; a quote that isn't there at all goes back to the model once, and a claim
  whose quote still isn't there is dropped with a warning. A chapter pass then picks and merges the
  claims by candidate id, so quotes are never rewritten, and writes the summary and learning
  objectives. `STUDIEPODCAST_PLAN_REVIEW_ROUNDS` (default 1) review rounds add what's missing and
  remove what's double. The lexicon reads the same parts.
- **Figures travel as their caption text.** The endpoint is text-only and always will be: prompts
  are plain text and a picture is never sent. Nothing else changes; the same schema validation,
  repair round and repetition guard protect every answer. A model that gets stuck repeating itself
  gets one retry, then an error that says so, rather than "raise the limit".
- **A long chapter takes minutes per call**, so `OPENCODE_TIMEOUT` defaults to 1800 seconds. Lower
  it if you want a stalled server noticed sooner. A Stop ends the job right away, also in the
  middle of a call; the endpoint does not stream, so OpenCode finishes that answer in the
  background and the pipeline drops it. Everything saved before the Stop stays.
- **A model that is not running is started.** OpenCode starts some local model servers (Strata, a
  llama.cpp launcher plugin) only on the first prompt in a session, and the stateless endpoint the
  pipeline writes through does not count as one. So when the model behind OpenCode refuses the
  connection, the pipeline sends one short prompt in a new OpenCode session, the way the app does,
  waits until the model answers (`OPENCODE_WAKE_TIMEOUT`, default 300 s) and deletes that session again.
  If a prompt does not start your model, `OPENCODE_MODEL_START` can hold the command that does. Every
  job checks the model before it starts, so a missing model shows in the first minute, not at chapter
  one; a book run that loses the model halfway stops at once with what to do, instead of failing every
  chapter after it. `start-app.bat` and `studiepodcast doctor` start the model as well, so it is warm
  before you upload.
- **The pipeline cannot unload the writing model**: what OpenCode keeps loaded is OpenCode's
  business. If that model runs on the same GPU as Chatterbox, unload it before a render.
  The render checks the card first (`nvidia-smi`) and warns in the log when more than 4 GB is
  already in use; `studiepodcast doctor` shows the same reading.

### Quality over speed

The defaults aim at a usable episode in reasonable time. Quality mode aims at the best episode your
model can write, for runs that may take all night: set `STUDIEPODCAST_QUALITY_MODE=1` in `.env`,
pass `--quality` (or `-Q`) to any CLI command, or tick "grondig" in the web app for a single run.
It changes exactly four numbers and one rule.

| Pass | Normal | Quality mode |
|---|---|---|
| `STUDIEPODCAST_PLAN_REVIEW_ROUNDS` | 1 | 3 |
| `STUDIEPODCAST_PLAN_PART_CHARS` | 8000 | 4000 — smaller pieces of the book, read better |
| `STUDIEPODCAST_REVISE_ROUNDS` | 1 | 3 |
| `OPENCODE_TIMEOUT` | 1800 | 3600 — a thinking local model is never cut off mid-answer |

And the rule: in quality mode the rewrite loop does not stop at "the audit no longer blocks". A
script that passes but speaks flat goes back to the writer as well — too few interruptions and
backchannels (`too_few_overlaps`), nobody reacting when something surprising lands
(`flat_emotional_reaction`), an interruption that breaks off mid-word (`cut_word_missing`), a quiz
question nobody waits for (`quiz_pause`). The notes are handed to the flattest body segments, which
get rewritten with the warning as instruction. This is the only place those fixes can come from:
the mixer cannot overlap two lines the script never marked, and no render setting adds a reaction
that was never written.

A value you set yourself always wins over the mode: quality mode only fills in the passes you left
alone (upwards only — it never relaxes a setting you already made more thorough), and `--reviews`,
`--part-chars` and `--revise-rounds` override single runs without touching `.env`.
`studiepodcast doctor` prints the passes currently in effect.

Budget for it: a chapter that takes ten minutes normally can take an hour or more this way. Running
a whole book overnight or over a weekend is the intended use. A chapter that fails along the way
does not stop the others, and running the book again picks up where it stopped (see *M6 Whole
book*).

### Keeping book content on your machine

The audio side (Chatterbox, faster-whisper, WhisperX) is always local. For the text side, honesty matters:
the OpenCode *server* runs on your machine, but if the model OpenCode is configured with is a
hosted one, your book text goes to that provider. For confidential material, point OpenCode at a
local model and run with `STUDIEPODCAST_OFFLINE=1`.

What offline mode enforces:

- **The OpenCode server has to be local.** `OPENCODE_URL` must resolve to localhost or a private
  address (10.x, 172.16-31.x, 192.168.x). Anything else is refused before a request goes out.
  Traffic to it never goes through an `HTTP(S)_PROXY` from the environment.
- **Hugging Face runs from its local cache**, with telemetry off. Chatterbox, Whisper and WhisperX
  weights you've already downloaded keep working; `doctor` lists them. To fetch a new one, turn
  offline mode off for that one run. Downloading weights sends no book content.
- **The web app serves no CDN-loaded pages.** The `/docs` API explorer is switched off.

Then check with `studiepodcast doctor`. It shows which URL and model are in use, whether the
server's address passes the offline check, and which models are in the Hugging Face cache.

## Milestones, in order

The brief says not to build around a voice you have not heard, and not to build the UI before the script is good. The CLI follows that order.

**M0 Voice audition.** Put 30 to 60 seconds of clean Dutch audio per host in `cast/refs/` (mono, no music, no room echo; the reference must be Dutch or the accent leaks into the clone), or upload a longer recording on the character's page in the web app, which cuts and levels it for you (see **Reference clips** under The cast). Then render one paragraph of the real book:

```bash
studiepodcast audition paragraaf.txt --ref tessa=cast/refs/tessa.wav --ref joris=cast/refs/joris.wav
```

It renders three exaggeration levels per voice into `data/auditions/` and prints the WER of a faster-whisper transcript per take (Dutch word error rate around 0.03 to 0.04 on the V3 weights is what the shipped references get). Chatterbox is the only voice, so this is the one place to judge the language before committing GPU minutes to a series. Once you are happy, freeze the reference files: re-cloning drifts the voice and the cast stops being the cast.

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

Chapter detection tries font-size heuristics, then the PDF's bookmark TOC, then the model on the first pages. Figures keep the caption text that is next to them; there is no vision pass, the endpoint is text-only. Tables are spelled out as sentences. Check the chapter boundaries in `data/books/statistiek1/book.json` before going on.

**What the ingest leaves out.** A study book prints things for a reader with a pen, and a podcast must not read them: exercises (from an "Oefening 3", "Opdrachten" or "Vragen bij ..." heading up to the next real heading), source lists (from a "Bronnen", "Literatuur", "Referenties" or "Noten" heading up to the next real heading), fill-in-the-blank copies of the text, paragraphs that repeat an earlier one, footnote numbers (superscript digits glued to a word; an exponent after a symbol stays) and footnote bodies. A caption under a picture becomes that figure's text; a row of names under portraits is dropped. A sentence cut by a page break is joined again. Word lists stay: they are the book's own definitions. `book.json` lists what was left out under `warnings`. A book without chapters becomes one episode with the book's title.

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

The episode is as long as its material: a claim is budgeted 15 to 85 seconds by difficulty, around that a frame of opening, quiz and outro, between 6 minutes and `STUDIEPODCAST_TARGET_MINUTES` (default 25). Eight easy pages make a short episode. The claims are served first: a guest and "wacht, opnieuw" are extras, and they are left out when the body would otherwise get less than 90% of the time its claims need. The quiz has 2 to 4 questions by length.

**The thread.** Before a line is written, one model call makes the outline (`script.outline` in the script json, shown on the script page as the question above each block): the one question the episode answers, the question each block answers, the order in which the claims have to be told (the setup of an experiment before its outcome, whatever the order in the book), the bridge from each block to the next, and the answer the outro lands on. Every claim sits in exactly one block; an outline that forgets a claim or invents an id is repaired, and without a usable answer the claims go into blocks in book order. A human moment (an anecdote, a real disagreement, a tease, the penny dropping) comes only where the outline puts one, in at most two blocks.

**One home per claim.** A guest is used only when someone in `cast/guests.yaml` has the field the plan asks for (no stand-in expert), tells the hardest claims, and those claims are then not told by the hosts first. "Wacht, opnieuw" comes only for a claim of difficulty 4 or 5 and is about the why: the brief carries the neighbouring claims and the passage from the book that hold the reason. Each misconception goes to the block whose claims it is about, each definition is explained in one block only, and the worked example goes to the block it is about.

**What a segment is asked.** The opening sets a recognisable scene in plain words and ends on the episode's question; it carries no claim and no argument the listener cannot follow yet. A body block answers its one question, starts from the bridge of the block before, and gets the paragraph of the book around its claims instead of one sentence per claim (one sentence is a sentence to read out; the paragraph has who did the study and why the outcome is what it is). A claim can also carry `context` from the plan: what a listener has to know first. The skeptic voices the misconception and the explainer corrects it, never the other way round. The quiz asks why and how about the hard claims, the planned wrong answer is a real misconception, and the skeptic confirms each answer. The outro answers the episode's question in a sentence or two and points at the next chapter.

**What the writer always sees, and what not.** Every call sees the rules (`WRITER_RULES`: who knows what, plain words, nothing out of thin air, the thread), how the hosts talk and what they want, and the outline. The hosts' lives, stories and shared history (`life`, `stories`, `relationship` in `cast/hosts.yaml`) are left out of that shared part: a writer that sees the courgettes and the stew in every call puts them in every segment. They go to the one block in which the outline places a story. The writer first writes the segment's `beats`, then the lines; the beats are kept on the segment in the script json.

When the facts are right, the table read (`STUDIEPODCAST_POLISH`, on by default) rewrites every segment once for how it sounds: every line answering the one before, plain words, no reference the listener has not been given. A line that carries a claim keeps its fact (numbers, names, direction) and its claim ids, and gets new wording when it was lifted from the book. The edited script is audited again, and if the edit added blocking issues the version before it is kept.

The table read and quality mode work from the audit's warnings about how the script sounds. From the text itself: a line that is the book's sentence (`source_copy`), a claim told again in another block (`claim_repeated`, `near_duplicate`), the skeptic stating a fact from a chapter he has not read (`skeptic_fact`) or using a term before anyone explained it (`term_before_intro`), semicolons and colon lists (`written_syntax`), more than three numbers in a line (`number_overload`), a word found in no Dutch word list, the book or the cast (`unknown_word`), and segments told as strings of three-word facts (`telegraphic`). From the model: lines that ignore what was just said (`ignores_previous`), written or solemn language (`stiff`, `formal`), a comparison that explains nothing (`forced_image`); a listener pass per segment that hears the episode once, in order, and reports a reference nobody set up (`no_antecedent`), a question nobody answered (`unanswered`), a sentence without meaning (`nonsense`) and a correction of something that was right (`contradicts`); and one call that checks whether the claims of difficulty 4 and up are explained and not only stated (`not_landed`). The listener's findings and `not_landed` send a segment back to the writer in every mode, not only in quality mode: the fix can need something the table read may not add. A guest's statements that carry no claim are checked against the chapter like any other fact.

**Model or prompt?** Invented words and broken grammar come from the writing model, and no rule removes them. To see which problems are the model's, write the same chapter once with a stronger model (`OPENCODE_MODEL=provider/model studiepodcast script <book> <chapter>`) and compare the two audits.

The audit blocks what a writing model does on autopilot: a host's tic more than once per segment (`tic_overuse`), most of a host's lines starting the same way (`repeated_opening`), the same quip template over and over such as "een X is geen Y" (`stock_phrase`), the hosts talking about "de tekst" or "de bron" instead of the subject (`source_talk`), a definition read out as "term: omschrijving" (`definition_recital`) and a line repeated word for word outside the quiz (`duplicate_line`). The support check accepts a paraphrase that gets the content right. Each rewrite round is audited again, and when a round made the script worse, the best revision is saved again as the newest one (`kept_best` in the log).

**M4 Human check.** Read the script and the audit output. Expect to rewrite `cast/hosts.yaml` here.

**M5 Chatterbox render.** Approve the script revision you checked, then commit the GPU minutes:

```bash
studiepodcast approve statistiek1 ch03
studiepodcast render statistiek1 ch03
```

Per speaker turn: cache lookup, three takes with seed and exaggeration jitter, faster-whisper transcription, WER against the script (reject above 5%), pick the take closest to the expected duration, retry once at lower exaggeration, flag the turn in the manifest if everything fails. Edited turns re-render; untouched turns come from `render/cache`.

**M6 Whole book.**

```bash
studiepodcast run statistiek1                     # every chapter up to the approval gate
studiepodcast run statistiek1 --fresh             # the same, redoing what is already done
studiepodcast timings statistiek1                 # when each pass started and ended, and how long it took
studiepodcast continuity                           # what the writer sees from earlier episodes
```

A book run survives its failures. A chapter that fails (OpenCode down for longer than its retries,
an answer that never fits the schema) is logged and skipped, the other chapters carry on, and the
run ends by naming the chapters that failed. Run the book again and only what is missing is made:
a stage is skipped when its output is on disk and at least as new as what it was made from
(`book.json`, then the plan, then the script). A re-ingested PDF therefore redoes everything, a new
plan redoes its script, and a script you edited by hand is audited again, never rewritten.
`--fresh` turns the skipping off.

`cast/continuity.jsonl` is appended after every episode. Episode 5 referencing episode 2 is the whole trick.

## Web app

```bash
studiepodcast serve            # http://127.0.0.1:8000
```

Or double-click `start-app.bat`: it checks that the OpenCode server is reachable, starts the app and opens your browser.

The app is a small website that walks you through the steps, one page at a time, with one green button for what to do now:

- **Je boeken** lists your books with how far each one is. With no book yet it explains the evening, night and morning and starts you on the first one.
- **Nieuw boek** asks for the PDF, how to begin (everything in one go, or only reading it in first) and then starts.
- **A book's page** is a path of five steps: read in, write the scripts, read and approve, record, listen. Only the step that needs you is open; the others are one line saying where they stand. While the night's run goes, the writing step shows how far it is and has a stop button. Below the path, every chapter with its summary and where it stands.
- **A chapter's page** has three steps: read and approve, record, listen. The step on top says what to do; below it the chapter itself, in four tabs:
  - **Script**: the dialogue as a transcript; problems found by the audit sit under their line in red, tips in orange. Click a line and the panel beside it shows the claims it carries, the passage from the book each one rests on, and what the audit says about it. Click in the text to change it; saving checks the script again. "Details per regel" shows the emotion tags, claim ids, overlap and pause per line.
  - **Boektekst**: the chapter's text as read from the PDF, with every passage the episode uses highlighted (red underline: in the plan but in no line of the script). Click a highlight to see the claim and the lines that carry it.
  - **Samenvatting**: the plan: summary, learning goals, claims (each with its place in the book and its lines in the script), terms and misconceptions.
  - **Controle**: the audit report.

  The pronunciation list, the timings, the files and redoing a step sit under **Meer** at the bottom.
- **Stemmen** is the voice lab: pick a host or guest, type a sentence, set four sliders (expression = `exaggeration`, hold on the voice = `cfg_weight`, variation = `temperature`, tempo = `speech_rate`) and a seed, and make a take. Every take is kept in `data/voicelab/<voice>/` with the settings and seed that made it and Whisper's check of it (the share of words it misheard; a real recording rejects a take above 5%). Keep the seed to hear what one slider does; "Overnemen" puts a take's settings back on the sliders; "Opslaan" writes them into that voice's block in `cast/hosts.yaml` or `cast/guests.yaml`, comments and all, and every recording after uses them. The model stays loaded between takes; a recording releases it first, and the page has a button to free the GPU before you write with OpenCode. A take is saved as soon as it is made; the Whisper check comes after and can only cost the check, never the take. The check never downloads Whisper (run `studiepodcast fetch-whisper` once), runs on the processor when the GPU has less than 2.5 GB free, gives up after 3 minutes of loading or 90 seconds of listening, and stops when you press Stop. A take that was not checked says why on its card; after a check that hung, the next takes skip it until you free the GPU on the page or restart the app. While a take runs, its row says which phase it is in and for how long. Under the knobs, **Klank van de show** sets the sound every voice and every episode gets (see Audio details); the Droog / Met klank switch above the takes plays them either way, with the voice's breath first and at its place in the stereo image.
- **A character's page** ("Personage instellen" on Stemmen, or "+ Nieuw personage" under the list) sets up one voice in four steps. **Wie is het?**: name and background for a host; name, domains and persona for a guest. **Karakter**: everything else the writer reads (voice, life, stories, behaviour, never, tics and the rest), one point per line. **Referentieclip**: upload a recording (WAV, FLAC, MP3 or OGG); the app turns it into the best reference it can and shows the upload and the result side by side, with what it found and what it changed. "Gebruik deze clip" makes it the character's reference; the old clip is kept in `cast/refs/backup/`. **Stem afstellen**: the same four sliders as the lab, saved to the cast file. A new character is always a guest: the writer works with exactly two hosts. Saving writes only the fields that changed into `cast/hosts.yaml` or `cast/guests.yaml`; comments and every other character stay as they were, and a file that would no longer load is put back.
- **Meekijken** (from the banner, or the writing step on a book's page; `#/boek/<book>/meekijken`) follows one run of a book. On top, while it runs: the chapter, what Luisterstof is doing in a sentence ("Blok 3 van 7 schrijven: Waarom je hart sneller gaat"), what the writing model is working on and for how long against how long that usually takes, the bar for the whole book with the finish time, and the first lines of the block that was just written. Below, per chapter, a timeline of the run: every step as it started and what came out of it ("Plan klaar: 15 beweringen", "Controle vond 4 problemen en 2 tips", "Herschrijfronde 1: 4 problemen gaan terug naar de schrijver"), with the end result and the time it took under the chapter's title. "Alle details" adds every question to the model and every block with its first lines. After the run the page stays: in the morning it tells you what happened in the night, and earlier runs of the book can be picked from the list.
- The lamp at the top right says whether the writing model and the voices are ready; click it for details. A running job shows a banner with its progress under the header. Jobs and the raw log are at the bottom of every page.

**Look.** Calm like a reading room, clear like a podcast app: linen backgrounds, bottle green for text and the main button, vermilion only for what is moving or waiting on you (progress, the current step, chapters to approve). Titles and text you read (the book, the script) are set in Boek Serif, everything you click in Instrument Sans. Both are in `app/static/fonts/` with their licences, so the app loads nothing from the internet. Buttons answer every click: they lift under the pointer, press in, show a spinner while their work runs, a tick when it is done and a shake when it failed; a short message at the bottom right says what happened. The logo, a ribbon microphone with a paragraph for a grille and a bookmark ribbon, is `app/static/logo.svg` (with the wordmark) and `app/static/logo-mark.svg` (the microphone alone); the browser tab shows `app/static/favicon.svg`.

**Overnight.** Add a book with "Alles in één keer" chosen (the default) and the job reads the book in and runs every chapter through plan, lexicon, script, the audit and rewrite loop and the table read, then stops at the approval gate. In the morning every chapter has a script and an audit. While a job runs the laptop does not go to sleep (Windows; the screen may still switch off). Closing the lid follows the power settings: set "when I close the lid" to "do nothing", and keep the charger in. If the night was cut short anyway (a restart, a Stop, a chapter that failed), **Verder schrijven** on the book's page carries on where it stopped without reading the PDF again; uploading the same PDF again would redo everything.

**Timings.** Every pass is recorded with its start, end and duration in `data/books/<book>/timings.jsonl`: reading in, plan, lexicon, writing, each audit, each rewrite round, the table read, continuity and render, plus the whole chapter, and a skipped pass when a run carried on past finished work. The log shows them as they finish, with clock times on every line. The book's page says when the latest run started and how long it took, and **Tijden** under a chapter's **Meer** lists every pass of its last three runs with the number of model calls. `studiepodcast timings <book>` prints the same in a terminal.

Upload a PDF, watch the stages stream over SSE, browse and download every artifact, edit the script line by line (text, speaker, tags, covers, overlap, pause), re-audit, approve and render. Saving a script bumps its revision and clears the approval, so an edited script is always re-audited before it renders. The "grondig" checkbox (on upload and next to the stage buttons) runs just that job in quality mode, hours-per-chapter thoroughness included, without changing the server's own settings.

While a job runs, a banner under the header says in plain Dutch where it is (book, chapter 3 of 9 and its title), what it is doing now, and what the writing model is working on and for how long. The bar covers the whole job: chapters, then the steps of a chapter weighted by how long they take (writing and checking the script is most of it); it never runs backwards. The finish time comes from how long a chapter took in this run, or in an earlier run of the book until this one has finished a chapter. The sentences come from `pipeline/narrate.py`, which turns every pipeline event into one; the CLI prints the same sentences, and `GET /api/books/{id}/activity` returns a run's events as sentences for the Meekijken page. **Stop** ends the job at once during a model call, and at the next take or step otherwise. The OpenCode endpoint does not stream, so OpenCode finishes an answer it was writing in the background; the next job can wait a moment for it. What was saved before the stop stays; the interrupted stage starts over when you run it again. A reloaded page picks up running jobs.

API summary: `POST /api/books` (upload, `quality=true` to run it in quality mode), `GET /api/books/{id}`, `GET /api/books/{id}/chapters/{ch}`, `PUT .../script`, `POST .../run` with `{"stage": "plan|glossary|script|audit|render|all", "quality": false}`, `POST .../approve`, `GET /api/jobs/{id}/events` (SSE, resumes from `Last-Event-ID`), `POST /api/jobs/{id}/cancel`, `GET /api/books/{id}/artifacts/{path}`, `POST /api/books/{id}/run` with `{"upto": "plan|glossary|script", "quality": false, "fresh": false}` (the whole book, carrying on where it stopped), `GET /api/books/{id}/timings`, `GET|PUT /api/characters/{id}`, `POST /api/characters` (a new guest), `POST /api/characters/{id}/clips` (multipart upload; returns the processed clip and its report), `POST /api/characters/{id}/clips/{clip}/use`. `GET /api/health` reports the model and the writing passes in effect.

## The cast

`cast/hosts.yaml` defines two hosts with different epistemic roles: Tessa explains with analogies and pushes them too far, Joris refuses hand-waves and asks the question the listener has. A role alone writes a robot, so each host also has a character file the writer sees in full: `voice` (how they talk), `life` (people, places, habits), `stories` (anecdotes they can tell), `wants`, `behaviour` (what they do when wrong, bored or moved) and `never`. The top-level `relationship` holds what they share: how they met, the running tease, what each admires in the other. All of it is a first draft; rewrite it until it sounds like your hosts. A host's own stories are not subject matter: they may tell them freely, as long as no fact about the material comes with them.

`cast/voice_samples.md` holds four scenes, one of them a story told at length, in the register the writer should hit. The writer is told to take the tone and nothing else, and the audit blocks a line that copies a sample line (`sample_copy`). Write your own scenes here once the voices are cast; the scenes in the voice-actor script fit.

**Reference clips.** Chatterbox takes the speaking style from the first 6 seconds of the reference, the timbre from the first 10 and the speaker embedding from the whole clip. `pipeline/audio/refclip.py` (used by the character's page, no model needed, a couple of seconds per clip) makes the most of that:

1. Mix to mono, or keep the better channel when the two channels are not the same recording.
2. Remove DC offset and filter below 70 Hz (rumble, handling, traffic).
3. Find the pauses from the signal's energy and cut the silence before the first word.
4. Measure the pitch range of every candidate opening. When a later stretch starts clearly livelier than the first sentence, the clip starts there and the part before it is added at the end, so no speech is lost.
5. Shorten pauses longer than 0.45 s and stop at a pause before 55 seconds.
6. Level to -20 LUFS with peaks at most -1.5 dBFS, short fades, 24-bit WAV.

The report says what it could not fix: clipping, background noise, room echo, a clip shorter than 30 seconds of speech, low bandwidth (a phone or a compressed file) and a flat start. Each upload is kept in `data/voicelab/<id>/clips/` with the original, the result and `report.json`; nothing goes to `cast/refs/` until you choose a clip.

`tic_markers` are phrases only that host may use; the linter blocks a script in which anyone else uses them, or the host itself more than once per segment. `cast/guests.yaml` is a small pool picked by domain when a chapter's plan sets `needs_expert`. `cast/banned_phrases.yaml` is the zero-tolerance list, editable.

## Audio details

**The sound of the show** (`cast/klank.yaml`, set on the Stemmen page). Chatterbox renders every line dry, so next to each other the voices sound like they come from nowhere. `pipeline/audio/finish.py` puts them in one recording: per line a shared microphone low cut, warmth (a little low end and a very light saturation) and a de-esser that only acts where s, z and sch rise above the voice; over the whole conversation one gentle compressor and one small made-up room (early reflections, then a tail that darkens as it dies away, the same for every episode); at the end a limiter that turns peaks down instead of the whole episode. Before a longer line that moves the conversation on, and only where the pause allows it, the speaker breathes in. `pipeline/audio/breaths.py` cuts those inhales from the speaker's own recordings (the reference clip, then the two newest uploads in `data/voicelab/<voice>/clips/`): a soft, pitchless stretch after a pause and just before the voice comes in, well below speech, swelling smoothly. A Dutch g or ch is too loud and runs into the vowel; mouth clicks jump from frame to frame; neither gets picked. A voice whose recordings hold no usable breath gets a soft made-up one, and the Stemmen page says which of the two each voice has. The same script gives the same breaths in every render. With width the episode is stereo: the explainer a little to the left, the skeptic a little to the right, a guest in the middle, never further out than halfway, with the room answering a little differently on each side and a compressor and limiter that move both channels together, so a voice stays where it sits. Width 0 gives mono, as before. Four presets (`droog` is exactly the old dry sound, `studio` is the default, `huiskamer`, `radio`) and seven knobs from 0 to 1 (room, size, warmth, deess, compression, breath, width). A `klank.yaml` written before breath and width existed takes those two from its own preset. On the Stemmen page every take plays dry or with the sound, switched while it plays; the server treats the dry take on request at the same loudness, so the comparison is fair and no new take is needed. Episodes recorded before keep their sound until they are recorded again. All of it runs on the processor: a take in a fraction of a second, an episode in a few seconds.

The voice comes from Resemble AI's Chatterbox Multilingual on the V3 T3 weights,
`t3_mtl23ls_v3.safetensors` (the June 2026 release). That checkpoint is fixed. The newest
`chatterbox-tts` on PyPI (0.1.7) still hardcodes V2; Resemble AI's GitHub main loads V3 natively and
generates the way V3 was tuned: no alignment stream analyzer (on V3 it cuts lines short), repetition
penalty 1.2, and the noisy last speech token trimmed. Install that one, at the commit the project
pins, with `--no-deps`: its own pins (`numpy<2`, `transformers==5.2.0`) would reshuffle a working venv,
and its source runs on the packages already there.

```
uv pip install --no-deps "chatterbox-tts @ https://github.com/resemble-ai/chatterbox/archive/ab55923c27757c90cfcc407baa57dc25a7d25217.zip"
uv pip install omegaconf
```

With a PyPI release (0.1.5 to 0.1.7) `pipeline/audio/synth.py` still loads V3 through its own copy of
the package's loader, and `studiepodcast doctor` says so. 0.1.4 and older are refused: they build T3
for the first multilingual vocabulary (2352 tokens, V3 has 2454) and tokenize without lowercasing.
The checkpoint name is part of the take cache key, so takes
written with another checkpoint re-render rather than silently mixing two voices in one episode, and
every worker in the pool loads the same file: a failed load is an error instead of a quiet fallback to
V2, because two hosts on two checkpoints would sound like two different people mid-episode. Point
`CHATTERBOX_CKPT_DIR` at a folder of weights (yours or a fine-tune) to skip the download entirely;
otherwise they come from the Hugging Face cache, which is offline-safe once fetched. `studiepodcast
doctor` prints which T3 file a render will load and whether the weights around it are all on disk, so
you can see which voice you are actually rendering with before an overnight run rather than after one.

- Speaker turns (consecutive lines by one speaker) render in one call for prosody continuity; the emotional state carries forward so exaggeration moves gradually. It carries two ways: mostly within a speaker's own successive turns (60% of the previous value survives into the next), and more lightly across speakers, 20% of the other host's immediately preceding turn bleeds into this one, so a line the script didn't explicitly tag still picks up some of the other host's mood instead of resetting to a flat baseline the instant the turn changes.
- Inter-turn gaps are sampled from 180 to 420 ms. Interrupts start 250 ms before the cut word's onset, duck the interrupted line 12 dB with an 80 ms fade and let the buried fragment run 400 to 600 ms. Backchannels sit at the first word boundary past 60% of the target line at -8 dB and do not advance the timeline.
- Quick handoffs get their own, much tighter gap. Every line is synthesized in isolation, Chatterbox has no way to hear what the other host just said, so a uniform 180-420 ms silence before every single handoff is what makes a factually correct script still read as two people taking turns reading lines rather than a conversation. When the previous line ends in a question mark, or is short and reactive (six words or fewer), the next line's gap is instead sampled from -60 to 90 ms: sometimes a touch of real overlap, always tight, with the outgoing line's tail ducked a gentle 4 dB (not the 12 dB an interrupt gets) so the handoff blends without burying a word or cutting anyone off. A genuine written beat (`pause_after_ms`) or a segment boundary always overrides this and keeps the deliberate gap.
- Every line is normalised to the same loudness before placement, room tone sits under the whole episode, optional stings live in `cast/stings/intro.wav` and `outro.wav`, and the episode is normalised to -16 LUFS mono.
- An episode that opens with a cold open gets music under it: the jingle in `jingle/hook_jingle.wav`, or your own `cast/stings/hook.wav` when you put one there. The music plays alone for 2.4 seconds (one bar of the jingle), then the first voice comes in and the music drops 14 dB under it. Half of the music's own rise and fall is evened out first, so its soft opening bars stay audible under speech and its loudest bars stay out of the way. It plays to its own end; when the cold open is over before the music is, it fades out there. `STUDIEPODCAST_HOOK_MUSIC=0` turns it off, and `studiepodcast doctor` prints which file a render will use. `python jingle/make_jingle.py` renders the jingle again from its score. Gaps above 1.2 s that were not written as a beat are reported in the transcript's `qa` list.

### If the render sounds like two monologues, not a dialogue

The timeline tightening above is a mixing-level fix and applies on the *next* `render`, it rebuilds the timeline from whatever takes are already cached, so it does not by itself require re-synthesizing anything or a long run. Three things to check, roughly in order of effort:

1. **Re-run `render` on an already-approved episode.** If the audio you're unhappy with predates this fix, this alone often measurably improves it, and finishes in seconds to minutes since every turn is a cache hit.
2. **Check the script has enough connective tissue.** `studiepodcast audit` now warns (`too_few_overlaps`, non-blocking) when a script has fewer interrupt/backchannel moments than `min_connective_per_10min` (default 2 per 10 minutes) calls for. No amount of mixing fixes a script that never has the hosts react to each other; if this warning fires, the fix is in the script, not the audio, edit in a few more short reactions and re-audit. Or let the writer do it: in quality mode these warnings are fed back into the rewrite loop (see *Quality over speed*).
3. **Push take quality further.** This is the one actually worth an overnight run: raise `STUDIEPODCAST_TAKES` (default 3) to 6-8 and re-render. Since take count is part of the cache key, this forces fresh Chatterbox generation for every turn, giving the take-selection loop a better shot at picking a delivery that already sounds natural at the source, on top of the timeline fix.

### Emotional continuity between hosts

Overlap and gap timing make the *rhythm* of a conversation; a script where the hosts never react to each other's mood still sounds like two monologues however tight the handoffs are. Two things address this, one in the writing, one in the render:

- The writer prompt (`WRITER_RULES` in `pipeline/script/writer.py`) has a rule that a host's emotion tag on one line is not isolated: the other host, replying, should pick it up in the next line, either by mirroring it (both getting more enthusiastic, or both turning serious together) or by deliberately countering it (staying calm, reining it in), whichever the topic calls for. `studiepodcast audit` checks this is actually happening: `flat_emotional_reaction` (non-blocking, needs at least 4 tagged-then-replied moments in the episode before it fires) warns when the other host picks up on a tag in fewer than `min_emotional_reaction_rate` (default 30%) of the opportunities. If it fires, the fix is the same as `too_few_overlaps`: edit the script, not the audio, or run the script stage in quality mode and let the rewrite loop chase it (see *Quality over speed*).
- The render itself now also carries a *little* of this even when the script doesn't tag the reacting line explicitly: 20% of the immediately preceding turn's exaggeration bleeds across into the next turn regardless of speaker (see Audio details above). This is a safety net under imperfect script compliance, not a replacement for the writer actually reacting, it is deliberately a light touch (20%, versus 60% for a speaker's own carry) so an untagged neutral reply doesn't get pulled into an emotion it was never written to have.

Together these are what let `exaggeration` and `temperature` (see below) run higher for a genuinely lively episode without every scene reading as one host performing next to a mannequin.

### Performance prototype (opt-in, not yet in the main pipeline)

A layer that makes delivery follow the conversation instead of every line sounding the same. It is built as a prototype that renders one scene in labelled variants. The normal `script`/`render` path is untouched until listening shows it clearly helps.

```bash
studiepodcast doctor                                   # alignment must be real: phrasing is skipped on estimated timing
studiepodcast reactions generate                     # candidates for every label, both hosts (or --speaker joris --label laugh,sigh)
studiepodcast reactions list
studiepodcast prototype-scene <book> <chapter>
```

`prototype-scene` has the writer produce one ~75 s scene with performance labels per line, and checks that it contains:

- an explanation
- a disagreement
- an interruption
- a realization
- a laugh
- a backchannel
- a quick overlapping handoff
- three or more different speaking rates

If any are missing, the writer gets one retry. The scene is saved, then rendered five ways into `data/books/<book>/prototype/<chapter>/`:

| file | what's on |
|---|---|
| `scene_current.mp3` | today's pipeline, labels ignored |
| `scene_timing.mp3` | response timing from labels + reaction bank |
| `scene_phrasing.mp3` | delivery/mood → exaggeration and rate, phrase-level re-timing |
| `scene_acoustics.mp3` | today's pipeline + shared recording chain only |
| `scene_full.mp3` | everything |

`report.md` next to them lists per turn the labels, exaggeration, per-phrase rate and pauses, where each reaction came from, whether alignment was real, and why phrasing was skipped where it was.

How the parts work:

- **Labels, not numbers.** The writer picks `delivery` (explain, think, excite, react, interrupt, realize, disagree, setup, punchline), `mood` (persistent speaker state), `timing` (immediate, hesitate, search, deliberate) and optional `phrases` inside a line. The numbers live per host under `performance:` in `cast/hosts.yaml`, merged over the defaults in `pipeline/performance.py`. Inside a range, the exact value comes from a seed, so reruns are identical.
- **Hybrid phrasing.** Chatterbox has no context between calls, so a separately generated fragment ends on sentence-final intonation. Each turn is therefore still rendered in one pass, then cut between aligned words and re-timed per phrase, with pauses inserted. On estimated word timing that cut would land mid-word, so the turn is left alone and the report says so.
- **Reaction bank.** Chatterbox Multilingual has no `[laugh]` tags, and one-word generations are unreliable. Laughs, sighs and short reactions come from `cast/reactions/<speaker>/<label>/*.wav`. Generate candidates, keep the good ones, or drop in real recordings. A label with an empty bank folder falls back to synthesis and the report says so.
- **Shared acoustics** (`cast/acoustics.yaml`). A partial per-speaker EQ correction toward the shared average spectrum removes chain differences without matching the timbres. Manual `eq:` bands per host go on top, e.g. `eq: [{freq_hz: 300, gain_db: -2, q: 1.0}]`. Then the same high-pass and presence profile for everyone, one compressor and one small room on the dialogue bus.

Alignment cache: an alignment that was estimated is no longer reused from cache. It is retried on the next render, so after fixing a broken WhisperX install those turns get real word timing without clearing anything.

### A take says the voice model runs "op de processor"

torch cannot use the GPU, so Chatterbox runs on the processor: a take of a few sentences takes a minute or more, a chapter hours. On Windows the reason is nearly always the CPU-only build of torch, which is what PyPI serves for Windows (`doctor` shows `built for CUDA none`). Install the GPU build with the command under Setup above, then check:

```bat
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That should print `2.7.1+cu128 True`. If it says `+cu128 False`, torch is right and the NVIDIA driver is too old or the card is not visible: update the driver. The same build also brings the cuDNN 9 that Whisper's GPU engine needs, so takes are then listened back on the GPU as well.

The other reason is a card the installed torch has no code for. Loading used to fail with `CUDA error: no kernel image is available for execution on the device`; now the voice model goes to the processor and the take says which card it is. An RTX 50 card (Blackwell, `sm_120`) needs the CUDA 12.8 build above; `doctor` prints the card, what this torch has code for, and the command.

### A take in Stemmen stays on "take terugluisteren"

That is Whisper listening back, after the take itself was made. Three causes, in the order they are likely:

1. **Whisper is not on the laptop yet.** Its first use used to download `large-v3` (about 3 GB) inside the take, with only a progress bar in the app's console window to show for it. Now the check is skipped instead; run `studiepodcast fetch-whisper` once. `studiepodcast doctor` says whether it is on disk.
2. **The GPU is full.** Chatterbox stays loaded between takes and the writing model in OpenCode may hold the card as well. On Windows CUDA then spills into shared system memory and everything on the card crawls. The check now moves to the processor when less than 2.5 GB is free; `doctor` shows how much of the card is in use. Freeing the GPU on the Stemmen page or closing the model in OpenCode helps the takes themselves too.
3. **torch and faster-whisper disagree about cuDNN.** The console says `Could not locate cudnn_ops_infer64_8.dll`. The GPU builds of torch bring cuDNN 9 (the CPU-only build brings none, see the entry above); whisperx 3.3.3 to 3.4.x pin CTranslate2 (faster-whisper's engine) below 4.5, and those versions want cuDNN 8. On a Windows GPU that does not fail, it hangs. The project now pins `whisperx==3.3.2` and `ctranslate2>=4.5`; an existing install is fixed with

   ```bat
   uv pip install --python .venv "whisperx==3.3.2" "ctranslate2>=4.5"
   ```

   `studiepodcast doctor` then says `whisper gpu: ok`. Until then the app finds the missing cuDNN before Whisper starts and lets Whisper listen on the processor (slower, it does not hang); takes and renders say so.

A take that is already hanging in a version before this fix cannot be stopped from the page: close the app's window and start it again. A job that was running is closed as failed at startup ("worker restarted"), not started again.

### Locating and verifying voice quality problems

Every rendered turn is judged twice and both judgements are recorded in the render manifest: faster-whisper re-transcribes the chosen take and scores it against the script text (WER), and WhisperX places every word in time for the timeline logic to cut on. Neither step crashes the render when it fails, an unreachable transcriber or a broken CUDA/cuDNN install degrades to "accept unverified" or "estimate word positions evenly", by design, so a bad machine still finishes a render. That also means a render can complete and sound wrong without ever raising an error. Three ways to see what actually happened, from least to most detail:

```bash
studiepodcast render <book> <chapter>          # prints a verification/alignment summary after rendering
studiepodcast inspect <book> <chapter>          # every turn: WER, take path, flags, alignment method
studiepodcast inspect <book> <chapter> --only-suspect   # only the turns worth listening to first
studiepodcast doctor                            # is whisperx/faster-whisper actually importing?
```

The web UI shows the same summary (verified turns, alignment fallback count) on a chapter's listen step, and `GET /api/books/{id}/chapters/{ch}` returns it as `manifest_quality`.

**Abrupt cutoffs mid-sentence.** This is very likely `inspect` reporting turns with "geschatte uitlijning" (estimated alignment). When the real forced aligner (WhisperX on CUDA) is not actually working, every interrupt cut-point and backchannel insertion falls back to word positions spaced evenly across the clip's duration, not where the words were actually spoken. A cut that lands mid-word sounds exactly like a clipped sentence. Run `doctor` first, on the machine with the cuDNN/ctranslate2 mismatch this is the same root cause as the missing `cudnn_ops_infer64_8.dll` error above: `uv pip install --python .venv "whisperx==3.3.2" "ctranslate2>=4.5"`, then re-render and check `inspect` again for the fallback count to drop to zero.

**Background noise, rustling ("gerritsel").** Chatterbox clones the reference clip, including its recording conditions; this is a named pitfall in `PLAN.md`, not a synthesis bug, so no render setting fixes it. Listen to `cast/refs/*.wav` directly. Any rustling, room tone, or mic handling noise there gets reproduced, often exaggerated, in every line cloned from it. Re-record 30 to 60 seconds in a quiet room, close to the mic, no fabric or paper handling, then re-run the M0 audition (`studiepodcast audition ...`) before trusting the new clip. The character's page in the web app measures the noise in a clip's pauses and says when it is too loud, and it filters low rumble, but it cannot take hiss or rustling out of the speech itself; audition and listen before freezing a clip.

**Insufficient intonation and expression.** Three levers, all per-host in `cast/hosts.yaml`: `exaggeration` controls how much emotional range Chatterbox reaches for, `cfg_weight` trades pace for adherence to the reference, and `temperature` (default 0.8, the model's real sampling temperature) controls how much delivery varies from take to take, lower is flatter but more predictable, higher varies more at the cost of consistency. If deliveries sound flat, sweep exaggeration upward first, then temperature (`studiepodcast audition ... --exaggerations 0.4,0.6,0.8 --temperatures 0.6,0.8,1.0`) and listen to the results in `data/auditions/` rather than guessing at a value. Two things that some Chatterbox hosting APIs advertise are not available here: `useHD` (48kHz upscaling) and a text "description" field for style, neither exists on the local `chatterbox-tts` package this project runs, they belong to paid hosted wrappers built on top of it (fal.ai's ChatterboxHD, Segmind, and similar). What the local model does respond to, confirmed by Resemble AI's own documentation and now part of the writer's prompt rules: CAPITALIZED words get more emphasis (louder, slower on that word), and punctuation shapes breath and pacing, a script with only long comma-sentences reads flatter than one that varies sentence length and stop points. The per-line emotion tags already available in scripts (`pipeline/tags.py`, e.g. `excited`, `deadpan`, `whisper`) are this project's own equivalent of a style/description control, each one nudges `exaggeration` for a single line. The reference clip itself still sets a ceiling on all of this: a flat, monotone reference clones as a flat, monotone voice, so if none of these levers help, the fix is a more expressive reference recording.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `STUDIEPODCAST_OFFLINE` | `0` | `1`: nothing leaves the local network (see above) |
| `STUDIEPODCAST_QUALITY_MODE` | `0` | `1`: quality over speed (see above); raises the passes below and rewrites flat scripts |
| `STUDIEPODCAST_PLAN_REVIEW_ROUNDS`, `STUDIEPODCAST_PLAN_PART_CHARS` | `1`, `8000` | How the content plan and lexicon are made; how much text one plan call reads |
| `STUDIEPODCAST_REVISE_ROUNDS` | `1` | How often a script is rewritten before it is accepted as it is: when the audit blocks it, when a listener cannot follow a line, or when a hard claim is stated but not explained |
| `OPENCODE_URL`, `OPENCODE_PASSWORD` | service.json, then the desktop app's log, then `http://127.0.0.1:4096` | OpenCode server (backend `opencode`) and its Basic-auth password |
| `OPENCODE_MODEL` | `strata/strata` | `provider/id` (optionally `#variant`) as OpenCode lists them |
| `OPENCODE_WAKE_TIMEOUT`, `OPENCODE_MODEL_START` | `300`, empty | How long a model that is being started may take; an optional command that starts it directly |
| `OPENCODE_TIMEOUT` | `1800` | Seconds per OpenCode call; a long chapter takes minutes |
| `STUDIEPODCAST_DEVICE` | `cuda` | Chatterbox device |
| `CHATTERBOX_CKPT_DIR` | the Hugging Face cache | Use a local folder of Chatterbox V3 weights instead of downloading |
| `HF_TOKEN` | | Only needed to fetch weights the first time, never for a render |
| `CHATTERBOX_WORKERS` | `3` | Parallel model instances (a few GB each) |
| `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE` | `large-v3`, `int8` | faster-whisper model |
| `STUDIEPODCAST_TAKES`, `STUDIEPODCAST_WER_THRESHOLD` | `3`, `0.05` | Take loop |
| `STUDIEPODCAST_TARGET_MINUTES` | `25` | Longest episode; shorter material makes a shorter one |
| `STUDIEPODCAST_POLISH` | `1` | The table read: one dialogue edit per segment once the facts are right |
| `STUDIEPODCAST_LISTENER` | `1` | The listener pass and the landing check in every audit: what a first-time listener cannot follow, and hard claims that are stated but not explained |
| `STUDIEPODCAST_DATA_DIR`, `STUDIEPODCAST_CAST_DIR` | `data`, `cast` | Storage |

## Tests

```bash
pytest
```

The suite runs the whole pipeline on a generated four-page study book with a fake model and a null synthesizer, including take rejection, cache reuse, timeline maths, loudness, the approval gate and the HTTP API. The OpenCode client has its own tests against a stub of the server (discovery, Basic auth, schema repair, the repetition guard, token progress, aborting a job while a call is in flight), and the Chatterbox loader is tested against a stub of the package for the checkpoint it asks for.

## What has not been exercised here

The WhisperX adapter is written against its documented API but was not exercised in this environment (the installed whisperx cannot load under this torch version, so alignment runs on the uniform fallback here). M0 and M5 are where it gets its first real test, which is also the point of it.

What *was* run here: Chatterbox V3 on a CUDA box rendering the shipped Dutch references, with faster-whisper scoring the takes (mean word error rate 0.033 on V3 against 0.042 on V2, which is why V3 is what loads), and the OpenCode backend against the real desktop server for `doctor`, a plan and a script. Script quality with your own model is still yours to judge: the audit output and a quality-mode rewrite are the places to look first.
