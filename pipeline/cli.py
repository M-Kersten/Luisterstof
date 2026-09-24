"""Command-line interface. Run the pipeline from here until the output is good."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from pipeline.config import Settings

app = typer.Typer(help="Studiepodcast: PDF -> content plan -> script -> audio, one episode per chapter.",
                  no_args_is_help=True, pretty_exceptions_enable=False)

state: dict[str, Any] = {}


def _pipeline():
    from pipeline.runner import Pipeline

    if "pipeline" not in state:
        def on_event(stage: str, status: str, data: dict) -> None:
            detail = " ".join(f"{k}={v}" for k, v in data.items() if k not in ("book_id",))
            typer.echo(f"[{stage:>10}] {status:<8} {detail}")

        state["pipeline"] = Pipeline(state["settings"], fake_llm=state["fake_llm"], fake_audio=state["fake_audio"], on_event=on_event)
    return state["pipeline"]


@app.callback()
def main(
    data_dir: Annotated[Path | None, typer.Option(help="Where data/books lives.")] = None,
    cast_dir: Annotated[Path | None, typer.Option(help="Where hosts.yaml, guests.yaml, continuity.jsonl live.")] = None,
    fake_llm: Annotated[bool, typer.Option("--fake-llm", help="Canned model output, no API key needed.")] = False,
    fake_audio: Annotated[bool, typer.Option("--fake-audio", help="Shaped noise instead of Piper/Chatterbox.")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    overrides = {}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if cast_dir is not None:
        overrides["cast_dir"] = cast_dir
    state["settings"] = Settings.from_env(**overrides)
    from pipeline.offline import apply_env

    apply_env(state["settings"])  # before chatterbox/whisper import huggingface_hub, which reads it once
    state["fake_llm"] = fake_llm
    state["fake_audio"] = fake_audio


@app.command()
def ingest(pdf: Path, book_id: Annotated[str, typer.Option(help="Identifier, e.g. statistiek1")],
           method: Annotated[str, typer.Option(help="auto | fonts | toc | llm")] = "auto",
           captions: Annotated[bool, typer.Option(help="Describe figures with the model.")] = True,
           title: str | None = None):
    """Stage 1: PDF -> book.json with chapter boundaries."""
    book = _pipeline().ingest(pdf, book_id, method=method, captions=captions, title=title)
    for ch in book.chapters:
        typer.echo(f"  {ch.id} [{ch.level}] p{ch.pages[0]}-{ch.pages[1]} {ch.title} ({len(ch.sections)} secties, {ch.char_count} tekens)")


@app.command()
def plan(book_id: str, chapter: Annotated[str | None, typer.Argument()] = None):
    """Stage 2: content plan (claims, definitions, misconceptions) for one or all chapters."""
    p = _pipeline()
    for ch in _chapters(p, book_id, chapter):
        result = p.plan(book_id, ch)
        typer.echo(f"  {ch}: {len(result.key_claims)} beweringen, expert={result.needs_expert} ({result.expert_domain})")


@app.command()
def glossary(book_id: str, chapter: Annotated[str | None, typer.Argument()] = None):
    """Stage 2b: extend the book lexicon from one or all chapters."""
    p = _pipeline()
    for ch in _chapters(p, book_id, chapter):
        g = p.glossary(book_id, ch)
    typer.echo(f"  lexicon: {len(g.entries)} entries -> {p.paths(book_id).glossary_json}")


@app.command()
def script(book_id: str, chapter: str, revise_rounds: int = 1, llm_checks: bool = True):
    """Stage 3 + 3b: write the script, audit it, revise once if needed, log continuity."""
    s, a = _pipeline().script(book_id, chapter, revise_rounds=revise_rounds, llm_checks=llm_checks)
    _print_audit(a)


@app.command()
def audit(book_id: str, chapter: str, llm_checks: bool = True):
    """Stage 3b alone: re-audit the script on disk (after editing it)."""
    _print_audit(_pipeline().audit(book_id, chapter, llm_checks=llm_checks))


@app.command()
def draft(book_id: str, chapter: str):
    """Piper draft render: seconds, free, for a first listen."""
    audio, transcript, _ = _pipeline().draft(book_id, chapter)
    typer.echo(f"  {audio}\n  {transcript}")


@app.command()
def approve(book_id: str, chapter: str, note: str | None = None, force: bool = False):
    """The approval gate: commit the GPU minutes for this script revision."""
    a = _pipeline().approve(book_id, chapter, note=note, force=force)
    typer.echo(f"  approved revision {a.script_revision} (audit passed: {a.audit_passed})")


@app.command()
def render(book_id: str, chapter: str, force: bool = False):
    """Chatterbox render with take selection, alignment, timeline assembly and mix."""
    audio, transcript, manifest = _pipeline().render(book_id, chapter, force=force)
    flagged = manifest.flagged_turns()
    verified, verifiable = manifest.verification_coverage()
    fallback = manifest.alignment_fallback_count()
    typer.echo(f"  {audio}\n  {transcript}\n  flagged turns: {len(flagged)}")
    if verifiable:
        mark = "OK" if verified == verifiable else "!!"
        typer.echo(f"  [{mark}] WER verification: {verified}/{verifiable} turns. "
                  f"{'run `studiepodcast doctor`, fix the transcriber, then re-render' if verified < verifiable else ''}")
    if fallback:
        typer.echo(f"  [!!] alignment: {fallback}/{len(manifest.turns)} turns used estimated (uniform) word "
                  f"timing, not real forced alignment. Interrupt/backchannel placement on those turns is approximate.")
    for t in flagged:
        typer.echo(f"    {t.turn_id} {t.speaker}: {t.flag_reason} :: {t.text_spoken[:80]}")


@app.command()
def eleven(book_id: str, chapter: str, blocks: list[str]):
    """Accent tier: render the given blocks (b001, b002, ...) with ElevenLabs v3 Text-to-Dialogue."""
    m = _pipeline().eleven(book_id, chapter, blocks)
    typer.echo(f"  overrides: {m.block_overrides}")


@app.command()
def run(book_id: str, pdf: Path | None = None, upto: str = "draft", chapters: str | None = None,
        method: str = "auto", llm_checks: bool = True):
    """Everything up to the approval gate (or further with --upto render)."""
    p = _pipeline()
    if pdf is not None:
        p.ingest(pdf, book_id, method=method)
    wanted = [c.strip() for c in chapters.split(",")] if chapters else None
    results = p.run_book(book_id, upto=upto, chapters=wanted, llm_checks=llm_checks)
    for ch, out in results.items():
        a = out.get("audit")
        typer.echo(f"  {ch}: audit={'ok' if a and a.passed else 'open'} draft={out.get('draft', '-')} final={out.get('render', '-')}")


@app.command()
def status(book_id: Annotated[str | None, typer.Argument()] = None):
    """Which artifacts exist per chapter."""
    p = _pipeline()
    for bid in ([book_id] if book_id else p.books()):
        s = p.status(bid)
        typer.echo(f"{bid}: {s['title']} (structure: {s['structure_method']}, lexicon: {s['glossary_entries']})")
        for ch in s["chapters"]:
            flags = " ".join(k for k in ("plan", "script", "audit", "draft", "approved", "final") if ch[k])
            aud = "" if ch["audit_passed"] is None else (" audit=ok" if ch["audit_passed"] else f" audit={ch['audit_blocking']} blocking")
            typer.echo(f"  {ch['id']} {ch['title'][:40]:<40} {flags}{aud}")


@app.command()
def inspect(book_id: str, chapter: str, tier: str = "final",
           only_suspect: Annotated[bool, typer.Option(help="Only turns that are flagged, unverified, or fallback-aligned.")] = False):
    """Per-turn audit: WER, acceptance, the exaggeration actually used, alignment method and the audio file for every rendered turn.

    Answers "waar hoor ik precies wat er mis is": every take the take-selection loop chose, with
    enough to go straight to the file and listen, instead of only the fully-flagged turns.
    """
    p = _pipeline()
    from pipeline.models import RenderManifest

    manifest_path = p.paths(book_id).manifest(chapter, tier)
    manifest = RenderManifest.load_or_none(manifest_path)
    if manifest is None:
        typer.echo(f"  geen manifest gevonden op {manifest_path}; eerst renderen.")
        raise typer.Exit(1)

    verified, verifiable = manifest.verification_coverage()
    fallback = manifest.alignment_fallback_count()
    typer.echo(f"  {book_id}/{chapter} ({tier}): {len(manifest.turns)} beurten, "
              f"verificatie {verified}/{verifiable}, {fallback} met geschatte uitlijning, "
              f"{len(manifest.flagged_turns())} gevlagd.")
    if verifiable and verified < verifiable:
        typer.echo("  [!!] niet elke beurt kon worden geverifieerd; zie `studiepodcast doctor`.")
    typer.echo("")

    for t in manifest.turns:
        take = t.chosen_take
        if only_suspect and not (t.flagged or t.aligned_with_fallback or (take and take.wer is None)):
            continue
        wer = "onbekend" if take is None or take.wer is None else f"{take.wer:.3f}"
        exag = "-" if take is None else f"{take.exaggeration:.2f}"
        marks = []
        if t.flagged:
            marks.append("GEVLAGD: " + (t.flag_reason or ""))
        if t.aligned_with_fallback:
            marks.append("geschatte uitlijning")
        if take is not None and take.wer is None:
            marks.append("niet geverifieerd")
        mark_text = f"  <- {', '.join(marks)}" if marks else ""
        path = take.path if take else "-"
        typer.echo(f"  {t.turn_id:>6} {t.speaker:<10} wer={wer:<8} exag={exag:<5} {path}{mark_text}")
        typer.echo(f"         {t.text_spoken[:110]}")


@app.command()
def audition(text_file: Path,
             ref: Annotated[list[str], typer.Option(help="name=path.wav, repeatable, e.g. --ref tessa=cast/refs/tessa.wav")],
             out_dir: Path = Path("data/auditions"), exaggerations: str = "0.3,0.5,0.7",
             cfg_weights: Annotated[str, typer.Option(help="Comma-separated. Chatterbox's own pacing lever, lower tends to slow delivery.")] = "0.5",
             speech_rates: Annotated[str, typer.Option(help="Comma-separated. Post-render time-stretch, <1.0 slower, 1.0=off. The lever that always works.")] = "1.0",
             temperatures: Annotated[str, typer.Option(help="Comma-separated. Sampling temperature, default 0.8. Higher varies delivery more, lower is flatter but more predictable.")] = "0.8",
             verify: bool = True):
    """M0: render one real paragraph with each reference voice (name=path.wav) before building anything.

    Sweeps every combination of the four axes. Start with exaggerations alone; if delivery is too fast,
    add --cfg-weights 0.3,0.4,0.5 next; if that alone isn't enough, add --speech-rates 0.85,0.9,1.0.
    If exaggeration is maxed out but deliveries still sound flat, sweep --temperatures 0.6,0.8,1.0 too.
    """
    from pipeline.audio.asr import make_transcriber
    from pipeline.audio.audition import audition as run_audition
    from pipeline.audio.synth import make_synth

    settings = state["settings"]
    refs = {}
    for item in ref:
        name, _, path = item.partition("=")
        refs[name] = Path(path)
    synth = make_synth("null" if state["fake_audio"] else "final", settings, pool=False)
    transcriber = make_transcriber(settings, fake=state["fake_audio"]) if verify else None
    rows = run_audition(text_file.read_text(encoding="utf-8"), refs, out_dir, synth,
                        exaggerations=tuple(float(x) for x in exaggerations.split(",")),
                        cfg_weights=tuple(float(x) for x in cfg_weights.split(",")),
                        speech_rates=tuple(float(x) for x in speech_rates.split(",")),
                        temperatures=tuple(float(x) for x in temperatures.split(",")),
                        transcriber=transcriber)
    for row in rows:
        typer.echo(json.dumps(row, ensure_ascii=False))


@app.command()
def doctor():
    """Show the platform profile and which optional pieces are installed (run this first on a new machine)."""
    import importlib
    import shutil
    import sys

    settings = state["settings"]
    if sys.version_info >= (3, 14):
        typer.echo(
            f"  WARNING: running on Python {sys.version_info.major}.{sys.version_info.minor}. torch commonly "
            "lags a new Python release by months and fails to build or import on a version it has no wheels "
            "for yet. If chatterbox crashes with something like \"'NoneType' object is not callable\" in "
            "perth, recreate the venv on 3.11-3.13, e.g.:\n"
            "    uv python install 3.12 && uv venv --python 3.12 .venv && source .venv/bin/activate && "
            'uv pip install -e ".[dev,mac]"'
        )
    typer.echo("profile:")
    for k, v in settings.profile().items():
        typer.echo(f"  {k}: {v}")
    from pipeline.config import apple_silicon

    typer.echo("packages:")
    # faster-whisper/whisperx (CUDA tier) and mlx_whisper (Apple Silicon tier) are mutually
    # exclusive by design; only check the pair that's actually meant to be installed here, so a
    # correctly-absent package on the other platform isn't reported as if something were missing.
    on_mac = apple_silicon()
    common = ("torch", "chatterbox", "perth", "piper", "soxr", "pyloudnorm")
    tier_specific = ("mlx_whisper",) if on_mac else ("faster_whisper", "whisperx")
    for name in common + tier_specific:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            typer.echo(f"  {name}: ok {version}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  {name}: missing ({type(exc).__name__}: {exc})")
    try:
        import perth  # type: ignore

        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            # perth/__init__.py swallows the real ImportError and leaves this None. Walk the same
            # import chain directly so the actual missing/broken package is visible instead of a
            # generic "'NoneType' object is not callable" three steps later, inside chatterbox.
            try:
                from perth.perth_net.perth_net_implicit.perth_watermarker import (
                    PerthImplicitWatermarker as _,  # noqa: F401
                )
                cause = "unknown (re-import succeeded the second time; try re-running chatterbox)"
            except Exception as exc:  # noqa: BLE001
                cause = f"{type(exc).__name__}: {exc}"
            typer.echo(f"  WARNING: perth.PerthImplicitWatermarker is None. Real cause: {cause}")
            typer.echo("    chatterbox-tts needs torch==2.6.0, torchaudio==2.6.0, librosa==0.11.0 and "
                      "resemble-perth's own transitive deps (pyyaml, scipy) all importable, not just "
                      "installed. If pip/uv reported no error, reinstall the missing one directly, e.g.:\n"
                      "        uv pip install librosa==0.11.0 torch==2.6.0 torchaudio==2.6.0")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # type: ignore

        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        typer.echo(f"torch devices: cuda={torch.cuda.is_available()} mps={mps}")
    except Exception:  # noqa: BLE001
        typer.echo("torch devices: torch not installed")
    typer.echo(f"piper binary: {shutil.which(settings.piper_bin) or 'not on PATH (python package is enough)'}")
    voices = sorted(p.stem for p in settings.piper_voices_dir.glob("*.onnx")) if settings.piper_voices_dir.exists() else []
    typer.echo(f"piper voices in {settings.piper_voices_dir}: {', '.join(voices) or 'none'}")
    from pipeline.script.cast import load_cast

    try:
        cast = load_cast(settings.cast_dir)
        for sp in list(cast.hosts) + list(cast.guests):
            ref = settings.cast_dir / sp.voice_ref if sp.voice_ref else None
            typer.echo(f"  ref {sp.id}: {'ok' if ref and ref.is_file() else 'missing'} ({ref})")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"cast: {exc}")
    typer.echo(f"keys: anthropic={'set' if settings.anthropic_api_key else 'missing'} elevenlabs={'set' if settings.elevenlabs_api_key else 'missing'}"
               + ("  (both blocked: offline)" if settings.offline else ""))
    _doctor_llm(settings)
    stings = [p.name for p in (settings.cast_dir / "stings").glob("*.wav")] if (settings.cast_dir / "stings").exists() else []
    typer.echo(f"stings: {', '.join(stings) or 'none'}")


def _doctor_llm(settings: Settings) -> None:
    """Where script-side data goes, and whether the local model server is ready."""
    import os

    from pipeline.offline import resolve_local

    if settings.llm_backend == "local":
        from urllib.parse import urlparse

        typer.echo(f"llm: local ({settings.local_llm_api}) {settings.local_llm_url} model={settings.local_llm_model} "
                   f"context={settings.local_llm_context} vision={'on' if settings.local_llm_vision else 'off'}")
        local, addresses = resolve_local(urlparse(settings.local_llm_url).hostname or "")
        typer.echo(f"  address: {', '.join(addresses) or 'does not resolve'} -> "
                   + ("local network" if local else "NOT the local network: book text would leave it"))
        try:
            from pipeline.local_llm import LocalLLM, think_value

            llm = LocalLLM(settings)
            ok, detail = llm.check()
            typer.echo(f"  server: {'ok' if ok else '!!'} {detail}")
            caps = llm.capabilities() if ok else None
            if caps is not None:
                typer.echo(f"  capabilities: {', '.join(sorted(caps)) or 'none reported'}")
                if settings.local_llm_vision and "vision" not in caps:
                    typer.echo("  note: this model can't read images, so figure captions are skipped")
                think = think_value(settings.local_llm_think)
                if "thinking" in caps and think is None:
                    typer.echo("  note: this model reasons before it answers (slower, usually more careful); "
                               "LOCAL_LLM_THINK=off makes it answer directly")
                if think and "thinking" not in caps:
                    typer.echo(f"  !! LOCAL_LLM_THINK={settings.local_llm_think}, but this model can't reason step by step: unset it")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  server: !! {type(exc).__name__}: {exc}")
    else:
        typer.echo(f"llm: Claude API ({settings.llm_model}): book text, plans and scripts are sent to Anthropic"
                   + ("  [!! blocked: offline mode needs STUDIEPODCAST_LLM_BACKEND=local]" if settings.offline else ""))
    if settings.offline:
        typer.echo("offline: on. Claude API and ElevenLabs refuse to start; Hugging Face runs from its cache "
                   f"(HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', 'unset')})")
    else:
        typer.echo("offline: off. Set STUDIEPODCAST_OFFLINE=1 to enforce that nothing leaves the local network")
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore

        repos = sorted(r.repo_id for r in scan_cache_dir().repos)
        typer.echo(f"  models in the Hugging Face cache (all offline mode can load): {', '.join(repos) or 'none'}")
    except ModuleNotFoundError:
        typer.echo("  Hugging Face cache: huggingface_hub not installed (it comes with the chatterbox install)")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  Hugging Face cache: not readable ({type(exc).__name__}: {exc})")


@app.command("prototype-scene")
def prototype_scene(book_id: str, chapter: str):
    """Performance prototype: write one ~75 s labelled scene and render it as labelled variants.

    Output in data/books/<book>/prototype/<chapter>/: scene_current, scene_timing, scene_phrasing,
    scene_acoustics, scene_full (.mp3), scene.script.json and report.md.
    """
    from pipeline.prototype import run_prototype

    result = run_prototype(_pipeline(), book_id, chapter)
    missing = [k for k, ok in result.coverage.items() if not ok]
    typer.echo(f"  scene: {len(list(result.scene.lines()))} regels"
               + (f", ontbreekt: {', '.join(missing)}" if missing else ", alle momenten aanwezig"))
    for name, path in result.files.items():
        m = result.manifests[name]
        typer.echo(f"  {name:<10} {path}  (geschatte uitlijning: {m.alignment_fallback_count()})")
    typer.echo(f"  rapport: {result.report}")
    if any(m.alignment_fallback_count() for m in result.manifests.values()):
        typer.echo("  [!!] een deel van de beurten heeft geschatte woordtiming; daar is phrasing overgeslagen. "
                   "Draai `studiepodcast doctor`.")


reactions_app = typer.Typer(help="Reaction bank: curated laughs, sighs and short reactions per speaker.", no_args_is_help=True)
app.add_typer(reactions_app, name="reactions")


@reactions_app.command("generate")
def reactions_generate(speaker: Annotated[str, typer.Option(help="Speaker id (e.g. tessa), comma-separated, or 'all' (hosts).")] = "all",
                       label: Annotated[str, typer.Option(help="laugh, chuckle, sigh, hm, ja, oh, wacht, precies; comma-separated or 'all'.")] = "all",
                       n: Annotated[int, typer.Option(help="Candidates per speaker per label.")] = 12):
    """Render candidates into cast/reactions/_candidates/<speaker>/<label>/; move the good ones into
    cast/reactions/<speaker>/<label>/ (real recorded clips can go there directly too).

    Without options: every label for both hosts, the model loaded once.
    """
    from pipeline.audio.reactions import ReactionBank, generate_candidates
    from pipeline.audio.synth import make_synth, voice_spec_for
    from pipeline.performance import REACTIONS
    from pipeline.script.cast import load_cast

    settings = state["settings"]
    cast = load_cast(settings.cast_dir)
    speakers = [h.id for h in cast.hosts] if speaker == "all" else [s.strip() for s in speaker.split(",") if s.strip()]
    labels = list(REACTIONS) if label == "all" else [x.strip() for x in label.split(",") if x.strip()]
    unknown = [x for x in labels if x not in REACTIONS]
    if unknown:
        raise typer.BadParameter(f"unknown label(s) {', '.join(unknown)}; choose from {', '.join(REACTIONS)} or all")
    voices = {s: voice_spec_for(cast.speaker(s), settings.cast_dir) for s in speakers}  # KeyError on a typo, before any GPU work
    bank = ReactionBank(settings.cast_dir)
    synth = make_synth("null" if state["fake_audio"] else "final", settings, pool=False)
    total, skipped = 0, 0
    for s in speakers:
        for lab in labels:
            paths, failures = generate_candidates(synth, voices[s], lab, bank.candidates_dir(s, lab), n=n)
            total += len(paths)
            skipped += len(failures)
            typer.echo(f"  {s:<10} {lab:<8} {len(paths)} kandidaten -> {bank.candidates_dir(s, lab)}")
            for failure in failures:
                typer.echo(f"    [overgeslagen] {failure}")
    typer.echo(f"  klaar: {total} kandidaten, {skipped} overgeslagen. Beluister ze en verplaats de goede naar "
               f"{bank.root / '<speaker>' / '<label>'}")


@reactions_app.command("list")
def reactions_list():
    """How many curated clips each speaker has per reaction label."""
    from pipeline.audio.reactions import ReactionBank
    from pipeline.script.cast import load_cast

    settings = state["settings"]
    cast = load_cast(settings.cast_dir)
    speakers = [s.id for s in list(cast.hosts) + list(cast.guests)]
    for speaker, labels in ReactionBank(settings.cast_dir).coverage(speakers).items():
        typer.echo(f"  {speaker:<14} " + "  ".join(f"{k}={v}" for k, v in labels.items()))


@app.command()
def continuity(last: int = 10):
    """Print the continuity log the writer will see."""
    from pipeline.script.cast import continuity_text, load_continuity

    typer.echo(continuity_text(load_continuity(state["settings"].cast_dir, last_n=last)))


@app.command()
def sample_pdf(out: Path = Path("data/sample_book.pdf"), toc: bool = True):
    """Generate the small synthetic Dutch study book used by the tests."""
    from pipeline.sample_book import build

    typer.echo(str(build(out, with_toc=toc)))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the web app (upload, watch stages stream, inspect artifacts, edit scripts)."""
    import os

    import uvicorn

    os.environ.setdefault("STUDIEPODCAST_DATA_DIR", str(state["settings"].data_dir))
    os.environ.setdefault("STUDIEPODCAST_CAST_DIR", str(state["settings"].cast_dir))
    if state["fake_llm"]:
        os.environ["STUDIEPODCAST_FAKE_LLM"] = "1"
    if state["fake_audio"]:
        os.environ["STUDIEPODCAST_FAKE_AUDIO"] = "1"
    uvicorn.run("app.api:app", host=host, port=port, reload=reload)


def _chapters(p, book_id: str, chapter: str | None) -> list[str]:
    if chapter:
        return [chapter]
    return [c.id for c in p.load_book(book_id).episode_chapters()]


def _print_audit(a) -> None:
    typer.echo(f"  audit: {'PASSED' if a.passed else 'BLOCKED'}  blocking={len(a.blocking())} warnings={len(a.warnings())} "
               f"coverage missing={a.coverage.missing} est={a.stats.get('estimated_minutes')} min")
    for issue in a.blocking():
        typer.echo(f"    [{issue.check}/{issue.rule}] {issue.line_id or issue.claim_id or ''}: {issue.message}")


if __name__ == "__main__":
    app()
