/* Luisterstof: a website that walks you through it. Hand in a book, let it write overnight, read and approve each
   chapter, let the voices record it, listen. One page per step, one orange button for what to do now. */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const state = {
  view: null, book: null, chapter: null, step: null, data: null, glossary: null, timings: null, status: null,
  books: [], jobs: [], health: null, names: {}, hosts: [], dirty: false, stageOpen: {}, runQuality: false,
  source: null, sourceFor: null, workTab: "script", selectedLine: null, selectedClaim: null,
  voices: null, voice: null, voiceTakes: [], voiceDraft: {},
  engineDetail: { writer: "", voice: "" },
};

const api = {
  async get(url) { const r = await fetch(url); if (!r.ok) throw new Error(await r.text()); return r.json(); },
  async send(url, method, body) {
    const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};
const enc = encodeURIComponent;
const bookUrl = (rest = "") => `/api/books/${enc(state.book)}${rest}`;
const chapterUrl = (rest = "") => bookUrl(`/chapters/${enc(state.chapter)}${rest}`);

// ------------------------------------------------------------------ helpers
function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function log(text) { const el = $("#log"); el.textContent += text + "\n"; el.scrollTop = el.scrollHeight; }
function chip(text, kind = "", title = "", extra = "") {
  const cls = ["chip", kind && `chip--${kind}`, extra].filter(Boolean).join(" ");
  return `<span class="${cls}"${title ? ` title="${escapeHtml(title)}"` : ""}>${escapeHtml(text)}</span>`;
}
function remember(key, value) { try { localStorage.setItem(`luisterstof.${key}`, value); } catch { /* private window */ } }
function recall(key) { try { return localStorage.getItem(`luisterstof.${key}`); } catch { return null; } }
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

let toastTimer = null;
function toast(text, bad = false) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.toggle("is-bad", bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 9000 : 4500);
}
// FastAPI answers {"detail": "..."}; show the sentence, not the JSON.
function errorText(err) {
  const raw = (err && err.message) || String(err);
  try { const j = JSON.parse(raw); if (j && j.detail) return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* plain text */ }
  return raw;
}
function fail(err, what) {
  const msg = errorText(err);
  log(`fout${what ? " bij " + what.toLowerCase() : ""}: ${msg}`);
  toast(`${what ? what + " mislukt: " : ""}${msg}`, true);
}
// What a failure means, in Dutch, next to the raw message.
function explainError(err) {
  if (!err) return "";
  if (/worker restarted/i.test(err)) return "De app stopte terwijl dit liep (herstart of gesloten venster). Verder schrijven gaat door waar het was.";
  if (/opencode|not reachable|does not answer|LLMUnavailable/i.test(err)) return "Het schrijfmodel antwoordde niet. Start OpenCode en probeer het opnieuw.";
  if (/chapters failed/i.test(err)) return "De andere hoofdstukken zijn wel gemaakt. Verder schrijven maakt alleen wat ontbreekt.";
  return "";
}
async function busy(btn, label, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = label;
  try { return await fn(); } finally { btn.disabled = false; btn.innerHTML = old; }
}

// ----------------------------------------------------------------- formats
function fmtSeconds(seconds) {
  const s = Math.round(seconds || 0), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h} u ${String(m).padStart(2, "0")} min` : m ? `${m} min ${String(sec).padStart(2, "0")} s` : `${sec} s`;
}
function fmtClock(iso) { return iso ? new Date(iso).toLocaleTimeString("nl-NL", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : ""; }
function fmtWhen(iso) {
  return iso ? new Date(iso).toLocaleString("nl-NL", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
}
function fmtDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

const STAGES = { ingest: "Inlezen", pipeline: "Boek verwerken", plan: "Plan", glossary: "Uitspraak", script: "Script schrijven",
  audit: "Controle", continuity: "Continuïteit", render: "Opname", approve: "Goedkeuren", all: "Hoofdstuk schrijven",
  book: "Boek", run: "Hoofdstuk", voice: "Stem proberen" };
const PASS_NAMES = { ingest: "inlezen", plan: "plan", glossary: "uitspraak", script: "script schrijven", audit: "controle",
  revise: "herschrijven", polish: "tafellezing", continuity: "continuïteit", render: "opname", chapter: "hele hoofdstuk" };
const passTitle = (p) => p.title || ((PASS_NAMES[p.pass_name] || p.pass_name) + (p.label ? ` (${p.label})` : ""));
const SEGMENTS = { cold_open: "Koude opening", recap: "Terugblik", body: "Blok", guest: "Gast", reexplain: "Nog eens uitgelegd",
  quiz: "Quiz", outro: "Afsluiting" };
const KINDS = { loanword_en: "Engels leenwoord", notation: "notatie", abbreviation: "afkorting" };
const OVERLAP = { none: "geen", interrupt: "onderbreekt", backchannel: "reactie" };
const JOB_STATUS = { queued: ["in de wachtrij", ""], running: ["bezig", "run"], done: ["klaar", "ok"], failed: ["mislukt", "bad"],
  cancelled: ["gestopt", "warn"] };
const TIMING_STATUS = { done: "", failed: chip("mislukt", "bad"), stopped: chip("gestopt", "warn"), skipped: chip("overgeslagen") };
const ACTIVE = ["running", "queued"];
const jobName = (j) => j.book_id === VOICELAB ? `Take voor ${state.names[j.chapter_id] || j.chapter_id}`
  : `${STAGES[j.stage] || j.stage}${j.chapter_id ? " " + j.chapter_id : ""}`;
const STEP_SLUG = { review: "nalezen", record: "opnemen", listen: "luisteren" };
const SLUG_STEP = Object.fromEntries(Object.entries(STEP_SLUG).map(([k, v]) => [v, k]));
const STEP_ORDER = ["review", "record", "listen"];

// ------------------------------------------------------------------ routes
// #/  #/nieuw  #/boek/<id>  #/boek/<id>/<hoofdstuk>  #/boek/<id>/<hoofdstuk>/<stap>
const bookHash = (id) => `#/boek/${enc(id)}`;
const chapterHash = (book, ch, step) => `${bookHash(book)}/${enc(ch)}${step && STEP_SLUG[step] ? "/" + STEP_SLUG[step] : ""}`;

function parseHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (h === "/nieuw") return { view: "new" };
  const v = h.match(/^\/stemmen(?:\/([^/]+))?$/);
  if (v) return { view: "voices", voice: v[1] || null };
  const m = h.match(/^\/boek\/([^/]+)(?:\/([^/]+))?(?:\/([^/]+))?$/);
  if (m) return { view: m[2] ? "chapter" : "book", book: m[1], chapter: m[2] || null, step: SLUG_STEP[m[3]] || null };
  return { view: "home" };
}

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

let lastHash = location.hash;
window.addEventListener("hashchange", () => {
  const r = parseHash();
  const stays = r.view === "chapter" && r.book === state.book && r.chapter === state.chapter;
  if (state.dirty && !stays) {
    if (!confirm("Je wijzigingen in het script zijn nog niet opgeslagen. Toch verlaten?")) { history.replaceState(null, "", lastHash); return; }
    setDirty(false);
  }
  lastHash = location.hash;
  route();
});

async function route() {
  const r = parseHash();
  $("#sys").open = false;
  if (r.view === "new") { showView("new"); renderEngineWarning(); return; }
  if (r.view === "home") { showView("home"); renderHome(); return; }
  if (r.view === "voices") { showView("voices"); await openVoices(r.voice); return; }
  if (!state.books.some(b => b.book_id === r.book)) {
    try { await loadBooks(); } catch { /* shown below */ }
    if (!state.books.some(b => b.book_id === r.book)) { toast(`Het boek ${r.book} bestaat niet (meer).`, true); go("#/"); return; }
  }
  if (r.book !== state.book || !state.status || r.view === "book") {
    try { await loadBook(r.book); } catch (err) { fail(err, "Boek openen"); return; }
  }
  if (r.view === "book") { state.chapter = null; state.data = null; showView("book"); renderBook(); return; }
  if (!state.status.chapters.some(c => c.id === r.chapter)) { toast(`Hoofdstuk ${r.chapter} bestaat niet.`, true); go(bookHash(r.book)); return; }
  await openChapter(r.chapter, r.step);
}

function showView(name) {
  const changed = name !== state.view;
  state.view = name;
  for (const v of ["home", "new", "book", "chapter", "voices"]) $(`#view-${v}`).hidden = v !== name;
  $$(".site-nav a").forEach(a => a.toggleAttribute("aria-current", (a.dataset.nav === "home" && ["home", "book", "chapter"].includes(name)) || a.dataset.nav === name || (a.dataset.nav === "new" && name === "new")));
  $("#savebar").hidden = !(state.dirty && name === "chapter");
  if (changed) window.scrollTo(0, 0);
  document.title = name === "chapter" && state.data ? `${$("#chapter-title").textContent} · Luisterstof`
    : name === "book" && state.status ? `${state.status.title || state.status.book_id} · Luisterstof` : "Luisterstof";
}

// ------------------------------------------------------------------ data
async function loadBooks() {
  state.books = await api.get("/api/books");
  return state.books;
}

async function loadBook(bookId) {
  if (bookId !== state.book) { state.chapter = null; state.data = null; state.stageOpen = {}; state.source = null; state.sourceFor = null; }
  state.book = bookId;
  remember("book", bookId);
  const [s, timings] = await Promise.all([
    api.get(`/api/books/${enc(bookId)}`),
    api.get(`/api/books/${enc(bookId)}/timings`).catch(() => null),
  ]);
  state.status = s;
  state.timings = timings;
  const i = state.books.findIndex(b => b.book_id === bookId);
  if (i >= 0) state.books[i] = s;
  await loadJobs();
}

let refreshTimer = null, refreshChapter = false;
// Debounced: a run replayed after a reload sends hundreds of events at once. With `chapter` the open chapter is
// reloaded too, unless you are editing it.
function refreshBookSoon(chapter = false) {
  refreshChapter = refreshChapter || chapter;
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    const withChapter = refreshChapter;
    refreshChapter = false;
    try {
      if (state.book) await loadBook(state.book);
      else await loadJobs();
      if (withChapter && state.view === "chapter" && state.chapter && !state.dirty) await openChapter(state.chapter, state.step, { keep: true });
      else afterJobsChanged();
    } catch { /* the next refresh catches up */ }
  }, 1200);
}

// The latest run that did work on a chapter. A resumed run that skipped it does not wipe its durations.
function chapterTiming(id) {
  const entries = ((state.timings && state.timings.entries) || []).filter(e => e.chapter === id);
  const worked = entries.filter(e => e.pass_name !== "chapter" && e.status !== "skipped").map(e => e.run).sort();
  const runId = worked[worked.length - 1];
  if (!runId) return (state.timings && state.timings.summary && state.timings.summary.chapters[id]) || null;
  const passes = entries.filter(e => e.run === runId);
  const seconds = passes.filter(p => p.pass_name !== "chapter" && p.status !== "skipped").reduce((n, p) => n + (p.seconds || 0), 0);
  return { run: runId, passes, seconds };
}
function chapterInfo(id) { return state.status ? state.status.chapters.find(c => c.id === id) : null; }
const isActive = (j) => ACTIVE.includes(j.status);
function chapterJob(id) { return state.jobs.find(j => j.book_id === state.book && j.chapter_id === id && isActive(j)); }
function bookJob(bookId = state.book) { return state.jobs.find(j => j.book_id === bookId && !j.chapter_id && isActive(j)); }
function chapterBusy(id) { return !!chapterJob(id) || (!!run.job && !run.job.chapter_id && run.job.book_id === state.book && run.chapterId === id); }
function bookBusy(bookId = state.book) { return state.jobs.some(j => j.book_id === bookId && isActive(j)); }

// The latest run that touched a chapter: did a pass fail or stop, and why.
function lastFailure(id) {
  const t = chapterTiming(id);
  if (!t) return null;
  const bad = t.passes.filter(p => p.status === "failed" || p.status === "stopped");
  if (!bad.length) return null;
  const pass = bad.find(p => p.pass_name !== "chapter") || bad[0];
  const whole = bad.find(p => p.pass_name === "chapter");
  return { status: pass.status, pass: pass.pass_name, title: passTitle(pass), error: pass.error || (whole && whole.error) || "" };
}
function failureText(f) {
  if (!f) return "";
  return `De laatste keer ${f.status === "stopped" ? "is het gestopt" : "liep het vast"} bij ${f.title}` +
    (f.error ? `: ${f.error}` : ".") + (explainError(f.error) ? ` ${explainError(f.error)}` : "");
}

// The newest run that went over the whole book (an upload or Verder schrijven).
function latestBookRun() {
  const entries = (state.timings && state.timings.entries) || [];
  const ids = [...new Set(entries.filter(e => e.pass_name === "chapter" || e.pass_name === "ingest").map(e => e.run))].sort();
  const id = ids[ids.length - 1];
  if (!id) return null;
  const rows = entries.filter(e => e.run === id);
  const started = rows.map(e => e.started_at).sort()[0];
  const finished = rows.map(e => e.finished_at).sort().pop();
  const chapters = rows.filter(e => e.pass_name === "chapter");
  const n = (st) => chapters.filter(e => e.status === st).length;
  const skipped = chapters.filter(c => rows.filter(e => e.chapter === c.chapter && e.pass_name !== "chapter").every(e => e.status === "skipped")).length;
  return { started, finished, seconds: (Date.parse(finished) - Date.parse(started)) / 1000,
    chapters: chapters.length, done: n("done") - skipped, skipped, failed: n("failed"), stopped: n("stopped") };
}

// Where a chapter stands, for the book page and the home page.
function phase(ch) {
  if (!ch.script || !ch.audit) return "write";
  if (!ch.approved) return "review";
  if (!ch.final) return "record";
  return "listen";
}
function bookCounts(s) {
  const c = { n: s.chapters.length, write: 0, review: 0, record: 0, listen: 0, scripted: 0 };
  for (const ch of s.chapters) { c[phase(ch)] += 1; if (ch.script) c.scripted += 1; }
  return c;
}

// ------------------------------------------------------------------ home
function renderHome() {
  const books = state.books;
  $("#home-welcome").hidden = books.length > 0;
  $("#home-books").hidden = !books.length;
  if (!books.length) return;
  const last = recall("book");
  const sorted = [...books].sort((a, b) => (b.book_id === last) - (a.book_id === last));
  $("#book-cards").innerHTML = sorted.map((s, i) => {
    const c = bookCounts(s);
    const working = bookBusy(s.book_id);
    let line, kind;
    if (!s.ingested) [line, kind] = working ? ["Wordt ingelezen", "run"] : ["Inlezen is niet gelukt", "bad"];
    else if (working) [line, kind] = ["Luisterstof is aan het werk", "run"];
    else if (c.review) [line, kind] = [`${plural(c.review, "hoofdstuk wacht", "hoofdstukken wachten")} op je goedkeuring`, "warn"];
    else if (c.record) [line, kind] = [`${plural(c.record, "hoofdstuk is", "hoofdstukken zijn")} klaar om op te nemen`, "warn"];
    else if (c.write) [line, kind] = [`${plural(c.write, "hoofdstuk moet", "hoofdstukken moeten")} nog geschreven worden`, ""];
    else [line, kind] = ["Alle afleveringen zijn klaar", "ok"];
    const pct = (x) => c.n ? (100 * x / c.n).toFixed(1) : 0;
    return `<a class="bookcard${i === 0 ? " is-recent" : ""}" href="${bookHash(s.book_id)}">
      <span class="bookcard__title">${escapeHtml(s.title || s.book_id)}</span>
      <span class="bookcard__meta">${s.ingested ? `${plural(c.n, "hoofdstuk", "hoofdstukken")} · ${plural(c.listen, "aflevering", "afleveringen")} klaar` : escapeHtml(s.book_id)}</span>
      ${s.ingested ? `<span class="meter" title="${c.listen} opgenomen, ${c.record} goedgekeurd, ${c.review} geschreven, ${c.write} nog te schrijven">
        <i class="meter__listen" style="width:${pct(c.listen)}%"></i><i class="meter__record" style="width:${pct(c.record)}%"></i><i class="meter__review" style="width:${pct(c.review)}%"></i></span>` : ""}
      <span class="bookcard__status">${chip(line, kind)}<span class="bookcard__go">${i === 0 ? "Verder" : "Openen"} →</span></span>
    </a>`;
  }).join("");
}

// ------------------------------------------------------------- book journey
function renderBook() {
  const s = state.status;
  if (!s) return;
  const title = s.title || s.book_id;
  $("#crumb-book").textContent = title;
  $("#book-title").textContent = title;
  const c = bookCounts(s);
  $("#book-meta").textContent = s.ingested
    ? `${plural(c.n, "hoofdstuk", "hoofdstukken")}, elk wordt één aflevering. Hieronder zie je waar je bent en wat de volgende stap is.`
    : "Hieronder zie je hoe het met het inlezen staat.";
  let primaryUsed = false;
  const button = (label, attrs, { primary = true, disabled = false } = {}) => {
    const isPrimary = primary && !disabled && !primaryUsed;
    if (isPrimary) primaryUsed = true;
    return `<button class="btn ${isPrimary ? "btn--primary" : "btn--secondary"}" ${attrs}${disabled ? " disabled" : ""}>${escapeHtml(label)}</button>`;
  };
  const stages = journeyStages(s, c, button);
  // One step open: the first one that needs you (and a step that is running, to follow it). The rest is a line.
  const focus = (stages.find(st => st.status === "failed" || st.status === "active") || {}).key;
  $("#journey").innerHTML = stages.map((st, i) => {
    const open = state.stageOpen[st.key] ?? (st.key === focus || st.status === "running");
    const mark = st.status === "done" ? "✓" : st.status === "failed" ? "!" : String(i + 1);
    return `<li class="jstep jstep--${st.status}">
      <details data-key="${st.key}"${open ? " open" : ""}>
        <summary><span class="jstep__mark" aria-hidden="true">${mark}</span>
          <span class="jstep__head"><span class="jstep__title">${escapeHtml(st.title)}</span><span class="jstep__summary">${st.summary}</span></span></summary>
        <div class="jstep__body">${st.body}</div>
      </details></li>`;
  }).join("");
  $("#run-book-quality") && ($("#run-book-quality").checked = state.runQuality);
  applyQualityMode();
  renderChapterCards(s);
}

const PHASE_CHIP = {
  write: ["nog te schrijven", ""], review: ["nalezen", "warn"], record: ["klaar om op te nemen", "warn"], listen: ["opgenomen", "ok"],
};
function renderChapterCards(s) {
  $("#book-chapters").hidden = !s.ingested || !s.chapters.length;
  $("#chapter-cards").innerHTML = s.chapters.map(ch => {
    let [label, kind] = PHASE_CHIP[phase(ch)];
    if (phase(ch) === "review" && ch.audit_passed === false) [label, kind] = [plural(ch.audit_blocking, "probleem", "problemen"), "bad"];
    if (phase(ch) === "write" && lastFailure(ch.id)) [label, kind] = ["liep vast", "bad"];
    if (chapterBusy(ch.id)) [label, kind] = ["bezig", "run"];
    const t = chapterTiming(ch.id);
    return `<li><a class="chcard" href="${chapterHash(s.book_id, ch.id)}">
      <span class="chcard__top"><span class="chcard__id">${escapeHtml(ch.id)}</span><span class="chcard__title">${escapeHtml(ch.title)}</span>${chip(label, kind)}</span>
      <span class="chcard__summary">${ch.summary ? escapeHtml(ch.summary) : '<span class="subtle">Nog geen samenvatting: die komt met het plan.</span>'}</span>
      <span class="chcard__meta">${pagesText(ch.pages)} · ${ch.chars.toLocaleString("nl-NL")} tekens${t && t.seconds ? ` · gemaakt in ${fmtSeconds(t.seconds)}` : ""}</span>
    </a></li>`;
  }).join("");
}

function chapterRows(items, action) {
  return `<ul class="chrows">${items.map(({ ch, note, btn }) => `<li>
    <span class="chrows__title"><span class="chrows__id">${escapeHtml(ch.id)}</span> ${escapeHtml(ch.title)}${note ? `<span class="chrows__note">${note}</span>` : ""}</span>
    ${btn || ""}</li>`).join("")}</ul>${action || ""}`;
}

function journeyStages(s, c, button) {
  const stages = [];
  const job = bookJob(s.book_id);
  const lastBookJob = state.jobs.find(j => j.book_id === s.book_id && !j.chapter_id);
  const writerDown = $("#engine-writer").dataset.state === "down";
  const writerWarn = writerDown ? `<div class="callout callout--bad"><b>Het schrijfmodel staat uit.</b> Start de OpenCode-app en klik bovenaan op het statuslampje om opnieuw te controleren.</div>` : "";

  // 1 read the book in
  if (s.ingested) {
    stages.push({ key: "ingest", title: "Boek inlezen", status: "done", summary: `${plural(c.n, "hoofdstuk", "hoofdstukken")} gevonden`,
      body: `<ol class="toc">${s.chapters.map(ch => `<li><a href="${chapterHash(s.book_id, ch.id)}">${escapeHtml(ch.title)}</a><span class="muted small">${pagesText(ch.pages)}</span></li>`).join("")}</ol>
        <p class="muted small">Klopt de indeling niet? Voeg het boek opnieuw toe en kies onder Meer opties een andere manier om hoofdstukken te vinden.</p>` });
  } else if (bookBusy(s.book_id)) {
    stages.push({ key: "ingest", title: "Boek inlezen", status: "running", summary: "Bezig",
      body: "<p>Luisterstof zoekt de hoofdstukken in de PDF. Dat duurt meestal een paar minuten.</p>" });
  } else {
    const err = lastBookJob && lastBookJob.error;
    stages.push({ key: "ingest", title: "Boek inlezen", status: "failed", summary: "Niet gelukt",
      body: `<p>Het inlezen van dit boek is niet gelukt.</p>${err ? `<div class="errorbox">${escapeHtml(err)}</div>` : ""}
        <p>Voeg de PDF opnieuw toe. Onder Meer opties kun je een andere manier kiezen om de hoofdstukken te vinden.</p>
        <div class="actions">${button("Opnieuw toevoegen", 'data-action="new"')}</div>` });
  }

  // 2 write the scripts
  const latest = latestBookRun();
  const runLine = latest ? `<p class="muted">De laatste run begon ${fmtWhen(latest.started)} en duurde ${fmtSeconds(latest.seconds)}.</p>` : "";
  if (!s.ingested) {
    stages.push({ key: "write", title: "Scripts laten schrijven", status: "locked", summary: "Na het inlezen", body: "<p>Zodra de hoofdstukken bekend zijn, schrijft Luisterstof per hoofdstuk een script.</p>" });
  } else if (job) {
    const at = run.job && run.job.id === job.id && run.book ? `Nu bij hoofdstuk ${run.book.index} van ${run.book.total}: ${escapeHtml(run.book.label)}.` : "Bezig.";
    stages.push({ key: "write", title: "Scripts laten schrijven", status: "running", summary: job.status === "queued" ? "In de wachtrij" : "Bezig",
      body: `<p>${at} Je kunt dit venster sluiten: het werk gaat door zolang de app draait. Kijk morgen weer.</p>
        <p class="muted">${plural(c.n - c.write, "script is", "scripts zijn")} al klaar, ${c.write} nog te gaan. Wat klaar is, verschijnt meteen hieronder bij Nalezen.</p>
        <div class="actions"><button class="btn btn--outline" data-action="stop" data-job="${escapeHtml(job.id)}">Stoppen</button></div>` });
  } else if (!c.write) {
    stages.push({ key: "write", title: "Scripts laten schrijven", status: "done", summary: `Alle ${c.n} scripts geschreven en gecontroleerd`, body: runLine || "<p>Alle scripts zijn er.</p>" });
  } else {
    const failed = s.chapters.filter(ch => phase(ch) === "write" && lastFailure(ch.id));
    const quality = `<label class="check" title="Meer controlerondes en herschrijfrondes. Kan uren per hoofdstuk duren."><input type="checkbox" id="run-book-quality" class="quality-box"> grondig (langzamer, beter)</label>`;
    const jobErr = lastBookJob && lastBookJob.status === "failed" && lastBookJob.error
      ? `<div class="errorbox"><b>De laatste run stopte:</b> ${escapeHtml(lastBookJob.error)}${explainError(lastBookJob.error) ? `<br>${escapeHtml(explainError(lastBookJob.error))}` : ""}</div>` : "";
    if (!c.scripted) {
      stages.push({ key: "write", title: "Scripts laten schrijven", status: "active", summary: "Nog niet begonnen",
        body: `<p>Luisterstof maakt voor elk hoofdstuk een plan van wat er feitelijk in moet, schrijft daar een gesprek tussen Tessa en Joris bij en controleert dat gesprek tegen het boek. Voor dit boek duurt dat uren: start het als de computer een nacht kan doorwerken.</p>
          ${writerWarn}${jobErr}<div class="actions">${button("Start met schrijven", 'data-action="run-book"')}${quality}</div>` });
    } else {
      stages.push({ key: "write", title: "Scripts laten schrijven", status: "active", summary: `${c.n - c.write} van ${c.n} geschreven`,
        body: `<p>${plural(c.write, "hoofdstuk heeft", "hoofdstukken hebben")} nog geen gecontroleerd script. Verder schrijven slaat over wat al klaar is.</p>
          ${failed.length ? `<ul class="fails">${failed.map(ch => `<li><b>${escapeHtml(ch.id)} ${escapeHtml(ch.title)}</b>: ${escapeHtml(failureText(lastFailure(ch.id)))}</li>`).join("")}</ul>` : ""}
          ${runLine}${writerWarn}${jobErr}<div class="actions">${button("Verder schrijven", 'data-action="run-book"')}${quality}</div>` });
    }
  }

  // 3 read and approve
  const toReview = s.chapters.filter(ch => phase(ch) === "review");
  const approved = s.chapters.filter(ch => ch.approved).length;
  if (!c.scripted) {
    stages.push({ key: "review", title: "Nalezen en goedkeuren", status: "locked", summary: "Zodra er scripts zijn",
      body: "<p>Jij leest elk script na voordat het opgenomen wordt. Luisterstof wijst aan waar het iets niet vertrouwt.</p>" });
  } else if (toReview.length) {
    const rows = toReview.map(ch => {
      const busyNow = chapterBusy(ch.id);
      const note = busyNow ? chip("bezig", "run") : ch.audit_passed === false ? chip(plural(ch.audit_blocking, "probleem", "problemen") + " om op te lossen", "bad") : chip("klaar om na te lezen", "warn");
      return { ch, note, btn: button(ch.audit_passed === false ? "Oplossen" : "Nalezen", `data-action="chapter" data-chapter="${escapeHtml(ch.id)}" data-step="review"`) };
    });
    stages.push({ key: "review", title: "Nalezen en goedkeuren", status: "active", summary: `${plural(toReview.length, "hoofdstuk wacht", "hoofdstukken wachten")} op jou`,
      body: `<p>Lees elk gesprek na zoals de luisteraar het hoort. Klopt het, dan keur je het goed en kan het opgenomen worden.</p>${chapterRows(rows)}
        ${approved ? `<p class="muted small">${plural(approved, "hoofdstuk is", "hoofdstukken zijn")} al goedgekeurd.</p>` : ""}` });
  } else if (approved === c.n) {
    stages.push({ key: "review", title: "Nalezen en goedkeuren", status: "done", summary: "Alles goedgekeurd", body: "<p>Je hebt alle scripts goedgekeurd. Je kunt ze altijd nog teruglezen via het hoofdstuk.</p>" });
  } else {
    stages.push({ key: "review", title: "Nalezen en goedkeuren", status: "waiting", summary: "Wacht op de volgende scripts",
      body: `<p>${plural(approved, "hoofdstuk is", "hoofdstukken zijn")} goedgekeurd. De rest verschijnt hier zodra het geschreven is.</p>` });
  }

  // 4 record
  const toRecord = s.chapters.filter(ch => phase(ch) === "record");
  const voiceDown = $("#engine-voice").dataset.state === "down";
  if (!toRecord.length && !c.listen) {
    stages.push({ key: "record", title: "Opnemen", status: "locked", summary: "Na je goedkeuring",
      body: "<p>Tessa en Joris lezen een goedgekeurd script in. Dat kost per hoofdstuk een paar uur rekentijd op de videokaart.</p>" });
  } else if (toRecord.length) {
    const rows = toRecord.map(ch => ({ ch, note: chapterBusy(ch.id) ? chip("wordt opgenomen", "run") : "",
      btn: button("Opnemen", `data-action="chapter" data-chapter="${escapeHtml(ch.id)}" data-step="record"`, { primary: !chapterBusy(ch.id) }) }));
    stages.push({ key: "record", title: "Opnemen", status: "active", summary: `${plural(toRecord.length, "hoofdstuk is", "hoofdstukken zijn")} klaar om op te nemen`,
      body: `<p>Elke beurt wordt een paar keer opgenomen en de opname die het script het best volgt, wordt gekozen. Reken op een paar uur per hoofdstuk.</p>
        ${voiceDown ? '<div class="callout callout--bad"><b>De stemmen zijn niet geïnstalleerd.</b> Draai <code>studiepodcast doctor</code> in de map van Luisterstof om te zien wat er ontbreekt.</div>' : ""}
        ${chapterRows(rows)}` });
  } else if (c.listen === c.n) {
    stages.push({ key: "record", title: "Opnemen", status: "done", summary: "Alles opgenomen", body: "<p>Alle afleveringen zijn opgenomen.</p>" });
  } else {
    stages.push({ key: "record", title: "Opnemen", status: "waiting", summary: "Wacht op je goedkeuring", body: "<p>Keur een script goed, dan kan het hier opgenomen worden.</p>" });
  }

  // 5 listen
  const done = s.chapters.filter(ch => ch.final);
  if (!done.length) {
    stages.push({ key: "listen", title: "Luisteren", status: "locked", summary: "Na het opnemen", body: "<p>Hier staan straks de afleveringen.</p>" });
  } else {
    const rows = done.map(ch => ({ ch, note: ch.flagged_turns ? chip(`${ch.flagged_turns} twijfelachtige ${ch.flagged_turns === 1 ? "beurt" : "beurten"}`, "warn", "Beurten waarvan geen opname het script goed genoeg volgde. Beluister ze even.") : "",
      btn: button("Beluisteren", `data-action="chapter" data-chapter="${escapeHtml(ch.id)}" data-step="listen"`, { primary: false }) }));
    stages.push({ key: "listen", title: "Luisteren", status: done.length === c.n ? "done" : "ready", summary: `${plural(done.length, "aflevering", "afleveringen")} klaar`,
      body: chapterRows(rows) });
  }
  return stages;
}

async function runBook(btn) {
  if (!state.book) return;
  try {
    const job = await busy(btn, "Starten...", () => api.send(bookUrl("/run"), "POST", { upto: "script", quality: state.runQuality }));
    log("schrijven gestart: wat al klaar is wordt overgeslagen");
    toast("Luisterstof is begonnen met schrijven. Je kunt dit venster sluiten.");
    watch(job);
  } catch (err) { fail(err, "Schrijven starten"); }
}

// --------------------------------------------------------------- chapter
function chapterPhase(data) {
  if (!data.script) return "missing";
  if (!data.approved) return "review";
  if (!data.final.audio) return "record";
  return "listen";
}

async function openChapter(chapterId, step = null, { keep = false } = {}) {
  let data;
  try { data = await api.get(`/api/books/${enc(state.book)}/chapters/${enc(chapterId)}`); } catch (err) { fail(err, "Hoofdstuk openen"); return; }
  const scroll = window.scrollY;
  const same = chapterId === state.chapter && state.view === "chapter";
  if (!same) { state.selectedLine = null; state.selectedClaim = null; state.workTab = data.script ? "script" : "source"; }
  state.chapter = chapterId;
  state.data = data;
  setDirty(false);
  await loadSource();
  const info = chapterInfo(chapterId);
  const ids = state.status.chapters.map(c => c.id), at = ids.indexOf(chapterId);
  for (const [el, other] of [[$("#prev-chapter"), ids[at - 1]], [$("#next-chapter"), ids[at + 1]]]) {
    el.hidden = !other;
    if (other) { el.href = chapterHash(state.book, other); el.title = (chapterInfo(other) || {}).title || other; }
  }
  const title = (data.plan && data.plan.chapter_title) || (info && info.title) || chapterId;
  $("#crumb-chapter-book").textContent = state.status.title || state.book;
  $("#crumb-chapter-book").href = bookHash(state.book);
  $("#crumb-chapter").textContent = `Hoofdstuk ${chapterId}`;
  $("#chapter-overline").textContent = `Hoofdstuk ${chapterId}`;
  $("#chapter-title").textContent = title;
  const meta = [];
  if (info) meta.push(`pagina ${info.pages[0]} tot ${info.pages[1]}`);
  if (data.script) meta.push(`aflevering van ongeveer ${data.script.target_minutes} minuten`);
  $("#chapter-meta").textContent = meta.join(" · ");
  const current = chapterPhase(data);
  const allowed = current === "missing" ? [] : STEP_ORDER.slice(0, STEP_ORDER.indexOf(current) + 1);
  state.step = current === "missing" ? "missing" : allowed.includes(step) ? step : current;
  showView("chapter");
  renderTracker(current);
  renderChapter();
  renderWorkspace();
  renderArtifacts();
  renderTimings(chapterId);
  loadGlossary();
  window.scrollTo(0, keep && same ? scroll : 0);
  const want = chapterHash(state.book, chapterId, state.step === "missing" ? null : state.step);
  if (location.hash !== want) { history.replaceState(null, "", want); lastHash = want; }
}

function renderTracker(current) {
  const ci = STEP_ORDER.indexOf(current);
  $$("#tracker button").forEach(b => {
    const i = STEP_ORDER.indexOf(b.dataset.step);
    const st = current === "missing" ? "todo" : i < ci ? "done" : i === ci ? "current" : "todo";
    b.dataset.state = st;
    b.disabled = st === "todo";
    b.toggleAttribute("aria-current", b.dataset.step === state.step);
    $(".tracker__n", b).textContent = st === "done" ? "✓" : String(i + 1);
  });
}

function renderChapter() {
  const data = state.data;
  if (!data) return;
  for (const s of ["missing", "review", "record", "listen"]) $(`#step-${s}`).hidden = s !== state.step;
  $("#tracker").hidden = state.step === "missing";
  $("#review-end").hidden = state.step !== "review";
  if (state.step === "missing") renderMissing(data);
  if (state.step === "review") renderReviewGuide(data);
  if (state.step === "record") renderRecord(data);
  if (state.step === "listen") renderAudio(data);
}

// ------------------------------------------------------------ workspace
// The chapter itself, next to whatever step you are on: the script, the book's text, the summary, the audit.
function renderWorkspace() {
  const data = state.data;
  if (data.script) renderScript(data);
  else $("#script-lines").innerHTML = `<p class="empty">Nog geen script. ${chapterBusy(state.chapter) ? "Het wordt nu geschreven." : "Zodra het geschreven is, lees je het hier."}</p>`;
  $(".readbar").hidden = !data.script;
  renderSource();
  renderPlan(data);
  renderAudit(data);
  renderLinePanel(state.selectedLine);
  renderClaimPanel(state.selectedClaim);
  showTab("work", state.workTab);
}

async function loadSource() {
  if (state.sourceFor === state.book && state.source) return;
  try { state.source = await api.get(bookUrl("/book")); state.sourceFor = state.book; } catch { state.source = null; }
}
function sourceChapter() {
  return state.source && state.source.chapters ? state.source.chapters.find(c => c.id === state.chapter) : null;
}
function sourceSection(id) {
  const ch = sourceChapter();
  return ch ? ch.sections.find(s => s.id === id) : null;
}
// The passage a claim rests on: the span in the book when it was found there, else the quote the plan gave.
function claimPassage(c) {
  const sp = c.source_span, sec = sourceSection(sp.section);
  const found = sec && sp.match_score > 0 && sp.end > sp.start && !(sp.start === 0 && sp.end >= sec.text.length);
  return { text: found ? sec.text.slice(sp.start, sp.end) : (sp.quote || ""), section: sec ? sec.title : sp.section,
    pages: sec && sec.pages ? sec.pages : null, found };
}
function claimById(id) { return state.data && state.data.plan ? state.data.plan.key_claims.find(c => c.id === id) : null; }
// Which lines carry a claim, read from the page so an edit shows up before you save.
function linesCovering(claimId) {
  return $$("#script-lines .line").filter(div => $(".covers", div).value.split(",").map(x => x.trim()).includes(claimId))
    .map(div => ({ id: div.dataset.id, speaker: $(".speaker", div).selectedOptions[0].text, text: $(".text", div).value }));
}
function issuesFor(lineId) {
  const d = state.data;
  return d.audit && auditState(d) !== "stale" ? d.audit.issues.filter(i => i.line_id === lineId) : [];
}
const pagesText = (pages) => pages ? (pages[0] === pages[1] ? `p. ${pages[0]}` : `p. ${pages[0]}-${pages[1]}`) : "";

function chapterAtAGlance() {
  const d = state.data, p = d.plan;
  if (!p) return '<p class="muted">Nog geen plan, dus nog geen samenvatting.</p>';
  const a = d.audit && auditState(d) !== "stale" ? d.audit : null;
  const blocking = a ? a.issues.filter(i => i.severity === "blocking").length : 0, tips = a ? a.issues.length - blocking : 0;
  const covered = new Set(d.script ? d.script.segments.flatMap(s => s.lines.flatMap(l => l.covers)) : []);
  const inScript = p.key_claims.filter(c => covered.has(c.id)).length;
  return `<h3>Dit hoofdstuk in het kort</h3><p class="panel__summary">${escapeHtml(p.summary)}</p>
    <dl class="facts">
      <div><dt>Beweringen</dt><dd>${p.key_claims.length}${d.script ? `, ${inScript} in het script` : ""}</dd></div>
      <div><dt>Begrippen</dt><dd>${p.definitions.length}</dd></div>
      <div><dt>Controle</dt><dd>${!d.script ? "nog geen script" : !a ? "nog niet gecontroleerd" : blocking ? `<span class="bad">${plural(blocking, "probleem", "problemen")}</span>, ${plural(tips, "tip", "tips")}` : `geslaagd, ${plural(tips, "tip", "tips")}`}</dd></div>
    </dl>`;
}

function renderLinePanel(lineId) {
  const panel = $("#line-panel"), d = state.data;
  $$("#script-lines .line.is-selected").forEach(el => el.classList.remove("is-selected"));
  const div = lineId ? $(`#script-lines .line[data-id="${CSS.escape(lineId)}"]`) : null;
  state.selectedLine = div ? lineId : null;
  if (!div) {
    panel.innerHTML = `<div class="panel">${chapterAtAGlance()}<p class="muted small">Klik op een regel om te zien op welk stuk uit het boek hij steunt en wat de controle ervan vindt.</p></div>`;
    return;
  }
  div.classList.add("is-selected");
  const speaker = $(".speaker", div).selectedOptions[0].text;
  const covers = $(".covers", div).value.split(",").map(x => x.trim()).filter(Boolean);
  const issues = issuesFor(lineId);
  const claims = covers.map(id => {
    if (id.includes(":")) return `<div class="srcquote"><b>${escapeHtml(id)}</b> <span class="muted">uit een eerder hoofdstuk</span></div>`;
    const c = claimById(id);
    if (!c) return `<div class="srcquote"><b>${escapeHtml(id)}</b> <span class="bad">staat niet in het plan</span></div>`;
    const ps = claimPassage(c);
    return `<div class="srcquote"><p><b>${escapeHtml(c.id)}</b> ${escapeHtml(c.claim)}</p>
      ${ps.text ? `<blockquote>${escapeHtml(ps.text)}</blockquote>` : ""}
      <p class="srcquote__where">${escapeHtml(ps.section)}${ps.pages ? ", " + pagesText(ps.pages) : ""}${ps.found ? ` · <button class="link" data-show-claim="${escapeHtml(c.id)}">in de boektekst</button>` : ""}</p></div>`;
  });
  panel.innerHTML = `<div class="panel">
    <div class="panel__head"><h3>${escapeHtml(speaker)}, regel ${escapeHtml(lineId)}</h3><button class="iconbtn" data-close-panel title="Terug naar de samenvatting" aria-label="Sluiten">&times;</button></div>
    ${issues.length ? `<div class="panel__issues">${issues.map(issueLine).join("")}</div>` : ""}
    <h4>Uit het boek</h4>
    ${claims.length ? claims.join("") : '<p class="muted">Deze regel brengt geen feit uit het boek over: een vraag, een reactie of een grap. Dat mag.</p>'}
  </div>`;
}

function renderSource() {
  const root = $("#source-view"), ch = sourceChapter(), d = state.data;
  if (!ch) { root.innerHTML = '<p class="empty">De tekst van dit hoofdstuk is niet gevonden.</p>'; return; }
  const claims = d.plan ? d.plan.key_claims : [];
  const covered = new Set(d.script ? d.script.segments.flatMap(s => s.lines.flatMap(l => l.covers)) : []);
  root.innerHTML = ch.sections.map(sec => {
    const spans = claims.filter(c => c.source_span.section === sec.id && claimPassage(c).found)
      .map(c => ({ c, start: c.source_span.start, end: c.source_span.end })).sort((a, b) => a.start - b.start);
    let html = "", at = 0;
    for (const sp of spans) {
      if (sp.start < at) continue;  // overlapping spans: the first one wins
      const missing = d.script && !covered.has(sp.c.id);
      html += escapeHtml(sec.text.slice(at, sp.start)) +
        `<mark class="srcmark${missing ? " srcmark--missing" : ""}${state.selectedClaim === sp.c.id ? " is-active" : ""}" data-claim="${escapeHtml(sp.c.id)}" id="src-${escapeHtml(sp.c.id)}" tabindex="0" title="${escapeHtml(sp.c.id + ": " + sp.c.claim)}">${escapeHtml(sec.text.slice(sp.start, sp.end))}</mark>`;
      at = sp.end;
    }
    html += escapeHtml(sec.text.slice(at));
    const title = sec.title && sec.title !== ch.title ? `<h3>${escapeHtml(sec.title)} <span class="muted small">${pagesText(sec.pages)}</span></h3>` : "";
    return `<section class="srcsec">${title}<div class="srcsec__text">${html}</div></section>`;
  }).join("");
}

function renderClaimPanel(claimId) {
  const panel = $("#claim-panel"), d = state.data, c = claimId ? claimById(claimId) : null;
  $$("#source-view mark.is-active").forEach(m => m.classList.remove("is-active"));
  state.selectedClaim = c ? claimId : null;
  if (!c) {
    const claims = d.plan ? d.plan.key_claims : [];
    const marked = claims.filter(x => claimPassage(x).found).length;
    panel.innerHTML = `<div class="panel">${chapterAtAGlance()}<p class="muted small">${d.plan ? `${marked} van de ${claims.length} beweringen zijn in de tekst gemarkeerd. ` : ""}Klik een markering om te zien welke bewering het is en waar die in het script terugkomt.</p></div>`;
    return;
  }
  const mark = $(`#src-${CSS.escape(claimId)}`);
  if (mark) mark.classList.add("is-active");
  const lines = linesCovering(claimId);
  panel.innerHTML = `<div class="panel">
    <div class="panel__head"><h3>Bewering ${escapeHtml(c.id)}</h3><button class="iconbtn" data-close-panel title="Terug naar de samenvatting" aria-label="Sluiten">&times;</button></div>
    <p>${escapeHtml(c.claim)}</p>
    <p>${chip(`moeilijkheid ${c.difficulty}`)} ${chip(`tentamen ${c.exam_relevance}`)}</p>
    <h4>In het script</h4>
    ${lines.length ? `<ul class="linelinks">${lines.map(l => `<li><button class="linelink" data-line="${escapeHtml(l.id)}"><b>${escapeHtml(l.speaker)}</b> ${escapeHtml(l.text.length > 140 ? l.text.slice(0, 138) + "..." : l.text)}</button></li>`).join("")}</ul>`
      : `<p class="bad">${d.script ? "Komt in geen enkele regel terug." : "Er is nog geen script."}</p>`}
  </div>`;
}

function showClaimInSource(claimId) {
  showTab("work", "source");
  renderClaimPanel(claimId);
  const mark = $(`#src-${CSS.escape(claimId)}`);
  if (mark) mark.scrollIntoView({ behavior: "smooth", block: "center" });
}

function actionButtons(el, items) {
  el.innerHTML = "";
  for (const it of items) {
    if (!it) continue;
    const b = document.createElement("button");
    b.className = `btn ${it.primary ? "btn--primary" : it.outline ? "btn--outline" : "btn--secondary"}`;
    b.textContent = it.label;
    b.disabled = !!it.disabled;
    if (it.title) b.title = it.title;
    b.onclick = it.action;
    el.appendChild(b);
  }
}
function setGuide(prefix, title, text) {
  $(`#${prefix}-title`).textContent = title;
  $(`#${prefix}-text`).textContent = text;
}

function renderMissing(data) {
  const busyNow = chapterBusy(state.chapter);
  const f = lastFailure(state.chapter);
  $("#missing-why").hidden = true;
  if (busyNow) {
    setGuide("missing", "Dit hoofdstuk wordt nu geschreven", "Kijk straks terug. De balk bovenaan laat zien hoe ver het is.");
    actionButtons($("#missing-actions"), [{ label: "Terug naar het boek", action: () => go(bookHash(state.book)) }]);
    return;
  }
  if (f) {
    setGuide("missing", "Het schrijven liep vast", "Dit hoofdstuk heeft nog geen script.");
    $("#missing-why").textContent = failureText(f);
    $("#missing-why").hidden = false;
  } else {
    setGuide("missing", "Nog geen script", "Dit hoofdstuk is nog niet geschreven. Je kunt het hele boek laten schrijven vanaf de boekpagina, of alleen dit hoofdstuk.");
  }
  actionButtons($("#missing-actions"), [
    { label: "Schrijf dit hoofdstuk", primary: true, action: () => runStage(data.plan ? "script" : "all") },
    { label: "Terug naar het boek", action: () => go(bookHash(state.book)) },
  ]);
}

// The audit only counts when it checked the script revision that is on disk now.
function auditState(data) {
  if (!data.script) return "none";
  if (!data.audit || data.audit.script_revision !== data.script.revision) return "stale";
  return data.audit.passed ? "passed" : "blocked";
}

function firstBlocking(data) {
  const blocking = data.audit ? data.audit.issues.filter(i => i.severity === "blocking") : [];
  return blocking.find(i => i.line_id && data.script.segments.some(s => s.lines.some(l => l.id === i.line_id)));
}

function renderReviewGuide(data) {
  const busyNow = chapterBusy(state.chapter);
  const a = auditState(data);
  const blocking = data.audit ? data.audit.issues.filter(i => i.severity === "blocking").length : 0;
  const toEnd = () => { showTab("work", "script"); $("#review-end").scrollIntoView({ behavior: "smooth", block: "center" }); };
  const recheck = (primary) => ({ label: "Opnieuw controleren", primary, action: () => runStage("audit") });
  let top = [], end = [];
  if (busyNow) {
    const job = chapterJob(state.chapter);
    setGuide("review", "Luisterstof werkt aan dit hoofdstuk", `${job ? STAGES[job.stage] || job.stage : "Schrijven"} is bezig. Het script kan nog veranderen, dus wacht even met aanpassen.`);
    setGuide("end", "Even wachten", "Goedkeuren kan zodra Luisterstof klaar is met dit hoofdstuk.");
    end = [{ label: "Goedkeuren", disabled: true }];
  } else if (data.approved) {
    setGuide("review", "Goedgekeurd", `Je hebt deze versie goedgekeurd. Pas je nog iets aan, dan vervalt de goedkeuring en wordt het script opnieuw gecontroleerd.`);
    top = [{ label: "Naar opnemen", primary: true, action: () => go(chapterHash(state.book, state.chapter, "record")) }];
    setGuide("end", "Goedgekeurd", "De stemmen kunnen dit hoofdstuk opnemen.");
    end = [{ label: "Naar opnemen", action: () => go(chapterHash(state.book, state.chapter, "record")) }];
  } else if (a === "stale") {
    setGuide("review", "Eerst opnieuw controleren", data.audit
      ? "Het script is aangepast sinds de laatste controle. Laat Luisterstof het opnieuw tegen het boek leggen voordat je goedkeurt."
      : "Dit script is nog niet gecontroleerd. Laat Luisterstof het tegen het boek leggen voordat je goedkeurt.");
    top = [recheck(true)];
    setGuide("end", "Eerst opnieuw controleren", "Goedkeuren kan als de controle van deze versie geslaagd is.");
    end = [recheck(false)];
  } else if (a === "blocked") {
    setGuide("review", `${plural(blocking, "probleem", "problemen")} om op te lossen`,
      "Luisterstof vond regels die niet kloppen met het boek of tegen de regels ingaan. Ze staan hieronder rood gemarkeerd, met uitleg. Pas de tekst aan en sla op; daarna wordt het opnieuw gecontroleerd. Liever een heel nieuw script? Dat kan onderaan bij Meer, onder Opnieuw maken.");
    const first = firstBlocking(data);
    top = [{ label: first ? "Naar het eerste probleem" : "Bekijk de problemen", primary: true,
      action: () => first ? jumpToLine(first.line_id) : showWork("audit") }];
    setGuide("end", "Nog niet goed te keuren", `Los eerst ${blocking === 1 ? "het rode probleem" : "de rode problemen"} op.`);
    end = [{ label: "Goedkeuren", disabled: true }];
  } else {
    const tips = data.audit ? data.audit.issues.filter(i => i.severity === "warning").length : 0;
    setGuide("review", "Lees het gesprek na", "Lees het zoals de luisteraar het straks hoort." +
      (tips ? ` Waar Luisterstof een tip heeft, staat die in oranje onder de regel; tips mogen blijven staan.` : "") +
      " Iets aanpassen? Klik in de tekst. Klopt alles, keur het dan onderaan goed.");
    top = [{ label: "Direct naar goedkeuren", action: toEnd }];
    setGuide("end", "Klaar met lezen?", "Na je goedkeuring kunnen de stemmen dit hoofdstuk opnemen.");
    end = [{ label: "Goedkeuren", primary: true, action: approve }];
  }
  if (state.dirty) {
    setGuide("end", "Eerst opslaan", "Sla je wijzigingen op; daarna wordt het script opnieuw gecontroleerd.");
    end = [{ label: "Goedkeuren", disabled: true }];
  }
  actionButtons($("#review-actions"), top);
  actionButtons($("#end-actions"), end);
}

function renderRecord(data) {
  const job = chapterJob(state.chapter);
  const voiceDown = $("#engine-voice").dataset.state === "down";
  const f = lastFailure(state.chapter);
  $("#record-why").hidden = true;
  if (job && job.stage === "render") {
    setGuide("record", "De stemmen nemen op", "Dit duurt een paar uur. De balk bovenaan laat zien hoe ver het is; je kunt dit venster sluiten.");
    actionButtons($("#record-actions"), []);
    return;
  }
  if (data.final.audio) {
    setGuide("record", "Opgenomen", "Deze aflevering is opgenomen. Opnieuw opnemen kan, bijvoorbeeld nadat je het script hebt aangepast en opnieuw goedgekeurd.");
    actionButtons($("#record-actions"), [
      { label: "Beluisteren", primary: true, action: () => go(chapterHash(state.book, state.chapter, "listen")) },
      { label: "Opnieuw opnemen", disabled: !!job, action: () => confirm("Opnieuw opnemen? Dit kost weer een paar uur.") && runStage("render", { force: true }) },
    ]);
    return;
  }
  setGuide("record", "Laat de stemmen het opnemen",
    "Tessa en Joris lezen het goedgekeurde script in. Elke beurt wordt een paar keer opgenomen en de opname die het script het best volgt, wordt gekozen. Reken op een paar uur rekentijd op de videokaart; start het als je de computer even niet nodig hebt.");
  if (voiceDown) {
    $("#record-why").textContent = "De stemmen zijn niet geïnstalleerd. Draai studiepodcast doctor in de map van Luisterstof om te zien wat er ontbreekt.";
    $("#record-why").hidden = false;
  } else if (f && f.pass === "render") {
    $("#record-why").textContent = failureText(f);
    $("#record-why").hidden = false;
  }
  actionButtons($("#record-actions"), [
    { label: "Start de opname", primary: true, disabled: !!job, action: () => runStage("render") },
    { label: "Het script nog eens lezen", action: () => go(chapterHash(state.book, state.chapter, "review")) },
  ]);
}

// ------------------------------------------------------------------ script
const FIELD_SIZING = !!(window.CSS && CSS.supports && CSS.supports("field-sizing", "content"));
function autosize(el) {
  if (FIELD_SIZING || !el.offsetParent) return;
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight + 2}px`;
}
function autosizeAll() { if (!FIELD_SIZING) $$("#script-lines textarea.text").forEach(autosize); }

function issueLine(i) {
  const blocking = i.severity === "blocking";
  return `<div class="lineissue lineissue--${blocking ? "bad" : "warn"}"><span class="lineissue__kind">${blocking ? "Probleem" : "Tip"}</span><span>${escapeHtml(i.message)}${i.suggestion ? ` <span class="muted">${escapeHtml(i.suggestion)}</span>` : ""} <span class="rule">${escapeHtml(i.rule)}</span></span></div>`;
}

function renderScript(data) {
  const root = $("#script-lines");
  root.innerHTML = "";
  const issues = {};
  for (const i of (data.audit && auditState(data) !== "stale" ? data.audit.issues : [])) if (i.line_id) (issues[i.line_id] ||= []).push(i);
  const total = data.script.segments.reduce((n, s) => n + s.lines.length, 0);
  $("#script-meta").textContent = `${total} regels · versie ${data.script.revision}`;
  // The cast order fixes each host's colour, so Tessa looks the same in every chapter.
  const speakers = [...new Set([...state.hosts, ...data.script.segments.flatMap(s => s.lines.map(l => l.speaker))])];
  const name = (id) => state.names[id] || id.charAt(0).toUpperCase() + id.slice(1);
  const frag = document.createDocumentFragment();
  let flagged = 0, blockNo = 0;
  for (const seg of data.script.segments) {
    const h = document.createElement("div");
    h.className = "segment" + (seg.lines.some(l => issues[l.id]) ? " has-issues" : "");
    const label = seg.type === "body" ? `Blok ${++blockNo}` : SEGMENTS[seg.type] || seg.type;
    h.innerHTML = `<span class="segment__type">${escapeHtml(label)}</span>` +
      (seg.covers.length ? `<span class="segment__covers">dekt ${escapeHtml(seg.covers.join(", "))}</span>` : "");
    frag.appendChild(h);
    for (const line of seg.lines) {
      const li = issues[line.id] || [];
      if (li.length) flagged += 1;
      const blocking = li.some(i => i.severity === "blocking");
      const div = document.createElement("div");
      div.className = "line" + (blocking ? " has-issue" : li.length ? " has-warning" : "") + (line.overlap.mode === "backchannel" ? " is-backchannel" : "");
      div.dataset.id = line.id;
      div.dataset.spk = String(speakers.indexOf(line.speaker));
      const mode = line.overlap.mode;
      div.innerHTML = `<div class="line__who">
          <select class="speaker" aria-label="Spreker van regel ${escapeHtml(line.id)}">${speakers.map(s => `<option value="${escapeHtml(s)}" ${s === line.speaker ? "selected" : ""}>${escapeHtml(name(s))}</option>`).join("")}</select>
          <span class="id">${escapeHtml(line.id)}</span></div>
        <div class="line__main">
          <textarea class="text" rows="1" aria-label="Tekst van regel ${escapeHtml(line.id)}">${escapeHtml(line.text)}</textarea>
          <div class="line__meta">
            <label class="mf" title="De emotie waarmee de regel wordt uitgesproken, gescheiden door komma's."><span>emotie</span><input class="tags" type="text" value="${escapeHtml(line.tags.join(", "))}"></label>
            <label class="mf" title="Welke beweringen uit het plan deze regel overbrengt (c1, c2)."><span>dekt</span><input class="covers" type="text" value="${escapeHtml(line.covers.join(", "))}"></label>
            <label class="mf" title="geen: een gewone beurt. onderbreekt: valt de vorige spreker in de rede. reactie: kort onder de ander door."><span>overlap</span><select class="overlap${mode === "none" ? " is-none" : ""}">${Object.entries(OVERLAP).map(([v, t]) => `<option value="${v}" ${mode === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>
            <label class="mf" title="Een bewuste stilte na de regel, in milliseconden."><span>pauze</span><input class="pause" type="number" min="0" step="250" placeholder="ms" value="${line.pause_after_ms || ""}"></label>
          </div>
          ${li.length ? `<div class="issues">${li.map(issueLine).join("")}</div>` : ""}
        </div>`;
      frag.appendChild(div);
    }
  }
  root.appendChild(frag);
  const box = $("#only-issues");
  box.disabled = !flagged;
  if (!flagged) box.checked = false;
  box.parentElement.lastChild.textContent = ` alleen regels met opmerkingen${flagged ? ` (${flagged})` : ""}`;
  root.classList.toggle("only-issues", box.checked);
  root.classList.toggle("show-details", $("#show-details").checked);
  autosizeAll();
}

function collectScript() {
  const script = JSON.parse(JSON.stringify(state.data.script));
  const byId = {};
  for (const seg of script.segments) for (const l of seg.lines) byId[l.id] = l;
  let prev = null;
  for (const div of $$("#script-lines .line")) {
    const l = byId[div.dataset.id];
    l.speaker = $(".speaker", div).value;
    l.text = $(".text", div).value.trim();
    l.tags = $(".tags", div).value.split(",").map(s => s.trim()).filter(Boolean);
    l.covers = $(".covers", div).value.split(",").map(s => s.trim()).filter(Boolean);
    const mode = $(".overlap", div).value;
    l.overlap = mode === "none" || !prev ? { mode: "none", target: null, cut_word: null } : { mode, target: prev.id, cut_word: mode === "interrupt" ? (prev.text.replace(/[—…\-. ]+$/, "").split(/\s+/).pop() || null) : null };
    l.pause_after_ms = parseInt($(".pause", div).value || "0", 10);
    prev = l;
  }
  return script;
}

function setDirty(on) {
  state.dirty = on;
  $("#savebar").hidden = !(on && state.view === "chapter");
  document.body.classList.toggle("has-savebar", on && state.view === "chapter");
}
function markDirty() {
  if (state.dirty || !state.data || !state.data.script) return;
  setDirty(true);
  if (state.step === "review") renderReviewGuide(state.data);
}

async function saveScript() {
  if (!state.data || !state.data.script || !state.dirty) return;
  if (chapterBusy(state.chapter)) {
    toast("Luisterstof werkt nog aan dit hoofdstuk. Opslaan kan als dat klaar is; je wijzigingen blijven staan.", true);
    return;
  }
  try {
    const out = await busy($("#save-script"), "Opslaan...", () => api.send(chapterUrl("/script"), "PUT", collectScript()));
    log(`script opgeslagen als versie ${out.revision}; controle en goedkeuring vervallen`);
    setDirty(false);
    await runStage("audit", {}, "Opgeslagen. Luisterstof controleert het script nu opnieuw.");
    await openChapter(state.chapter, "review", { keep: true });
    refreshBookSoon();
  } catch (err) { fail(err, "Script opslaan"); }
}

async function discardEdits() {
  if (!confirm("Je wijzigingen weggooien?")) return;
  setDirty(false);
  await openChapter(state.chapter, "review", { keep: true });
}

function jumpToLine(id) {
  showTab("work", "script");
  const el = $(`#script-lines .line[data-id="${CSS.escape(id)}"]`);
  if (!el) return;
  renderLinePanel(id);
  if ($("#only-issues").checked && !el.matches(".has-issue, .has-warning")) { $("#only-issues").checked = false; $("#script-lines").classList.remove("only-issues"); }
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("is-flash");
  void el.offsetWidth;
  el.classList.add("is-flash");
  $(".text", el).focus({ preventScroll: true });
}

// ------------------------------------------------------------------- audit
function renderAudit(data) {
  const a = data.audit;
  const count = $("#audit-count");
  if (!a) {
    $("#audit-summary").innerHTML = `<p class="empty">Nog geen controle. ${data.script ? "Laat het script controleren via de stappen bovenaan." : "Eerst een script."}</p>`;
    $("#audit-issues").innerHTML = "";
    count.hidden = true;
    return;
  }
  const blocking = a.issues.filter(i => i.severity === "blocking");
  const warnings = a.issues.filter(i => i.severity === "warning");
  count.hidden = !a.issues.length;
  count.textContent = blocking.length || warnings.length;
  count.className = `count${blocking.length ? " count--bad" : ""}`;
  const stale = data.script && a.script_revision !== data.script.revision;
  $("#audit-summary").innerHTML = `<section class="card">
    <div class="verdict">${a.passed ? chip("GESLAAGD", "ok", "", "chip--lg") : chip("GEBLOKKEERD", "bad", "", "chip--lg")}
      <h2>${plural(blocking.length, "probleem", "problemen")}, ${plural(warnings.length, "tip", "tips")}</h2>
      ${stale ? chip("hoort bij een oudere versie", "warn", "Het script is daarna aangepast. Laat het opnieuw controleren.") : ""}</div>
    <div class="auditstats">
      <span>dekking: ${a.coverage.covered.length} van ${a.coverage.required.length} beweringen${a.coverage.missing.length ? ", ontbreekt: " + escapeHtml(a.coverage.missing.join(", ")) : ""}</span>
      <span>tegen het boek gelegd: ${a.support.checked} regels, ${a.support.unsupported} niet onderbouwd${a.support.skipped ? " (overgeslagen, geen model)" : ""}</span>
    </div>
  </section>`;
  const group = (title, items) => items.length ? `<div class="issuegroup"><h3>${title} (${items.length})</h3>${items.map(i => `
    <div class="issue ${i.severity}">
      ${chip(i.severity === "blocking" ? "probleem" : "tip", i.severity === "blocking" ? "bad" : "warn")}
      <div class="issue__head"><span class="issue__rule">${escapeHtml(i.rule)}</span>
        ${i.line_id ? `<a href="#" class="lineref" data-line="${escapeHtml(i.line_id)}" title="Naar deze regel in het script.">regel ${escapeHtml(i.line_id)}</a>` : ""}
        ${i.claim_id ? chip(i.claim_id, "", "Bewering uit het plan.") : ""}</div>
      <div class="issue__msg">${escapeHtml(i.message)}</div>
      ${i.suggestion ? `<div class="issue__tip">${escapeHtml(i.suggestion)}</div>` : ""}
    </div>`).join("")}</div>` : "";
  $("#audit-issues").innerHTML = group("Problemen", blocking) + group("Tips", warnings) ||
    '<p class="empty">De controle vond niets. Lees het script toch even door voordat je goedkeurt.</p>';
}

// -------------------------------------------------------------------- plan
function renderPlan(data) {
  const p = data.plan;
  const root = $("#plan-view");
  if (!p) { root.innerHTML = '<p class="empty">Nog geen plan.</p>'; return; }
  root.innerHTML = `<div class="plan">
    <p class="plan__summary">${escapeHtml(p.summary)}</p>
    <section><h3>Leerdoelen</h3><ul class="bullets">${p.learning_objectives.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></section>
    <section><h3>Beweringen (${p.key_claims.length})</h3>${p.key_claims.map(c => {
      const lines = data.script ? linesCovering(c.id) : [];
      const ps = claimPassage(c);
      return `<div class="claim"><span class="id">${escapeHtml(c.id)}</span><span>${escapeHtml(c.claim)}</span>
      <div class="claim__meta">${chip(`moeilijkheid ${c.difficulty}`)} ${chip(`tentamen ${c.exam_relevance}`)}
        <span title="${escapeHtml(scoreTitle(c.source_span.match_score))}">${escapeHtml(ps.section)}${ps.pages ? ", " + pagesText(ps.pages) : ""}</span>
        ${ps.found ? `<button class="link" data-show-claim="${escapeHtml(c.id)}">in de boektekst</button>` : chip("citaat niet gevonden", "warn", scoreTitle(c.source_span.match_score))}
        ${data.script ? (lines.length ? `<span>in het script: ${lines.map(l => `<button class="link" data-line="${escapeHtml(l.id)}">${escapeHtml(l.id)}</button>`).join(" ")}</span>` : chip("niet in het script", "bad")) : ""}</div></div>`;
    }).join("")}</section>
    <section><h3>Definities</h3><ul class="bullets">${p.definitions.map(d => `<li><b>${escapeHtml(d.term)}</b>: ${escapeHtml(d.definition)}</li>`).join("") || '<li class="muted">geen</li>'}</ul></section>
    <section><h3>Misvattingen</h3><ul class="bullets">${p.misconceptions.map(m => `<li><b>fout:</b> ${escapeHtml(m.wrong)} <b>klopt wel:</b> ${escapeHtml(m.right)} <span class="muted">${escapeHtml(m.why_tempting)}</span></li>`).join("") || '<li class="muted">geen</li>'}</ul></section>
    <div class="planfacts">${chip(`gastexpert: ${p.needs_expert ? `ja${p.expert_domain ? ` (${p.expert_domain})` : ""}` : "nee"}`)} ${chip(`veel formules: ${p.formula_dense_sections.join(", ") || "nee"}`)}</div>
  </div>`;
}

function scoreTitle(score) {
  if (score === null || score === undefined) return "Citaatscore onbekend.";
  if (score >= 100) return "Het citaat staat letterlijk in deze sectie.";
  if (score > 0) return `Het citaat staat er bijna letterlijk in (${score}).`;
  return "Het citaat is niet gevonden. De hele sectie geldt als bron, dus de controle is zwakker.";
}

// ------------------------------------------------------------------- audio
function renderAudio(data) {
  const url = data.final.audio;
  $("#final-audio").innerHTML = url
    ? `<div class="player"><audio controls preload="metadata" src="${escapeHtml(url)}"></audio><a class="btn btn--secondary btn--sm" href="${escapeHtml(url)}" download>Downloaden</a></div>`
    : '<p class="muted">Nog niet opgenomen.</p>';
  const q = data.manifest_quality;
  if (q && q.verifiable_turns) {
    const verOk = q.verified_turns === q.verifiable_turns;
    const bits = [chip(`${q.verified_turns} van ${q.verifiable_turns} beurten nagecontroleerd`, verOk ? "ok" : "bad", verOk ? "" : "Niet elke opname kon worden teruggeluisterd; zie studiepodcast doctor.")];
    if (q.alignment_fallback_turns) bits.push(chip(`${q.alignment_fallback_turns} met geschatte timing`, "warn"));
    $("#audio-quality").innerHTML = bits.join(" ");
  } else {
    $("#audio-quality").innerHTML = "";
  }
  const flagged = data.manifest ? data.manifest.turns.filter(t => t.flagged) : [];
  $("#flagged").innerHTML = flagged.length ? `<section class="card"><div class="cardhead"><h2>Even beluisteren</h2>${chip(String(flagged.length), "warn")}</div>
    <p class="muted">Bij deze beurten volgde geen enkele opname het script goed genoeg. Luister ze na; klinkt het fout, pas dan de tekst aan en neem opnieuw op.</p>
    <div class="flagged">${flagged.map(t => `<div class="issue warning">${chip(t.turn_id, "warn")}<div class="issue__head"><b>${escapeHtml(t.speaker)}</b> <span class="muted">${escapeHtml(t.flag_reason || "")}</span></div><div class="issue__msg">${escapeHtml(t.text_spoken)}</div></div>`).join("")}</div></section>` : "";
  const root = $("#blocks");
  root.innerHTML = "";
  const spans = (data.final.transcript && data.final.transcript.blocks) || [];
  const paths = Object.fromEntries(spans.map(s => [s.block_id, s.path]));
  for (const b of data.blocks) {
    const div = document.createElement("div");
    div.className = "block";
    const src = paths[b.id] ? bookUrl(`/artifacts/render/${enc(state.chapter)}/blocks/${enc(b.id)}.mp3`) : null;
    div.innerHTML = `<b>${escapeHtml(b.id)}</b><span class="block__info">${b.line_ids.length} regels, ${escapeHtml(b.segment_types.map(t => SEGMENTS[t] || t).join(" / "))}</span>
      ${src ? `<audio controls preload="none" src="${src}"></audio>` : '<span class="muted small">geen los bestand</span>'}`;
    root.appendChild(div);
  }
}

// ----------------------------------------------------------------- timings
function renderTimings(chapterId) {
  const root = $("#timings-view");
  const entries = ((state.timings && state.timings.entries) || []).filter(e => e.chapter === chapterId);
  if (!entries.length) { root.innerHTML = '<p class="empty">Nog geen tijden voor dit hoofdstuk.</p>'; return; }
  const runs = [...new Set(entries.map(e => e.run))].sort().reverse().slice(0, 3);
  root.innerHTML = runs.map(runId => {
    const rows = entries.filter(e => e.run === runId);
    const total = rows.find(e => e.pass_name === "chapter");
    const passes = rows.filter(e => e.pass_name !== "chapter");
    const longest = Math.max(1, ...passes.map(e => e.seconds || 0));
    const head = `Run van ${fmtWhen(rows[0].started_at)}` + (total ? `, in totaal ${fmtSeconds(total.seconds)}` : "");
    return `<section class="card card--flush runblock"><div class="cardhead"><h3>${escapeHtml(head)}</h3>${total ? TIMING_STATUS[total.status] || "" : ""}</div>
      <div class="tablewrap"><table class="table timings"><thead><tr><th>Stap</th><th class="opt-3">Begon</th><th class="opt-3">Klaar</th><th class="num">Duur</th><th class="opt-2" style="width:20%"><span class="sr-only">Verhouding</span></th><th></th></tr></thead><tbody>` +
      passes.map(e => `<tr class="${e.status === "failed" ? "is-failed" : ""}"><td>${escapeHtml(passTitle(e))}${e.error ? `<div class="why">${escapeHtml(e.error)}</div>` : ""}</td>
        <td class="opt-3">${fmtClock(e.started_at)}</td><td class="opt-3">${fmtClock(e.finished_at)}</td>
        <td class="num">${e.status === "skipped" ? "-" : fmtSeconds(e.seconds)}</td>
        <td class="opt-2">${e.status === "skipped" ? "" : `<div class="timebar"><i style="width:${Math.max(2, (e.seconds / longest) * 100).toFixed(1)}%"></i></div>`}</td>
        <td>${TIMING_STATUS[e.status] || ""}</td></tr>`).join("") + "</tbody></table></div></section>";
  }).join("");
}

async function renderArtifacts() {
  try {
    const items = await api.get(bookUrl("/artifacts"));
    $("#artifact-list").innerHTML = items.map(a => `<li><a href="${bookUrl("/artifacts/" + a.path.split("/").map(enc).join("/"))}" target="_blank" rel="noopener">${escapeHtml(a.path)}</a>
      <span class="muted">${a.bytes >= 1048576 ? (a.bytes / 1048576).toFixed(1) + " MB" : (a.bytes / 1024).toFixed(0) + " kB"}</span></li>`).join("") ||
      '<li class="muted">Nog geen bestanden.</li>';
  } catch (err) { $("#artifact-list").innerHTML = `<li class="muted">${escapeHtml(errorText(err))}</li>`; }
}

// ----------------------------------------------------------------- lexicon
async function loadGlossary() {
  try {
    const g = await api.get(bookUrl("/glossary"));
    state.glossary = g;
    $("#glossary tbody").innerHTML = "";
    loadGlossaryRows(g.entries);
  } catch (err) { fail(err, "Uitspraak laden"); }
}

function loadGlossaryRows(entries) {
  const tbody = $("#glossary tbody");
  for (const e of entries) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><input type="text" value="${escapeHtml(e.surface)}" data-k="surface" aria-label="Term"></td>
      <td><select data-k="kind" aria-label="Soort">${Object.entries({ ...KINDS, ...(KINDS[e.kind] ? {} : { [e.kind]: e.kind }) }).map(([k, t]) => `<option value="${escapeHtml(k)}" ${k === e.kind ? "selected" : ""}>${escapeHtml(t)}</option>`).join("")}</select></td>
      <td><input type="text" value="${escapeHtml(e.spoken)}" data-k="spoken" aria-label="Uitspraak"></td>
      <td class="center"><input type="checkbox" ${e.lock ? "checked" : ""} data-k="lock" aria-label="Vast"></td>
      <td class="center"><button class="iconbtn" data-del="1" title="Term verwijderen" aria-label="Term verwijderen">&times;</button></td>`;
    tbody.appendChild(tr);
  }
}

function collectGlossary() {
  const entries = [];
  for (const tr of $$("#glossary tbody tr")) {
    const get = (k) => tr.querySelector(`[data-k=${k}]`);
    const surface = get("surface").value.trim();
    if (!surface) continue;
    entries.push({ surface, kind: get("kind").value, spoken: get("spoken").value.trim() || surface, lock: get("lock").checked });
  }
  return { ...state.glossary, entries };
}

// -------------------------------------------------------------------- jobs
async function loadJobs() {
  let jobs;
  try { jobs = await api.get("/api/jobs"); } catch { return state.jobs; }
  state.jobs = jobs;
  $("#jobs").innerHTML = jobs.slice(0, 20).map(j => {
    const [label, kind] = JOB_STATUS[j.status] || [j.status, ""];
    const stop = isActive(j)
      ? `<button class="link" data-cancel="${escapeHtml(j.id)}" title="${j.status === "queued" ? "Haalt deze taak uit de wachtrij." : "Stopt deze taak bij het volgende controlepunt."}">stop</button>` : "";
    return `<li class="${j.status}"><span class="job__name" title="Taak ${escapeHtml(j.id)}">${escapeHtml(j.book_id)} · ${escapeHtml(jobName(j))}</span>
      <span class="job__side">${chip(label, kind)}${stop}</span>
      <span class="job__when">${fmtWhen(j.created_at)}</span>
      ${j.error ? `<span class="job__err" title="${escapeHtml(j.error)}">${escapeHtml(j.error)}</span>` : ""}</li>`;
  }).join("") || '<li class="muted">Nog geen taken.</li>';
  const running = jobs.filter(j => j.status === "running").length;
  const queued = jobs.filter(j => j.status === "queued").length;
  $("#drawer-summary").innerHTML = [running && chip(`${running} bezig`, "run"), queued && chip(`${queued} in de wachtrij`)].filter(Boolean).join("");
  return jobs;
}

// Events that announce the step about to start; "turn" reports a finished one.
const DONE_KINDS = new Set(["turn"]);
const run = { job: null, started: 0, stage: null, book: null, outer: null, inner: null, innerStart: 0, tokens: null, stopping: false, timer: null, chapterId: null };
const watching = new Set();

function startRun(job, ts) {
  Object.assign(run, { job, started: ts || Date.now(), stage: job.stage, book: null, outer: null, inner: null, tokens: null, stopping: false, chapterId: job.chapter_id || null });
  $("#stop-job").disabled = false;
  $("#stop-job").textContent = "Stop";
  $("#progress-panel").classList.remove("stopping");
  $("#progress-panel").hidden = false;
  clearInterval(run.timer);
  run.timer = setInterval(renderProgress, 1000);
}

function endRun(job) {
  if (!run.job || run.job.id !== job.id) return;
  clearInterval(run.timer);
  run.job = null;
  run.chapterId = null;
  $("#progress-panel").hidden = true;
}

function trackEvent(job, ev) {
  const d = ev.data || {};
  const ts = Date.parse(ev.ts) || Date.now();
  if (!run.job || run.job.id !== job.id) startRun(job, ts);
  if (ev.stage === "llm") trackWriter(ev.status, d);
  if (ev.stage === "llm" && ev.status === "tokens") { run.tokens = { ...d, at: Date.now() }; renderProgress(); return; }
  if (ev.stage === "job" && ev.status === "stopping") run.stopping = true;
  else if (ev.stage === "book") {
    if (d.index) run.book = { index: d.index, total: d.total, label: d.label };
    if (d.chapter && d.chapter !== run.chapterId && ev.status === "chapter") {
      const left = run.chapterId;
      run.chapterId = d.chapter;
      afterJobsChanged();
      refreshBookSoon(left && left === state.chapter);
    }
    if (ev.status === "chapter_failed") refreshBookSoon();
  }
  else if (ev.stage === "run") { run.outer = { index: d.index, total: d.total, label: d.label }; run.inner = null; }
  else if (ev.stage === "timing") { if (d.pass_name === "chapter" || d.pass_name === "ingest") refreshBookSoon(); return; }
  else if (ev.status === "start") { run.stage = ev.stage; run.inner = null; run.tokens = null; }
  else if (d.index && d.total) {
    const done = DONE_KINDS.has(ev.status) ? d.index : d.index - 1;
    if (!run.inner || run.inner.stage !== ev.stage || run.inner.total !== d.total) run.innerStart = ts;
    run.stage = ev.stage;
    const label = d.label || (ev.status === "turn" ? `beurt ${d.turn}${d.speaker ? " (" + d.speaker + ")" : ""}` : "");
    run.inner = { stage: ev.stage, index: d.index, total: d.total, done, label, at: ts };
  }
  renderProgress();
}

function renderProgress() {
  if (!run.job) return;
  const now = Date.now();
  const book = run.job.book_id === VOICELAB ? "Stemmen" : (state.books.find(b => b.book_id === run.job.book_id) || {}).title || run.job.book_id;
  $("#progress-title").textContent = `${STAGES[run.stage] || STAGES[run.job.stage] || run.job.stage} · ${book}${run.job.chapter_id ? ", " + run.job.chapter_id : ""}`;
  $("#progress-panel").classList.toggle("stopping", run.stopping);
  const outer = [];
  if (run.book) outer.push(`hoofdstuk ${run.book.index} van ${run.book.total}: ${run.book.label}`);
  if (run.outer) outer.push(`stap ${run.outer.index} van ${run.outer.total}: ${run.outer.label}`);
  $("#progress-outer").textContent = outer.join(" · ");
  const bar = $("#progress-bar");
  const detail = [];
  if (run.inner) {
    bar.max = run.inner.total;
    bar.value = run.inner.done;
    detail.push(`${run.inner.index} van ${run.inner.total}${run.inner.label ? ": " + run.inner.label : ""}`);
    if (run.inner.done > 0 && run.inner.done < run.inner.total) {
      const per = (now - run.innerStart) / run.inner.done;
      detail.push(`nog ~${fmtDuration(per * (run.inner.total - run.inner.done))}`);
    }
  } else {
    bar.removeAttribute("value");  // no known total: an indeterminate bar
  }
  if (run.tokens && now - run.tokens.at < 10000) detail.push(`model schrijft: ${run.tokens.tokens} tokens`);
  detail.push(`bezig ${fmtDuration(now - run.started)}`);
  if (run.stopping) detail.push("stopt bij het volgende controlepunt");
  detail.push("je kunt dit venster sluiten");
  $("#progress-detail").textContent = detail.join(" · ");
}

async function stopJob(jobId) {
  const job0 = state.jobs.find(j => j.id === jobId);
  if (job0 && job0.status === "running" && !confirm("Stoppen? Wat al klaar is, blijft bewaard; de stap die bezig is, begint later opnieuw.")) return;
  try {
    const job = await api.send(`/api/jobs/${enc(jobId)}/cancel`, "POST", {});
    log(`■ stop gevraagd voor ${STAGES[job.stage] || job.stage} ${job.chapter_id || ""} (${job.status})`);
    if (run.job && run.job.id === jobId) {
      run.stopping = true;
      $("#stop-job").disabled = true;
      $("#stop-job").textContent = "Stoppen...";
      renderProgress();
    }
    await loadJobs();
    afterJobsChanged();
  } catch (err) { fail(err, "Stoppen"); }
}

function watch(job) {
  if (watching.has(job.id)) return;
  watching.add(job.id);
  log(`▶ ${job.book_id} · ${jobName(job)} (taak ${job.id})`);
  // The browser reconnects on its own and sends the last event id, so a render that outlives one stream continues.
  const es = new EventSource(`/api/jobs/${enc(job.id)}/events`);
  es.addEventListener("stage", (e) => {
    const ev = JSON.parse(e.data);
    trackEvent(job, ev);
    if (ev.stage === "llm" && ev.status === "tokens") return;  // heartbeat: progress banner only
    const at = ev.ts ? fmtClock(ev.ts) + " " : "";
    if (ev.stage === "timing") {
      const t = ev.data;
      const status = { done: "", failed: " MISLUKT", stopped: " gestopt", skipped: " overgeslagen" }[t.status] || "";
      log(`${at}⏱ ${t.chapter || "boek"} ${t.title}: ${fmtClock(t.started_at)} → ${fmtClock(t.finished_at)} (${fmtSeconds(t.seconds)})${status}`);
      return;
    }
    const d = { ...ev.data };
    delete d.book_id;
    delete d.trace;
    log(`${at}[${ev.stage}] ${ev.status} ${Object.entries(d).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ")}`);
  });
  es.addEventListener("job", (e) => {
    const j = JSON.parse(e.data);
    if (["done", "failed", "cancelled"].includes(j.status)) {
      log(`■ ${jobName(j)} ${j.status}${j.error ? ": " + j.error : ""}`);
      es.close();
      watching.delete(j.id);
      endRun(j);
      if (j.status === "done") toast(`${jobName(j)} is klaar.`);
      else if (j.status === "failed") toast(`${jobName(j)} is mislukt. ${explainError(j.error) || j.error || "Onbekende fout."}`, true);
      else toast(`${jobName(j)} is gestopt. Wat klaar was, is bewaard.`);
      if (j.status === "failed" && /opencode|not reachable|does not answer/i.test(j.error || "")) {
        setEngine("writer", "down", "staat uit", j.error);
        clearTimeout(engineTimer);
        engineTimer = setTimeout(checkEngines, 60000);
      }
      refreshAfterJob(j);
    }
  });
  loadJobs().then(afterJobsChanged);
}

async function refreshAfterJob(j) {
  await loadJobs();
  if (j.book_id === VOICELAB) {
    if (state.view === "voices") { await loadVoices(); renderVoiceList(); if (state.voice === j.chapter_id) await loadTakes(); renderVoiceState(); }
    return;
  }
  if (["ingest", "pipeline"].includes(j.stage) && j.book_id === state.sourceFor) state.source = null;
  try {
    await loadBooks();
    if (state.book && j.book_id === state.book) await loadBook(state.book);
    if (state.view === "chapter" && state.chapter && !state.dirty && j.book_id === state.book) {
      // A recording that just finished takes you to the episode; anything else leaves you where you were.
      const next = j.status === "done" && j.stage === "render" && j.chapter_id === state.chapter && state.step === "record" ? "listen" : state.step;
      await openChapter(state.chapter, next, { keep: next === state.step });
    }
    else afterJobsChanged();
  } catch { /* the next refresh catches up */ }
}

// Jobs decide what is busy: the page you look at follows them.
function afterJobsChanged() {
  if (state.view === "voices") renderVoiceState();
  if (state.view === "home") renderHome();
  else if (state.view === "book") renderBook();
  else if (state.view === "chapter" && state.data) {
    renderTracker(chapterPhase(state.data));
    if (state.step === "review") renderReviewGuide(state.data);
    else if (state.step === "missing") renderMissing(state.data);
    else if (state.step === "record") renderRecord(state.data);
  }
}

async function runStage(stage, extra = {}, message = null) {
  const box = $("#quality");
  const opts = { ...(box && box.checked && !box.disabled ? { quality: true } : {}), ...extra };
  try {
    const job = await api.send(chapterUrl("/run"), "POST", { stage, ...opts });
    toast(message || `${STAGES[stage] || stage} is gestart. De balk bovenaan laat zien hoe ver het is.`);
    watch(job);
    return job;
  } catch (err) { fail(err, STAGES[stage] || stage); return null; }
}

async function approve(e) {
  try {
    await busy(e.currentTarget, "Bezig...", () => api.send(chapterUrl("/approve"), "POST", {}));
    log("goedgekeurd; opnemen kan nu");
    toast("Goedgekeurd. Je kunt het nu laten opnemen.");
    await loadBook(state.book);
    go(chapterHash(state.book, state.chapter, "record"));
  } catch (err) { fail(err, "Goedkeuren"); }
}

// --------------------------------------------------------------- voice lab
// One sentence, one voice, the four knobs by hand. Every take is kept with the settings and the seed that made it;
// the best one's settings are saved for that voice and every recording after uses them.
const VOICELAB = "stemlab";
const KNOBS = {
  exaggeration: ["Expressie", "Hoeveel emotie en nadruk. Hoger is levendiger; te hoog klinkt overdreven of struikelt."],
  cfg_weight: ["Houvast aan de stem", "Hoe strak de take de referentieclip volgt. Lager klinkt vaak rustiger en trager."],
  temperature: ["Variatie", "Hoeveel takes van elkaar verschillen. Lager is voorspelbaar, hoger levendiger maar wisselvalliger."],
  speech_rate: ["Tempo", "Na de opname sneller of trager gemaakt, zonder de toonhoogte te veranderen. 1 is ongewijzigd."],
};
const knobText = (k, v) => `${KNOBS[k][0].toLowerCase()} ${Number(v).toFixed(2).replace(/0$/, "")}`;
const sameParams = (a, b) => a && b && Object.keys(KNOBS).every(k => Math.abs(a[k] - b[k]) < 1e-6);

async function loadVoices() {
  try { state.voices = await api.get("/api/voices"); } catch (err) { fail(err, "Stemmen laden"); }
  return state.voices;
}
function currentVoice() { return state.voices ? state.voices.voices.find(v => v.id === state.voice) : null; }

async function openVoices(voiceId) {
  if (!state.voices) await loadVoices();
  if (!state.voices) return;
  const list = state.voices.voices;
  if (!voiceId) voiceId = recall("voice");
  if (!list.some(v => v.id === voiceId)) voiceId = (list.find(v => v.has_ref) || list[0] || {}).id || null;
  saveVoiceDraft();  // the knobs as you left them, also when you come back to the same voice
  if (voiceId !== state.voice) { state.voice = voiceId; state.voiceTakes = []; }
  if (voiceId) { remember("voice", voiceId); const h = `#/stemmen/${enc(voiceId)}`; if (location.hash !== h) { history.replaceState(null, "", h); lastHash = h; } }
  renderVoiceList();
  renderVoiceEditor();
  if (voiceId) await loadTakes();
  renderVoiceState();
}

function renderVoiceList() {
  const kinds = { host: "Host", guest: "Gast" };
  $("#voice-list").innerHTML = state.voices.voices.map(v => `<a class="voiceitem" href="#/stemmen/${enc(v.id)}"${v.id === state.voice ? ' aria-current="page"' : ""}>
      <span class="voiceitem__name">${escapeHtml(v.name)}</span>
      <span class="voiceitem__meta">${kinds[v.kind] || v.kind}${v.role ? `, ${v.role === "explainer" ? "legt uit" : "vraagt door"}` : ""}</span>
      ${v.has_ref ? "" : chip("geen clip", "warn", `Zet een referentieclip in cast/${v.voice_ref || "refs/"}`)}
    </a>`).join("");
  const loaded = state.voices.model_loaded;
  $("#voice-model").hidden = !loaded;
  if (loaded) $("#voice-model").innerHTML = 'Het stemmodel staat geladen op de videokaart, zodat de volgende take snel is. Ga je schrijven met OpenCode, geef de kaart dan eerst vrij. <button class="link" id="voice-release">Videokaart vrijgeven</button>';
}

function sliderHtml(key, spec, value, saved) {
  const pct = (x) => (100 * (x - spec.min) / (spec.max - spec.min)).toFixed(1);
  return `<div class="slider" data-knob="${key}">
    <div class="slider__head"><label for="knob-${key}">${KNOBS[key][0]}</label><output id="out-${key}">${value}</output></div>
    <div class="slider__track"><input type="range" id="knob-${key}" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${value}">
      <span class="slider__saved" style="left:${pct(saved)}%" title="opgeslagen: ${saved}"></span></div>
    <p class="slider__help">${KNOBS[key][1]} <span class="slider__savedtext">Opgeslagen: ${saved}.</span></p>
  </div>`;
}

function saveVoiceDraft() {
  if (!state.voice || $("#voice-editor").hidden) return;
  state.voiceDraft[state.voice] = { text: $("#voice-text").value, params: knobValues(), seed: $("#voice-seed").value };
  try { localStorage.setItem("luisterstof.voiceText", $("#voice-text").value); } catch { /* private window */ }
}

function renderVoiceEditor() {
  const v = currentVoice();
  $("#voice-empty").hidden = !!v;
  $("#voice-editor").hidden = !v;
  if (!v) return;
  const draft = state.voiceDraft[v.id] || {};
  const params = draft.params || v.params;
  $("#voice-kind").textContent = v.kind === "host" ? "Host" : "Gast";
  $("#voice-name").textContent = v.name;
  $("#voice-about").textContent = v.about;
  $("#voice-ref").innerHTML = v.has_ref
    ? `<span class="muted small">De referentieclip, waar elke take op lijkt:</span><audio controls preload="none" src="/api/voices/${enc(v.id)}/ref"></audio>`
    : `<div class="callout callout--bad"><b>Nog geen referentieclip.</b> Zet 30 tot 60 seconden schone opname in <code>cast/${escapeHtml(v.voice_ref || "refs/" + v.id + ".wav")}</code>. De Opnamegids legt uit hoe.${state.voices.fake_audio ? " In testmodus kun je toch takes maken." : ""}</div>`;
  let text = draft.text;
  if (text === undefined) { try { text = localStorage.getItem("luisterstof.voiceText") || ""; } catch { text = ""; } }
  $("#voice-text").value = text || $("[data-example]").dataset.example;
  $("#voice-text").maxLength = state.voices.max_text;
  $("#voice-sliders").innerHTML = Object.entries(state.voices.params).map(([k, spec]) => sliderHtml(k, spec, params[k], v.params[k])).join("");
  $("#voice-seed").value = draft.seed || Math.floor(Math.random() * 999998) + 1;
  $("#voice-save").textContent = `Opslaan voor ${v.name}`;
  updateVoiceDirty();
}

function knobValues() {
  const out = {};
  for (const k of Object.keys(KNOBS)) { const el = $(`#knob-${k}`); if (el) out[k] = parseFloat(el.value); }
  return out;
}

function updateVoiceDirty() {
  const v = currentVoice();
  if (!v) return;
  const changed = !sameParams(knobValues(), v.params);
  $("#voice-save").disabled = !changed;
  $("#voice-reset").disabled = !changed;
  for (const k of Object.keys(KNOBS)) $(`#out-${k}`).textContent = Number(knobValues()[k]).toFixed(2);
  $("#voice-state").textContent = changed ? "Andere instellingen dan opgeslagen." : "Dit zijn de opgeslagen instellingen.";
}

function voiceJob() {
  return state.jobs.find(j => j.book_id === VOICELAB && j.chapter_id === state.voice && isActive(j));
}

function renderVoiceState() {
  if (state.view !== "voices" || !currentVoice()) return;
  const job = voiceJob();
  const btn = $("#voice-go");
  btn.disabled = !!job;
  btn.textContent = job ? (job.status === "queued" ? "In de wachtrij..." : "Bezig...") : "Maak take";
  renderTakes();
}

async function loadTakes() {
  if (!state.voice) return;
  try { state.voiceTakes = await api.get(`/api/voices/${enc(state.voice)}/takes`); } catch (err) { fail(err, "Takes laden"); }
  renderTakes();
}

function renderTakes() {
  const v = currentVoice();
  const root = $("#voice-takes");
  if (!v) return;
  const job = voiceJob();
  const pending = job ? `<div class="take take--pending"><span class="chip chip--run">${job.status === "queued" ? "in de wachtrij" : "bezig"}</span>
    <span class="muted">${state.voices.model_loaded ? "De take wordt gemaakt." : "De eerste take laadt ook het stemmodel; dat duurt een minuut of twee."}</span></div>` : "";
  if (!state.voiceTakes.length) { root.innerHTML = pending || '<p class="empty">Nog geen takes voor deze stem.</p>'; return; }
  root.innerHTML = pending + state.voiceTakes.map(t => {
    const saved = sameParams(t.params, v.params);
    const wer = t.wer === null || t.wer === undefined ? chip("niet teruggeluisterd")
      : t.wer <= 0.05 ? chip(`goed verstaan (${Math.round(t.wer * 100)}% fout)`, "ok", "Whisper verstond bijna alles; zo komt de take door de controle van een echte opname.")
      : chip(`${Math.round(t.wer * 100)}% verkeerd verstaan`, "bad", "Een echte opname keurt een take boven 5% af.");
    return `<div class="take${saved ? " is-saved" : ""}" data-take="${escapeHtml(t.id)}">
      <div class="take__play"><audio controls preload="none" src="${escapeHtml(t.audio)}"></audio></div>
      <div class="take__body">
        <p class="take__text">${escapeHtml(t.text)}</p>
        ${t.wer > 0 && t.transcript ? `<p class="take__heard">Whisper hoorde: ${escapeHtml(t.transcript)}</p>` : ""}
        <p class="take__meta">${Object.keys(KNOBS).map(k => `<span>${escapeHtml(knobText(k, t.params[k]))}</span>`).join("")}<span>seed ${t.seed}</span><span>${String(t.duration_s).replace(".", ",")} s</span><span>${fmtWhen(t.created_at)}</span></p>
        <p class="take__chips">${wer}${saved ? chip("opgeslagen instellingen", "ok") : ""}</p>
      </div>
      <div class="take__actions">
        <button class="btn btn--secondary btn--sm" data-use="${escapeHtml(t.id)}" title="Zet de knoppen, de seed en de zin terug zoals bij deze take.">Overnemen</button>
        <button class="btn btn--ghost btn--sm" data-save-take="${escapeHtml(t.id)}"${saved ? " disabled" : ""}>Opslaan voor ${escapeHtml(v.name)}</button>
        <button class="iconbtn" data-del-take="${escapeHtml(t.id)}" title="Take verwijderen" aria-label="Take verwijderen">&times;</button>
      </div>
    </div>`;
  }).join("");
}

async function makeTake() {
  const v = currentVoice();
  if (!v) return;
  const text = $("#voice-text").value.trim();
  if (text.split(/\s+/).filter(Boolean).length < 3) { toast("Typ minstens drie woorden: heel korte zinnen laten Chatterbox vastlopen.", true); return; }
  const seed = parseInt($("#voice-seed").value, 10) || null;
  try {
    const job = await api.send(`/api/voices/${enc(v.id)}/try`, "POST", { text, seed, ...knobValues() });
    state.jobs.unshift(job);
    watch(job);
    renderVoiceState();
    if (!$("#seed-keep").checked) $("#voice-seed").value = Math.floor(Math.random() * 999998) + 1;
  } catch (err) { fail(err, "Take maken"); }
}

async function saveVoice(params) {
  const v = currentVoice();
  if (!v) return;
  try {
    await api.send(`/api/voices/${enc(v.id)}/params`, "PUT", params);
    toast(`Opgeslagen voor ${v.name}. Elke volgende opname gebruikt deze instellingen.`);
    log(`stem ${v.id}: ${Object.keys(KNOBS).map(k => `${k}=${params[k]}`).join(" ")} opgeslagen`);
    await loadVoices();
    loadCast();
    renderVoiceList();
    updateVoiceDirty();
    $$("#voice-sliders .slider").forEach(el => {
      const k = el.dataset.knob, spec = state.voices.params[k], saved = currentVoice().params[k];
      $(".slider__saved", el).style.left = `${(100 * (saved - spec.min) / (spec.max - spec.min)).toFixed(1)}%`;
      $(".slider__savedtext", el).textContent = `Opgeslagen: ${saved}.`;
    });
    renderTakes();
  } catch (err) { fail(err, "Opslaan"); }
}

function useTake(take) {
  for (const k of Object.keys(KNOBS)) $(`#knob-${k}`).value = take.params[k];
  $("#voice-seed").value = take.seed;
  $("#voice-text").value = take.text;
  updateVoiceDirty();
  $(".voicecard").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ----------------------------------------------------------------- engines
const ENGINE_TEXT = {
  writer: { reachable: "bereikbaar", ready: "klaar", test: "testmodus", down: "staat uit", checking: "controleren...", waking: "wordt gestart", unknown: "onbekend" },
  voice: { ready: "geïnstalleerd", test: "testmodus", down: "niet geïnstalleerd", checking: "controleren...", unknown: "onbekend" },
};

function setEngine(which, st, label, detail) {
  const el = $(`#engine-${which}`);
  el.dataset.state = st;
  $(".engine__state", el).textContent = label || ENGINE_TEXT[which][st] || st;
  if (detail !== undefined) state.engineDetail[which] = detail || "";
  el.title = state.engineDetail[which] || "";
  renderSys();
  if (which === "writer") renderEngineWarning();
}

// One line in the header: can Luisterstof work right now?
function renderSys() {
  const w = $("#engine-writer").dataset.state, v = $("#engine-voice").dataset.state;
  let text, kind;
  if (w === "down") [text, kind] = ["Schrijfmodel staat uit", "bad"];
  else if (w === "checking" || w === "unknown") [text, kind] = ["controleren...", "wait"];
  else if (w === "waking") [text, kind] = ["Schrijfmodel start op", "wait"];
  else if (w === "test" && v === "test") [text, kind] = ["Testmodus", "test"];
  else if (v === "down") [text, kind] = ["Klaar om te schrijven, stemmen ontbreken", "warn"];
  else [text, kind] = ["Klaar", "ok"];
  $("#sys-text").textContent = text;
  $("#sys").dataset.state = kind;
}

function trackWriter(status, d) {
  const now = $("#engine-writer").dataset.state;
  if (now === "test") return;
  if (status === "check") setEngine("writer", "checking", "controleren...", d.message);
  else if (status === "wake") setEngine("writer", "waking", "wordt gestart", d.message);
  else if (status === "ready") setEngine("writer", "ready", `klaar (${d.seconds} s)`, `antwoordde in ${d.seconds} s`);
  else if (status === "tokens" && now !== "ready") setEngine("writer", "ready", "klaar", "schrijft nu");
}

let engineTimer = null;
async function checkEngines() {
  setEngine("writer", "checking");
  setEngine("voice", "checking");
  try {
    const e = await api.get("/api/engines");
    setEngine("writer", e.writer.state, null, e.writer.message);
    setEngine("voice", e.voice.state, null, e.voice.message);
  } catch (err) {
    setEngine("writer", "unknown", null, errorText(err));
    setEngine("voice", "unknown", null, errorText(err));
  }
  // OpenCode started after the app: notice it without a click.
  clearTimeout(engineTimer);
  if ($("#engine-writer").dataset.state === "down") engineTimer = setTimeout(checkEngines, 60000);
  afterJobsChanged();
}

function renderEngineWarning() {
  const el = $("#engine-warning");
  const down = $("#engine-writer").dataset.state === "down";
  const all = ($("#upload input[name=start]:checked") || {}).value !== "ingest";
  el.hidden = !(down && all);
  if (!el.hidden) {
    el.innerHTML = "<b>Het schrijfmodel staat uit.</b> Start de OpenCode-app en klik bovenaan op het statuslampje om opnieuw te controleren. " +
      "Zonder model kun je het boek wel alvast inlezen: kies dan Eerst alleen inlezen.";
  }
}

async function loadCast() {
  try {
    const c = (await api.get("/api/cast")).cast;
    for (const sp of [...(c.hosts || []), ...(c.guests || [])]) state.names[sp.id] = sp.name;
    state.hosts = (c.hosts || []).map(h => h.id);
  } catch { /* ids are shown instead */ }
}

async function loadHealth() {
  try { state.health = await api.get("/api/health"); } catch { return; }
  $("#engine-offline").hidden = !state.health.offline;
  applyQualityMode();
}

// The server already runs thorough: show that, and don't pretend you can turn it off per run.
function applyQualityMode() {
  if (!state.health || !state.health.quality_mode) return;
  for (const box of $$(".quality-box")) {
    box.checked = true;
    box.disabled = true;
    const label = box.closest("label");
    if (label && !label.dataset.qm) { label.dataset.qm = "1"; label.title += " De app draait nu al altijd grondig (STUDIEPODCAST_QUALITY_MODE=1)."; }
  }
}

// ------------------------------------------------------------------ upload
const fileInput = $("#upload-file");
function slug(name) {
  return name.replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase().slice(0, 40) || "boek";
}
function showChosenFile() {
  const f = fileInput.files[0];
  $("#drop").classList.toggle("has-file", !!f);
  $("#drop-text").innerHTML = f
    ? `<b>${escapeHtml(f.name)}</b><br>${f.size >= 1048576 ? (f.size / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(f.size / 1024)) + " kB"} · klik om een ander bestand te kiezen`
    : "<b>Sleep de PDF hierheen</b><br>of klik om een bestand te kiezen";
  $("#upload-name").placeholder = f ? `leeg laten: ${slug(f.name)}` : "bijvoorbeeld statistiek1";
  $("#upload-error").hidden = true;
}
function uploadError(text) {
  const el = $("#upload-error");
  el.textContent = text;
  el.hidden = false;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}
function updateStartText() {
  const all = $("#upload input[name=start]:checked").value === "all";
  $("#start-explain").textContent = all
    ? "Luisterstof leest het boek in en schrijft daarna alle scripts. Je kunt dit venster sluiten: het werk gaat door zolang de app draait."
    : "Luisterstof leest het boek in. Dat duurt een paar minuten; daarna zie je welke hoofdstukken er gevonden zijn.";
  $("#upload-submit").textContent = all ? "Start" : "Inlezen";
  renderEngineWarning();
}

// ------------------------------------------------------------------ wiring
function showTab(group, name) {
  $$(`.tabs[data-group="${group}"] button`).forEach(b => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  $$(`.tab[data-group="${group}"]`).forEach(t => t.hidden = t.id !== `tab-${name}`);
  if (group === "work") { state.workTab = name; if (name === "script") autosizeAll(); }
}
function showWork(tab) {
  showTab("work", tab);
  $("#workspace").scrollIntoView({ behavior: "smooth", block: "start" });
}
$$(".tabs").forEach(nav => $$("button", nav).forEach(b => b.onclick = () => showTab(nav.dataset.group, b.dataset.tab)));
$("#script-lines").addEventListener("focusin", (e) => { const l = e.target.closest(".line"); if (l && l.dataset.id !== state.selectedLine) renderLinePanel(l.dataset.id); });
$("#script-lines").addEventListener("click", (e) => { const l = e.target.closest(".line"); if (l && l.dataset.id !== state.selectedLine) renderLinePanel(l.dataset.id); });
for (const panel of ["#line-panel", "#claim-panel"]) {
  $(panel).addEventListener("click", (e) => {
    if (e.target.closest("[data-close-panel]")) { panel === "#line-panel" ? renderLinePanel(null) : renderClaimPanel(null); return; }
    const show = e.target.closest("[data-show-claim]");
    if (show) { showClaimInSource(show.dataset.showClaim); return; }
    const line = e.target.closest("[data-line]");
    if (line) jumpToLine(line.dataset.line);
  });
}
$("#source-view").addEventListener("click", (e) => { const m = e.target.closest("mark[data-claim]"); if (m) renderClaimPanel(m.dataset.claim); });
$("#source-view").addEventListener("keydown", (e) => { const m = e.target.closest("mark[data-claim]"); if (m && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); renderClaimPanel(m.dataset.claim); } });
$("#plan-view").addEventListener("click", (e) => {
  const show = e.target.closest("[data-show-claim]");
  if (show) { showClaimInSource(show.dataset.showClaim); return; }
  const line = e.target.closest("[data-line]");
  if (line) jumpToLine(line.dataset.line);
});
$$("[data-run]").forEach(b => b.onclick = () => {
  if (b.dataset.confirm && !confirm(b.dataset.confirm)) return;
  runStage(b.dataset.run);
});
$$("#tracker button").forEach(b => b.onclick = () => go(chapterHash(state.book, state.chapter, b.dataset.step)));

$("#journey").addEventListener("click", (e) => {
  const summary = e.target.closest("summary");
  if (summary) { const d = summary.parentElement; state.stageOpen[d.dataset.key] = !d.open; return; }
  const b = e.target.closest("[data-action]");
  if (!b) return;
  const a = b.dataset.action;
  if (a === "run-book") runBook(b);
  else if (a === "stop") stopJob(b.dataset.job);
  else if (a === "new") go("#/nieuw");
  else if (a === "chapter") go(chapterHash(state.book, b.dataset.chapter, b.dataset.step));
});
$("#journey").addEventListener("change", (e) => { if (e.target.id === "run-book-quality") state.runQuality = e.target.checked; });
$("#recheck").onclick = checkEngines;
$("#voice-sliders").addEventListener("input", updateVoiceDirty);
$("#voice-go").onclick = makeTake;
$("#voice-text").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); makeTake(); } });
$("#seed-new").onclick = () => { $("#voice-seed").value = Math.floor(Math.random() * 999998) + 1; };
$("#voice-save").onclick = () => saveVoice(knobValues());
$("#voice-reset").onclick = () => { const v = currentVoice(); if (!v) return; for (const k of Object.keys(KNOBS)) $(`#knob-${k}`).value = v.params[k]; updateVoiceDirty(); };
$$("[data-example]").forEach(b => b.onclick = () => { $("#voice-text").value = b.dataset.example; $("#voice-text").focus(); });
$("#voice-model").addEventListener("click", async (e) => {
  if (e.target.id !== "voice-release") return;
  try { await api.send("/api/voices/release", "POST", {}); await loadVoices(); renderVoiceList(); toast("De videokaart is vrijgegeven."); } catch (err) { fail(err, "Vrijgeven"); }
});
$("#voice-takes").addEventListener("click", async (e) => {
  const find = (id) => state.voiceTakes.find(t => t.id === id);
  const use = e.target.closest("[data-use]"), save = e.target.closest("[data-save-take]"), del = e.target.closest("[data-del-take]");
  if (use) useTake(find(use.dataset.use));
  else if (save) { const t = find(save.dataset.saveTake); if (t) saveVoice(t.params); }
  else if (del) {
    if (!confirm("Deze take verwijderen?")) return;
    try { await api.send(`/api/voices/${enc(state.voice)}/takes/${enc(del.dataset.delTake)}`, "DELETE", {}); await loadTakes(); } catch (err) { fail(err, "Verwijderen"); }
  }
});
// One take plays at a time, so two takes can be compared back to back.
document.addEventListener("play", (e) => { if (e.target.tagName === "AUDIO") $$("audio").forEach(a => { if (a !== e.target) a.pause(); }); }, true);

fileInput.onchange = showChosenFile;
$$("#upload input[name=start]").forEach(r => r.onchange = updateStartText);
for (const t of ["dragenter", "dragover"]) $("#drop").addEventListener(t, () => $("#drop").classList.add("is-over"));
for (const t of ["dragleave", "drop"]) $("#drop").addEventListener(t, () => $("#drop").classList.remove("is-over"));
// A PDF dropped anywhere opens the new-book page with that file instead of the browser opening the PDF.
window.addEventListener("dragover", (e) => { if (e.target !== fileInput) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (e.target === fileInput) return;
  e.preventDefault();
  const files = e.dataTransfer && e.dataTransfer.files;
  if (!files || !files.length) return;
  if (state.view !== "new") go("#/nieuw");
  fileInput.files = files;
  showChosenFile();
});

$("#upload").onsubmit = async (e) => {
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) { uploadError("Kies eerst de PDF van het boek."); return; }
  if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") { uploadError("Dit is geen PDF. Kies het studieboek als PDF-bestand."); return; }
  const fd = new FormData(e.target);
  const all = fd.get("start") === "all";
  fd.delete("start");
  fd.set("auto", all ? "true" : "false");
  fd.set("quality", fd.get("quality") ? "true" : "false");
  try {
    const out = await busy($("#upload-submit"), "Bezig met uploaden...", async () => {
      const r = await fetch("/api/books", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    });
    e.target.reset();
    showChosenFile();
    updateStartText();
    applyQualityMode();
    await loadBooks();
    state.status = null;
    go(bookHash(out.book_id));
    watch(out.job);
    toast(all ? "Het boek is binnen. Inlezen en schrijven zijn begonnen; dit duurt uren." : "Het boek is binnen en wordt ingelezen.");
  } catch (err) {
    uploadError(`Uploaden mislukt: ${errorText(err)}`);
    log("upload mislukt: " + errorText(err));
  }
};

$("#script-lines").addEventListener("input", (e) => { markDirty(); if (e.target.matches("textarea")) autosize(e.target); });
$("#script-lines").addEventListener("change", (e) => {
  markDirty();
  if (e.target.matches("select.overlap")) e.target.classList.toggle("is-none", e.target.value === "none");
});
$("#only-issues").onchange = (e) => $("#script-lines").classList.toggle("only-issues", e.target.checked);
$("#show-details").onchange = (e) => { $("#script-lines").classList.toggle("show-details", e.target.checked); autosizeAll(); };
$("#save-script").onclick = saveScript;
$("#discard").onclick = discardEdits;
$("#audit-issues").onclick = (e) => {
  const id = e.target.dataset && e.target.dataset.line;
  if (!id) return;
  e.preventDefault();
  jumpToLine(id);
};
$("#save-glossary").onclick = async () => {
  try {
    await busy($("#save-glossary"), "Opslaan...", () => api.send(bookUrl("/glossary"), "PUT", collectGlossary()));
    log("uitspraak opgeslagen");
    toast("Uitspraak opgeslagen. Het geldt vanaf de volgende keer dat er geschreven of opgenomen wordt.");
    await loadGlossary();
  } catch (err) { fail(err, "Uitspraak opslaan"); }
};
$("#add-entry").onclick = () => {
  loadGlossaryRows([{ surface: "", kind: "loanword_en", spoken: "", lock: true }]);
  const rows = $$("#glossary tbody tr");
  $("input", rows[rows.length - 1]).focus();
};
$("#glossary").onclick = (e) => {
  const del = e.target.closest("[data-del]");
  if (del) del.closest("tr").remove();
};
$("#clear-log").onclick = (e) => { e.preventDefault(); $("#log").textContent = ""; };
$("#stop-job").onclick = () => run.job && stopJob(run.job.id);
$("#jobs").onclick = (e) => {
  const id = e.target.dataset && e.target.dataset.cancel;
  if (id) stopJob(id);
};
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && state.view === "chapter" && state.dirty) {
    e.preventDefault();
    saveScript();
  }
});
document.addEventListener("click", (e) => { if ($("#sys").open && !e.target.closest("#sys")) $("#sys").open = false; });
window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });

// ------------------------------------------------------------------- start
(async () => {
  updateStartText();
  loadHealth();
  checkEngines();
  const cast = loadCast();
  try { await loadBooks(); } catch (err) { fail(err, "Boeken laden"); }
  await Promise.all([loadJobs(), cast]);
  // No address: straight to the last book you worked on, else your books (or the welcome).
  if (!location.hash || location.hash === "#" || location.hash === "#/") {
    const last = recall("book");
    if (last && state.books.some(b => b.book_id === last)) history.replaceState(null, "", bookHash(last));
  }
  lastHash = location.hash;
  await route();
  // After a reload: pick up jobs that are still running or waiting, oldest first.
  state.jobs.filter(isActive).reverse().forEach(watch);
})();
