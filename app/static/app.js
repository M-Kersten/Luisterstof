/* Luisterstof UI: hand in a book, follow the night's run, review and edit a chapter's script, render and listen. */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const state = {
  book: null, chapter: null, data: null, glossary: null, sources: {}, timings: null, status: null,
  books: [], jobs: [], health: null, view: null, dirty: false, nextAction: null, engineDetail: { writer: "", voice: "" },
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
const dash = '<span class="subtle">-</span>';
function remember(key, value) { try { localStorage.setItem(`luisterstof.${key}`, value); } catch { /* private window */ } }
function recall(key) { try { return localStorage.getItem(`luisterstof.${key}`); } catch { return null; } }

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
// What a failure means for the user, in Dutch, next to the raw message.
function explainError(err) {
  if (!err) return "";
  if (/worker restarted/i.test(err)) return "De app stopte terwijl deze taak liep (herstart of gesloten venster). Hele boek doorlopen gaat verder waar hij was.";
  if (/opencode|not reachable|does not answer|LLMUnavailable/i.test(err)) return "Het schrijfmodel antwoordde niet. Start OpenCode en probeer het opnieuw.";
  if (/chapters failed/i.test(err)) return "De andere hoofdstukken zijn wel gemaakt. Hele boek doorlopen maakt alleen wat ontbreekt.";
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

const STAGES = { ingest: "Inlezen", pipeline: "Boek verwerken", plan: "Plan", glossary: "Lexicon", script: "Script",
  audit: "Audit", continuity: "Continuïteit", render: "Chatterbox render",
  approve: "Goedkeuren", all: "Hoofdstuk", book: "Boek", run: "Hoofdstuk" };
const PASS_NAMES = { ingest: "inlezen", plan: "plan", glossary: "lexicon", script: "script schrijven", audit: "audit",
  revise: "herschrijven", polish: "tafellezing", continuity: "continuïteit", render: "render", chapter: "hele hoofdstuk" };
const passTitle = (p) => p.title || ((PASS_NAMES[p.pass_name] || p.pass_name) + (p.label ? ` (${p.label})` : ""));
const SEGMENTS = { cold_open: "Koude opening", recap: "Terugblik", body: "Blok", guest: "Gast", reexplain: "Nog eens uitgelegd",
  quiz: "Quiz", outro: "Afsluiting" };
const KINDS = { loanword_en: "Engels leenwoord", notation: "notatie", abbreviation: "afkorting" };
const OVERLAP = { none: "geen", interrupt: "onderbreekt", backchannel: "reactie" };
const JOB_STATUS = { queued: ["in de wachtrij", ""], running: ["bezig", "run"], done: ["klaar", "ok"], failed: ["mislukt", "bad"],
  cancelled: ["gestopt", "warn"] };
const TIMING_STATUS = { done: "", failed: chip("mislukt", "bad"), stopped: chip("gestopt", "warn"), skipped: chip("overgeslagen") };
const ACTIVE = ["running", "queued"];
const jobName = (j) => `${STAGES[j.stage] || j.stage}${j.chapter_id ? " " + j.chapter_id : ""}`;

// ------------------------------------------------------------------- views
function showView(name) {
  const changed = name !== state.view;
  state.view = name;
  $("#upload-view").hidden = name !== "upload";
  $("#book-panel").hidden = name !== "book";
  $("#chapter-panel").hidden = name !== "chapter";
  const noBooks = !state.books.length;
  $("#sidebar").hidden = noBooks;
  $("#cols").classList.toggle("is-single", noBooks);
  $("#upload-cancel").hidden = noBooks;
  if (changed) $("#content").scrollTop = 0;
  markNav();
  syncHash();
}

function syncHash() {
  let h = "";
  if (state.view === "upload") h = "#nieuw";
  else if (state.book) h = `#/${enc(state.book)}` + (state.view === "chapter" && state.chapter ? `/${enc(state.chapter)}` : "");
  if (location.hash !== h) history.replaceState(null, "", h || location.pathname + location.search);
}

function markNav() {
  const set = (el, on) => on ? el.setAttribute("aria-current", "page") : el.removeAttribute("aria-current");
  set($("#nav-overview"), state.view === "book");
  $$("#chapter-list [data-chapter]").forEach(b => set(b, state.view === "chapter" && b.dataset.chapter === state.chapter));
}

// Leaving a chapter with unsaved edits asks first.
function leaveChapter() {
  if (!state.dirty) return true;
  if (!confirm("Je wijzigingen in het script zijn nog niet opgeslagen. Toch verlaten?")) return false;
  setDirty(false);
  return true;
}
function goOverview() {
  if (!leaveChapter()) return;
  if (!state.book) { showView("upload"); return; }
  state.chapter = null;
  state.data = null;
  renderOverview();
  showView("book");
}
async function goChapter(id) {
  if (id !== state.chapter && !leaveChapter()) return;
  await openChapter(id);
}
function goUpload() {
  if (!leaveChapter()) return;
  showView("upload");
  renderEngineWarning();
}

// ------------------------------------------------------------------ books
async function loadBooks() {
  const books = await api.get("/api/books");
  state.books = books;
  const sel = $("#books");
  sel.innerHTML = books.map(b => {
    const name = b.title ? `${b.title.length > 48 ? b.title.slice(0, 46) + "..." : b.title} (${b.book_id})` : `${b.book_id} (nog niet ingelezen)`;
    return `<option value="${escapeHtml(b.book_id)}" ${b.book_id === state.book ? "selected" : ""}>${escapeHtml(name)}</option>`;
  }).join("");
  $("#bookpick").hidden = !books.length;
  return books;
}

// Status, timings and jobs of a book; renders the chapter list and the overview but leaves the view alone.
async function loadBook(bookId) {
  if (bookId !== state.book) { state.chapter = null; state.data = null; }
  state.book = bookId;
  remember("book", bookId);
  const [s, timings] = await Promise.all([
    api.get(`/api/books/${enc(bookId)}`),
    api.get(`/api/books/${enc(bookId)}/timings`).catch(() => null),
  ]);
  state.status = s;
  state.timings = timings;
  await loadJobs();
  renderSidebar();
  renderOverview();
  if ($("#books").value !== bookId) loadBooks();
}

async function openBook(bookId) {
  try { await loadBook(bookId); } catch (err) { fail(err, "Boek openen"); return; }
  showView("book");
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
    if (!state.book) return;
    try {
      await loadBook(state.book);
      if (withChapter && state.view === "chapter" && state.chapter && !state.dirty) await openChapter(state.chapter, { keepTab: true });
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
function chapterJob(id) { return state.jobs.find(j => j.chapter_id === id && ACTIVE.includes(j.status)); }
function chapterBusy(id) { return !!chapterJob(id) || (!!run.job && !run.job.chapter_id && run.job.book_id === state.book && run.chapterId === id); }
function bookBusy() { return state.jobs.some(j => j.book_id === state.book && ACTIVE.includes(j.status)); }

// The latest run that touched a chapter: did a pass fail or stop, and why.
function lastFailure(id) {
  const t = chapterTiming(id);
  if (!t) return null;
  const bad = t.passes.filter(p => p.status === "failed" || p.status === "stopped");
  if (!bad.length) return null;
  const pass = bad.find(p => p.pass_name !== "chapter") || bad[0];
  const whole = bad.find(p => p.pass_name === "chapter");
  return { status: pass.status, title: passTitle(pass), error: pass.error || (whole && whole.error) || "" };
}

// One status per chapter for the list and the overview: a label, a colour, the next step, and whether it waits on you.
function chapterState(ch) {
  let st;
  if (!ch.plan && !ch.script) st = { label: "nog niet gemaakt", kind: "", next: "Hele boek doorlopen" };
  else if (!ch.script) st = { label: "plan klaar", kind: "", next: "Script schrijven" };
  else if (!ch.audit) st = { label: "audit nodig", kind: "warn", next: "Audit draaien" };
  else if (ch.audit_passed === false) st = { label: `${ch.audit_blocking} blokkerend`, kind: "bad", next: "Problemen oplossen", you: true };
  else if (!ch.approved) st = { label: "nakijken", kind: "warn", next: "Lezen en goedkeuren", you: true };
  else if (!ch.final) st = { label: "goedgekeurd", kind: "ok", next: "Render starten", you: true };
  else if (ch.flagged_turns) st = { label: `klaar, ${ch.flagged_turns} gevlagd`, kind: "warn", next: "Gevlagde beurten beluisteren", you: true };
  else st = { label: "klaar", kind: "ok", next: "Beluisteren" };
  const fail = lastFailure(ch.id);
  if (fail) {
    st.fail = fail;
    if (!ch.script || !ch.audit) {
      Object.assign(st, { label: fail.status === "stopped" ? "gestopt" : "mislukt", kind: fail.status === "stopped" ? "warn" : "bad",
        next: "Hele boek doorlopen", you: false, broken: fail.status === "failed" });
    }
  }
  if (chapterBusy(ch.id)) Object.assign(st, { label: "bezig", kind: "run", busy: true, you: false, broken: false });
  return st;
}

function renderSidebar() {
  const s = state.status;
  const ul = $("#chapter-list");
  if (!s) { ul.innerHTML = ""; return; }
  $("#chapter-count").textContent = s.chapters.length || "";
  if (!s.chapters.length) {
    ul.innerHTML = `<li class="navempty">${bookBusy()
      ? "Het boek wordt ingelezen. De hoofdstukken verschijnen hier zodra dat klaar is."
      : "Nog geen hoofdstukken: het boek is niet ingelezen. Het overzicht zegt waarom."}</li>`;
    markNav();
    return;
  }
  ul.innerHTML = s.chapters.map(ch => {
    const st = chapterState(ch);
    const t = chapterTiming(ch.id);
    return `<li><button class="navitem" data-chapter="${escapeHtml(ch.id)}" title="${escapeHtml(`${ch.id} ${ch.title}`)}">
      <span class="navitem__label"><span class="navitem__id">${escapeHtml(ch.id)}</span><span class="navitem__title">${escapeHtml(ch.title)}</span></span>
      <span class="navitem__state">${chip(st.label, st.kind)}${t && t.seconds ? `<span>${fmtSeconds(t.seconds)}</span>` : ""}</span>
    </button></li>`;
  }).join("");
  markNav();
}

// ----------------------------------------------------------- book overview
function renderOverview() {
  const s = state.status;
  if (!s) return;
  const list = s.chapters.map(ch => ({ ch, st: chapterState(ch) }));
  $("#book-title").textContent = s.title || s.book_id;
  $("#book-meta").textContent = s.ingested
    ? `${s.chapters.length} hoofdstukken · lexicon ${s.glossary_entries} termen · structuur gevonden via ${s.structure_method || "-"} · ${s.book_id}`
    : `${s.book_id} · nog niet ingelezen`;

  // Last run of the whole book: when, how long, how many chapters, and how the last book job ended.
  const latest = latestBookRun();
  const going = state.jobs.some(j => j.book_id === s.book_id && !j.chapter_id && j.status === "running");
  const tally = (r) => [r.done && `${r.done} ${r.done === 1 ? "hoofdstuk" : "hoofdstukken"} gemaakt`,
    r.skipped && `${r.skipped} overgeslagen omdat ${r.skipped === 1 ? "het" : "ze"} al klaar ${r.skipped === 1 ? "was" : "waren"}`,
    r.failed && `${r.failed} mislukt`, r.stopped && `${r.stopped} gestopt`].filter(Boolean).join(", ");
  $("#book-run").textContent = !latest ? (going ? "De run is net begonnen." : "Nog geen run van het hele boek.")
    : going ? `Loopt sinds ${fmtWhen(latest.started)}.` + (tally(latest) ? ` Tot nu toe: ${tally(latest)}.` : "")
    : `Begonnen ${fmtWhen(latest.started)}, laatste stap klaar ${fmtWhen(latest.finished)}, ${fmtSeconds(latest.seconds)} in totaal.` +
      (tally(latest) ? ` ${tally(latest).charAt(0).toUpperCase()}${tally(latest).slice(1)}.` : "");
  const job = state.jobs.find(j => j.book_id === s.book_id && !j.chapter_id);
  const [label, kind] = job ? JOB_STATUS[job.status] || [job.status, ""] : ["", ""];
  $("#run-chip").innerHTML = job ? chip(`${STAGES[job.stage] || job.stage}: ${label}`, kind, `Taak ${job.id}, ${fmtWhen(job.updated_at)}`) : "";
  $("#run-error").innerHTML = !job ? "" : job.status === "failed"
    ? `<div class="errorbox"><b>Mislukt:</b> ${escapeHtml(job.error || "onbekende fout")}${explainError(job.error) ? `<br><span class="muted">${escapeHtml(explainError(job.error))}</span>` : ""}</div>`
    : job.status === "cancelled" ? '<p class="muted">Gestopt. Wat klaar was, is bewaard; Hele boek doorlopen gaat verder waar de run was.</p>' : "";
  const n = s.chapters.length;
  const count = (f) => list.filter(f).length;
  const tiles = [
    ["Hoofdstukken", n, ""],
    ["Met script", `${count(x => x.ch.script)}/${n}`, ""],
    ["Nakijken", count(x => x.ch.audit_passed && !x.ch.approved), "warn"],
    ["Blokkeert", count(x => x.ch.audit_passed === false), "bad"],
    ["Mislukt", count(x => x.st.broken), "bad"],
    ["Goedgekeurd", count(x => x.ch.approved), "ok"],
    ["Gerenderd", count(x => x.ch.final), "ok"],
  ];
  $("#book-stats").innerHTML = s.ingested ? tiles.map(([dt, dd, k]) => {
    const show = k && parseInt(dd, 10) > 0 ? ` tile--${k}` : "";
    return `<div class="tile${show}"><dt>${dt}</dt><dd>${dd}</dd></div>`;
  }).join("") : "";

  renderBookNext(s, list);
  $("#chapters-card").hidden = !s.chapters.length;

  const tbody = $("#chapters tbody");
  tbody.innerHTML = list.map(({ ch, st }) => {
    const audit = ch.audit_passed === null ? dash : ch.audit_passed
      ? chip("ok", "ok", "Geen blokkerende problemen gevonden.")
      : chip(`${ch.audit_blocking} blokkerend`, "bad", "Open het hoofdstuk en kijk bij Audit wat er mis is.");
    const why = st.fail && (!ch.script || !ch.audit) ? `<span class="why">${st.fail.status === "stopped" ? "Gestopt" : "Mislukt"} bij ${escapeHtml(st.fail.title)}${st.fail.error ? `: ${escapeHtml(st.fail.error)}` : ""}</span>` : "";
    return `<tr class="clickable" tabindex="0" data-chapter="${escapeHtml(ch.id)}">
      <td><div class="chtitle"><b>${escapeHtml(ch.id)} ${escapeHtml(ch.title)}</b><span class="muted small">p${ch.pages[0]}-${ch.pages[1]}, ${ch.chars.toLocaleString("nl-NL")} tekens</span>${why}</div></td>
      <td class="opt-1">${ch.plan ? chip("klaar", "ok") : dash}</td>
      <td class="opt-2">${ch.script ? chip(`rev ${ch.script_revision}`, "ok", `Revisie ${ch.script_revision}: zo vaak is het script opgeslagen.`) : dash}</td>
      <td>${audit}</td>
      <td class="opt-1">${ch.approved ? chip("ja", "ok") : dash}</td>
      <td class="opt-2">${ch.final ? chip("klaar", "ok") : dash}${ch.flagged_turns ? " " + chip(`${ch.flagged_turns} gevlagd`, "warn", "Beurten waarvan geen enkele opname het script goed genoeg volgde. Beluister ze bij Audio.") : ""}</td>
      <td class="num">${durationCell(chapterTiming(ch.id))}</td>
      <td class="nextcell${st.you ? " is-you" : ""}">${st.busy ? chip("bezig", "run") : escapeHtml(st.next)}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="8" class="muted">Nog geen hoofdstukken.</td></tr>`;
}

// The newest run that went over the book (an upload or Hele boek doorlopen), not a single step on one chapter.
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

function durationCell(chapter) {
  if (!chapter) return dash;
  const parts = chapter.passes.filter(p => p.pass_name !== "chapter" && p.status !== "skipped")
    .map(p => `${passTitle(p)}: ${fmtSeconds(p.seconds)}`);
  return `<span title="${escapeHtml(parts.join("\n"))}">${fmtSeconds(chapter.seconds)}</span>`;
}

// One next step for the whole book, with one primary button.
function renderBookNext(s, list) {
  const box = $("#book-next"), text = $("#book-next-text"), action = $("#book-next-action"), runBtn = $("#run-book");
  box.classList.remove("callout--done");
  action.hidden = true;
  action.onclick = null;
  let runIsPrimary = false;
  const mine = list.filter(x => x.st.you);
  const firstMine = mine[0];
  const offerMine = (primary) => {
    if (!firstMine) return;
    action.hidden = false;
    action.className = `btn ${primary ? "btn--primary" : "btn--secondary"}`;
    action.textContent = `${firstMine.ch.id} openen`;
    action.title = `${firstMine.ch.id} ${firstMine.ch.title}: ${firstMine.st.next.toLowerCase()}.`;
    action.onclick = () => goChapter(firstMine.ch.id);
  };
  if (!s.ingested) {
    text.textContent = bookBusy()
      ? "Het boek wordt ingelezen. Daarna begint het schrijven vanzelf, als dat bij het uploaden aan stond."
      : "Het boek is nog niet ingelezen. Hieronder staat wat er misging: los dat op en upload de PDF opnieuw.";
  } else if (bookBusy()) {
    text.textContent = "Er loopt een taak voor dit boek. Je kunt dit venster sluiten: de taak gaat door zolang de app draait. Kijk later hier wat er klaar is.";
    offerMine(false);
  } else {
    const todo = list.filter(x => !x.ch.script || (x.st.fail && !x.ch.audit));
    const broken = todo.filter(x => x.st.broken).length;
    if (todo.length) {
      runIsPrimary = true;
      text.textContent = `${todo.length === 1 ? "1 hoofdstuk heeft" : `${todo.length} hoofdstukken hebben`} nog geen script met audit` +
        (broken ? ` (${broken} mislukt)` : "") + ". Hele boek doorlopen maakt wat ontbreekt en slaat over wat al klaar is." +
        (mine.length ? ` Intussen ${mine.length === 1 ? "wacht 1 hoofdstuk" : `wachten ${mine.length} hoofdstukken`} op jou.` : "");
      offerMine(false);
    } else if (firstMine) {
      text.textContent = `${mine.length === 1 ? "1 hoofdstuk wacht" : `${mine.length} hoofdstukken wachten`} op jou. Begin bij ${firstMine.ch.id} ${firstMine.ch.title}: ${firstMine.st.next.toLowerCase()}.`;
      offerMine(true);
    } else {
      box.classList.add("callout--done");
      text.textContent = "Alle afleveringen zijn klaar. Open een hoofdstuk om te luisteren.";
    }
  }
  runBtn.className = `btn ${runIsPrimary ? "btn--primary" : "btn--secondary"}`;
  const bookJob = state.jobs.some(j => j.book_id === s.book_id && !j.chapter_id && ACTIVE.includes(j.status));
  runBtn.disabled = !s.ingested || bookJob;
  runBtn.dataset.busy = bookJob ? "1" : "";
}

// --------------------------------------------------------------- chapters
async function openChapter(chapterId, { keepTab = false } = {}) {
  const same = chapterId === state.chapter && state.view === "chapter";
  const scroll = $("#content").scrollTop;
  let data;
  try { data = await api.get(`/api/books/${enc(state.book)}/chapters/${enc(chapterId)}`); } catch (err) { fail(err, "Hoofdstuk openen"); return; }
  state.chapter = chapterId;
  state.data = data;
  setDirty(false);
  showView("chapter");
  const info = chapterInfo(chapterId);
  $("#chapter-overline").textContent = `Hoofdstuk ${chapterId}`;
  $("#chapter-title").textContent = (data.plan && data.plan.chapter_title) || (info && info.title) || chapterId;
  const meta = [];
  if (info) meta.push(`p${info.pages[0]}-${info.pages[1]}`, `${info.chars.toLocaleString("nl-NL")} tekens`);
  if (data.script) meta.push(`script revisie ${data.script.revision}`, `streef ${data.script.target_minutes} min`);
  $("#chapter-meta").textContent = meta.join(" · ");
  renderChapterActions(data);
  renderScript(data);
  renderAudit(data);
  renderPlan(data);
  renderAudio(data);
  renderArtifacts();
  renderTimings(chapterId);
  loadGlossary();
  renderSidebar();
  if (same || keepTab) { $("#content").scrollTop = same ? scroll : 0; autosizeAll(); }
  else { showTab("script"); $("#content").scrollTop = 0; }
}

const RENDER_TITLE = $("#render").title;
function setStepLabel(btn, text) { btn.innerHTML = `<i></i>${escapeHtml(text)}`; }

function renderChapterActions(data) {
  const approveBtn = $("#approve"), renderBtn = $("#render");
  approveBtn.disabled = !(data.audit && data.audit.passed) || data.approved;
  setStepLabel(approveBtn, data.approved ? "5 Goedgekeurd" : "5 Goedkeuren");
  approveBtn.title = approveReason(data);
  renderBtn.disabled = !data.approved;
  renderBtn.title = data.approved ? RENDER_TITLE : "Stap 6. Kan pas na je goedkeuring van deze scriptversie (stap 5). De render kost uren GPU-tijd, lees daarom eerst het script en de audit.";
  const a = auditState(data);
  const set = (step, st) => { $(`.step[data-step="${step}"]`).dataset.state = st || ""; };
  set("plan", data.plan ? "done" : "next");
  set("glossary", state.status && state.status.glossary_entries ? "done" : "");
  set("script", data.script ? "done" : data.plan ? "next" : "");
  set("audit", a === "passed" ? "done" : a === "blocked" ? "bad" : a === "stale" ? "next" : "");
  set("approve", data.approved ? "done" : a === "passed" ? "next" : "");
  set("render", data.final.audio ? "done" : data.approved ? "next" : "");
  renderNext(data);
}

// The audit only counts when it checked the script revision that is on disk now.
function auditState(data) {
  if (!data.script) return "none";
  if (!data.audit || data.audit.script_revision !== data.script.revision) return "stale";
  return data.audit.passed ? "passed" : "blocked";
}

function approveReason(data) {
  if (data.approved) return `Stap 5. Revisie ${data.script.revision} is goedgekeurd. Sla je het script opnieuw op, dan vervalt dit.`;
  if (!data.script) return "Stap 5. Er is nog geen script om goed te keuren.";
  const a = auditState(data);
  if (a === "blocked") return "Stap 5. Kan nog niet: de audit vond blokkerende problemen. Los ze op in het script, sla op en draai de audit opnieuw.";
  if (!data.audit) return "Stap 5. Kan nog niet: draai eerst de audit.";
  if (a === "stale") return "Stap 5. De audit hoort bij een oudere revisie. Goedkeuren draait de audit eerst opnieuw op deze versie.";
  return "Stap 5. Jouw akkoord op deze scriptversie. Lees eerst het script en de audit; daarna mag de Chatterbox-render.";
}

// The one next step for this chapter: a sentence and the primary button that does it.
function nextStep(data) {
  if (state.dirty) {
    return { text: "Je hebt het script aangepast. Sla het op; daarna draai je de audit opnieuw op de nieuwe versie.",
      label: "Script opslaan", action: saveScript };
  }
  const job = chapterJob(state.chapter);
  if (job || chapterBusy(state.chapter)) {
    return { text: job ? `${STAGES[job.stage] || job.stage} ${job.status === "queued" ? "staat in de wachtrij" : "is bezig"} voor dit hoofdstuk. De voortgang staat bovenaan.`
      : "De run van het hele boek werkt nu aan dit hoofdstuk. De voortgang staat bovenaan.", label: "Bezig...", disabled: true };
  }
  if (!data.plan) {
    return { text: "Dit hoofdstuk heeft nog geen plan. Zonder plan weet het script niet wat het feitelijk moet overbrengen.",
      label: "Plan maken", action: () => runStage("plan"), title: $('.step[data-step="plan"]').title };
  }
  if (!data.script) {
    return { text: "Het plan is klaar. Wil je eerst de uitspraak van vaktermen laten zoeken, gebruik dan stap 2 Lexicon. Het script wordt meteen geaudit.",
      label: "Script schrijven", action: () => runStage("script"), title: $('.step[data-step="script"]').title };
  }
  const a = auditState(data);
  if (a === "stale") {
    return { text: "Het script is gewijzigd sinds de laatste controle. Draai de audit op deze versie.",
      label: "Audit draaien", action: () => runStage("audit"), title: $('.step[data-step="audit"]').title };
  }
  if (a === "blocked") {
    const blocking = data.audit.issues.filter(i => i.severity === "blocking");
    const first = blocking.find(i => i.line_id && data.script.segments.some(s => s.lines.some(l => l.id === i.line_id)));
    return { text: `${blocking.length === 1 ? "1 blokkerend probleem" : `${blocking.length} blokkerende problemen`} oplossen. Pas de regels aan in het script, sla op en draai de audit opnieuw. Of laat stap 3 Script alles opnieuw schrijven.`,
      label: first ? "Naar het eerste probleem" : "Problemen bekijken", action: () => first ? jumpToLine(first.line_id) : showTab("audit") };
  }
  if (!data.approved) {
    return { text: "Lees het script en de bevindingen van de audit. Klopt de inhoud, keur dan goed; daarna mag de render.",
      label: "Goedkeuren", action: approve, title: approveReason(data) };
  }
  if (!data.final.audio) {
    return { text: "Goedgekeurd. De render met Chatterbox duurt uren; de voortgang staat bovenaan.",
      label: "Render starten", action: () => runStage("render"), title: RENDER_TITLE };
  }
  const flagged = data.manifest ? data.manifest.turns.filter(t => t.flagged).length : 0;
  if (flagged) {
    return { text: `De aflevering is klaar. ${flagged === 1 ? "1 beurt is" : `${flagged} beurten zijn`} gevlagd: beluister ze bij Audio.`,
      label: "Naar de audio", action: () => showTab("audio") };
  }
  return { text: "Klaar. De aflevering staat bij Audio.", label: "Beluisteren", action: () => showTab("audio"), done: true };
}

function renderNext(data) {
  if (!data) return;
  const n = nextStep(data);
  $("#next-step").textContent = n.text;
  const failure = !state.dirty && (!data.script || !data.audit) ? lastFailure(state.chapter) : null;
  $("#next-why").hidden = !failure;
  if (failure) {
    $("#next-why").textContent = `De laatste run ${failure.status === "stopped" ? "is gestopt" : "liep vast"} bij ${failure.title}` +
      (failure.error ? `: ${failure.error}` : ".") + (explainError(failure.error) ? ` ${explainError(failure.error)}` : "");
  }
  const btn = $("#next-action");
  btn.hidden = !n.label;
  btn.textContent = n.label || "";
  btn.title = n.title || "";
  btn.disabled = !!n.disabled;
  state.nextAction = n.action || null;
  $("#next-box").classList.toggle("callout--done", !!n.done);
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
  return `<div class="lineissue">${chip(blocking ? "blokkerend" : "waarschuwing", blocking ? "bad" : "warn")}<span><span class="rule">${escapeHtml(i.rule)}</span> ${escapeHtml(i.message)}${i.suggestion ? ` <span class="muted">${escapeHtml(i.suggestion)}</span>` : ""}</span></div>`;
}

function renderScript(data) {
  const root = $("#script-lines");
  root.innerHTML = "";
  $("#save-script").disabled = !data.script;
  if (!data.script) {
    root.innerHTML = `<p class="empty">Nog geen script. ${chapterBusy(state.chapter) ? "Het wordt nu gemaakt; de voortgang staat bovenaan." : data.plan ? "Kies hierboven Script schrijven." : "Eerst een plan, dan een script."}</p>`;
    $("#script-meta").textContent = "";
    updateIssueFilter(0);
    return;
  }
  const issues = {};
  for (const i of (data.audit ? data.audit.issues : [])) if (i.line_id) (issues[i.line_id] ||= []).push(i);
  const total = data.script.segments.reduce((n, s) => n + s.lines.length, 0);
  const stale = auditState(data) === "stale" && data.audit;
  $("#script-meta").textContent = `revisie ${data.script.revision} · ${total} regels · streef ${data.script.target_minutes} min` +
    (stale ? " · opmerkingen horen bij een oudere revisie" : "");
  const speakers = [...new Set(data.script.segments.flatMap(s => s.lines.map(l => l.speaker)))];
  const frag = document.createDocumentFragment();
  let flagged = 0;
  for (const seg of data.script.segments) {
    const h = document.createElement("div");
    h.className = "segment" + (seg.lines.some(l => issues[l.id]) ? " has-issues" : "");
    h.innerHTML = `<span class="segment__type">${escapeHtml(SEGMENTS[seg.type] || seg.type)}</span>` +
      (seg.title && seg.title.toLowerCase() !== (SEGMENTS[seg.type] || seg.type).toLowerCase() ? `<span class="segment__title">${escapeHtml(seg.title)}</span>` : "") +
      (seg.covers.length ? `<span class="segment__covers">dekt ${escapeHtml(seg.covers.join(", "))}</span>` : "");
    frag.appendChild(h);
    for (const line of seg.lines) {
      const li = issues[line.id] || [];
      if (li.length) flagged += 1;
      const blocking = li.some(i => i.severity === "blocking");
      const div = document.createElement("div");
      div.className = "line" + (blocking ? " has-issue" : li.length ? " has-warning" : "");
      div.dataset.id = line.id;
      div.dataset.spk = String(speakers.indexOf(line.speaker));
      const mode = line.overlap.mode;
      div.innerHTML = `<span class="id">${escapeHtml(line.id)}</span>
        <div class="meta-who"><select class="speaker" aria-label="Spreker van ${escapeHtml(line.id)}">${speakers.map(s => `<option ${s === line.speaker ? "selected" : ""}>${escapeHtml(s)}</option>`).join("")}</select></div>
        <textarea class="text" rows="1" aria-label="Tekst van ${escapeHtml(line.id)}">${escapeHtml(line.text)}</textarea>
        <div class="line__meta">
          <label class="mf" title="De emotie waarmee de regel wordt uitgesproken, gescheiden door komma's."><span>tags</span><input class="tags" type="text" value="${escapeHtml(line.tags.join(", "))}"></label>
          <label class="mf" title="Welke beweringen uit het plan deze regel overbrengt (c1, c2)."><span>dekt</span><input class="covers" type="text" value="${escapeHtml(line.covers.join(", "))}"></label>
          <label class="mf" title="geen: een gewone beurt. onderbreekt: valt de vorige spreker in de rede. reactie: kort onder de ander door (ja, precies, een lach)."><span>overlap</span><select class="overlap${mode === "none" ? " is-none" : ""}">${Object.entries(OVERLAP).map(([v, t]) => `<option value="${v}" ${mode === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>
          <label class="mf" title="Een bewuste stilte na de regel, in milliseconden."><span>pauze</span><input class="pause" type="number" min="0" step="250" placeholder="ms" value="${line.pause_after_ms || ""}"></label>
        </div>
        ${li.length ? `<div class="issues">${li.map(issueLine).join("")}</div>` : ""}`;
      frag.appendChild(div);
    }
  }
  root.appendChild(frag);
  updateIssueFilter(flagged);
  autosizeAll();
}

function updateIssueFilter(n) {
  const box = $("#only-issues");
  box.disabled = !n;
  if (!n) box.checked = false;
  box.parentElement.lastChild.textContent = ` alleen regels met opmerkingen${n ? ` (${n})` : ""}`;
  $("#script-lines").classList.toggle("only-issues", box.checked);
}

function collectScript() {
  const script = JSON.parse(JSON.stringify(state.data.script));
  const byId = {};
  for (const seg of script.segments) for (const l of seg.lines) byId[l.id] = l;
  let prev = null;
  for (const div of document.querySelectorAll("#script-lines .line")) {
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
  $("#dirty-note").hidden = !on;
  $("#save-script").className = `btn ${on ? "btn--primary" : "btn--secondary"} btn--sm`;
}
function markDirty() {
  if (state.dirty || !state.data || !state.data.script) return;
  setDirty(true);
  renderNext(state.data);
}

async function saveScript() {
  if (!state.data || !state.data.script) return;
  if (chapterBusy(state.chapter)) {
    toast("Er loopt een taak voor dit hoofdstuk. Opslaan kan als die klaar is; je wijzigingen blijven staan.", true);
    return;
  }
  try {
    const out = await busy($("#save-script"), "Opslaan...", () => api.send(chapterUrl("/script"), "PUT", collectScript()));
    log(`script opgeslagen als revisie ${out.revision}; audit en goedkeuring vervallen`);
    toast(`Script opgeslagen als revisie ${out.revision}. Draai de audit opnieuw voor deze versie.`);
    setDirty(false);
    await openChapter(state.chapter, { keepTab: true });
    refreshBookSoon();
  } catch (err) { fail(err, "Script opslaan"); }
}

function jumpToLine(id) {
  showTab("script");
  const el = document.querySelector(`#script-lines .line[data-id="${CSS.escape(id)}"]`);
  if (!el) return;
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
    $("#audit-summary").innerHTML = `<p class="empty">Nog geen audit. ${data.script ? "Draai stap 4 Audit." : "Eerst een script."}</p>`;
    $("#audit-issues").innerHTML = "";
    count.hidden = true;
    return;
  }
  const blocking = a.issues.filter(i => i.severity === "blocking");
  const warnings = a.issues.filter(i => i.severity === "warning");
  count.hidden = !a.issues.length;
  count.textContent = blocking.length || warnings.length;
  count.className = `count${blocking.length ? " count--bad" : ""}`;
  count.title = `${blocking.length} blokkerend, ${warnings.length} waarschuwingen`;
  const stale = data.script && a.script_revision !== data.script.revision;
  const stats = Object.entries(a.stats).map(([k, v]) => `${k}=${v}`).join(" · ");
  $("#audit-summary").innerHTML = `<section class="card">
    <div class="verdict">${a.passed ? chip("GESLAAGD", "ok", "", "chip--lg") : chip("GEBLOKKEERD", "bad", "", "chip--lg")}
      <h2>${blocking.length} blokkerend, ${warnings.length} ${warnings.length === 1 ? "waarschuwing" : "waarschuwingen"}</h2>
      ${a.script_revision ? `<span class="muted small">revisie ${a.script_revision}</span>` : ""}
      ${stale ? chip("hoort bij een oudere revisie", "warn", "Het script is daarna opgeslagen. Draai de audit opnieuw.") : ""}</div>
    <div class="auditstats">
      <span>dekking: ${a.coverage.covered.length}/${a.coverage.required.length} vereiste beweringen${a.coverage.missing.length ? ", ontbreekt: " + escapeHtml(a.coverage.missing.join(", ")) : ""}</span>
      <span>ondersteuning: ${a.support.checked} regels gecontroleerd, ${a.support.unsupported} niet ondersteund${a.support.skipped ? " (overgeslagen, geen LLM)" : ""}</span>
    </div>
    ${stats ? `<div class="auditstats">${escapeHtml(stats)}</div>` : ""}
  </section>`;
  const group = (title, items) => items.length ? `<div class="issuegroup"><h3>${title} (${items.length})</h3>${items.map(i => `
    <div class="issue ${i.severity}">
      ${chip(i.severity === "blocking" ? "blokkerend" : "waarschuwing", i.severity === "blocking" ? "bad" : "warn")}
      <div class="issue__head"><span class="issue__rule">${escapeHtml(i.check)}/${escapeHtml(i.rule)}</span>
        ${i.line_id ? `<a href="#" class="lineref" data-line="${escapeHtml(i.line_id)}" title="Naar deze regel in het script.">${escapeHtml(i.line_id)}</a>` : ""}
        ${i.claim_id ? chip(i.claim_id, "", "Bewering uit het plan.") : ""}</div>
      <div class="issue__msg">${escapeHtml(i.message)}</div>
      ${i.suggestion ? `<div class="issue__tip">${escapeHtml(i.suggestion)}</div>` : ""}
    </div>`).join("")}</div>` : "";
  $("#audit-issues").innerHTML = group("Blokkerend", blocking) + group("Waarschuwingen", warnings) ||
    '<p class="empty">De audit vond niets. Lees het script toch even door voordat je goedkeurt.</p>';
}

// -------------------------------------------------------------------- plan
function renderPlan(data) {
  const p = data.plan;
  const root = $("#plan-view");
  if (!p) { root.innerHTML = '<p class="empty">Nog geen plan.</p>'; return; }
  root.innerHTML = `<div class="plan">
    <p class="plan__summary">${escapeHtml(p.summary)}</p>
    <section><h3>Leerdoelen</h3><ul class="bullets">${p.learning_objectives.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></section>
    <section><h3>Beweringen (${p.key_claims.length})</h3>${p.key_claims.map(c => `<div class="claim"><span class="id">${escapeHtml(c.id)}</span><span>${escapeHtml(c.claim)}</span>
      <div class="claim__meta">${chip(`moeilijkheid ${c.difficulty}`, "", "1 tot 5: hoe lastig voor een eerstejaars.")} ${chip(`tentamen ${c.exam_relevance}`, "", "1 tot 5: hoe zeker dit tentamenstof is.")}
        <span title="${escapeHtml(scoreTitle(c.source_span.match_score))}">${escapeHtml(c.source_span.section)} (citaatscore ${c.source_span.match_score ?? "-"})</span></div></div>`).join("")}</section>
    <section><h3>Definities</h3><ul class="bullets">${p.definitions.map(d => `<li><b>${escapeHtml(d.term)}</b>: ${escapeHtml(d.definition)}</li>`).join("") || '<li class="muted">geen</li>'}</ul></section>
    <section><h3>Misvattingen</h3><ul class="bullets">${p.misconceptions.map(m => `<li><b>fout:</b> ${escapeHtml(m.wrong)} <b>juist:</b> ${escapeHtml(m.right)} <span class="muted">${escapeHtml(m.why_tempting)}</span></li>`).join("") || '<li class="muted">geen</li>'}</ul></section>
    <div class="planfacts">${chip(`expert nodig: ${p.needs_expert ? `ja${p.expert_domain ? ` (${p.expert_domain})` : ""}` : "nee"}`)} ${chip(`notatiezwaar: ${p.formula_dense_sections.join(", ") || "-"}`)}</div>
    ${p.warnings.length ? `<p class="muted">waarschuwingen: ${p.warnings.map(escapeHtml).join("; ")}</p>` : ""}
  </div>`;
}

function scoreTitle(score) {
  if (score === null || score === undefined) return "Citaatscore onbekend.";
  if (score >= 100) return "Citaatscore 100: het citaat staat letterlijk in deze sectie.";
  if (score > 0) return `Citaatscore ${score}: het citaat staat er bijna letterlijk in. De audit legt de regel naast die passage.`;
  return "Citaatscore 0: het citaat is niet gevonden. De hele sectie geldt als bron, dus de controle is zwakker.";
}

// ------------------------------------------------------------------- audio
function renderAudio(data) {
  const url = data.final.audio;
  $("#final-audio").innerHTML = url
    ? `<div class="player"><audio controls preload="metadata" src="${escapeHtml(url)}"></audio><a class="btn btn--secondary btn--sm" href="${escapeHtml(url)}" download>Downloaden</a></div>`
    : `<p class="muted">Nog niet gerenderd. ${data.approved ? "Start de render met stap 6." : "Keur eerst het script goed (stap 5), dan kan de render."}</p>`;
  const q = data.manifest_quality;
  const qualityEl = $("#audio-quality");
  if (q && q.verifiable_turns) {
    const verOk = q.verified_turns === q.verifiable_turns;
    const bits = [chip(`verificatie ${q.verified_turns}/${q.verifiable_turns}`, verOk ? "ok" : "bad", verOk ? "" : "Niet elke beurt kon worden geverifieerd, zie `studiepodcast doctor`.")];
    if (q.alignment_fallback_turns) bits.push(chip(`${q.alignment_fallback_turns} geschatte uitlijning`, "warn"));
    qualityEl.innerHTML = bits.join(" ");
  } else {
    qualityEl.innerHTML = "";
  }
  const flagged = data.manifest ? data.manifest.turns.filter(t => t.flagged) : [];
  $("#flagged").innerHTML = flagged.length ? `<section class="card"><div class="cardhead"><h2>Gevlagde beurten</h2>${chip(String(flagged.length), "warn")}</div>
    <div class="flagged">${flagged.map(t => `<div class="issue warning">${chip(t.turn_id, "warn")}<div class="issue__head"><b>${escapeHtml(t.speaker)}</b> <span class="muted">${escapeHtml(t.flag_reason || "")}</span></div><div class="issue__msg">${escapeHtml(t.text_spoken)}</div></div>`).join("")}</div></section>` : "";
  const root = $("#blocks");
  root.innerHTML = "";
  const spans = (data.final.transcript && data.final.transcript.blocks) || [];
  const paths = Object.fromEntries(spans.map(s => [s.block_id, s.path]));
  if (!data.blocks.length) { root.innerHTML = '<p class="empty" style="margin:0 16px 16px">Nog geen blokken: eerst een script.</p>'; return; }
  for (const b of data.blocks) {
    const div = document.createElement("div");
    div.className = "block";
    const src = paths[b.id] ? bookUrl(`/artifacts/render/${enc(state.chapter)}/blocks/${enc(b.id)}.mp3`) : null;
    div.innerHTML = `<b>${escapeHtml(b.id)}</b><span class="block__info">${b.line_ids.length} regels, ${b.chars} tekens, ${escapeHtml(b.segment_types.map(t => SEGMENTS[t] || t).join(" / "))}
      ${b.too_long ? chip("te lang", "bad") : ""}</span>
      ${src ? `<audio controls preload="none" src="${src}"></audio>` : '<span class="muted small">nog geen audio</span>'}`;
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
    const head = `Run van ${fmtWhen(rows[0].started_at)}` + (total ? `, hoofdstuk in ${fmtSeconds(total.seconds)}` : "");
    return `<section class="card card--flush runblock"><div class="cardhead"><h3>${escapeHtml(head)}</h3>${total ? TIMING_STATUS[total.status] || "" : ""}</div>
      <div class="tablewrap"><table class="table timings"><thead><tr><th>Stap</th><th class="opt-3">Begon</th><th class="opt-3">Klaar</th><th class="num">Duur</th><th class="opt-2" style="width:20%"><span class="sr-only">Verhouding</span></th>
      <th class="num opt-2" title="Hoeveel keer het schrijfmodel werd aangeroepen in deze stap.">Modelaanroepen</th><th></th></tr></thead><tbody>` +
      passes.map(e => `<tr class="${e.status === "failed" ? "is-failed" : ""}"><td>${escapeHtml(passTitle(e))}${e.error ? `<div class="why">${escapeHtml(e.error)}</div>` : ""}</td>
        <td class="opt-3">${fmtClock(e.started_at)}</td><td class="opt-3">${fmtClock(e.finished_at)}</td>
        <td class="num">${e.status === "skipped" ? "-" : fmtSeconds(e.seconds)}</td>
        <td class="opt-2">${e.status === "skipped" ? "" : `<div class="timebar" title="${Math.round((e.seconds / longest) * 100)}% van de langste stap"><i style="width:${Math.max(2, (e.seconds / longest) * 100).toFixed(1)}%"></i></div>`}</td>
        <td class="num opt-2">${e.llm_calls || ""}</td><td>${TIMING_STATUS[e.status] || ""}</td></tr>`).join("") + "</tbody></table></div></section>";
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
  } catch (err) { fail(err, "Lexicon laden"); }
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
  for (const tr of document.querySelectorAll("#glossary tbody tr")) {
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
  try { jobs = await api.get("/api/jobs" + (state.book ? `?book_id=${enc(state.book)}` : "")); } catch { return state.jobs; }
  state.jobs = jobs;
  $("#jobs").innerHTML = jobs.slice(0, 20).map(j => {
    const [label, kind] = JOB_STATUS[j.status] || [j.status, ""];
    const stop = ACTIVE.includes(j.status)
      ? `<button class="link" data-cancel="${escapeHtml(j.id)}" title="${j.status === "queued" ? "Haalt deze taak uit de wachtrij." : "Stopt deze taak bij het volgende controlepunt."}">stop</button>` : "";
    return `<li class="${j.status}"><span class="job__name" title="Taak ${escapeHtml(j.id)}">${escapeHtml(jobName(j))}</span>
      <span class="job__side">${chip(label, kind)}${stop}</span>
      <span class="job__when">${fmtWhen(j.created_at)}</span>
      ${j.error ? `<span class="job__err" title="${escapeHtml(j.error)}">${escapeHtml(j.error)}</span>` : ""}</li>`;
  }).join("") || '<li class="muted">Nog geen taken.</li>';
  renderDrawerSummary();
  return jobs;
}

function renderDrawerSummary() {
  const running = state.jobs.filter(j => j.status === "running").length;
  const queued = state.jobs.filter(j => j.status === "queued").length;
  const last = state.jobs.find(j => !ACTIVE.includes(j.status));
  const parts = [];
  if (running) parts.push(chip(`${running} bezig`, "run"));
  if (queued) parts.push(chip(`${queued} in de wachtrij`));
  if (last) {
    const [label, kind] = JOB_STATUS[last.status] || [last.status, ""];
    parts.push(`<span class="last">laatste: ${escapeHtml(jobName(last))} ${chip(label, kind)} ${escapeHtml(fmtWhen(last.updated_at))}</span>`);
  }
  $("#drawer-summary").innerHTML = parts.join("");
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
  const where = run.job.chapter_id ? ` · ${run.job.chapter_id}` : ` · ${run.job.book_id}`;
  $("#progress-title").textContent = `${STAGES[run.stage] || STAGES[run.job.stage] || run.job.stage}${where}`;
  $("#progress-panel").classList.toggle("stopping", run.stopping);
  const outer = [];
  if (run.book) outer.push(`hoofdstuk ${run.book.index} van ${run.book.total}: ${run.book.label}`);
  if (run.outer) outer.push(`stap ${run.outer.index} van ${run.outer.total}: ${run.outer.label}`);
  $("#progress-outer").textContent = outer.join(" · ");
  $("#progress-outer").title = outer.join("\n");
  const bar = $("#progress-bar");
  const detail = [];
  if (run.inner) {
    bar.max = run.inner.total;
    bar.value = run.inner.done;
    detail.push(`${run.inner.index} van ${run.inner.total}${run.inner.label ? ": " + run.inner.label : ""}`);
    if (run.inner.done > 0 && run.inner.done < run.inner.total) {
      const per = (now - run.innerStart) / run.inner.done;
      detail.push(`nog ~${fmtDuration(per * (run.inner.total - run.inner.done))} voor ${STAGES[run.inner.stage] || run.inner.stage}`);
    }
  } else {
    bar.removeAttribute("value");  // no known total: an indeterminate bar
  }
  if (run.tokens && now - run.tokens.at < 10000) {
    const rate = run.tokens.seconds ? ` (${(run.tokens.tokens / run.tokens.seconds).toFixed(1)} per s)` : "";
    detail.push(`model schrijft: ${run.tokens.tokens} tokens${rate}`);
  }
  detail.push(`bezig ${fmtDuration(now - run.started)}`);
  const queued = state.jobs.filter(j => j.status === "queued").length;
  if (queued) detail.push(`${queued} in de wachtrij`);
  if (run.stopping) detail.push("stopt bij het volgende controlepunt");
  $("#progress-detail").textContent = detail.join(" · ");
}

async function stopJob(jobId) {
  const job0 = state.jobs.find(j => j.id === jobId);
  if (job0 && job0.status === "running" && !confirm(`${jobName(job0)} stoppen? Wat al klaar is, blijft bewaard; de stap die bezig is, start je later opnieuw.`)) return;
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
  log(`▶ job ${job.id} ${STAGES[job.stage] || job.stage} ${job.chapter_id || ""}`);
  // The browser reconnects on its own and sends the last event id, so a render that outlives one stream continues.
  const es = new EventSource(`/api/jobs/${enc(job.id)}/events`);
  es.addEventListener("stage", (e) => {
    const ev = JSON.parse(e.data);
    trackEvent(job, ev);
    if (ev.stage === "llm" && ev.status === "tokens") return;  // heartbeat: progress strip only
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
      log(`■ job ${j.id} ${j.status}${j.error ? ": " + j.error : ""}`);
      es.close();
      watching.delete(j.id);
      endRun(j);
      if (j.status === "done") toast(`${jobName(j)} is klaar.`);
      else if (j.status === "failed") toast(`${jobName(j)} is mislukt. ${explainError(j.error) || j.error || "Onbekende fout."}`, true);
      else toast(`${jobName(j)} is gestopt. Wat klaar was, is bewaard.`);
      if (j.status === "failed" && /opencode|not reachable|does not answer/i.test(j.error || "")) {
        setEngine("writer", "down", "antwoordt niet", j.error);
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
  if (!state.book) { await loadBooks(); return; }
  try {
    await loadBooks();
    if (j.book_id === state.book) await loadBook(state.book);
    if (state.view === "chapter" && state.chapter && !state.dirty) await openChapter(state.chapter, { keepTab: true });
    else afterJobsChanged();
  } catch { /* the next refresh catches up */ }
}

// Jobs decide what is busy: the overview, the chapter's next step and the list follow them.
function afterJobsChanged() {
  if (!state.status) return;
  renderSidebar();
  renderOverview();
  if (state.view === "chapter") renderNext(state.data);
}

async function runStage(stage, extra = {}) {
  const box = $("#quality");
  const opts = { ...(box && box.checked && !box.disabled ? { quality: true } : {}), ...extra };
  try {
    const job = await api.send(chapterUrl("/run"), "POST", { stage, ...opts });
    toast(`${STAGES[stage] || stage} voor ${state.chapter} is gestart. De voortgang staat bovenaan.`);
    watch(job);
  } catch (err) { fail(err, STAGES[stage] || stage); }
}

async function approve() {
  try {
    await busy($("#next-action"), "Bezig...", () => api.send(chapterUrl("/approve"), "POST", {}));
    log("goedgekeurd; de Chatterbox-render kan nu");
    toast("Goedgekeurd. De Chatterbox-render kan nu.");
    await openChapter(state.chapter, { keepTab: true });
    refreshBookSoon();
  } catch (err) { fail(err, "Goedkeuren"); }
}

// ----------------------------------------------------------------- engines
const ENGINE_TEXT = {
  writer: { reachable: "bereikbaar", ready: "klaar", test: "testmodus", down: "niet bereikbaar", checking: "controleren...", waking: "wordt gestart", unknown: "onbekend" },
  voice: { ready: "geïnstalleerd", test: "testmodus", down: "niet geïnstalleerd", checking: "controleren...", unknown: "onbekend" },
};
const ENGINE_INTRO = {
  writer: "Het schrijfmodel maakt plan, lexicon, script en audit.",
  voice: "De stemmen: Chatterbox maakt de echte aflevering.",
};

function healthText() {
  const h = state.health;
  if (!h) return "";
  const where = `OpenCode (${h.model}): het schrijfmodel draait op deze machine` + (h.offline ? ", offline: niets verlaat het netwerk" : "");
  const p = h.writing_passes || {};
  const passes = h.quality_mode
    ? ` · grondig: ${p.plan_review_rounds}× planreview, ${p.script_revise_rounds}× herschrijven, ${Math.round(p.seconds_per_llm_call / 60)} min per modelaanroep`
    : "";
  const fake = [h.fake_llm && "nep-LLM", h.fake_audio && "nep-audio"].filter(Boolean);
  return `${where} · plan per sectie (grondig)${passes}${fake.length ? ` · testmodus: ${fake.join(", ")}` : ""}`;
}

function setEngine(which, st, label, detail) {
  const el = $(`#engine-${which}`);
  el.dataset.state = st;
  $(".engine__state", el).textContent = label || ENGINE_TEXT[which][st] || st;
  if (detail !== undefined) state.engineDetail[which] = detail || "";
  engineTitle(which);
  if (which === "writer") renderEngineWarning();
}
function engineTitle(which) {
  const lines = [ENGINE_INTRO[which]];
  if (which === "writer" && healthText()) lines.push(healthText());
  if (state.engineDetail[which]) lines.push(`Details: ${state.engineDetail[which]}`);
  lines.push("Klik op het pijltje rechts om opnieuw te controleren.");
  $(`#engine-${which}`).title = lines.join("\n");
}

// Writer state from a running job: checking, starting a sleeping model, answering.
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
}

function renderEngineWarning() {
  const el = $("#engine-warning");
  const down = $("#engine-writer").dataset.state === "down";
  el.hidden = !down;
  if (down) {
    el.innerHTML = "<b>Het schrijfmodel is niet bereikbaar.</b> Start de OpenCode-app (of <code>opencode serve</code>) en klik bovenaan op het pijltje om opnieuw te controleren. " +
      "Zonder model stopt de taak meteen. Wil je het boek alvast alleen inlezen, zet dan bij Opties <i>direct doorlopen</i> uit.";
  }
}

async function loadHealth() {
  try {
    state.health = await api.get("/api/health");
  } catch { return; }
  const h = state.health;
  $("#engine-offline").hidden = !h.offline;
  if (h.tags && h.tags.length) $("#tag-list").textContent = h.tags.join(", ");
  applyQualityMode();
  engineTitle("writer");
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
}
function updateUploadLabel() {
  $("#upload-submit").textContent = $("#upload-auto").checked ? "Uploaden en starten" : "Uploaden en inlezen";
}

// ------------------------------------------------------------------ wiring
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".tab").forEach(t => t.hidden = t.id !== `tab-${name}`);
  if (name === "script") autosizeAll();
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
document.querySelectorAll("[data-run]").forEach(b => b.onclick = () => runStage(b.dataset.run));
$("#next-action").onclick = () => state.nextAction && state.nextAction();

$("#nav-overview").onclick = goOverview;
$("#brand").onclick = (e) => { e.preventDefault(); goOverview(); };
$("#chapter-list").onclick = (e) => { const b = e.target.closest("[data-chapter]"); if (b) goChapter(b.dataset.chapter); };
$("#chapters tbody").onclick = (e) => { const tr = e.target.closest("tr[data-chapter]"); if (tr) goChapter(tr.dataset.chapter); };
$("#chapters tbody").onkeydown = (e) => {
  const tr = e.target.closest("tr[data-chapter]");
  if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); goChapter(tr.dataset.chapter); }
};
$("#books").onchange = async (e) => {
  if (!leaveChapter()) { e.target.value = state.book; return; }
  state.chapter = null;
  await openBook(e.target.value);
};
$("#new-book").onclick = goUpload;
$("#upload-cancel").onclick = () => showView(state.book ? (state.chapter ? "chapter" : "book") : "upload");
$("#recheck").onclick = checkEngines;

fileInput.onchange = showChosenFile;
$("#upload-auto").onchange = updateUploadLabel;
for (const t of ["dragenter", "dragover"]) $("#drop").addEventListener(t, () => $("#drop").classList.add("is-over"));
for (const t of ["dragleave", "drop"]) $("#drop").addEventListener(t, () => $("#drop").classList.remove("is-over"));
// A PDF dropped anywhere opens the upload with that file instead of the browser opening the PDF.
window.addEventListener("dragover", (e) => { if (e.target !== fileInput) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (e.target === fileInput) return;
  e.preventDefault();
  const files = e.dataTransfer && e.dataTransfer.files;
  if (!files || !files.length) return;
  if (state.view !== "upload") goUpload();
  if (state.view === "upload") { fileInput.files = files; showChosenFile(); }
});

$("#upload").onsubmit = async (e) => {
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) { uploadError("Kies eerst de PDF van het boek."); return; }
  if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") { uploadError("Dit is geen PDF. Kies het studieboek als PDF-bestand."); return; }
  const fd = new FormData(e.target);
  fd.set("auto", fd.get("auto") ? "true" : "false");
  fd.set("quality", fd.get("quality") ? "true" : "false");
  try {
    const out = await busy($("#upload-submit"), "Bezig met uploaden...", async () => {
      const r = await fetch("/api/books", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    });
    e.target.reset();
    showChosenFile();
    updateUploadLabel();
    applyQualityMode();
    await loadBooks();
    await openBook(out.book_id);
    watch(out.job);
    toast(fd.get("auto") === "true" ? "Het boek is binnen. Inlezen en schrijven zijn gestart; dit duurt uren." : "Het boek is binnen en wordt ingelezen.");
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
$("#save-script").onclick = saveScript;
$("#approve").onclick = approve;
$("#audit-issues").onclick = (e) => {
  const id = e.target.dataset && e.target.dataset.line;
  if (!id) return;
  e.preventDefault();
  jumpToLine(id);
};
$("#save-glossary").onclick = async () => {
  try {
    await busy($("#save-glossary"), "Opslaan...", () => api.send(bookUrl("/glossary"), "PUT", collectGlossary()));
    log("lexicon opgeslagen");
    toast("Lexicon opgeslagen. Het geldt vanaf de volgende script- of renderbeurt.");
    await loadGlossary();
  } catch (err) { fail(err, "Lexicon opslaan"); }
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
$("#run-book").onclick = async () => {
  if (!state.book) return;
  try {
    const job = await api.send(bookUrl("/run"), "POST", { upto: "script", quality: $("#run-book-quality").checked });
    log("hele boek doorlopen: wat al klaar is wordt overgeslagen");
    toast("Hele boek doorlopen is gestart. Wat al klaar is, wordt overgeslagen.");
    watch(job);
  } catch (err) { fail(err, "Hele boek doorlopen"); }
};
$("#stop-job").onclick = () => run.job && stopJob(run.job.id);
$("#jobs").onclick = (e) => {
  const id = e.target.dataset && e.target.dataset.cancel;
  if (id) stopJob(id);
};
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && state.view === "chapter" && state.data && state.data.script) {
    e.preventDefault();
    saveScript();
  }
});
window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });

// ------------------------------------------------------------------- start
(async () => {
  updateUploadLabel();
  loadHealth();
  checkEngines();
  let books = [];
  try { books = await loadBooks(); } catch (err) { fail(err, "Boeken laden"); }
  // Where to start: the address (a reload), else the last book, else the first; no book means the upload.
  const m = decodeURIComponent(location.hash).match(/^#\/([^/]+)(?:\/([^/]+))?$/);
  const ids = books.map(b => b.book_id);
  const wanted = m && ids.includes(m[1]) ? m[1] : ids.includes(recall("book")) ? recall("book") : ids[0];
  if (location.hash === "#nieuw" || !wanted) {
    if (wanted) { try { await loadBook(wanted); } catch { /* shown when opened */ } }
    goUpload();
  } else {
    await openBook(wanted);
    if (m && m[1] === wanted && m[2] && state.status && state.status.chapters.some(c => c.id === m[2])) await openChapter(m[2]);
  }
  // After a reload: pick up jobs that are still running or waiting, oldest first.
  const jobs = await loadJobs();
  jobs.filter(j => ACTIVE.includes(j.status)).reverse().forEach(watch);
  afterJobsChanged();
})();
