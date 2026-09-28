/* Studiepodcast UI: upload, watch stages stream, inspect artifacts, edit scripts line by line. */
const $ = (sel, root = document) => root.querySelector(sel);
const state = { book: null, chapter: null, data: null, glossary: null, sources: {} };

const api = {
  async get(url) { const r = await fetch(url); if (!r.ok) throw new Error(await r.text()); return r.json(); },
  async send(url, method, body) {
    const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};

function log(text) { const el = $("#log"); el.textContent += text + "\n"; el.scrollTop = el.scrollHeight; }
function badge(value, labels = ["ja", "nee"]) {
  if (value === null || value === undefined) return '<span class="badge">-</span>';
  return `<span class="badge ${value ? "ok" : ""}">${value ? labels[0] : labels[1]}</span>`;
}

// ------------------------------------------------------------------ books
async function loadBooks() {
  const books = await api.get("/api/books");
  const ul = $("#books");
  ul.innerHTML = "";
  for (const b of books) {
    const li = document.createElement("li");
    li.textContent = `${b.book_id} · ${b.title || "(nog niet ingelezen)"}`;
    li.className = state.book === b.book_id ? "active" : "";
    li.onclick = () => openBook(b.book_id);
    ul.appendChild(li);
  }
}

async function openBook(bookId) {
  if (bookId !== state.book) {
    state.chapter = null;
    $("#chapter-panel").hidden = true;
  }
  state.book = bookId;
  const s = await api.get(`/api/books/${bookId}`);
  $("#book-panel").hidden = false;
  $("#book-title").textContent = `${s.title || bookId} (structuur: ${s.structure_method || "-"}, lexicon: ${s.glossary_entries})`;
  const tbody = $("#chapters tbody");
  tbody.innerHTML = "";
  for (const ch of s.chapters) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    const audit = ch.audit_passed === null ? "-" : ch.audit_passed
      ? '<span class="badge ok" title="Geen blokkerende problemen gevonden.">ok</span>'
      : `<span class="badge bad" title="Open het hoofdstuk en kijk in het Audit-tab wat er mis is.">${ch.audit_blocking} blokkerend</span>`;
    tr.innerHTML = `<td>${ch.id} ${ch.title}<br><span class="muted">p${ch.pages[0]}-${ch.pages[1]}, ${ch.chars} tekens</span></td>
      <td>${badge(ch.plan)}</td><td>${ch.script ? `<span class="badge ok" title="Revisie ${ch.script_revision}: zo vaak is het script opgeslagen.">rev ${ch.script_revision}</span>` : badge(false)}</td>
      <td>${audit}</td><td>${badge(ch.draft)}</td><td>${badge(ch.approved)}</td>
      <td>${badge(ch.final)}${ch.flagged_turns ? ` <span class="badge warn" title="Beurten waarvan geen enkele opname het script goed genoeg volgde. Beluister ze in het Audio-tab.">${ch.flagged_turns} gevlagd</span>` : ""}</td>`;
    tr.onclick = () => openChapter(ch.id);
    tbody.appendChild(tr);
  }
  loadBooks();
  loadJobs();
}

// --------------------------------------------------------------- chapters
async function openChapter(chapterId) {
  state.chapter = chapterId;
  const data = await api.get(`/api/books/${state.book}/chapters/${chapterId}`);
  state.data = data;
  $("#chapter-panel").hidden = false;
  $("#chapter-title").textContent = `${chapterId}: ${(data.plan && data.plan.chapter_title) || ""}`;
  $("#approve").disabled = !(data.audit && data.audit.passed) || data.approved;
  $("#approve").textContent = data.approved ? "6 Goedgekeurd" : "6 Goedkeuren";
  $("#approve").title = approveReason(data);
  $("#render").disabled = !data.approved;
  $("#render").title = data.approved ? RENDER_TITLE : "Stap 7. Kan pas na je goedkeuring van deze scriptversie (stap 6). De render kost uren GPU-tijd, daarom eerst de draft beluisteren.";
  $("#next-step").innerHTML = nextStep(data);
  renderScript(data);
  renderAudit(data);
  renderPlan(data);
  renderAudio(data);
  renderArtifacts();
  loadGlossary();
}

const RENDER_TITLE = $("#render").title;

// The audit only counts when it checked the script revision that is on disk now.
function auditState(data) {
  if (!data.script) return "none";
  if (!data.audit || data.audit.script_revision !== data.script.revision) return "stale";
  return data.audit.passed ? "passed" : "blocked";
}

function approveReason(data) {
  if (data.approved) return `Stap 6. Revisie ${data.script.revision} is goedgekeurd. Sla je het script opnieuw op, dan vervalt dit.`;
  if (!data.script) return "Stap 6. Er is nog geen script om goed te keuren.";
  const a = auditState(data);
  if (a === "blocked") return "Stap 6. Kan nog niet: de audit vond blokkerende problemen. Los ze op in het Script-tab, sla op en draai Audit opnieuw.";
  if (!data.audit) return "Stap 6. Kan nog niet: draai eerst de Audit.";
  if (a === "stale") return "Stap 6. De audit hoort bij een oudere revisie. Goedkeuren draait de audit eerst opnieuw op deze versie.";
  return "Stap 6. Jouw akkoord op deze scriptversie. Luister eerst de Piper-draft; daarna mag de Chatterbox-render.";
}

function nextStep(data) {
  const step = (text) => `<b>Volgende stap:</b> ${text}`;
  if (!data.plan) return step("<b>1 Plan</b>. Zonder plan weet het script niet wat het feitelijk moet overbrengen.");
  if (!data.script) return step("eventueel <b>2 Lexicon</b> voor de uitspraak van vaktermen, dan <b>3 Script</b>. Het script wordt meteen geaudit.");
  const a = auditState(data);
  if (a === "stale") return step("<b>4 Audit</b>. Het script is gewijzigd sinds de laatste controle.");
  if (a === "blocked") {
    const n = data.audit.issues.filter(i => i.severity === "blocking").length;
    return step(`${n} blokkerend${n === 1 ? " probleem" : "e problemen"} oplossen. Zie het Audit-tab, pas de regels aan in het Script-tab, sla op en draai <b>4 Audit</b>. Of laat <b>3 Script</b> alles opnieuw schrijven.`);
  }
  if (!data.draft.audio) return step("<b>5 Piper draft</b> en beluister hem in het Audio-tab.");
  if (!data.approved) return step("beluister de draft in het Audio-tab. Klopt de inhoud, klik dan <b>6 Goedkeuren</b>.");
  if (!data.final.audio) return step("<b>7 Chatterbox render</b>. Dit duurt uren; de voortgang staat in de log.");
  const flagged = data.manifest ? data.manifest.turns.filter(t => t.flagged).length : 0;
  return flagged
    ? step(`de aflevering is klaar. Beluister de ${flagged} gevlagde beurten in het Audio-tab.`)
    : "<b>Klaar.</b> De aflevering staat in het Audio-tab.";
}

function renderScript(data) {
  const root = $("#script-lines");
  root.innerHTML = "";
  if (!data.script) { root.innerHTML = '<p class="muted">Nog geen script.</p>'; $("#script-meta").textContent = ""; return; }
  const issues = {};
  for (const i of (data.audit ? data.audit.issues : [])) if (i.line_id) (issues[i.line_id] ||= []).push(i);
  $("#script-meta").textContent = `revisie ${data.script.revision} · ${data.script.segments.reduce((n, s) => n + s.lines.length, 0)} regels · streef ${data.script.target_minutes} min`;
  const speakers = [...new Set(data.script.segments.flatMap(s => s.lines.map(l => l.speaker)))];
  for (const seg of data.script.segments) {
    const h = document.createElement("div");
    h.className = "segment";
    h.textContent = `${seg.type} ${seg.title ? "· " + seg.title : ""} ${seg.covers.length ? "· dekt " + seg.covers.join(", ") : ""}`;
    root.appendChild(h);
    for (const line of seg.lines) {
      const div = document.createElement("div");
      div.className = "line" + (issues[line.id] ? " has-issue" : "");
      div.dataset.id = line.id;
      div.innerHTML = `<span class="id">${line.id}</span>
        <div class="meta"><select class="speaker">${speakers.map(s => `<option ${s === line.speaker ? "selected" : ""}>${s}</option>`).join("")}</select></div>
        <textarea class="text">${escapeHtml(line.text)}</textarea>
        <div class="meta">
          <input class="tags" type="text" placeholder="tags" value="${line.tags.join(", ")}">
          <input class="covers" type="text" placeholder="covers (c1, c2)" value="${line.covers.join(", ")}">
          <select class="overlap"><option ${line.overlap.mode === "none" ? "selected" : ""}>none</option><option ${line.overlap.mode === "interrupt" ? "selected" : ""}>interrupt</option><option ${line.overlap.mode === "backchannel" ? "selected" : ""}>backchannel</option></select>
          <input class="pause" type="number" min="0" step="250" placeholder="pauze ms" value="${line.pause_after_ms || ""}">
        </div>
        ${issues[line.id] ? `<div class="issues">${issues[line.id].map(i => `[${i.rule}] ${escapeHtml(i.message)}`).join("<br>")}</div>` : ""}`;
      root.appendChild(div);
    }
  }
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

function renderAudit(data) {
  const a = data.audit;
  if (!a) { $("#audit-summary").textContent = "Nog geen audit."; $("#audit-issues").innerHTML = ""; return; }
  $("#audit-summary").textContent = `${a.passed ? "GESLAAGD" : "GEBLOKKEERD"} · ${a.issues.filter(i => i.severity === "blocking").length} blokkerend, ${a.issues.filter(i => i.severity === "warning").length} waarschuwingen\n` +
    `dekking: ${a.coverage.covered.length}/${a.coverage.required.length} vereiste beweringen${a.coverage.missing.length ? " · ontbreekt: " + a.coverage.missing.join(", ") : ""}\n` +
    `ondersteuning: ${a.support.checked} regels gecontroleerd, ${a.support.unsupported} niet ondersteund${a.support.skipped ? " (overgeslagen, geen LLM)" : ""}\n` +
    Object.entries(a.stats).map(([k, v]) => `${k}=${v}`).join(" · ");
  const root = $("#audit-issues");
  root.innerHTML = "";
  for (const i of a.issues) {
    const div = document.createElement("div");
    div.className = `issue ${i.severity}`;
    div.innerHTML = `<b>${i.check}/${i.rule}</b> ${i.line_id ? `<a href="#" data-line="${i.line_id}">${i.line_id}</a>` : ""} ${i.claim_id || ""}: ${escapeHtml(i.message)}${i.suggestion ? `<br><span class="muted">${escapeHtml(i.suggestion)}</span>` : ""}`;
    root.appendChild(div);
  }
  root.onclick = (e) => {
    const id = e.target.dataset && e.target.dataset.line;
    if (!id) return;
    e.preventDefault();
    showTab("script");
    const el = document.querySelector(`.line[data-id="${id}"]`);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); $(".text", el).focus(); }
  };
}

function renderPlan(data) {
  const p = data.plan;
  const root = $("#plan-view");
  if (!p) { root.innerHTML = '<p class="muted">Nog geen plan.</p>'; return; }
  root.innerHTML = `<p>${escapeHtml(p.summary)}</p>
    <h3>Leerdoelen</h3><ul>${p.learning_objectives.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul>
    <h3>Beweringen</h3>${p.key_claims.map(c => `<div class="claim"><span class="id">${c.id}</span> ${escapeHtml(c.claim)} <span class="badge" title="1 tot 5: hoe lastig voor een eerstejaars.">moeilijkheid ${c.difficulty}</span> <span class="badge" title="1 tot 5: hoe zeker dit tentamenstof is.">tentamen ${c.exam_relevance}</span> <span class="muted" title="${escapeHtml(scoreTitle(c.source_span.match_score))}">${c.source_span.section} (${c.source_span.match_score ?? "-"})</span></div>`).join("")}
    <h3>Definities</h3><ul>${p.definitions.map(d => `<li><b>${escapeHtml(d.term)}</b>: ${escapeHtml(d.definition)}</li>`).join("")}</ul>
    <h3>Misvattingen</h3><ul>${p.misconceptions.map(m => `<li><b>fout:</b> ${escapeHtml(m.wrong)} <b>juist:</b> ${escapeHtml(m.right)} <span class="muted">${escapeHtml(m.why_tempting)}</span></li>`).join("")}</ul>
    <p>expert nodig: ${p.needs_expert ? `ja (${escapeHtml(p.expert_domain || "")})` : "nee"} · notatiezwaar: ${p.formula_dense_sections.join(", ") || "-"}</p>
    ${p.warnings.length ? `<p class="muted">waarschuwingen: ${p.warnings.map(escapeHtml).join("; ")}</p>` : ""}`;
}

function scoreTitle(score) {
  if (score === null || score === undefined) return "Citaatscore onbekend.";
  if (score >= 100) return "Citaatscore 100: het citaat staat letterlijk in deze sectie.";
  if (score > 0) return `Citaatscore ${score}: het citaat staat er bijna letterlijk in. De audit legt de regel naast die passage.`;
  return "Citaatscore 0: het citaat is niet gevonden. De hele sectie geldt als bron, dus de controle is zwakker.";
}

function renderAudio(data) {
  const player = (url) => url ? `<audio controls src="${url}"></audio> <a href="${url}" download>download</a>` : '<span class="muted">nog niet gerenderd</span>';
  $("#draft-audio").innerHTML = player(data.draft.audio);
  $("#final-audio").innerHTML = player(data.final.audio);
  const q = data.manifest_quality;
  const qualityEl = $("#audio-quality");
  if (qualityEl) {
    if (q && q.verifiable_turns) {
      const verOk = q.verified_turns === q.verifiable_turns;
      const bits = [`<span class="badge ${verOk ? "ok" : "bad"}">verificatie ${q.verified_turns}/${q.verifiable_turns}</span>`];
      if (q.alignment_fallback_turns) bits.push(`<span class="badge warn">${q.alignment_fallback_turns} geschatte uitlijning</span>`);
      qualityEl.innerHTML = bits.join(" ") + (verOk ? "" : ' <span class="muted">niet elke beurt kon worden geverifieerd, zie `studiepodcast doctor`</span>');
    } else {
      qualityEl.innerHTML = "";
    }
  }
  const flagged = data.manifest ? data.manifest.turns.filter(t => t.flagged) : [];
  $("#flagged").innerHTML = flagged.length ? `<h3>Gevlagde beurten</h3>${flagged.map(t => `<div class="issue">${t.turn_id} ${t.speaker}: ${escapeHtml(t.flag_reason || "")}<br><span class="muted">${escapeHtml(t.text_spoken)}</span></div>`).join("")}` : "";
  const root = $("#blocks");
  root.innerHTML = "";
  const spans = (data.final.transcript && data.final.transcript.blocks) || [];
  const paths = Object.fromEntries(spans.map(s => [s.block_id, s.path]));
  const overrides = (data.manifest && data.manifest.block_overrides) || {};
  for (const b of data.blocks) {
    const div = document.createElement("div");
    div.className = "block";
    const url = paths[b.id] ? `/api/books/${state.book}/artifacts/render/${state.chapter}/blocks/${b.id}.mp3` : null;
    div.innerHTML = `<input type="checkbox" value="${b.id}"> <b>${b.id}</b> <span class="muted">${b.line_ids.length} regels, ${b.chars} tekens, ${b.segment_types.join("/")}</span>
      ${overrides[b.id] ? '<span class="badge ok">ElevenLabs</span>' : ""} ${b.too_long ? '<span class="badge bad">te lang</span>' : ""} ${url ? `<audio controls src="${url}"></audio>` : ""}`;
    root.appendChild(div);
  }
}

async function renderArtifacts() {
  const items = await api.get(`/api/books/${state.book}/artifacts`);
  $("#artifact-list").innerHTML = items.map(a => `<li><a href="/api/books/${state.book}/artifacts/${a.path}" target="_blank">${a.path}</a> <span class="muted">${(a.bytes / 1024).toFixed(0)} kB</span></li>`).join("");
}

async function loadGlossary() {
  const g = await api.get(`/api/books/${state.book}/glossary`);
  state.glossary = g;
  $("#glossary tbody").innerHTML = "";
  loadGlossaryRows(g.entries);
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

// ------------------------------------------------------------------- jobs
async function loadJobs() {
  const jobs = await api.get("/api/jobs" + (state.book ? `?book_id=${state.book}` : ""));
  $("#jobs").innerHTML = jobs.slice(0, 12).map(j => `<li class="${j.status}">${STAGES[j.stage] || j.stage} ${j.chapter_id || ""} · ${j.status}` +
    (["queued", "running"].includes(j.status) ? ` <button class="link" data-cancel="${j.id}" title="${j.status === "queued" ? "Haalt deze job uit de wachtrij." : "Stopt deze job bij het volgende controlepunt."}">stop</button>` : "") +
    `${j.error ? " · " + escapeHtml(j.error.slice(0, 80)) : ""}</li>`).join("");
  return jobs;
}

const STAGES = { ingest: "Inlezen", pipeline: "Boek verwerken", plan: "Plan", glossary: "Lexicon", script: "Script",
  audit: "Audit", continuity: "Continuïteit", draft: "Piper draft", render: "Chatterbox render", eleven: "ElevenLabs",
  approve: "Goedkeuren", all: "Hoofdstuk", book: "Boek", run: "Hoofdstuk" };
// Events that announce the step about to start; "turn" reports a finished one.
const DONE_KINDS = new Set(["turn"]);
const run = { job: null, started: 0, stage: null, book: null, outer: null, inner: null, innerStart: 0, tokens: null, stopping: false, timer: null };
const watching = new Set();

function fmtDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

function startRun(job, ts) {
  Object.assign(run, { job, started: ts || Date.now(), stage: job.stage, book: null, outer: null, inner: null, tokens: null, stopping: false });
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
  $("#progress-panel").hidden = true;
}

function trackEvent(job, ev) {
  const d = ev.data || {};
  const ts = Date.parse(ev.ts) || Date.now();
  if (!run.job || run.job.id !== job.id) startRun(job, ts);
  if (ev.stage === "llm" && ev.status === "tokens") { run.tokens = { ...d, at: Date.now() }; renderProgress(); return; }
  if (ev.stage === "job" && ev.status === "stopping") run.stopping = true;
  else if (ev.stage === "book") run.book = { index: d.index, total: d.total, label: d.label };
  else if (ev.stage === "run") { run.outer = { index: d.index, total: d.total, label: d.label }; run.inner = null; }
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
  if (run.stopping) detail.push("stopt bij het volgende controlepunt");
  $("#progress-detail").textContent = detail.join(" · ");
}

async function stopJob(jobId) {
  try {
    const job = await api.send(`/api/jobs/${jobId}/cancel`, "POST", {});
    log(`■ stop gevraagd voor ${STAGES[job.stage] || job.stage} ${job.chapter_id || ""} (${job.status})`);
    if (run.job && run.job.id === jobId) {
      run.stopping = true;
      $("#stop-job").disabled = true;
      $("#stop-job").textContent = "Stoppen...";
      renderProgress();
    }
    loadJobs();
  } catch (err) { log("fout: " + err.message); }
}

function watch(job) {
  if (watching.has(job.id)) return;
  watching.add(job.id);
  log(`▶ job ${job.id} ${STAGES[job.stage] || job.stage} ${job.chapter_id || ""}`);
  // The browser reconnects on its own and sends the last event id, so a render that outlives one stream continues.
  const es = new EventSource(`/api/jobs/${job.id}/events`);
  es.addEventListener("stage", (e) => {
    const ev = JSON.parse(e.data);
    trackEvent(job, ev);
    if (ev.stage === "llm" && ev.status === "tokens") return;  // heartbeat: progress panel only
    const d = { ...ev.data };
    delete d.book_id;
    delete d.trace;
    log(`[${ev.stage}] ${ev.status} ${Object.entries(d).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ")}`);
  });
  es.addEventListener("job", (e) => {
    const j = JSON.parse(e.data);
    if (["done", "failed", "cancelled"].includes(j.status)) {
      log(`■ job ${j.id} ${j.status}${j.error ? ": " + j.error : ""}`);
      es.close();
      watching.delete(j.id);
      endRun(j);
      loadJobs();
      if (state.book) openBook(state.book).then(() => state.chapter && openChapter(state.chapter));
    }
  });
  loadJobs();
}

async function runStage(stage, extra = {}) {
  try {
    const job = await api.send(`/api/books/${state.book}/chapters/${state.chapter}/run`, "POST", { stage, ...extra });
    watch(job);
  } catch (err) { log("fout: " + err.message); }
}

// ------------------------------------------------------------------ wiring
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.hidden = t.id !== `tab-${name}`);
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
document.querySelectorAll("[data-run]").forEach(b => b.onclick = () => runStage(b.dataset.run));

$("#upload").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  fd.set("auto", fd.get("auto") ? "true" : "false");
  const r = await fetch("/api/books", { method: "POST", body: fd });
  if (!r.ok) { log("upload mislukt: " + await r.text()); return; }
  const out = await r.json();
  await loadBooks();
  openBook(out.book_id);
  watch(out.job);
};

$("#save-script").onclick = async () => {
  try {
    const out = await api.send(`/api/books/${state.book}/chapters/${state.chapter}/script`, "PUT", collectScript());
    log(`script opgeslagen als revisie ${out.revision}; audit en goedkeuring vervallen`);
    await openChapter(state.chapter);
  } catch (err) { log("fout: " + err.message); }
};
$("#approve").onclick = async () => {
  try {
    await api.send(`/api/books/${state.book}/chapters/${state.chapter}/approve`, "POST", {});
    log("goedgekeurd; de Chatterbox-render kan nu");
    await openChapter(state.chapter);
  } catch (err) { log("fout: " + err.message); }
};
$("#save-glossary").onclick = async () => {
  try { await api.send(`/api/books/${state.book}/glossary`, "PUT", collectGlossary()); log("lexicon opgeslagen"); await loadGlossary(); }
  catch (err) { log("fout: " + err.message); }
};
$("#add-entry").onclick = () => loadGlossaryRows([{ surface: "", kind: "loanword_en", spoken: "", lock: true }]);
function loadGlossaryRows(entries) {
  const tbody = $("#glossary tbody");
  for (const e of entries) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><input type="text" value="${escapeHtml(e.surface)}" data-k="surface"></td>
      <td><select data-k="kind">${["loanword_en", "notation", "abbreviation"].map(k => `<option ${k === e.kind ? "selected" : ""}>${k}</option>`).join("")}</select></td>
      <td><input type="text" value="${escapeHtml(e.spoken)}" data-k="spoken"></td>
      <td><input type="checkbox" ${e.lock ? "checked" : ""} data-k="lock"></td>
      <td><button class="link" data-del="1">x</button></td>`;
    tbody.appendChild(tr);
  }
}
$("#glossary").onclick = (e) => {
  if (e.target.dataset && e.target.dataset.del !== undefined) { e.target.closest("tr").remove(); }
};
$("#eleven").onclick = async () => {
  const blocks = [...document.querySelectorAll("#blocks input:checked")].map(i => i.value);
  if (!blocks.length) { log("selecteer eerst blokken"); return; }
  try { watch(await api.send(`/api/books/${state.book}/chapters/${state.chapter}/eleven`, "POST", { blocks })); }
  catch (err) { log("fout: " + err.message); }
};
$("#clear-log").onclick = () => { $("#log").textContent = ""; };
$("#stop-job").onclick = () => run.job && stopJob(run.job.id);
$("#jobs").onclick = (e) => {
  const id = e.target.dataset && e.target.dataset.cancel;
  if (id) stopJob(id);
};

function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

(async () => {
  try {
    const h = await api.get("/api/health");
    const where = h.llm_backend === "local"
      ? `lokaal model ${h.model}${h.offline ? ", offline: niets verlaat het netwerk" : ""}`
      : `Claude API (${h.model}): boektekst gaat naar Anthropic`;
    const plan = h.plan_mode === "thorough" ? "plan per sectie (grondig)" : "plan in één keer";
    const fake = [h.fake_llm && "nep-LLM", h.fake_audio && "nep-audio"].filter(Boolean);
    $("#health").textContent = `tekst: ${where} · ${plan}${fake.length ? ` · testmodus: ${fake.join(", ")}` : ""}`;
    if (h.tags && h.tags.length) $("#tag-list").textContent = h.tags.join(", ");
  } catch {}
  await loadBooks();
  const jobs = await loadJobs();
  // After a reload: pick up jobs that are still running or waiting, oldest first.
  jobs.filter(j => ["running", "queued"].includes(j.status)).reverse().forEach(watch);
})();
