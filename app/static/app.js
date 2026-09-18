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
    const audit = ch.audit_passed === null ? "-" : ch.audit_passed ? '<span class="badge ok">ok</span>' : `<span class="badge bad">${ch.audit_blocking} blokkerend</span>`;
    tr.innerHTML = `<td>${ch.id} ${ch.title}<br><span class="muted">p${ch.pages[0]}-${ch.pages[1]}, ${ch.chars} tekens</span></td>
      <td>${badge(ch.plan)}</td><td>${ch.script ? `<span class="badge ok">rev ${ch.script_revision}</span>` : badge(false)}</td>
      <td>${audit}</td><td>${badge(ch.draft)}</td><td>${badge(ch.approved)}</td>
      <td>${badge(ch.final)}${ch.flagged_turns ? ` <span class="badge warn">${ch.flagged_turns} gevlagd</span>` : ""}</td>`;
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
  $("#approve").textContent = data.approved ? "Goedgekeurd" : "Goedkeuren";
  $("#render").disabled = !data.approved;
  renderScript(data);
  renderAudit(data);
  renderPlan(data);
  renderAudio(data);
  renderArtifacts();
  loadGlossary();
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
    <h3>Beweringen</h3>${p.key_claims.map(c => `<div class="claim"><span class="id">${c.id}</span> ${escapeHtml(c.claim)} <span class="badge">moeilijkheid ${c.difficulty}</span> <span class="badge">tentamen ${c.exam_relevance}</span> <span class="muted">${c.source_span.section} (${c.source_span.match_score ?? "-"})</span></div>`).join("")}
    <h3>Definities</h3><ul>${p.definitions.map(d => `<li><b>${escapeHtml(d.term)}</b>: ${escapeHtml(d.definition)}</li>`).join("")}</ul>
    <h3>Misvattingen</h3><ul>${p.misconceptions.map(m => `<li><b>fout:</b> ${escapeHtml(m.wrong)} <b>juist:</b> ${escapeHtml(m.right)} <span class="muted">${escapeHtml(m.why_tempting)}</span></li>`).join("")}</ul>
    <p>expert nodig: ${p.needs_expert ? `ja (${escapeHtml(p.expert_domain || "")})` : "nee"} · notatiezwaar: ${p.formula_dense_sections.join(", ") || "-"}</p>
    ${p.warnings.length ? `<p class="muted">waarschuwingen: ${p.warnings.map(escapeHtml).join("; ")}</p>` : ""}`;
}

function renderAudio(data) {
  const player = (url) => url ? `<audio controls src="${url}"></audio> <a href="${url}" download>download</a>` : '<span class="muted">nog niet gerenderd</span>';
  $("#draft-audio").innerHTML = player(data.draft.audio);
  $("#final-audio").innerHTML = player(data.final.audio);
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
  $("#jobs").innerHTML = jobs.slice(0, 12).map(j => `<li class="${j.status}">${j.stage} ${j.chapter_id || ""} · ${j.status}${j.error ? " · " + escapeHtml(j.error.slice(0, 80)) : ""}</li>`).join("");
}

function watch(job) {
  log(`▶ job ${job.id} ${job.stage} ${job.chapter_id || ""}`);
  const es = new EventSource(`/api/jobs/${job.id}/events`);
  es.addEventListener("stage", (e) => {
    const ev = JSON.parse(e.data);
    const d = { ...ev.data };
    delete d.book_id;
    log(`[${ev.stage}] ${ev.status} ${Object.entries(d).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ")}`);
  });
  es.addEventListener("job", (e) => {
    const j = JSON.parse(e.data);
    if (["done", "failed", "cancelled"].includes(j.status)) {
      log(`■ job ${j.id} ${j.status}${j.error ? ": " + j.error : ""}`);
      es.close();
      loadJobs();
      if (state.book) openBook(state.book).then(() => state.chapter && openChapter(state.chapter));
    }
  });
  es.onerror = () => es.close();
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

function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

(async () => {
  try { const h = await api.get("/api/health"); $("#health").textContent = `model ${h.model}${h.fake_llm ? " (fake llm)" : ""}${h.fake_audio ? " (fake audio)" : ""}`; } catch {}
  await loadBooks();
  await loadJobs();
})();
