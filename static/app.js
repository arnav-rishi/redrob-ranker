"use strict";

const $ = (sel) => document.querySelector(sel);

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    pill.classList.add("active");
    $("#tab-" + pill.dataset.tab).classList.add("active");
  });
});

// ── Shared rendering ──────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function formatReason(text) {
  const safe = escapeHtml(text);
  const idx = safe.indexOf("Concern:");
  if (idx === -1) return safe;
  return safe.slice(0, idx) + '<span class="concern">' + safe.slice(idx) + "</span>";
}

function renderTable(host, rows, maxScore) {
  if (!rows.length) {
    host.innerHTML = '<p class="muted" style="padding:16px">No matching rows.</p>';
    return;
  }
  const top = maxScore || Math.max(...rows.map((r) => r.score), 0.0001);
  const body = rows.map((r) => {
    const w = Math.max(4, Math.round((r.score / top) * 64));
    return `<tr class="${r.rank <= 10 ? "rank-top" : ""}">
      <td class="col-rank">${r.rank}</td>
      <td class="col-id">${escapeHtml(r.candidate_id)}</td>
      <td class="col-score"><span class="scorebar" style="width:${w}px"></span><span class="score-val">${r.score.toFixed(4)}</span></td>
      <td class="reason">${formatReason(r.reasoning)}</td>
    </tr>`;
  }).join("");
  host.innerHTML = `<table>
    <thead><tr><th>#</th><th>Candidate</th><th>Score</th><th>Reasoning</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function toCsv(rows) {
  const esc = (v) => {
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const head = "candidate_id,rank,score,reasoning";
  const lines = rows.map((r) =>
    [r.candidate_id, r.rank, r.score.toFixed(4), r.reasoning].map(esc).join(","));
  return [head, ...lines].join("\n");
}

function download(filename, text) {
  const blob = new Blob([text], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Precomputed results tab ───────────────────────────────────────────────
let ALL_RESULTS = [];
let RESULTS_MAX = 1;

async function loadResults() {
  try {
    const res = await fetch("/api/results");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    ALL_RESULTS = data.results;
    RESULTS_MAX = Math.max(...ALL_RESULTS.map((r) => r.score), 0.0001);
    $("#results-stats").innerHTML =
      `<b>${data.count}</b> ranked from <b>${(data.source || "").toLowerCase().includes("100") ? "100,000" : data.count}</b> candidates · top score <b>${RESULTS_MAX.toFixed(4)}</b>`;
    renderTable($("#results-table"), ALL_RESULTS, RESULTS_MAX);
  } catch (e) {
    $("#results-table").innerHTML =
      `<p class="muted" style="padding:16px">Could not load results: ${escapeHtml(e.message)}</p>`;
  }
}

$("#filter").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const rows = q
    ? ALL_RESULTS.filter((r) =>
        (r.candidate_id + " " + r.reasoning).toLowerCase().includes(q))
    : ALL_RESULTS;
  renderTable($("#results-table"), rows, RESULTS_MAX);
});

$("#dl-results").addEventListener("click", () => {
  if (ALL_RESULTS.length) download("redrob_top100.csv", toCsv(ALL_RESULTS));
});

// ── Upload tab ────────────────────────────────────────────────────────────
const drop = $("#drop");
const fileInput = $("#file");
let UPLOAD_RESULTS = [];

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hot"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("hot");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  const status = $("#upload-status");
  status.className = "upload-status busy";
  status.innerHTML = `<span class="spinner"></span>Ranking <b>${escapeHtml(file.name)}</b> — extracting features, BM25, scoring…`;
  $("#upload-bar").classList.add("hidden");
  $("#upload-table").innerHTML = "";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/rank", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));

    UPLOAD_RESULTS = data.results;
    const max = Math.max(...UPLOAD_RESULTS.map((r) => r.score), 0.0001);
    status.className = "upload-status ok";
    status.textContent = `✓ Ranked ${data.candidates_processed.toLocaleString()} candidates → top ${data.count}.`;
    $("#upload-stats").innerHTML =
      `<b>${data.count}</b> shown from <b>${data.candidates_processed.toLocaleString()}</b> processed · top score <b>${max.toFixed(4)}</b>`;
    $("#upload-bar").classList.remove("hidden");
    renderTable($("#upload-table"), UPLOAD_RESULTS, max);
  } catch (e) {
    status.className = "upload-status err";
    status.textContent = "✗ " + e.message;
  }
}

$("#dl-upload").addEventListener("click", () => {
  if (UPLOAD_RESULTS.length) download("redrob_ranked.csv", toCsv(UPLOAD_RESULTS));
});

// ── Init ──────────────────────────────────────────────────────────────────
loadResults();
