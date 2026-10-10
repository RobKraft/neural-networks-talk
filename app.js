// Perceptron Playground
// The same training rule as console/train-all-digits.js, wired up to a UI.

// Segment order: [a b c d e f g]
const SEGMENTS = ["a", "b", "c", "d", "e", "f", "g"];

const DIGITS = {
  0: [1, 1, 1, 1, 1, 1, 0],
  1: [0, 1, 1, 0, 0, 0, 0],
  2: [1, 1, 0, 1, 1, 0, 1],
  3: [1, 1, 1, 1, 0, 0, 1],
  4: [0, 1, 1, 0, 0, 1, 1],
  5: [1, 0, 1, 1, 0, 1, 1],
  6: [1, 0, 1, 1, 1, 1, 1],
  7: [1, 1, 1, 0, 0, 0, 0],
  8: [1, 1, 1, 1, 1, 1, 1],
  9: [1, 1, 1, 1, 0, 1, 1]
};

const DIGIT_SETS = {
  three: [3, 8, 9],
  ten: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
};

// The warm-up example's training order, from console/train-3-8-9.js
const TALK_SEQUENCE = [3, 8, 9, 3, 9, 8, 3, 8, 9];

const MAX_STEPS = 100000; // Safety cap, same as the console version
const MAX_LOG_ENTRIES = 400;

// Segment shapes for a 120x200 viewBox
function hSeg(y, x1, x2) {
  return `${x1},${y} ${x1 + 8},${y - 8} ${x2 - 8},${y - 8} ${x2},${y} ${x2 - 8},${y + 8} ${x1 + 8},${y + 8}`;
}
function vSeg(x, y1, y2) {
  return `${x},${y1} ${x + 8},${y1 + 8} ${x + 8},${y2 - 8} ${x},${y2} ${x - 8},${y2 - 8} ${x - 8},${y1 + 8}`;
}
const SEGMENT_POINTS = [
  hSeg(12, 22, 98),   // a
  vSeg(106, 16, 96),  // b
  vSeg(106, 104, 184),// c
  hSeg(188, 22, 98),  // d
  vSeg(14, 104, 184), // e
  vSeg(14, 16, 96),   // f
  hSeg(100, 22, 98)   // g
];

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// The perceptron
// ---------------------------------------------------------------------------

function dot(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function shuffle(list) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

let model;

function createModel() {
  const setName = $("#opt-set").value;
  const active = DIGIT_SETS[setName];
  const weights = {};
  for (const d of active) {
    weights[d] = $("#opt-init").value === "random"
      ? Array.from({ length: 7 }, () => Math.floor(Math.random() * 5) - 2)
      : Array(7).fill(0);
  }
  return {
    setName,
    active,
    order: $("#opt-order").value,
    weights,
    step: 0,
    epoch: 0,
    updates: 0,
    queue: [],
    converged: false,
    convergedAt: null
  };
}

function scoresFor(x) {
  const scores = {};
  for (const d of model.active) scores[d] = dot(model.weights[d], x);
  return scores;
}

// Highest score wins; ties go to the lowest digit (same as the console version)
function predict(x) {
  let bestDigit = model.active[0];
  let bestScore = -Infinity;
  for (const d of model.active) {
    const score = dot(model.weights[d], x);
    if (score > bestScore || (score === bestScore && d < bestDigit)) {
      bestScore = score;
      bestDigit = d;
    }
  }
  return bestDigit;
}

function correctCount() {
  return model.active.filter((d) => predict(DIGITS[d]) === d).length;
}

function nextEpochOrder() {
  if (model.order === "shuffle") return shuffle(model.active);
  return model.setName === "three" ? [...TALK_SEQUENCE] : [...model.active];
}

// One training step. Returns a record of what happened.
function trainStep() {
  if (model.queue.length === 0) {
    model.epoch++;
    model.queue = nextEpochOrder();
  }
  const trainDigit = model.queue.shift();
  const x = DIGITS[trainDigit];
  const scores = scoresFor(x);
  const predicted = predict(x);
  const wrong = predicted !== trainDigit;
  model.step++;

  if (wrong) {
    // Correct digit += x, wrong prediction -= x
    model.weights[trainDigit] = model.weights[trainDigit].map((w, i) => w + x[i]);
    model.weights[predicted] = model.weights[predicted].map((w, i) => w - x[i]);
    model.updates++;
  }

  const record = { step: model.step, epoch: model.epoch, trainDigit, predicted, wrong, scores, x };
  if (!model.converged && correctCount() === model.active.length) {
    model.converged = true;
    model.convergedAt = model.step;
    record.justConverged = true;
  }
  return record;
}

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

let input = [...DIGITS[8]];
let inputSource = { kind: "user" };
let lastRecord = null;
let playTimer = null;

function speedDelay() {
  const v = Number($("#opt-speed").value) / 100;
  return Math.round(1500 * Math.pow(0.02, v)); // 1500ms .. 30ms
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, ns) {
  const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function drawSeven(svg, pattern, interactive) {
  svg.textContent = "";
  SEGMENT_POINTS.forEach((points, i) => {
    const p = el("polygon", { points }, SVG_NS);
    if (pattern[i]) p.classList.add("on");
    if (interactive) {
      p.setAttribute("tabindex", "0");
      p.setAttribute("role", "switch");
      p.setAttribute("aria-label", `Segment ${SEGMENTS[i]}`);
      p.setAttribute("aria-checked", pattern[i] ? "true" : "false");
      p.dataset.index = i;
    }
    svg.appendChild(p);
  });
}

function updateDisplay() {
  $("#display").querySelectorAll("polygon").forEach((p, i) => {
    p.classList.toggle("on", !!input[i]);
    p.setAttribute("aria-checked", input[i] ? "true" : "false");
  });
}

function matchingDigit(x) {
  for (const d of Object.keys(DIGITS)) {
    if (DIGITS[d].every((v, i) => v === x[i])) return Number(d);
  }
  return null;
}

function renderPrediction() {
  const scores = scoresFor(input);
  const predicted = predict(input);
  const values = Object.values(scores);
  const allTied = values.every((v) => v === values[0]);

  $("#pred-digit").textContent = predicted;

  const note = $("#pred-note");
  note.className = "pred-note";
  const actual = matchingDigit(input);
  if (allTied) {
    note.textContent = "Every score ties, so it just picks the lowest digit. It hasn't learned anything yet.";
  } else if (actual === null) {
    note.textContent = "That's not a digit, but it has to pick one anyway.";
  } else if (!model.active.includes(actual)) {
    note.textContent = `That's a ${actual}, but this network was never taught about ${actual}s.`;
  } else if (actual === predicted) {
    note.textContent = "Correct ✓";
    note.classList.add("good");
  } else {
    note.textContent = `Wrong ✗ (it's a ${actual})`;
    note.classList.add("bad");
  }

  const src = $("#input-source");
  src.textContent = inputSource.kind === "train" ? `Training on ${inputSource.digit}` : "Your input";
  src.className = "pill" + (inputSource.kind === "train" ? " busy" : "");

  // Score bars
  const maxAbs = Math.max(1, ...values.map(Math.abs));
  const box = $("#scores");
  box.textContent = "";
  for (const d of model.active) {
    const s = scores[d];
    const row = el("div", { class: "score-row" + (d === predicted ? " best" : "") });
    const pct = (Math.abs(s) / maxAbs) * 50;
    const left = s >= 0 ? 50 : 50 - pct;
    row.innerHTML =
      `<span class="d">${d}</span>` +
      `<span class="bar"><span class="${s >= 0 ? "pos" : "neg"}" style="left:${left}%;width:${pct}%"></span></span>` +
      `<span class="v">${s}</span>`;
    box.appendChild(row);
  }

  return predicted;
}

function maxAbsWeight() {
  let m = 1;
  for (const d of model.active) for (const w of model.weights[d]) m = Math.max(m, Math.abs(w));
  return m;
}

function renderNetwork(predicted) {
  const svg = $("#network");
  svg.textContent = "";
  const outputs = model.active;
  const rowH = 34;
  const rows = Math.max(SEGMENTS.length, outputs.length);
  const height = rows * rowH + 16;
  const width = 400;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const inX = 40, outX = 310;
  const yFor = (i, n) => 8 + (rows - n) * rowH / 2 + i * rowH + rowH / 2;
  const maxW = maxAbsWeight();
  const scores = scoresFor(input);

  const edges = el("g", {}, SVG_NS);
  outputs.forEach((d, oi) => {
    SEGMENTS.forEach((_, ii) => {
      const w = model.weights[d][ii];
      const line = el("line", {
        class: "edge",
        x1: inX, y1: yFor(ii, SEGMENTS.length),
        x2: outX, y2: yFor(oi, outputs.length)
      }, SVG_NS);
      if (w === 0) {
        line.setAttribute("style", "stroke:var(--border)");
        line.setAttribute("stroke-width", 1);
        line.setAttribute("opacity", input[ii] ? 0.8 : 0.35);
      } else {
        line.setAttribute("style", `stroke:var(${w > 0 ? "--pos" : "--neg"})`);
        line.setAttribute("stroke-width", (0.75 + (Math.abs(w) / maxW) * 5).toFixed(2));
        line.setAttribute("opacity", input[ii] ? 0.9 : 0.15);
      }
      edges.appendChild(line);
    });
  });
  svg.appendChild(edges);

  SEGMENTS.forEach((name, i) => {
    const y = yFor(i, SEGMENTS.length);
    svg.appendChild(el("circle", { class: "node" + (input[i] ? " on" : ""), cx: inX, cy: y, r: 13 }, SVG_NS));
    const t = el("text", { x: inX, y: y + 4.5, "text-anchor": "middle" }, SVG_NS);
    t.textContent = name;
    svg.appendChild(t);
  });

  outputs.forEach((d, i) => {
    const y = yFor(i, outputs.length);
    svg.appendChild(el("circle", { class: "node" + (d === predicted ? " best" : ""), cx: outX, cy: y, r: 13 }, SVG_NS));
    const t = el("text", { x: outX, y: y + 4.5, "text-anchor": "middle" }, SVG_NS);
    t.textContent = d;
    svg.appendChild(t);
    const s = el("text", { class: "score-label", x: outX + 22, y: y + 4.5 }, SVG_NS);
    s.textContent = `score ${scores[d]}`;
    svg.appendChild(s);
  });
}

function renderWeights(changedDigits, predicted) {
  const table = $("#weights");
  table.textContent = "";
  const head = el("tr");
  head.appendChild(el("th"));
  for (const s of SEGMENTS) {
    const th = el("th", { scope: "col" });
    th.textContent = s;
    head.appendChild(th);
  }
  table.appendChild(head);

  const maxW = maxAbsWeight();
  for (const d of model.active) {
    const tr = el("tr", { class: d === predicted ? "best" : "" });
    const th = el("th", { class: "row", scope: "row" });
    th.textContent = `w${d}`;
    tr.appendChild(th);
    model.weights[d].forEach((w, i) => {
      const td = el("td");
      td.textContent = w;
      if (w !== 0) {
        const pct = Math.round(15 + (Math.abs(w) / maxW) * 55);
        td.style.background = `color-mix(in srgb, var(${w > 0 ? "--pos" : "--neg"}) ${pct}%, var(--surface-2))`;
      }
      if (changedDigits.includes(d) && lastRecord && lastRecord.x[i]) td.classList.add("changed");
      tr.appendChild(td);
    });
    table.appendChild(tr);
  }
}

function renderStats() {
  $("#stat-step").textContent = model.step;
  $("#stat-epoch").textContent = model.epoch;
  $("#stat-acc").textContent = `${correctCount()}/${model.active.length}`;
  $("#stat-updates").textContent = model.updates;

  const status = $("#status");
  if (model.converged) {
    status.textContent = `Learned ✓ in ${model.convergedAt} steps`;
    status.className = "pill good";
  } else if (playTimer) {
    status.textContent = "Training…";
    status.className = "pill busy";
  } else if (model.step === 0) {
    status.textContent = "Untrained";
    status.className = "pill";
  } else {
    status.textContent = "Still learning";
    status.className = "pill";
  }

  $("#btn-play").textContent = playTimer ? "Pause" : "Play";
  document.querySelectorAll(".digit-btn").forEach((b) => {
    b.classList.toggle("inactive", !model.active.includes(Number(b.dataset.digit)));
  });
}

function renderExplain() {
  const box = $("#explain");
  if (!lastRecord) return;
  const r = lastRecord;
  const scoreText = model.active.map((d) => `${d}=${r.scores[d]}`).join(", ");
  box.className = "explain " + (r.wrong ? "bad" : "good");
  let html = `<div class="head">Step ${r.step} · Epoch ${r.epoch}: shown a ${r.trainDigit}</div>`;
  html += `<div>Scores: <code>${scoreText}</code></div>`;
  if (r.wrong) {
    html += `<div>It guessed <strong>${r.predicted}</strong> ✗. So it <strong>added</strong> the ${r.trainDigit}'s lit segments to <code>w${r.trainDigit}</code> and <strong>subtracted</strong> them from <code>w${r.predicted}</code>.</div>`;
  } else {
    html += `<div>It guessed <strong>${r.predicted}</strong> ✓. Correct, so no weights change.</div>`;
  }
  if (r.justConverged) {
    html += `<div class="head" style="margin-top:6px">All ${model.active.length} digits are now correct. Training is done.</div>`;
  }
  box.innerHTML = html;
}

function appendLog(r) {
  const log = $("#log");
  log.querySelector(".empty")?.remove();
  const entry = el("div", { class: "entry" });
  const scoreText = model.active.map((d) => `${d}=${r.scores[d]}`).join(", ");
  let text = `Step ${r.step} (Epoch ${r.epoch}): train on ${r.trainDigit}\n  Scores: ${scoreText}\n  Predicted: ${r.predicted} `;
  entry.textContent = text;
  const verdict = el("span", { class: r.wrong ? "bad" : "good" });
  verdict.textContent = r.wrong ? "(wrong, updating)" : "(correct)";
  entry.appendChild(verdict);
  if (r.wrong) {
    entry.appendChild(document.createTextNode(
      `\n    w${r.trainDigit}: [${model.weights[r.trainDigit].join(", ")}]` +
      `\n    w${r.predicted}: [${model.weights[r.predicted].join(", ")}]`
    ));
  }
  if (r.justConverged) {
    entry.appendChild(document.createTextNode(`\n=== Converged in ${r.step} steps across ${r.epoch} epochs ===`));
  }
  log.appendChild(entry);
  while (log.children.length > MAX_LOG_ENTRIES) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function clearLog() {
  $("#log").innerHTML = '<div class="empty">Training steps will appear here, just like the console version\'s output.</div>';
}

function renderAll(changedDigits = []) {
  updateDisplay();
  const predicted = renderPrediction();
  renderNetwork(predicted);
  renderWeights(changedDigits, predicted);
  renderStats();
  renderExplain();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function doStep() {
  if (model.step >= MAX_STEPS) return stopPlay();
  const r = trainStep();
  lastRecord = r;
  input = [...r.x];
  inputSource = { kind: "train", digit: r.trainDigit };
  appendLog(r);
  if (r.justConverged) stopPlay();
  renderAll(r.wrong ? [r.trainDigit, r.predicted] : []);
}

function startPlay() {
  if (playTimer) return;
  if (model.converged) resetModel();
  const tick = () => {
    doStep();
    if (playTimer) playTimer = setTimeout(tick, speedDelay());
  };
  playTimer = setTimeout(tick, 0);
  renderStats();
}

function stopPlay() {
  if (playTimer) clearTimeout(playTimer);
  playTimer = null;
  renderStats();
}

function togglePlay() {
  playTimer ? stopPlay() : startPlay();
}

function finish() {
  stopPlay();
  let r = null;
  while (!model.converged && model.step < MAX_STEPS) {
    r = trainStep();
    appendLog(r);
  }
  if (!r) return;
  lastRecord = r;
  input = [...r.x];
  inputSource = { kind: "train", digit: r.trainDigit };
  renderAll();
}

function resetModel() {
  if (playTimer) clearTimeout(playTimer);
  playTimer = null;
  model = createModel();
  lastRecord = null;
  clearLog();
  $("#explain").className = "explain";
  $("#explain").innerHTML =
    "Press <strong>Step</strong>. The network sees a digit, scores every possible answer, " +
    "and picks the highest. If it's wrong, it <em>adds</em> the input to the right answer's " +
    "weights and <em>subtracts</em> it from the wrong guess's weights.";
  renderAll();
}

function setUserInput(pattern) {
  input = [...pattern];
  inputSource = { kind: "user" };
  renderAll();
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function init() {
  drawSeven($("#display"), input, true);

  const buttons = $("#digit-buttons");
  for (let d = 0; d <= 9; d++) {
    const b = el("button", { class: "digit-btn", "data-digit": d, "aria-label": `Show digit ${d}`, title: `Show ${d}` });
    const svg = el("svg", { class: "seven", viewBox: "0 0 120 200", "aria-hidden": "true" }, SVG_NS);
    drawSeven(svg, DIGITS[d], false);
    b.appendChild(svg);
    b.addEventListener("click", () => setUserInput(DIGITS[d]));
    buttons.appendChild(b);
  }

  const toggleSegment = (target) => {
    if (!target || target.dataset.index === undefined) return;
    const i = Number(target.dataset.index);
    input[i] = input[i] ? 0 : 1;
    inputSource = { kind: "user" };
    renderAll();
  };
  $("#display").addEventListener("click", (e) => toggleSegment(e.target.closest("polygon")));
  $("#display").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleSegment(e.target.closest("polygon"));
    }
  });

  $("#btn-step").addEventListener("click", () => { stopPlay(); doStep(); });
  $("#btn-play").addEventListener("click", togglePlay);
  $("#btn-finish").addEventListener("click", finish);
  $("#btn-reset").addEventListener("click", resetModel);
  $("#btn-clear-log").addEventListener("click", clearLog);
  for (const id of ["#opt-set", "#opt-order", "#opt-init"]) {
    $(id).addEventListener("change", resetModel);
  }

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest("button, select, input, textarea, polygon")) return;
    if (e.key === "s" || e.key === "S" || e.key === "ArrowRight") { stopPlay(); doStep(); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
    else if (e.key === "r" || e.key === "R") { resetModel(); e.preventDefault(); }
  });

  resetModel();
}

init();
