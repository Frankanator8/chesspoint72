import { fetchEngines, fillSelect, getQueryParams, setActiveNav, setStatus, SseStream } from "/static/js/shared.js";

setActiveNav(window.location.pathname);

const statusPill = document.getElementById("statusPill");
const boardEl = document.getElementById("board");
const fenLabel = document.getElementById("fenLabel");
const copyFenBtn = document.getElementById("copyFenBtn");
const moveList = document.getElementById("moveList");
const scoreLine = document.getElementById("scoreLine");

const engine1Select = document.getElementById("engine1Select");
const engine2Select = document.getElementById("engine2Select");
const depthRange = document.getElementById("depthRange");
const depthLabel = document.getElementById("depthLabel");
const delayRange = document.getElementById("delayRange");
const delayLabel = document.getElementById("delayLabel");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let engines = [];
let sessionId = null;
let stream = null;
let fen = "start";
let pending = []; // buffered events
let replayTimer = null;

let wdl = { w: 0, d: 0, l: 0 };
let ply = 0;

function fenToBoard(fenStr) {
  const placement = String(fenStr || "").split(" ")[0] || "";
  const ranks = placement.split("/");
  if (ranks.length !== 8) return null;
  const board = [];
  for (const r of ranks) {
    const row = [];
    for (const ch of r) {
      if (ch >= "1" && ch <= "8") {
        for (let i = 0; i < Number(ch); i++) row.push(null);
      } else {
        row.push(ch);
      }
    }
    if (row.length !== 8) return null;
    board.push(row);
  }
  return board; // [rank8..rank1][file a..h]
}

function pieceGlyph(p) {
  const map = {
    P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
    p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  };
  return map[p] || "";
}

function squareName(fileIdx, rankIdxFromWhite) {
  return String.fromCharCode("a".charCodeAt(0) + fileIdx) + String(rankIdxFromWhite + 1);
}

function renderBoard() {
  const b = fen === "start"
    ? fenToBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    : fenToBoard(fen);
  if (!b) return;
  boardEl.innerHTML = "";
  for (let rankFromWhite = 7; rankFromWhite >= 0; rankFromWhite--) {
    const row = b[7 - rankFromWhite];
    for (let fileIdx = 0; fileIdx < 8; fileIdx++) {
      const sq = squareName(fileIdx, rankFromWhite);
      const piece = row[fileIdx];
      const isLight = (fileIdx + rankFromWhite) % 2 === 0;
      const div = document.createElement("div");
      div.className = `board-square ${isLight ? "light" : "dark"}`;
      div.setAttribute("data-sq", sq);
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece === piece.toUpperCase() ? "white" : "black"}`;
        span.textContent = pieceGlyph(piece);
        div.appendChild(span);
      }
      boardEl.appendChild(div);
    }
  }
}

function pushMoveText(uci) {
  if (!uci) return;
  ply += 1;
  const rows = moveList.innerHTML ? [moveList.innerHTML] : [];
  rows.push(`<div class="small-muted">${ply}. <span class="text-light">${uci}</span></div>`);
  moveList.innerHTML = rows.join("");
  moveList.scrollTop = moveList.scrollHeight;
}

function scheduleReplay() {
  if (replayTimer) return;
  const delay = Number(delayRange.value);
  replayTimer = setTimeout(() => {
    replayTimer = null;
    const evt = pending.shift();
    if (evt) applyEvent(evt);
    if (pending.length) scheduleReplay();
  }, delay);
}

function applyEvent(evt) {
  if (evt.type === "move") {
    if (evt.fen) fen = evt.fen;
    fenLabel.textContent = fen;
    copyFenBtn.disabled = false;
    renderBoard();
    pushMoveText(evt.uci);
    setStatus(statusPill, "Running", "accent");
  } else if (evt.type === "game_over") {
    const r = evt.result || "*";
    if (r === "1-0") wdl.w += 1;
    else if (r === "0-1") wdl.l += 1;
    else if (r === "1/2-1/2") wdl.d += 1;
    scoreLine.textContent = `W ${wdl.w} · D ${wdl.d} · L ${wdl.l}`;
    setStatus(statusPill, `Game over (${r})`, "success");
    stopBtn.disabled = true;
    startBtn.disabled = false;
  }
}

async function startMatch() {
  const engine1_id = engine1Select.value;
  const engine2_id = engine2Select.value;
  const depth = Number(depthRange.value);
  if (!engine1_id || !engine2_id) return;

  setStatus(statusPill, "Starting…", "accent");
  startBtn.disabled = true;
  stopBtn.disabled = false;
  moveList.innerHTML = "";
  pending = [];
  ply = 0;

  const res = await fetch("/api/eve/new", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ engine1_id, engine2_id, depth }),
  });
  if (!res.ok) {
    setStatus(statusPill, "Error", "danger");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  fen = data.fen;
  fenLabel.textContent = fen;
  renderBoard();

  if (stream) stream.close();
  stream = new SseStream().connect(`/api/eve/${sessionId}/stream`);
  stream
    .on("move", (e) => {
      const payload = JSON.parse(e.data);
      pending.push({ type: "move", fen: payload.fen, uci: payload.uci });
      scheduleReplay();
    })
    .on("game_over", (e) => {
      const payload = JSON.parse(e.data);
      pending.push({ type: "game_over", result: payload.result });
      scheduleReplay();
    })
    .on("error", () => setStatus(statusPill, "Disconnected", "danger"));
}

async function stopMatch() {
  if (!sessionId) return;
  await fetch(`/api/eve/${sessionId}/stop`, { method: "POST" });
  setStatus(statusPill, "Stopping…", "accent");
  stopBtn.disabled = true;
}

async function init() {
  depthRange.oninput = () => (depthLabel.textContent = `(${depthRange.value})`);
  delayRange.oninput = () => (delayLabel.textContent = `(${delayRange.value}ms)`);
  depthRange.oninput();
  delayRange.oninput();

  engines = await fetchEngines();
  fillSelect(engine1Select, engines, { placeholder: "Choose white engine…" });
  fillSelect(engine2Select, engines, { placeholder: "Choose black engine…" });

  const qp = getQueryParams();
  if (qp.engine1) engine1Select.value = qp.engine1;
  if (qp.engine2) engine2Select.value = qp.engine2;

  // defaults if not prefilled
  if (!engine1Select.value) engine1Select.value = engines[0]?.id ?? "";
  if (!engine2Select.value) engine2Select.value = engines[0]?.id ?? "";

  startBtn.onclick = () => startMatch();
  stopBtn.onclick = () => stopMatch();

  copyFenBtn.onclick = async () => {
    if (!fen) return;
    await navigator.clipboard.writeText(fen);
    setStatus(statusPill, "FEN copied", "success");
    setTimeout(() => setStatus(statusPill, "Ready", "accent"), 600);
  };

  fenLabel.textContent = "";
  renderBoard();
  scoreLine.textContent = `W ${wdl.w} · D ${wdl.d} · L ${wdl.l}`;
}

init().catch((e) => {
  setStatus(statusPill, "Error", "danger");
  console.error(e);
});

