import { fetchEngines, fillSelect, setActiveNav, setStatus, SseStream } from "/static/js/shared.js";

setActiveNav(window.location.pathname);

const statusPill = document.getElementById("statusPill");
const statusLine = document.getElementById("statusLine");
const thinkingLine = document.getElementById("thinkingLine");
const fenLabel = document.getElementById("fenLabel");
const copyFenBtn = document.getElementById("copyFenBtn");
const moveList = document.getElementById("moveList");

const engineSelect = document.getElementById("engineSelect");
const depthRange = document.getElementById("depthRange");
const depthLabel = document.getElementById("depthLabel");
const timeRange = document.getElementById("timeRange");
const timeLabel = document.getElementById("timeLabel");
const newGameBtn = document.getElementById("newGameBtn");
const resignBtn = document.getElementById("resignBtn");

const hceModulesWrap = document.getElementById("hceModulesWrap");
const hceModulesGrid = document.getElementById("hceModulesGrid");
const presetClassicBtn = document.getElementById("presetClassicBtn");
const presetAdvancedBtn = document.getElementById("presetAdvancedBtn");
const presetAllBtn = document.getElementById("presetAllBtn");

let engines = [];
let hceMeta = null;

let sessionId = null;
let stream = null;
const boardEl = document.getElementById("board");
let fen = "start";
let humanColor = "white";
let pendingPromotion = null; // {from,to}
let selectedSq = null; // algebraic like "e2"
let lastMove = null; // {from,to}

function parseUci(uci) {
  const s = String(uci || "");
  if (s.length < 4) return null;
  return { from: s.slice(0, 2), to: s.slice(2, 4), promotion: s.length >= 5 ? s[4] : undefined };
}

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

function fenTurn(fenStr) {
  const parts = String(fenStr || "").split(" ");
  return parts[1] === "b" ? "black" : "white";
}

function pieceGlyph(p) {
  // Unicode chess pieces
  const map = {
    P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
    p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  };
  return map[p] || "";
}

function squareName(fileIdx, rankIdxFromWhite) {
  // fileIdx 0=a..7=h, rankIdxFromWhite 0=1..7=8
  return String.fromCharCode("a".charCodeAt(0) + fileIdx) + String(rankIdxFromWhite + 1);
}

function renderBoard() {
  const b = fen === "start"
    ? fenToBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    : fenToBoard(fen);
  if (!b) {
    boardEl.innerHTML = `<div class="d-flex align-items-center justify-content-center h-100"><div class="small-muted">Invalid FEN</div></div>`;
    return;
  }
  const orientWhite = humanColor === "white";
  boardEl.innerHTML = "";

  // ranks to display: 8..1 if white, else 1..8
  for (let displayRank = 0; displayRank < 8; displayRank++) {
    const rankFromWhite = orientWhite ? (7 - displayRank) : displayRank; // 7=8th..0=1st
    const row = b[7 - rankFromWhite]; // b is [rank8..rank1]
    for (let displayFile = 0; displayFile < 8; displayFile++) {
      const fileIdx = orientWhite ? displayFile : (7 - displayFile);
      const sq = squareName(fileIdx, rankFromWhite);
      const piece = row[fileIdx];
      const isLight = (fileIdx + rankFromWhite) % 2 === 0;
      const div = document.createElement("div");
      div.className = `board-square ${isLight ? "light" : "dark"} selectable`;
      div.setAttribute("data-sq", sq);
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece === piece.toUpperCase() ? "white" : "black"}`;
        span.textContent = pieceGlyph(piece);
        div.appendChild(span);
      } else {
        div.textContent = "";
      }
      if (selectedSq === sq) div.classList.add("selected");
      if (lastMove && (lastMove.from === sq || lastMove.to === sq)) div.classList.add("lastmove");
      boardEl.appendChild(div);
    }
  }
}

function renderMoves() {
  // We don't compute SAN client-side without chess.js; keep it simple.
  moveList.innerHTML = `<div class="small-muted">Move list will populate in later polish (SAN). For now, watch the board.</div>`;
}

function updateFen() {
  fenLabel.textContent = fen === "start" ? "startpos" : fen;
}

function updateTurnStatus() {
  const turn = fenTurn(fen);
  const label = turn === "white" ? "White" : "Black";
  setStatus(statusPill, `${label} to move`, "accent");
  statusLine.textContent = `${label} to move.`;
}

function setThinking(text) {
  thinkingLine.textContent = text || "";
}

function selectedHumanColor() {
  return document.getElementById("hcBlack").checked ? "black" : "white";
}

function getSelectedHceModulesCsv() {
  const checks = hceModulesGrid.querySelectorAll("input[type=checkbox][data-mod]");
  const selected = [];
  checks.forEach((c) => { if (c.checked) selected.push(c.getAttribute("data-mod")); });
  return selected.join(",");
}

function applyPreset(groupName) {
  if (!hceMeta) return;
  const group = hceMeta.module_groups?.[groupName] ?? [];
  const set = new Set(group);
  const checks = hceModulesGrid.querySelectorAll("input[type=checkbox][data-mod]");
  checks.forEach((c) => { c.checked = set.has(c.getAttribute("data-mod")); });
}

function renderHcePicker(meta) {
  hceModulesGrid.innerHTML = "";
  const groups = meta.module_groups ?? {};
  const classic = groups.classic ?? [];
  const advanced = groups.advanced ?? [];

  function addGroup(title, items) {
    const head = document.createElement("div");
    head.className = "col-12 mt-2";
    head.innerHTML = `<div class="small-muted fw-semibold">${title}</div>`;
    hceModulesGrid.appendChild(head);

    for (const m of items) {
      const id = `hce_${m}`;
      const col = document.createElement("div");
      col.className = "col-6 col-md-4";
      col.innerHTML = `
        <div class="form-check">
          <input class="form-check-input" type="checkbox" id="${id}" data-mod="${m}" checked />
          <label class="form-check-label" for="${id}">${m}</label>
        </div>`;
      hceModulesGrid.appendChild(col);
    }
  }

  addGroup("Classic", classic);
  addGroup("Advanced", advanced);

  presetClassicBtn.onclick = () => applyPreset("classic");
  presetAdvancedBtn.onclick = () => applyPreset("advanced");
  presetAllBtn.onclick = () => {
    const checks = hceModulesGrid.querySelectorAll("input[type=checkbox][data-mod]");
    checks.forEach((c) => { c.checked = true; });
  };
}

async function startNewGame() {
  if (!engineSelect.value) return;
  humanColor = selectedHumanColor();
  const depth = Number(depthRange.value);
  const think_time = Number(timeRange.value);

  const body = {
    engine_id: engineSelect.value,
    human_color: humanColor,
    depth,
    think_time,
  };
  if (engineSelect.value === "hce" && hceMeta?.supports_modules) {
    body.hce_modules = getSelectedHceModulesCsv();
  }

  setStatus(statusPill, "Starting…", "accent");
  setThinking("");
  const res = await fetch("/api/hvse/new", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    setStatus(statusPill, "Error", "danger");
    statusLine.textContent = `Failed to start: ${res.status}`;
    return;
  }

  const data = await res.json();
  sessionId = data.session_id;
  fen = data.fen;
  selectedSq = null;
  lastMove = null;
  renderBoard();

  resignBtn.disabled = false;
  renderMoves();
  updateFen();
  updateTurnStatus();

  if (stream) stream.close();
  stream = new SseStream().connect(`/api/hvse/${sessionId}/stream`);
  stream
    .on("thinking", (evt) => {
      try {
        const t = JSON.parse(evt.data);
        if (t.started_at_ms) setThinking("Engine thinking…");
        if (t.finished_in_ms) setThinking(`Engine moved (thinking ${t.finished_in_ms}ms).`);
      } catch {
        setThinking("Engine thinking…");
      }
    })
    .on("engine_move", (evt) => {
      const payload = JSON.parse(evt.data);
      const m = parseUci(payload.uci);
      if (m) lastMove = { from: m.from, to: m.to };
      fen = payload.fen ?? fen;
      renderMoves();
      updateFen();
      updateTurnStatus();
      renderBoard();
      setThinking("");
    })
    .on("game_over", (evt) => {
      const payload = JSON.parse(evt.data);
      setStatus(statusPill, "Game over", "success");
      statusLine.textContent = `Game over. Result: ${payload.result ?? "*"}`;
      setThinking("");
    })
    .on("error", () => {
      setStatus(statusPill, "Disconnected", "danger");
    });
}

function isHumanTurn() {
  return fenTurn(fen) === humanColor;
}

async function submitHumanMove(uci) {
  setThinking("Sending move…");
  const res = await fetch(`/api/hvse/${sessionId}/move`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ uci }),
  });
  const payload = await res.json();
  if (!payload.ok) {
    setThinking("");
    setStatus(statusPill, "Illegal move", "danger");
    statusLine.textContent = "Illegal move.";
    return false;
  }
  setStatus(statusPill, "Engine thinking…", "accent");
  statusLine.textContent = "Engine thinking…";
  setThinking("Waiting for engine…");
  return true;
}

function isPromotionMove(from, to) {
  // Lightweight: if a pawn is moving to last rank, prompt.
  // We don't know piece types client-side without a full rules engine; so only
  // prompt when the move reaches rank 1/8 from the mover's POV.
  const targetRank = to[1];
  return targetRank === "1" || targetRank === "8";
}

function showPromotionModal() {
  const modalEl = document.getElementById("promoModal");
  const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
  modal.show();

  modalEl.querySelectorAll("[data-promo]").forEach((btn) => {
    btn.onclick = () => {
      const p = btn.getAttribute("data-promo");
      modal.hide();
      const { from, to } = pendingPromotion;
      pendingPromotion = null;
      lastMove = { from, to };
      submitHumanMove(`${from}${to}${p}`);
    };
  });
}

async function resign() {
  if (!sessionId) return;
  await fetch(`/api/hvse/${sessionId}`, { method: "DELETE" });
  if (stream) stream.close();
  sessionId = null;
  resignBtn.disabled = true;
  setStatus(statusPill, "Resigned", "danger");
  statusLine.textContent = "Resigned.";
  setThinking("");
}

async function init() {
  depthRange.oninput = () => { depthLabel.textContent = `(${depthRange.value})`; };
  timeRange.oninput = () => { timeLabel.textContent = `(${Number(timeRange.value).toFixed(1)}s)`; };
  depthRange.oninput();
  timeRange.oninput();

  newGameBtn.onclick = () => startNewGame();
  resignBtn.onclick = () => resign();

  copyFenBtn.onclick = async () => {
    await navigator.clipboard.writeText(fen === "start" ? "startpos" : fen);
    setStatus(statusPill, "FEN copied", "success");
    setTimeout(() => updateTurnStatus(), 700);
  };

  engines = await fetchEngines();
  fillSelect(engineSelect, engines, { placeholder: "Choose an engine…" });

  hceMeta = engines.find((e) => e.id === "hce") ?? null;

  engineSelect.onchange = () => {
    const e = engines.find((x) => x.id === engineSelect.value);
    const supports = !!e?.supports_modules;
    hceModulesWrap.style.display = supports ? "" : "none";
    if (supports) renderHcePicker(e);
  };
  engineSelect.value = "material";
  engineSelect.onchange();

  // Board click handler (no external libs)
  boardEl.addEventListener("click", (e) => {
    const node = e.target?.closest?.("[data-sq]");
    const sq = node?.getAttribute?.("data-sq");
    if (!sq) return;
    if (!sessionId) {
      setStatus(statusPill, "Start a new game", "accent");
      statusLine.textContent = "Click “New game” first.";
      return;
    }
    if (!isHumanTurn()) return;

    if (!selectedSq) {
      selectedSq = sq;
      renderBoard();
      return;
    }

    const from = selectedSq;
    const to = sq;
    selectedSq = null;
    renderBoard();

    if (from === to) return;

    // Promotion modal when reaching back rank.
    if (isPromotionMove(from, to)) {
      pendingPromotion = { from, to };
      showPromotionModal();
      return;
    }

    lastMove = { from, to };
    submitHumanMove(`${from}${to}`);
  });

  renderMoves();
  updateFen();
  updateTurnStatus();
  renderBoard();
}

init().catch((e) => {
  setStatus(statusPill, "Error", "danger");
  statusLine.textContent = String(e);
});

