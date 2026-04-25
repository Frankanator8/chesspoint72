import { fetchEngines, setActiveNav, setStatus, SseStream } from "/static/js/shared.js";

setActiveNav(window.location.pathname);

const MODULE_DESCRIPTIONS = {
  material:    "Raw piece values",
  pst:         "Piece-square tables",
  pawns:       "Pawn structure (doubled/isolated/passed)",
  king_safety: "King attack exposure",
  mobility:    "Piece mobility bonuses",
  rooks:       "Rooks on 7th / open files",
  bishops:     "Bishop pair bonus",
  ewpm:        "Early-game pawn mobility",
  srcm:        "Short-range coordination",
  idam:        "Initiative & development",
  otvm:        "Open & semi-open file value",
  lmdm:        "Late middlegame dynamics",
  lscm:        "Long-range square control",
  clcm:        "Connected pawn chains",
  desm:        "Danger & escape squares",
};

// ── DOM refs ──────────────────────────────────────────────────
const statusPill    = document.getElementById("statusPill");
const statusLine    = document.getElementById("statusLine");
const thinkingLine  = document.getElementById("thinkingLine");
const fenLabel      = document.getElementById("fenLabel");
const copyFenBtn    = document.getElementById("copyFenBtn");
const moveList      = document.getElementById("moveList");
const boardEl       = document.getElementById("board");
const boardOverlay  = document.getElementById("boardOverlay");

const paletteClassic  = document.getElementById("modulePaletteClassic");
const paletteAdvanced = document.getElementById("modulePaletteAdvanced");
const dropZone        = document.getElementById("engineDropZone");

const presetClassicBtn  = document.getElementById("presetClassicBtn");
const presetAdvancedBtn = document.getElementById("presetAdvancedBtn");
const presetAllBtn      = document.getElementById("presetAllBtn");
const presetClearBtn    = document.getElementById("presetClearBtn");

const depthRange  = document.getElementById("depthRange");
const depthLabel  = document.getElementById("depthLabel");
const timeRange   = document.getElementById("timeRange");
const timeLabel   = document.getElementById("timeLabel");
const startBtn    = document.getElementById("startBtn");
const resignBtn   = document.getElementById("resignBtn");

// ── Engine builder state ──────────────────────────────────────
let allModules = { classic: [], advanced: [] };
const selectedModules = new Set();

// ── Game state ────────────────────────────────────────────────
let sessionId = null;
let stream = null;
let fen = "start";
let humanColor = "white";
let pendingPromotion = null;
let selectedSq = null;
let lastMove = null;

// ── Board helpers (ported from hvse.js) ──────────────────────

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
  return board;
}

function fenTurn(fenStr) {
  const parts = String(fenStr || "").split(" ");
  return parts[1] === "b" ? "black" : "white";
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
  if (!b) {
    boardEl.innerHTML = `<div class="d-flex align-items-center justify-content-center h-100"><div class="small-muted">Invalid FEN</div></div>`;
    return;
  }
  const orientWhite = humanColor === "white";
  boardEl.innerHTML = "";
  for (let displayRank = 0; displayRank < 8; displayRank++) {
    const rankFromWhite = orientWhite ? (7 - displayRank) : displayRank;
    const row = b[7 - rankFromWhite];
    for (let displayFile = 0; displayFile < 8; displayFile++) {
      const fileIdx = orientWhite ? displayFile : (7 - displayFile);
      const sq = squareName(fileIdx, rankFromWhite);
      const piece = row[fileIdx];
      const isLight = (fileIdx + rankFromWhite) % 2 === 0;
      const div = document.createElement("div");
      div.className = `board-square ${isLight ? "light" : "dark"}${sessionId ? " selectable" : ""}`;
      div.setAttribute("data-sq", sq);
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece === piece.toUpperCase() ? "white" : "black"}`;
        span.textContent = pieceGlyph(piece);
        div.appendChild(span);
      }
      if (selectedSq === sq) div.classList.add("selected");
      if (lastMove && (lastMove.from === sq || lastMove.to === sq)) div.classList.add("lastmove");
      boardEl.appendChild(div);
    }
  }
}

function updateFen() {
  fenLabel.textContent = fen === "start" ? "startpos" : fen;
}

function updateTurnStatus() {
  if (!sessionId) return;
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

function isHumanTurn() {
  return fenTurn(fen) === humanColor;
}

function isPromotionMove(from, to) {
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

// ── Module builder ────────────────────────────────────────────

function renderPalette() {
  function buildChips(container, mods, labelText) {
    container.innerHTML = `<span class="small-muted me-1" style="font-size:0.78rem;align-self:center">${labelText}</span>`;
    for (const mod of mods) {
      const chip = document.createElement("span");
      chip.className = "module-chip";
      chip.textContent = mod;
      chip.title = MODULE_DESCRIPTIONS[mod] ?? mod;
      chip.draggable = true;
      chip.setAttribute("data-mod", mod);

      chip.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", mod);
        e.dataTransfer.effectAllowed = "copy";
        chip.classList.add("dragging");
      });
      chip.addEventListener("dragend", () => chip.classList.remove("dragging"));

      container.appendChild(chip);
    }
  }

  buildChips(paletteClassic, allModules.classic, "Classic");
  buildChips(paletteAdvanced, allModules.advanced, "Advanced");
}

function renderDropZone() {
  dropZone.innerHTML = "";
  if (selectedModules.size === 0) {
    dropZone.classList.add("empty");
    startBtn.disabled = true;
    return;
  }
  dropZone.classList.remove("empty");
  startBtn.disabled = !!sessionId;

  for (const mod of selectedModules) {
    const chip = document.createElement("span");
    chip.className = "drop-chip";
    chip.textContent = mod;
    chip.title = MODULE_DESCRIPTIONS[mod] ?? mod;

    const remove = document.createElement("button");
    remove.className = "drop-chip-remove";
    remove.textContent = "×";
    remove.title = `Remove ${mod}`;
    remove.addEventListener("click", () => {
      selectedModules.delete(mod);
      renderDropZone();
    });

    chip.appendChild(remove);
    dropZone.appendChild(chip);
  }
}

function applyPreset(mods) {
  selectedModules.clear();
  for (const m of mods) selectedModules.add(m);
  renderDropZone();
}

function setupDropZone() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const mod = e.dataTransfer.getData("text/plain");
    if (mod && (allModules.classic.includes(mod) || allModules.advanced.includes(mod))) {
      selectedModules.add(mod);
      renderDropZone();
    }
  });
}

// ── Game lifecycle ────────────────────────────────────────────

async function startGame() {
  if (selectedModules.size === 0) return;
  humanColor = selectedHumanColor();
  const depth = Number(depthRange.value);
  const think_time = Number(timeRange.value);
  const hce_modules = [...selectedModules].join(",");

  setStatus(statusPill, "Starting…", "accent");
  setThinking("");

  const res = await fetch("/api/hvse/new", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ engine_id: "hce", human_color: humanColor, depth, think_time, hce_modules }),
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

  boardOverlay.style.display = "none";
  startBtn.disabled = true;
  resignBtn.disabled = false;

  renderBoard();
  updateFen();
  updateTurnStatus();
  moveList.innerHTML = `<div class="small-muted">Watch the board for moves.</div>`;

  if (stream) stream.close();
  stream = new SseStream().connect(`/api/hvse/${sessionId}/stream`);
  stream
    .on("thinking", (evt) => {
      try {
        const t = JSON.parse(evt.data);
        if (t.started_at_ms) setThinking("Engine thinking…");
        if (t.finished_in_ms) setThinking(`Engine moved (${t.finished_in_ms}ms).`);
      } catch {
        setThinking("Engine thinking…");
      }
    })
    .on("engine_move", (evt) => {
      const payload = JSON.parse(evt.data);
      const m = parseUci(payload.uci);
      if (m) lastMove = { from: m.from, to: m.to };
      fen = payload.fen ?? fen;
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
      resignBtn.disabled = true;
      sessionId = null;
      startBtn.disabled = selectedModules.size === 0;
    })
    .on("error", () => {
      setStatus(statusPill, "Disconnected", "danger");
    });
}

async function resign() {
  if (!sessionId) return;
  await fetch(`/api/hvse/${sessionId}`, { method: "DELETE" });
  if (stream) stream.close();
  sessionId = null;
  resignBtn.disabled = true;
  startBtn.disabled = selectedModules.size === 0;
  setStatus(statusPill, "Resigned", "danger");
  statusLine.textContent = "Resigned.";
  setThinking("");
}

// ── Init ──────────────────────────────────────────────────────

async function init() {
  depthRange.oninput = () => { depthLabel.textContent = `(${depthRange.value})`; };
  timeRange.oninput  = () => { timeLabel.textContent = `(${Number(timeRange.value).toFixed(1)}s)`; };
  depthRange.oninput();
  timeRange.oninput();

  startBtn.onclick  = () => startGame();
  resignBtn.onclick = () => resign();

  copyFenBtn.onclick = async () => {
    await navigator.clipboard.writeText(fen === "start" ? "startpos" : fen);
    setStatus(statusPill, "FEN copied", "success");
    setTimeout(() => { if (sessionId) updateTurnStatus(); }, 700);
  };

  const engines = await fetchEngines();
  const hceMeta = engines.find((e) => e.id === "hce");
  if (hceMeta?.module_groups) {
    allModules.classic  = hceMeta.module_groups.classic  ?? [];
    allModules.advanced = hceMeta.module_groups.advanced ?? [];
  }

  renderPalette();
  renderDropZone();
  setupDropZone();

  presetClassicBtn.onclick  = () => applyPreset(allModules.classic);
  presetAdvancedBtn.onclick = () => applyPreset(allModules.advanced);
  presetAllBtn.onclick      = () => applyPreset([...allModules.classic, ...allModules.advanced]);
  presetClearBtn.onclick    = () => { selectedModules.clear(); renderDropZone(); };

  boardEl.addEventListener("click", (e) => {
    const node = e.target?.closest?.("[data-sq]");
    const sq = node?.getAttribute?.("data-sq");
    if (!sq) return;
    if (!sessionId) return;
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

    if (isPromotionMove(from, to)) {
      pendingPromotion = { from, to };
      showPromotionModal();
      return;
    }

    lastMove = { from, to };
    submitHumanMove(`${from}${to}`);
  });

  renderBoard();
  updateFen();
}

init().catch((e) => {
  setStatus(statusPill, "Error", "danger");
  statusLine.textContent = String(e);
});
