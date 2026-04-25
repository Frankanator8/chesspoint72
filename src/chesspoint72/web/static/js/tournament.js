import { setActiveNav, setStatus } from "/static/js/shared.js";

setActiveNav(window.location.pathname);

async function loadBracket() {
  const container = document.getElementById("bracketContainer");
  const pill = document.getElementById("statusPill");
  try {
    const res = await fetch("/api/tournament/bracket");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderBracket(container, data);
    if (data.champion) {
      setStatus(pill, `Champion: ${data.champion}`, "success");
    } else if (data.rounds.length > 0) {
      setStatus(pill, "In progress", "accent");
    } else {
      setStatus(pill, "No data", "neutral");
    }
  } catch (e) {
    container.innerHTML = `<div class="small-muted">Failed to load bracket: ${e.message}</div>`;
    setStatus(pill, "Error", "danger");
  }
}

function makeSeed(name, isWinner, isPending, scoreStr) {
  const el = document.createElement("div");
  el.className = "bracket-seed";
  if (isWinner) el.classList.add("winner");
  else if (isPending) el.classList.add("pending");

  const nameSpan = document.createElement("span");
  nameSpan.textContent = name;
  el.appendChild(nameSpan);

  if (scoreStr) {
    const s = document.createElement("span");
    s.className = "bracket-score";
    s.textContent = scoreStr;
    el.appendChild(s);
  }
  return el;
}

function renderBracket(container, data) {
  container.innerHTML = "";
  const bracket = document.createElement("div");
  bracket.className = "bracket";

  const totalRounds = data.rounds.length;

  for (const round of data.rounds) {
    const col = document.createElement("div");
    col.className = "bracket-round";

    const label = document.createElement("div");
    label.className = "bracket-round-label";
    label.textContent = round.round === totalRounds ? "Final" : `Round ${round.round}`;
    col.appendChild(label);

    for (const match of round.matches) {
      const matchEl = document.createElement("div");
      matchEl.className = "bracket-match";

      if (match.bye) {
        const seed = makeSeed(match.e1, false, false, null);
        seed.classList.add("bye");
        const byeTag = document.createElement("span");
        byeTag.className = "bracket-score";
        byeTag.textContent = "bye";
        seed.appendChild(byeTag);
        matchEl.appendChild(seed);
      } else {
        const e2Name = match.e2 ?? "TBD";
        const hasScore = match.games !== null;
        const score1 = hasScore ? `${match.w1}W ${match.draws}D` : null;
        const score2 = hasScore ? `${match.w2}W ${match.draws}D` : null;
        const isPending = match.winner === null;

        matchEl.appendChild(makeSeed(match.e1, match.winner === match.e1, isPending, score1));
        matchEl.appendChild(makeSeed(e2Name, match.winner === match.e2, isPending, score2));
      }

      col.appendChild(matchEl);
    }

    bracket.appendChild(col);
  }

  if (data.champion) {
    const col = document.createElement("div");
    col.className = "bracket-round";
    const label = document.createElement("div");
    label.className = "bracket-round-label";
    label.textContent = "Champion";
    col.appendChild(label);
    const champ = document.createElement("div");
    champ.className = "bracket-champion";
    champ.textContent = `\u{1F3C6} ${data.champion}`;
    col.appendChild(champ);
    bracket.appendChild(col);
  }

  container.appendChild(bracket);
}

loadBracket();
