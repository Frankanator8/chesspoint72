export async function fetchEngines() {
  const res = await fetch("/api/engines");
  if (!res.ok) throw new Error(`Failed to load engines: ${res.status}`);
  return await res.json();
}

export class SseStream {
  constructor() {
    this.es = null;
    this.handlers = new Map();
  }

  connect(url) {
    this.close();
    this.es = new EventSource(url);
    this.es.onmessage = (evt) => {
      const h = this.handlers.get("message");
      if (h) h(evt);
    };
    this.es.onerror = (evt) => {
      const h = this.handlers.get("error");
      if (h) h(evt);
    };
    return this;
  }

  on(eventName, handler) {
    if (!this.es) {
      this.handlers.set(eventName, handler);
      return this;
    }
    if (eventName === "message" || eventName === "error") {
      this.handlers.set(eventName, handler);
      return this;
    }
    this.es.addEventListener(eventName, (evt) => handler(evt));
    return this;
  }

  close() {
    if (this.es) this.es.close();
    this.es = null;
    return this;
  }
}

export function setActiveNav(pathname) {
  const links = document.querySelectorAll("[data-nav]");
  links.forEach((a) => {
    const target = a.getAttribute("href");
    if (target === pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
}

export function setStatus(el, text, kind = "neutral") {
  const dot = el.querySelector("[data-status-dot]");
  const label = el.querySelector("[data-status-text]");
  label.textContent = text;
  dot.classList.remove("accent", "success", "danger");
  if (kind === "accent") dot.classList.add("accent");
  if (kind === "success") dot.classList.add("success");
  if (kind === "danger") dot.classList.add("danger");
}

export function fillSelect(selectEl, items, { placeholder = "Select..." } = {}) {
  selectEl.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = placeholder;
  selectEl.appendChild(opt0);
  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = it.id;
    opt.textContent = it.label ?? it.id;
    selectEl.appendChild(opt);
  }
}

export function getQueryParams() {
  const p = new URLSearchParams(window.location.search);
  const out = {};
  for (const [k, v] of p.entries()) out[k] = v;
  return out;
}

