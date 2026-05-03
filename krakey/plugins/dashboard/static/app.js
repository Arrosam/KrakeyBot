// Krakey Dashboard SPA — vanilla JS, no build step.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============== AUTH (cookie session) ==============
//
// The server gates the entire app on a per-installation token, set
// as an HttpOnly+SameSite=Strict cookie on first valid auth. The
// browser attaches the cookie automatically to fetches and to the
// WebSocket handshake — no JS-level plumbing required.
//
// The one-click URL the runtime prints (?token=<T>) lands on "/",
// the middleware validates the query param and sets the cookie in
// the same response. We then strip the token from the address bar
// here so it doesn't sit in the user's browser history.
(function stripTokenFromUrl() {
  const u = new URL(location.href);
  if (u.searchParams.has("token")) {
    u.searchParams.delete("token");
    history.replaceState({}, "", u.pathname + u.search + u.hash);
  }
})();

// 401 from a fetch means the cookie is gone or invalid — reload so
// the server-rendered auth page replaces the SPA. We can't render a
// JS modal inside this page since the SPA shouldn't be running with
// a stale session anyway.
const _origFetch = window.fetch.bind(window);
window.fetch = function (url, opts) {
  return _origFetch(url, opts).then((r) => {
    if (r.status === 401 && typeof url === "string" && url.startsWith("/")) {
      location.reload();
    }
    return r;
  });
};

// WS URL helper — same-origin, no token in URL (cookie carries it).
function _wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

// ============== TAB SWITCHING ==============

$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    const id = "tab-" + btn.dataset.tab;
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === id));
    if (btn.dataset.tab === "memory") loadMemory(currentMemView);
    if (btn.dataset.tab === "settings") loadSettings();
    if (btn.dataset.tab === "prompts") loadPrompts();
    if (btn.dataset.tab === "chat") {
      // History was rendered while panel was hidden (scrollHeight=0);
      // scroll to bottom now that it's visible.
      requestAnimationFrame(() => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
      });
      // Opening the tab is the user acknowledging anything that
      // arrived while they were elsewhere — clear the unread badge,
      // dot, and document-title counter.
      if (typeof _clearChatUnread === "function") _clearChatUnread();
    }
  });
});

// ============== STATUS BAR ==============

const statusBar = $("#status-bar");
let lastStats = {};

// Connection-state segment: label + Bootstrap-Icons SVG. Built as
// HTML so the SVG renders (textContent would print escape-decoded
// markup). All inputs here are static strings, no XSS surface.
function _connSegment(label, ok) {
  const icon = window.biIcon(
    ok ? "check-circle-fill" : "x-circle-fill", 11,
  );
  return `<span class="conn ${ok ? "ok" : "off"}">${label} ${icon}</span>`;
}
function setStatus() {
  const parts = [];
  if (lastStats.heartbeat_id != null) parts.push(`HB #${lastStats.heartbeat_id}`);
  if (lastStats.node_count != null) parts.push(`gm=${lastStats.node_count}n/${lastStats.edge_count}e`);
  if (lastStats.fatigue_pct != null) parts.push(`fatigue=${lastStats.fatigue_pct}%`);
  // Single connection indicator: green when both WSes are open, red
  // otherwise. The previous "events stale" state was misleading
  // (long hibernate intervals could trigger it on a perfectly
  // healthy runtime), and a periodic re-render timer was needed to
  // drive it; both are gone now.
  const eventsOpen = !!(eventsWS && eventsWS.readyState === 1);
  const chatOpen = !!(chatWS && chatWS.readyState === 1);
  parts.push(_connSegment("connection", eventsOpen && chatOpen));
  statusBar.innerHTML = parts.join("  |  ");
  renderStatusPanel();
}

// One periodic refresh so the status indicator can flip from the
// initial "— connecting —" placeholder to a real state even when
// no WS events are flowing (e.g. the user wired a broken LLM and
// the runtime can't heartbeat). Cheap — just rewrites the bar's
// innerHTML based on cached lastStats + current WS readyState.
// 1s cadence is invisible to a human but quick enough that a
// failing connect is obvious within the time it takes to read the
// header.
setInterval(setStatus, 1000);

// Format helpers for the big Status panel in the Inner Thoughts view.
// Renders the same data as the top-bar but as a persistent readable
// block so Samuel can watch GM growth + fatigue over time without
// squinting at the narrow header.
function _fmtSince(iso) {
  if (!iso) return "—";
  try {
    const then = new Date(iso).getTime();
    const ms = Date.now() - then;
    if (ms < 0 || isNaN(ms)) return iso;
    const sec = Math.floor(ms / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ${min % 60}m ago`;
    const d = Math.floor(hr / 24);
    return `${d}d ${hr % 24}h ago`;
  } catch { return iso; }
}

function _setPair(key, value, extraCls) {
  // Upsert a dt/dd pair. Keeping insertion order stable so the panel
  // doesn't flicker as new keys arrive — we stamp a `data-key`
  // attribute on the dt and bind the dd right after it.
  let dt = statusPanel.querySelector(`dt[data-key="${key}"]`);
  let dd;
  if (!dt) {
    dt = document.createElement("dt");
    dt.dataset.key = key;
    dt.textContent = key;
    dd = document.createElement("dd");
    dd.dataset.key = key;
    statusPanel.appendChild(dt);
    statusPanel.appendChild(dd);
  } else {
    dd = statusPanel.querySelector(`dd[data-key="${key}"]`);
  }
  dd.textContent = value;
  dd.className = extraCls || "";
}

function renderStatusPanel() {
  if (!statusPanel) return;
  _setPair("heartbeat", lastStats.heartbeat_id != null
    ? `#${lastStats.heartbeat_id}` : "—");
  _setPair("gm nodes", lastStats.node_count != null
    ? String(lastStats.node_count) : "—");
  _setPair("gm edges", lastStats.edge_count != null
    ? String(lastStats.edge_count) : "—");
  const fp = lastStats.fatigue_pct;
  let fCls = "";
  if (fp != null) {
    if (fp >= 75) fCls = "fatigue-high";
    else if (fp >= 50) fCls = "fatigue-mid";
  }
  _setPair("fatigue", fp != null ? `${fp}%` : "—", fCls);
  _setPair("last sleep", lastStats.last_sleep
    ? _fmtSince(lastStats.last_sleep) : "never",
    lastStats.last_sleep ? "" : "stale");
  _setPair("mode", lastStats.mode || "normal");
  _setPair("events ws", eventsWS && eventsWS.readyState === 1 ? "connected" : "disconnected");
}

// ============== INNER THOUGHTS — /ws/events ==============

let eventsWS = null;
const thinkingEl = $("#thinking-stream");
const decisionEl = $("#decision-stream");
// Tool Usage panel — replaces the old "Hypothalamus (dispatch)"
// stream. Logs dispatch + tool_result + idle + sleep events.
const toolEl = $("#tool-stream");
// Stimulus Stream — chronological feed of stimuli as they enter the
// runtime queue. Replaces the old "Pending Stimuli" snapshot list.
const stimEl = $("#stim-stream");
const statusPanel = $("#status-panel");

// Fingerprint set for stimulus-stream dedup. The runtime emits
// `stimuli_queued` once per heartbeat with the *current* queue
// snapshot — anything that survived this beat is in there. We only
// want to log each stimulus once, so we track seen fingerprints
// (type|source|ts) and append only the unseen.
const _seenStimuli = new Set();
const _SEEN_STIMULI_CAP = 1000;
// Section titles that change every heartbeat — open by default.
// Anything else (DNA, SELF-MODEL, HEARTBEAT question, BOOTSTRAP) is
// collapsed since it's noise during normal inspection.
const DYNAMIC_SECTIONS = ["STATUS", "GRAPH MEMORY", "HISTORY", "STIMULUS"];

function splitPromptSections(text) {
  if (!text) return [];
  const parts = text.split(/\n\n(?=#\s)/);
  const out = [];
  for (const p of parts) {
    const m = p.match(/^#\s+\[?([^\]\n]+?)\]?\s*\n([\s\S]*)$/);
    if (m) {
      out.push({ title: m[1].trim(), body: m[2] });
    } else {
      out.push({ title: "DNA / system prompt", body: p });
    }
  }
  return out;
}

function appendEntry(panel, hbId, text) {
  const div = document.createElement("div");
  div.className = "entry";
  const tag = document.createElement("span");
  tag.className = "hb-tag";
  tag.textContent = `#${hbId}`;
  div.appendChild(tag);
  div.appendChild(document.createTextNode(text));
  panel.appendChild(div);
  // keep last 200
  while (panel.children.length > 200) panel.removeChild(panel.firstChild);
  panel.scrollTop = panel.scrollHeight;
}

function appendStimulusToStream(stims) {
  // Each stimuli_queued event carries the current pending queue.
  // Track fingerprints so we only log each one once across the
  // session — the same stimulus survives multiple snapshots until
  // it gets drained by a heartbeat.
  for (const s of stims || []) {
    const fp = `${s.type}|${s.source}|${s.ts}`;
    if (_seenStimuli.has(fp)) continue;
    _seenStimuli.add(fp);
    if (_seenStimuli.size > _SEEN_STIMULI_CAP) {
      // Set iterator gives oldest-first; drop one to bound memory.
      const oldest = _seenStimuli.values().next().value;
      _seenStimuli.delete(oldest);
    }
    const div = document.createElement("div");
    div.className = "entry";
    if (s.adrenalin) div.classList.add("adrenalin");
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = `[${s.type}] ${s.source}`;
    div.appendChild(src);
    div.appendChild(document.createTextNode(
      " " + (s.content || "").slice(0, 200)
    ));
    stimEl.appendChild(div);
    while (stimEl.children.length > 200) stimEl.removeChild(stimEl.firstChild);
    stimEl.scrollTop = stimEl.scrollHeight;
  }
}

function handleEvent(e) {
  switch (e.kind) {
    case "heartbeat_start":
      lastStats.heartbeat_id = e.heartbeat_id;
      setStatus();
      break;
    case "gm_stats":
      lastStats.node_count = e.node_count;
      lastStats.edge_count = e.edge_count;
      lastStats.fatigue_pct = e.fatigue_pct;
      setStatus();
      break;
    case "stimuli_queued":
      appendStimulusToStream(e.stimuli);
      break;
    case "thinking":
      appendEntry(thinkingEl, e.heartbeat_id, e.text);
      break;
    case "decision":
      appendEntry(decisionEl, e.heartbeat_id, e.text);
      break;
    case "note":
      appendEntry(decisionEl, e.heartbeat_id, "[NOTE] " + e.text);
      break;
    case "dispatch":
      // Tool dispatch — Self decided to call this tool. Logged with
      // an outbound arrow so the paired result (←, below) reads
      // chronologically once it lands.
      appendEntry(toolEl, e.heartbeat_id,
        `→ ${e.tool} : ${e.intent}${e.adrenalin ? " (adrenalin)" : ""}`);
      break;
    case "tool_result":
      // Result returned by the tool. Truncate to keep the panel
      // scannable; full content shows up in the Log tab if needed.
      appendEntry(toolEl, "—",
        `← ${e.tool} : ${(e.content || "").slice(0, 200)}`);
      break;
    case "idle":
      // Self decided how long to idle before the next heartbeat.
      // Useful to see the rhythm of the loop alongside the tool
      // calls that fire each beat.
      appendEntry(toolEl, e.heartbeat_id,
        `⏱ idle ${Number(e.interval_seconds).toFixed(1)}s`);
      break;
    case "prompt_built":
      // Live-append to the Prompts tab cache so users see new beats
      // without a re-fetch. The tab's own loader still hits /api/prompts
      // on activation to sync with the server-side ring buffer.
      liveAppendPrompt({
        heartbeat_id: e.heartbeat_id,
        ts: new Date().toISOString(),
        full_prompt: e.layers.full_prompt || "",
      });
      break;
    case "sleep_start":
      appendEntry(toolEl, "—", "💤 sleep started: " + e.reason);
      lastStats.mode = "sleeping";
      setStatus();
      break;
    case "sleep_done":
      appendEntry(toolEl, "—", "🌅 sleep done: " + JSON.stringify(e.stats));
      lastStats.mode = "normal";
      lastStats.last_sleep = new Date().toISOString();
      setStatus();
      break;
  }
}

function connectEvents() {
  eventsWS = new WebSocket(_wsUrl("/ws/events"));
  eventsWS.onopen = setStatus;
  eventsWS.onclose = (ev) => {
    setStatus();
    // 1008 = policy violation = expired/invalid session cookie.
    // Reload so the server renders the auth page in place of the
    // (now stale) SPA. Don't reconnect-loop in the meantime.
    if (ev && ev.code === 1008) { location.reload(); return; }
    setTimeout(connectEvents, 2000);
  };
  eventsWS.onerror = setStatus;
  eventsWS.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.kind === "history") {
      for (const e of data.events) handleEvent(e);
    } else {
      handleEvent(data);
    }
  };
}
connectEvents();

// ============== CHAT — /ws/chat ==============

let chatWS = null;
const chatHistory = $("#chat-history");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatMeta = $("#chat-meta");

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  } catch { return iso; }
}

function renderChatMessage(msg) {
  const div = document.createElement("div");
  div.className = "bubble " + (msg.sender === "user" ? "user" : "krakey");
  if (msg.content) {
    div.appendChild(document.createTextNode(msg.content));
  }
  if (msg.attachments && msg.attachments.length) {
    const wrap = document.createElement("div");
    wrap.className = "attachments";
    for (const a of msg.attachments) {
      if ((a.type || "").startsWith("image/")) {
        const img = document.createElement("img");
        img.src = a.url; img.alt = a.name;
        wrap.appendChild(img);
      }
      const link = document.createElement("a");
      link.href = a.url; link.target = "_blank";
      link.textContent = `📎 ${a.name} (${formatBytes(a.size)})`;
      wrap.appendChild(link);
    }
    div.appendChild(wrap);
  }
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = fmtTime(msg.ts);
  div.appendChild(ts);
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function formatBytes(n) {
  if (n == null) return "?";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

// ----- chat unread / notifications -----
//
// Goal: when a Krakey-side message lands while the user isn't on
// the Chat tab (different tab, or the window is in the background),
// surface it without forcing the user back to the tab.
//   * tab badge — pulse + count on the Chat tab.
//   * audio ping — short Web Audio sine beep, no external asset.
//   * desktop notification — Notification API; permission asked
//     once on first user click so we don't surprise-popup on load.
//   * document title — "(N) Krakey Dashboard" so an out-of-focus
//     window's tab in the OS task bar shows the count too.
//
// Cleared the moment the user clicks the Chat tab, switches focus
// back to the window while Chat is already active, or sends a
// reply themselves.

const chatTabBtn = document.querySelector('.tab-btn[data-tab="chat"]');
let _chatUnread = 0;
const _DEFAULT_TITLE = document.title;

function _chatTabIsActive() {
  return chatTabBtn && chatTabBtn.classList.contains("active");
}

function _bumpChatUnread(msg) {
  _chatUnread += 1;
  if (!chatTabBtn) return;
  chatTabBtn.classList.add("has-unread");
  let dot = chatTabBtn.querySelector(".unread-dot");
  if (!dot) {
    dot = document.createElement("span");
    dot.className = "unread-dot";
    chatTabBtn.appendChild(dot);
  }
  dot.textContent = _chatUnread > 9 ? "9+" : String(_chatUnread);
  document.title = `(${_chatUnread > 9 ? "9+" : _chatUnread}) ${_DEFAULT_TITLE}`;
  _playChatPing();
  _showSystemNotification(msg);
}

function _clearChatUnread() {
  if (_chatUnread === 0 && !chatTabBtn?.classList.contains("has-unread")) return;
  _chatUnread = 0;
  if (chatTabBtn) {
    chatTabBtn.classList.remove("has-unread");
    const dot = chatTabBtn.querySelector(".unread-dot");
    if (dot) dot.remove();
  }
  document.title = _DEFAULT_TITLE;
}

// Web Audio sine ping (~A5, 250ms with quick attack/decay so it's a
// short "blip" not a sustained tone). Ctx is created lazily; browsers
// require a user gesture before audio plays, so we also resume it
// on the first document click.
let _audioCtx = null;
function _ensureAudio() {
  try {
    if (!_audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      _audioCtx = new Ctx();
    }
    if (_audioCtx.state === "suspended") _audioCtx.resume().catch(() => {});
    return _audioCtx;
  } catch (e) {
    return null;
  }
}
document.addEventListener("click", _ensureAudio, { once: true });

function _playChatPing() {
  const ctx = _ensureAudio();
  if (!ctx) return;
  try {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;  // A5
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) { /* autoplay-blocked etc. — silent */ }
}

// Desktop notification — only when the dashboard tab is hidden or
// not on Chat. Permission requested once on the user's first click
// so we don't surprise-popup on page load.
document.addEventListener("click", () => {
  if (!("Notification" in window)) return;
  if (Notification.permission === "default") {
    Notification.requestPermission().catch(() => {});
  }
}, { once: true });

function _showSystemNotification(msg) {
  if (!("Notification" in window)) return;
  if (Notification.permission !== "granted") return;
  // Don't double-notify when the user is actively reading.
  if (document.visibilityState === "visible" && _chatTabIsActive()) return;
  try {
    const body = (msg && msg.content || "").slice(0, 140);
    const n = new Notification("Krakey: new message", {
      body,
      icon: "/static/logo.png",
      tag: "krakey-chat",        // collapse repeats into one toast
      renotify: true,
    });
    n.onclick = () => {
      window.focus();
      if (chatTabBtn) chatTabBtn.click();
      n.close();
    };
  } catch (e) { /* notif quota / focus-lost — silent */ }
}

// Visibility change: if the user comes back to the window WITH the
// chat tab already active, treat that as "they saw it" and clear.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && _chatTabIsActive()) {
    _clearChatUnread();
  }
});

function connectChat() {
  chatWS = new WebSocket(_wsUrl("/ws/chat"));
  chatWS.onopen = () => { chatMeta.textContent = "connected"; setStatus(); };
  chatWS.onclose = (ev) => {
    chatMeta.textContent = "disconnected — reconnecting...";
    setStatus();
    // 1008 → expired session; events-WS handler also reloads.
    if (ev && ev.code === 1008) { location.reload(); return; }
    setTimeout(connectChat, 2000);
  };
  chatWS.onerror = () => { chatMeta.textContent = "error"; setStatus(); };
  chatWS.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.kind === "history") {
      chatHistory.innerHTML = "";
      for (const m of data.messages) renderChatMessage(m);
    } else if (data.kind === "message") {
      renderChatMessage(data.message);
      // Krakey-side messages while the user is away → notify.
      // User-side messages are echoes of what they just typed so
      // they're never "new" from the user's perspective.
      const isFromKrakey = data.message && data.message.sender !== "user";
      const away = !_chatTabIsActive() || document.visibilityState !== "visible";
      if (isFromKrakey && away) {
        _bumpChatUnread(data.message);
      }
    }
  };
}
connectChat();

// ---------- chat input: auto-expand + Enter/Shift+Enter ----------

function autoResize() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 240) + "px";
}
chatInput.addEventListener("input", autoResize);
chatInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    chatForm.requestSubmit();
  }
});

// ---------- attachments staging ----------

const attachBtn = $("#chat-attach-btn");
const fileInput = $("#chat-file");
const attachStrip = $("#chat-attachments");
let pendingAttachments = []; // [{name, url, type, size}]

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  fileInput.value = "";
  await uploadChatFiles(files);
});

async function uploadChatFiles(files) {
  if (!files || !files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  try {
    const r = await fetch("/api/chat/upload", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) { alert("upload failed: " + (body.detail || r.statusText)); return; }
    pendingAttachments.push(...body.files);
    renderAttachStrip();
  } catch (e) {
    alert("upload network: " + e);
  }
}

// Chat-tab drag-drop: dropping files anywhere on the Chat panel
// kicks off the same upload path as the attach button. The "release
// to send file" overlay is rendered purely via the .drag-active
// class — no extra DOM nodes are ever inserted, so the panel
// stays empty when nothing is being dragged.
//
// dragCounter is incremented on dragenter and decremented on
// dragleave to handle the browser firing those events for every
// child element the cursor crosses; otherwise the overlay would
// flicker as the drag pointer moves between the bubble list and
// the input form.
(function () {
  const chatTab = document.getElementById("tab-chat");
  if (!chatTab) return;
  let dragCounter = 0;

  function hasFiles(ev) {
    const types = ev.dataTransfer && ev.dataTransfer.types;
    if (!types) return false;
    for (let i = 0; i < types.length; i++) {
      if (types[i] === "Files") return true;
    }
    return false;
  }

  chatTab.addEventListener("dragenter", (ev) => {
    if (!hasFiles(ev)) return;
    dragCounter++;
    chatTab.classList.add("drag-active");
  });
  chatTab.addEventListener("dragover", (ev) => {
    if (!hasFiles(ev)) return;
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  });
  chatTab.addEventListener("dragleave", (ev) => {
    if (!hasFiles(ev)) return;
    dragCounter = Math.max(0, dragCounter - 1);
    if (dragCounter === 0) chatTab.classList.remove("drag-active");
  });
  chatTab.addEventListener("drop", (ev) => {
    if (!hasFiles(ev)) return;
    ev.preventDefault();
    dragCounter = 0;
    chatTab.classList.remove("drag-active");
    const files = Array.from(ev.dataTransfer.files || []);
    if (files.length) uploadChatFiles(files);
  });

  // Stop the browser's default open-the-file behaviour when files are
  // dropped outside the chat tab — we don't want the dashboard to
  // navigate away to view a dropped image.
  window.addEventListener("dragover", (ev) => {
    if (hasFiles(ev)) ev.preventDefault();
  });
  window.addEventListener("drop", (ev) => {
    if (hasFiles(ev)) ev.preventDefault();
  });
})();

function renderAttachStrip() {
  attachStrip.innerHTML = "";
  pendingAttachments.forEach((a, i) => {
    const chip = document.createElement("span");
    chip.className = "attach-chip";
    chip.appendChild(document.createTextNode(`📎 ${a.name} (${formatBytes(a.size)})`));
    const x = document.createElement("span");
    x.className = "x"; x.textContent = "×"; x.title = "remove";
    x.addEventListener("click", () => {
      pendingAttachments.splice(i, 1);
      renderAttachStrip();
    });
    chip.appendChild(x);
    attachStrip.appendChild(chip);
  });
}

chatForm.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = chatInput.value.trim();
  if ((!text && !pendingAttachments.length) || !chatWS || chatWS.readyState !== 1) return;
  chatWS.send(JSON.stringify({ text, attachments: pendingAttachments }));
  chatInput.value = "";
  autoResize();
  pendingAttachments = [];
  renderAttachStrip();
});

// ============== MEMORY ==============

let currentMemView = "graph";
// Hold onto the active cytoscape instance so we can destroy it cleanly
// when the user switches sub-views (otherwise its event listeners +
// internal canvas leak across re-renders).
let _gmCy = null;
// Last view that was actually rendered into #mem-content. Switching
// to the memory tab while we're already showing this view is a no-op
// — fetches + cytoscape rebuild are both expensive (especially with
// the no-truncation node count).
let _lastRenderedMemView = null;

$$(".mem-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".mem-btn").forEach((b) => b.classList.toggle("active", b === btn));
    currentMemView = btn.dataset.mem;
    loadMemory(currentMemView);
  });
});

function _disposeMemView() {
  if (_gmCy) {
    try { _gmCy.destroy(); } catch (e) { /* already gone */ }
    _gmCy = null;
  }
}

async function loadMemory(view, opts) {
  const force = !!(opts && opts.force);
  const target = $("#mem-content");
  if (!force && view === _lastRenderedMemView) return;
  _disposeMemView();
  target.classList.toggle("graph-mode", view === "graph");
  target.textContent = "loading...";
  try {
    if (view === "graph") {
      await renderGraph(target);
    } else if (view === "kbs") {
      const r = await fetch("/api/kbs").then((r) => r.json());
      target.innerHTML = renderKBs(r);
      $$(".kb-card button").forEach((btn) => {
        btn.addEventListener("click", () => loadKBEntries(btn.dataset.kbid));
      });
    }
    _lastRenderedMemView = view;
  } catch (e) {
    target.textContent = "error: " + e;
    _lastRenderedMemView = null;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ----- GM Graph view (cytoscape) -----

// Map gm category → fill colour, mirroring the cat-* classes in
// memory/view.css. Pre-resolved hex so we don't have to read computed
// CSS vars per-node.
const _CAT_COLORS = {
  FACT: "#7ec77e",
  RELATION: "#c585c5",
  KNOWLEDGE: "#6cd5d5",
  TARGET: "#e8c060",
  FOCUS: "#d27575",
};
const _CAT_DEFAULT = "#6b7280";

async function renderGraph(target) {
  // Build the shell first so the canvas + inspector + stats overlay
  // exist before cytoscape mounts. _disposeMemView() above already
  // tore down any previous cytoscape instance.
  target.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "gm-graph-wrap";
  const canvas = document.createElement("div");
  canvas.id = "gm-graph";
  const hint = document.createElement("div");
  hint.className = "gm-graph-hint";
  hint.textContent = "drag to pan · scroll to zoom · drag a node to move it";
  const stats = document.createElement("div");
  stats.className = "gm-graph-stats";
  stats.innerHTML = "<i>loading stats…</i>";
  canvas.appendChild(hint);
  canvas.appendChild(stats);
  const inspect = document.createElement("div");
  inspect.className = "gm-graph-inspect";
  inspect.innerHTML = "<h4>Inspector</h4><p style='color:var(--muted)'>Click a node or edge.</p>";
  wrap.appendChild(canvas);
  wrap.appendChild(inspect);
  target.appendChild(wrap);

  // Render the full GM — no truncation. The server's auth gate keeps
  // these big queries off the open Internet; a million is a generous
  // ceiling that's still finite enough that a runaway integer
  // overflow in the route can't happen.
  const [nodesRes, edgesRes, statsRes] = await Promise.all([
    fetch("/api/gm/nodes?limit=1000000").then((r) => r.json()),
    fetch("/api/gm/edges?limit=1000000").then((r) => r.json()),
    fetch("/api/gm/stats").then((r) => r.json()),
  ]);
  _renderGraphStats(stats, statsRes);

  if (typeof cytoscape === "undefined") {
    canvas.removeChild(hint);
    canvas.removeChild(stats);
    canvas.innerHTML =
      "<p style='padding:12px;color:var(--muted)'>" +
      "graph library failed to load (CDN blocked?). Refresh the " +
      "page when you have connectivity, or switch to KBs." +
      "</p>";
    return;
  }
  if (!nodesRes.nodes.length) {
    canvas.removeChild(hint);
    canvas.removeChild(stats);
    canvas.innerHTML =
      "<p style='padding:12px;color:var(--muted)'>(no nodes — GM is empty)</p>";
    return;
  }

  // Cytoscape needs name-based ids since edges are name triples (see
  // gm.list_edges_named). Drop edges whose endpoints we didn't fetch
  // (limit cap can leave dangling references).
  const cyNodes = nodesRes.nodes.map((n) => ({
    data: {
      id: n.name,
      label: n.name,
      raw: n,
      color: _CAT_COLORS[n.category] || _CAT_DEFAULT,
      size: 16 + Math.min(20, Math.max(0, (n.importance || 0)) * 2),
    },
  }));
  const known = new Set(cyNodes.map((n) => n.data.id));
  const cyEdges = edgesRes.edges
    .filter((e) => known.has(e.source) && known.has(e.target))
    .map((e, i) => ({
      data: {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        label: e.predicate || "",
        raw: e,
      },
    }));

  _gmCy = cytoscape({
    container: canvas,
    elements: { nodes: cyNodes, edges: cyEdges },
    minZoom: 0.1,
    maxZoom: 4,
    wheelSensitivity: 0.3,
    style: [
      {
        selector: "node",
        style: {
          "background-color": "data(color)",
          label: "data(label)",
          color: "#d8dee9",
          "font-size": 9,
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "ellipsis",
          "text-max-width": 80,
          "text-outline-color": "#0d0f12",
          "text-outline-width": 2,
          width: "data(size)",
          height: "data(size)",
          "border-width": 1,
          "border-color": "#262c34",
        },
      },
      {
        selector: "node:selected",
        style: { "border-color": "#6cd5d5", "border-width": 2 },
      },
      {
        selector: "edge",
        style: {
          width: 1,
          "line-color": "#3a4250",
          "target-arrow-color": "#3a4250",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          label: "data(label)",
          "font-size": 8,
          color: "#6b7280",
          "text-rotation": "autorotate",
          "text-background-color": "#0d0f12",
          "text-background-opacity": 0.6,
          "text-background-padding": 1,
        },
      },
      {
        selector: "edge:selected",
        style: { "line-color": "#6cd5d5", "target-arrow-color": "#6cd5d5" },
      },
    ],
    layout: {
      name: "cose",
      animate: false,
      idealEdgeLength: 80,
      nodeRepulsion: 8000,
      gravity: 0.25,
      numIter: 1500,
    },
  });

  _gmCy.on("tap", "node", (ev) => _showNodeInspect(inspect, ev.target.data("raw")));
  _gmCy.on("tap", "edge", (ev) => _showEdgeInspect(inspect, ev.target.data("raw")));
  _gmCy.on("tap", (ev) => {
    if (ev.target === _gmCy) _showInspectEmpty(inspect);
  });
}

function _renderGraphStats(host, s) {
  host.innerHTML = "";
  const total = document.createElement("div");
  total.className = "row";
  total.innerHTML =
    `<span class="label">total</span>` +
    `<span><b>${s.total_nodes ?? 0}</b> nodes · ` +
    `<b>${s.total_edges ?? 0}</b> edges</span>`;
  host.appendChild(total);
  const cats = s.by_category || {};
  if (Object.keys(cats).length) {
    const catsRow = document.createElement("div");
    catsRow.className = "cats";
    for (const [k, v] of Object.entries(cats)) {
      const span = document.createElement("span");
      span.className = "cat-" + k;
      span.textContent = `${k}=${v}`;
      catsRow.appendChild(span);
    }
    host.appendChild(catsRow);
  }
}

// Inspector helpers — build via DOM + textContent so node/edge data
// (which can include LLM-generated text) can't inject markup.
function _kvList(host, pairs) {
  host.innerHTML = "";
  const h4 = document.createElement("h4");
  h4.textContent = pairs._title || "Detail";
  host.appendChild(h4);
  const dl = document.createElement("dl");
  for (const [k, v] of pairs._rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v == null ? "—" : String(v);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
  host.appendChild(dl);
}

function _showNodeInspect(host, n) {
  if (!n) return;
  _kvList(host, {
    _title: "Node",
    _rows: [
      ["id", n.id],
      ["name", n.name],
      ["category", n.category],
      ["source", n.source_type],
      ["importance", n.importance != null ? Number(n.importance).toFixed(2) : "—"],
      // Render the full description. The inspector column has its
      // own overflow / word-break so a long LLM-generated summary
      // scrolls inside its column instead of being silently elided.
      ["description", n.description || ""],
    ],
  });
}

function _showEdgeInspect(host, e) {
  if (!e) return;
  _kvList(host, {
    _title: "Edge",
    _rows: [
      ["source", e.source],
      ["predicate", e.predicate],
      ["target", e.target],
    ],
  });
}

function _showInspectEmpty(host) {
  host.innerHTML =
    "<h4>Inspector</h4><p style='color:var(--muted)'>Click a node or edge.</p>";
}

function renderKBs(r) {
  if (!r.kbs.length) return "<p>no KBs yet (Sleep hasn't run)</p>";
  return r.kbs.map((k) => `
    <div class="kb-card">
      <h4>${escapeHtml(k.name)} <small style="color:var(--muted)">(${k.kb_id})</small></h4>
      <div class="meta">${k.entry_count} entries · ${escapeHtml(k.description || "")}</div>
      <button data-kbid="${escapeHtml(k.kb_id)}">View entries</button>
      <div id="kb-entries-${escapeHtml(k.kb_id)}"></div>
    </div>`).join("");
}

async function loadKBEntries(kbid) {
  const target = document.getElementById(`kb-entries-${kbid}`);
  if (!target) return;
  target.innerHTML = "loading...";
  try {
    const r = await fetch(`/api/kb/${encodeURIComponent(kbid)}/entries?limit=200`).then((r) => r.json());
    if (!r.entries.length) { target.innerHTML = "<i>(no entries)</i>"; return; }
    target.innerHTML = r.entries.map((e) => `
      <div class="kb-entry">
        <span class="tags">${(e.tags || []).join(", ")}</span>
        ${escapeHtml(e.content)}
      </div>`).join("");
  } catch (e) {
    target.textContent = "error: " + e;
  }
}

// Memory tab is lazy: don't fetch nodes/edges/stats on page load —
// only when the user actually opens the tab (tab-switch handler at
// the top of this file does that). Pre-fetching here was paying the
// /api/gm/* round-trip + cytoscape build on every page load even
// for users who never opened Memory, and it could pile on top of a
// busy runtime (auto-recall + LLM thinking) and stall the dashboard.

// ============== LOG — /ws/logs ==============
//
// Live stdout/stderr from the runtime, captured server-side by a tee
// in log_capture.py. ANSI escape sequences (the runtime's heartbeat
// logger emits cyan/green/yellow/magenta) are parsed here into
// CSS-classed spans so the on-screen log feels like a tinted
// terminal rather than a plain text dump.

const logStream = $("#log-stream");
const logCount = $("#log-count");
const logAutoscroll = $("#log-autoscroll");
const LOG_UI_CAP = 2000;  // cap rendered lines so a long-running tab can't OOM the browser
let logWS = null;
let _logLineCount = 0;

// Map ANSI SGR colour codes to the same theme tokens used elsewhere.
// Only handles foreground colours and reset — that's all the runtime
// console.colors module emits.
const _ANSI_CLASS = {
  "31": "ansi-red",
  "32": "ansi-green",
  "33": "ansi-yellow",
  "34": "ansi-cyan",   // map to cyan; we don't have a real blue token
  "35": "ansi-magenta",
  "36": "ansi-cyan",
  "37": null,           // white = default
  "0":  null,           // reset → close current span
};

function _renderAnsiLine(line) {
  // Escape HTML first so any < > & in log content can't inject markup.
  let safe = line
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  // Then walk the ANSI codes left-to-right, opening/closing spans as
  // we see codes. A single \x1b[<n>m sequence either opens a new
  // colour span (closing any previous) or, for code 0, closes.
  const re = /\[([0-9;]*)m/g;
  let out = "";
  let cursor = 0;
  let openSpan = false;
  let m;
  while ((m = re.exec(safe)) !== null) {
    out += safe.slice(cursor, m.index);
    cursor = m.index + m[0].length;
    // Multi-code sequences (e.g. "\x1b[1;36m") — take the last
    // colour-meaningful one. The runtime only emits single-code
    // sequences in practice, but be defensive.
    const codes = m[1].split(";").filter(Boolean);
    let cls = null;
    let isReset = false;
    for (const c of codes) {
      if (c === "0" || c === "") { isReset = true; cls = null; continue; }
      if (_ANSI_CLASS[c] !== undefined) { cls = _ANSI_CLASS[c]; isReset = false; }
    }
    if (openSpan) { out += "</span>"; openSpan = false; }
    if (!isReset && cls) { out += `<span class="${cls}">`; openSpan = true; }
  }
  out += safe.slice(cursor);
  if (openSpan) out += "</span>";
  return out;
}

function _appendLogLine(line) {
  const div = document.createElement("div");
  div.className = "log-line";
  // Tag warning-ish lines from stderr-style prefixes so they pop
  // even when the runtime didn't bother with ANSI codes.
  if (/\[runtime\]|error|traceback/i.test(line)) {
    div.classList.add("error");
  } else if (/\[warn\]|warning/i.test(line)) {
    div.classList.add("warn");
  }
  div.innerHTML = _renderAnsiLine(line);
  logStream.appendChild(div);
  while (logStream.children.length > LOG_UI_CAP) {
    logStream.removeChild(logStream.firstChild);
  }
  _logLineCount += 1;
  if (logCount) logCount.textContent = `${_logLineCount} lines`;
  if (logAutoscroll && logAutoscroll.checked) {
    logStream.scrollTop = logStream.scrollHeight;
  }
}

function connectLogs() {
  logWS = new WebSocket(_wsUrl("/ws/logs"));
  logWS.onopen = () => {
    if (logStream.textContent === "— connecting —") {
      logStream.textContent = "";
    }
  };
  logWS.onclose = (ev) => {
    if (ev && ev.code === 1008) { location.reload(); return; }
    setTimeout(connectLogs, 2000);
  };
  logWS.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.kind === "history") {
      logStream.textContent = "";
      for (const line of data.lines) _appendLogLine(line);
    } else if (data.kind === "line") {
      _appendLogLine(data.line);
    }
  };
}
connectLogs();

if (logStream) {
  $("#log-clear").addEventListener("click", () => {
    logStream.innerHTML = "";
    _logLineCount = 0;
    if (logCount) logCount.textContent = "0 lines";
  });
}

// ============== PROMPTS ==============

const promptsList = $("#prompts-list");
let promptsCache = [];   // newest first; trimmed to PROMPT_UI_CAP
const PROMPT_UI_CAP = 200;

async function loadPrompts() {
  promptsList.textContent = "loading...";
  try {
    const r = await fetch("/api/prompts?limit=200").then((r) => r.json());
    promptsCache = r.prompts || [];
    renderPromptsList();
  } catch (e) {
    promptsList.textContent = "error: " + e;
  }
}

function liveAppendPrompt(p) {
  // Merge / dedupe by heartbeat_id (server may replay on reconnect)
  const idx = promptsCache.findIndex((x) => x.heartbeat_id === p.heartbeat_id);
  if (idx !== -1) promptsCache.splice(idx, 1);
  promptsCache.unshift(p);
  if (promptsCache.length > PROMPT_UI_CAP) promptsCache.length = PROMPT_UI_CAP;
  // Only re-render if the Prompts tab is currently visible (cheap skip)
  if ($("#tab-prompts").classList.contains("active")) renderPromptsList();
}

function fmtTs(iso) {
  try {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  } catch { return iso; }
}

function renderPromptsList() {
  promptsList.innerHTML = "";
  if (!promptsCache.length) {
    promptsList.textContent = "(no prompts yet — wait one heartbeat)";
    return;
  }
  for (const p of promptsCache) {
    const card = document.createElement("details");
    card.className = "prompt-card";
    const sum = document.createElement("summary");
    const tag = document.createElement("span");
    tag.className = "hb-tag";
    tag.textContent = `#${p.heartbeat_id}`;
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = fmtTs(p.ts);
    sum.appendChild(tag);
    sum.appendChild(ts);
    sum.appendChild(document.createTextNode(
      "  — " + (p.full_prompt ? `${p.full_prompt.length} chars` : "(empty)")
    ));
    card.appendChild(sum);
    // Inner sections (DNA / STATUS / STIMULUS etc) — collapsible
    const inner = document.createElement("div");
    inner.className = "prompt-card-body";
    for (const s of splitPromptSections(p.full_prompt)) {
      const sec = document.createElement("details");
      sec.className = "prompt-section";
      if (DYNAMIC_SECTIONS.some((k) => s.title.toUpperCase().includes(k))) {
        sec.open = true;
      }
      const ss = document.createElement("summary");
      ss.textContent = s.title;
      const pre = document.createElement("pre");
      pre.textContent = s.body;
      sec.appendChild(ss);
      sec.appendChild(pre);
      inner.appendChild(sec);
    }
    card.appendChild(inner);
    promptsList.appendChild(card);
  }
}

// ============== SETTINGS (form-based) ==============

const settingsForm = $("#settings-form");
const settingsToast = $("#settings-toast");

// Mutable working copy of config; all widgets bind here.
let cfgState = null;

// Defaults to seed missing sections so toggles/numbers don't read as
// "unset" and mislead the user into thinking the runtime is off.
const SECTION_DEFAULTS = {
  idle: { min_interval: 2, max_interval: 300, default_interval: 10 },
  fatigue: { gm_node_soft_limit: 1000, force_sleep_threshold: 1200, thresholds: {} },
  // `sliding_window` section removed — history budget is now derived
  // from Self role's max_input_tokens × history_token_fraction (see
  // role params UI under LLM).
  graph_memory: {
    db_path: "workspace/data/graph_memory.sqlite",
    auto_ingest_similarity_threshold: 0.92,
    recall_per_stimulus_k: 50, recall_screening_token_multiplier: 3.0,
    neighbor_expand_depth: 1,
    // `max_recall_nodes` removed — recall is now capped by Self role's
    // recall_token_budget (absolute token cap, not a node count).
  },
  knowledge_base: { dir: "workspace/data/knowledge_bases" },
  // Defaults mirror SleepSection in krakey/models/config/memory.py —
  // editing here without keeping the dataclass in sync makes the UI
  // pre-populate the wrong values on a fresh config.
  sleep: {
    max_duration_seconds: 7200,
    min_community_size: 2,
    kb_consolidation_threshold: 0.85,
    kb_index_max: 30,
    kb_archive_pct: 10,
    kb_revive_threshold: 0.80,
  },
  // Same dataclass-mirror rule — SafetySection defaults are 500/50.
  safety: { gm_node_hard_limit: 500, max_consecutive_no_action: 50 },
  // Top-level `environments:` block — replaces the old `sandbox:`
  // section as of the runtime's environments-refactor. ``local`` is
  // always present (just an allow-list); ``sandbox`` is optional and
  // holds the VM connectivity fields the runtime previously read
  // off the top-level sandbox section.
  environments: {
    local: { allowed_plugins: [] },
    sandbox: null,
  },
};

// Hover tooltip text per "section.field" key.
const HELP = {
  "idle.min_interval": "Minimum idle interval (seconds). Self uses [IDLE] N to control each beat, but it will never go below this value.",
  "idle.max_interval": "Maximum idle interval (seconds). Even if Self requests a longer idle, it will not exceed this value.",
  "idle.default_interval": "Default idle interval (seconds) when Self does not specify one.",
  "fatigue.gm_node_soft_limit": "Soft upper bound on GM nodes. fatigue% = nodes / soft_limit * 100. Self uses fatigue% to decide whether to sleep proactively.",
  "fatigue.force_sleep_threshold": "Force-sleep threshold (fatigue%). Above this, runtime enters sleep without waiting for Self's consent.",
  "graph_memory.db_path": "Path to the GM SQLite file.",
  "graph_memory.auto_ingest_similarity_threshold": "Similarity threshold (0-1) for stimulus auto_ingest. Below this, the stimulus is treated as a new GM node.",
  "graph_memory.recall_per_stimulus_k": "Hard cap on per-stimulus vec_search top_k. The actual top_k is computed dynamically from recall_screening_token_multiplier and never exceeds this.",
  "graph_memory.recall_screening_token_multiplier": "Token multiplier for the screening pool: each stimulus tries to surface multiplier × recall_token_budget tokens of candidates so the final budget cut has a rich pool to choose from. 1.0 = no over-sampling; default 3.0.",
  "graph_memory.neighbor_expand_depth": "Neighbor-expansion depth at recall time (how many edges to traverse).",
  "knowledge_base.dir": "Directory for KB SQLite files; sleep migration writes here.",
  "sleep.max_duration_seconds": "Maximum allowed duration for a single sleep (seconds), to prevent sleep from hanging.",
  "sleep.min_community_size": "Communities below this many GM nodes stay in graph memory; only larger communities migrate to a KB. Default 2 = skip pure singletons.",
  "sleep.kb_consolidation_threshold": "KB consolidation: pairwise-merge active KBs whose index vectors (mean of member entry embeddings) are at least this cosine-close. 0–1; higher = stricter merging. Default 0.85.",
  "sleep.kb_index_max": "When the active KB count exceeds this, sleep archives the least-important `kb_archive_pct` percent. Archived KBs keep their files but stop showing up in recall.",
  "sleep.kb_archive_pct": "Percentage of active KBs to archive each pass once kb_index_max is exceeded. Importance = entry_count × mean entry importance.",
  "sleep.kb_revive_threshold": "When sleep would build a fresh KB for a new community, first compare the summary embedding against archived KBs' index vectors; revive an archived KB if cosine ≥ this. Models the 'forgot, then re-encountered' relearning shortcut. Default 0.80.",
  "safety.gm_node_hard_limit": "Hard upper bound on GM nodes. Above this, sleep refuses to add more nodes (prevents runaway growth).",
  "safety.max_consecutive_no_action": "After this many consecutive 'No action' beats, runtime considers Self stuck and triggers a self-rescue sleep.",
  "provider.type": "Provider implementation type. Currently only openai_compatible is supported.",
  "provider.base_url": "API root URL (without trailing /v1 etc.; LLMClient appends it automatically).",
  "provider.api_key": "API key. Supports a ${ENV_VAR} placeholder to read from environment variables.",
  "model.name": "Model ID, matching the provider's API.",
  "model.capabilities": "Capability tags for the model. Currently informational only; not strictly validated.",
  "role.provider": "Pick a provider for this role.",
  "role.model": "Pick a model under the chosen provider.",
  "channel.enabled": "Whether this channel is enabled.",
  "channel.default_adrenalin": "Whether stimuli pushed by this channel default to adrenalin=true (interrupting idle).",
  "tool.enabled": "Whether to register this tool for Hypothalamus to use.",
  "tool.max_results": "Maximum number of search results.",
  "tool.sandbox_dir": "Working directory for code / file operations.",
  "tool.timeout_seconds": "Subprocess timeout (seconds).",
  "tool.max_output_chars": "Truncation length for stdout/stderr.",
  "tool.screenshot_dir": "Directory for GUI screenshots.",
  "tool.history_path": "JSONL persistence path for web chat.",
  "tool.sandbox": "Whether this tool's non-idempotent operations are confined to the sandbox VM. Default true — turning off is dangerous (code / GUI runs on your host).",
  "environments.local.allowed_plugins": "Plugins permitted to use the always-on Local execution env (host-process access). Empty = no plugin can run on the host.",
  "environments.sandbox.allowed_plugins": "Plugins permitted to use the Sandbox VM env. Empty = sandbox VM is registered but no plugin can drive it.",
  "environments.sandbox.guest_os": "Sandbox guest OS: linux / macos / windows. Required when the sandbox env is enabled.",
  "environments.sandbox.provider": "VM manager: qemu (recommended) / virtualbox / utm.",
  "environments.sandbox.vm_name": "VM instance name (must be pre-provisioned).",
  "environments.sandbox.display": "headed = VM desktop shown in a window so you can watch / intervene; headless = VM hidden, only the agent interacts. Choose by your usage preference.",
  "environments.sandbox.resources.cpu": "vCPU count assigned to the VM.",
  "environments.sandbox.resources.memory_mb": "RAM (MB) assigned to the VM.",
  "environments.sandbox.resources.disk_gb": "VM disk size (GB).",
  "environments.sandbox.agent.url": "HTTP URL of the in-VM guest agent, e.g. http://10.0.2.10:8765. Must be on the host-only subnet.",
  "environments.sandbox.agent.token": "Shared token between host and agent. Use ${ENV_VAR} to read from the environment.",
  "environments.sandbox.network_mode": "VM network policy: nat_allowlist (egress allow-list) / host_only (no internet) / isolated (no network).",
  "environments.sandbox.allowlist_domains": "When network_mode=nat_allowlist, the sandbox VM can reach exactly these hostnames. Egress to anything else is dropped.",
};

// Fixed numeric/string dataclass schemas — drives generic renderer.
const SCHEMAS = {
  idle: [
    ["min_interval", "number"],
    ["max_interval", "number"],
    ["default_interval", "number"],
  ],
  fatigue_scalars: [
    ["gm_node_soft_limit", "number"],
    ["force_sleep_threshold", "number"],
  ],
  graph_memory: [
    ["db_path", "text"],
    ["auto_ingest_similarity_threshold", "number_float"],
    ["recall_per_stimulus_k", "number"],
    ["recall_screening_token_multiplier", "number_float"],
    ["neighbor_expand_depth", "number"],
  ],
  knowledge_base: [
    ["dir", "text"],
  ],
  sleep: [
    ["max_duration_seconds", "number"],
    // KB lifecycle tuning — fields drive sleep's compaction engine
    // (see krakey/memory/sleep/sleep_manager.py + sibling modules).
    // Power users tune these to control how aggressively Krakey
    // consolidates / archives knowledge bases between sleeps.
    ["min_community_size", "number"],
    ["kb_consolidation_threshold", "number_float"],
    ["kb_index_max", "number"],
    ["kb_archive_pct", "number"],
    ["kb_revive_threshold", "number_float"],
  ],
  safety: [
    ["gm_node_hard_limit", "number"],
    ["max_consecutive_no_action", "number"],
  ],
  // Schemas under `environments.sandbox.*`. The top-level sandbox
  // section is gone in the runtime (rewrites to environments.sandbox),
  // so the dashboard's sandbox UI is now a sub-block of Environments.
  env_sandbox_scalars: [
    ["guest_os", "text"],
    ["provider", "text"],
    ["vm_name", "text"],
    ["display", "text"],
    ["network_mode", "text"],
  ],
  env_sandbox_resources: [
    ["cpu", "number"],
    ["memory_mb", "number"],
    ["disk_gb", "number"],
  ],
  env_sandbox_agent: [
    ["url", "text"],
    ["token", "password"],
  ],
};

// Live load report (loaded ✓ / failed ✗) sourced from /api/plugins.
// The unified Plugins panel surfaces this as a per-row status badge —
// edits are NOT made through this object; per-plugin config edits live
// in modifierConfigEdits, enable/disable lives in cfgState.{modifiers,
// plugins}.
let pluginReport = { tools: [], channels: [] };

// Config-schema introspection cache. Populated on loadSettings() from
// GET /api/config/schema. The LLM role params UI reads this instead of
// hardcoding field lists — adding a field to LLMParams on the Python
// side automatically surfaces it here without JS edits.
//   shape: { llm_params: [{field, type, default, help, choices?}] }
//   (llm_role_defaults removed in the tag-based LLM refactor 2026-04-26)
let configSchema = { llm_params: [] };

async function loadSettings() {
  settingsToast.textContent = "";
  settingsForm.innerHTML = "loading...";
  try {
    // Load config + plugin discovery + schema in parallel
    const [cfgRes, pluginRes, schemaRes] = await Promise.all([
      fetch("/api/settings"),
      fetch("/api/plugins").catch(() => null),
      fetch("/api/config/schema").catch(() => null),
    ]);
    if (cfgRes.status === 503) {
      settingsForm.innerHTML = "<i>(config_path not provided to dashboard)</i>";
      return;
    }
    const data = await cfgRes.json();
    cfgState = data.parsed || {};
    // Migration: the runtime stopped reading top-level `sandbox:` —
    // see krakey/models/config/__init__.py — and moved every field
    // under `environments.sandbox`. If the user's on-disk yaml still
    // has the legacy block, lift its contents into the new shape so
    // the dashboard renders something instead of an empty toggle.
    // Drop the legacy key from the edit state so the next save
    // produces a clean yaml.
    if (cfgState.sandbox && typeof cfgState.sandbox === "object") {
      cfgState.environments = cfgState.environments || {};
      if (cfgState.environments.sandbox == null) {
        cfgState.environments.sandbox = {
          allowed_plugins: [],
          allowlist_domains: [],
          ...cfgState.sandbox,
        };
      }
      delete cfgState.sandbox;
    }
    if (pluginRes && pluginRes.ok) {
      pluginReport = await pluginRes.json();
    } else {
      pluginReport = { tools: [], channels: [] };
    }
    if (schemaRes && schemaRes.ok) {
      configSchema = await schemaRes.json();
    } else {
      configSchema = { llm_params: [] };
    }
    await loadAvailableModifiers();
    renderSettingsForm();
  } catch (e) {
    settingsForm.innerHTML = "error loading: " + escapeHtml(String(e));
  }
}

function renderSettingsForm() {
  settingsForm.innerHTML = "";
  // LLM (tag-based shape, post 2026-04-26 refactor)
  const llm = ensure(cfgState, "llm",
    () => ({ providers: {}, tags: {}, core_purposes: {} }));
  ensure(llm, "providers", () => ({}));
  ensure(llm, "tags", () => ({}));
  ensure(llm, "core_purposes", () => ({}));
  settingsForm.appendChild(renderLLMSection(llm));

  // Plugins — single unified panel covering modifiers, tools, and
  // channels (a plugin is the on-disk meta.yaml unit; its
  // ``components`` array carries one or more of those kinds). One
  // row per plugin: enable checkbox, kind badges, live load status,
  // expandable per-plugin config_schema + LLM purpose bindings.
  settingsForm.appendChild(renderPluginsSection());

  // Generic sections (each seeded from SECTION_DEFAULTS so missing fields
  // pre-populate to runtime defaults instead of looking "off"/empty)
  ensureSection("idle");
  settingsForm.appendChild(renderGenericSection("idle", "Idle",
    cfgState.idle, SCHEMAS.idle));

  ensureSection("fatigue");
  const fatSec = renderGenericSection("fatigue", "Fatigue",
    cfgState.fatigue, SCHEMAS.fatigue_scalars);
  fatSec.querySelector(".body").appendChild(renderFatigueThresholds(cfgState.fatigue));
  settingsForm.appendChild(fatSec);

  ensureSection("graph_memory");
  settingsForm.appendChild(renderGenericSection("graph_memory", "Graph Memory",
    cfgState.graph_memory, SCHEMAS.graph_memory));
  ensureSection("knowledge_base");
  settingsForm.appendChild(renderGenericSection("knowledge_base", "Knowledge Base",
    cfgState.knowledge_base, SCHEMAS.knowledge_base));

  ensureSection("sleep");
  settingsForm.appendChild(renderGenericSection("sleep", "Sleep",
    cfgState.sleep, SCHEMAS.sleep));
  ensureSection("safety");
  const safetySec = renderGenericSection("safety", "Safety",
    cfgState.safety, SCHEMAS.safety);
  // Hint at the top of the section body: the SafetySection dataclass
  // is parsed and persisted but the runtime currently has zero
  // consumers of config.safety.*. Document the status so users
  // don't expect their hard-limit to fire.
  const advisoryHint = document.createElement("p");
  advisoryHint.className = "section-hint";
  advisoryHint.textContent =
    "advisory only — runtime does not yet enforce these limits";
  const safetyBody = safetySec.querySelector(".body");
  safetyBody.insertBefore(advisoryHint, safetyBody.firstChild);
  settingsForm.appendChild(safetySec);

  // Environments — top-level execution-env block. Replaces the old
  // top-level `sandbox:` section the runtime no longer reads.
  // Two sub-blocks:
  //   * Local — always present, just an allow-list of plugins
  //     permitted to run on the host process.
  //   * Sandbox VM — optional. Toggle to register/deregister; when
  //     registered, holds VM connectivity + per-plugin allow-list.
  ensureSection("environments");
  settingsForm.appendChild(renderEnvironmentsSection(cfgState.environments));
}

function ensure(obj, key, factory) {
  if (obj[key] == null) obj[key] = factory();
  return obj[key];
}

function ensureSection(key) {
  const defaults = SECTION_DEFAULTS[key] || {};
  if (cfgState[key] == null) cfgState[key] = {};
  for (const [k, v] of Object.entries(defaults)) {
    if (cfgState[key][k] == null) cfgState[key][k] = v;
  }
}

// Per-section leading icon (Bootstrap Icons name). Looked up by title;
// sections without an entry just render no icon.
const SECTION_ICONS = {
  "LLM": "cpu",
  "Plugins": "gear",
  "Idle": "moon",
  "Fatigue": "bar-chart",
  "Graph Memory": "geo-alt",
  "Knowledge Base": "book",
  "Sleep": "moon",
  "Safety": "shield-check",
  "Environments": "hdd",
};

// Synthwave-ish accent per section. Renders on the heading icon +
// title only — the body keeps the neutral text colour so dense
// configuration stays readable. Token names match the CSS variables
// defined in shared/theme.css.
const SECTION_TINT = {
  "LLM": "cyan",
  "Plugins": "magenta",
  "Idle": "muted",
  "Fatigue": "yellow",
  "Graph Memory": "magenta",
  "Knowledge Base": "green",
  "Sleep": "muted",
  "Safety": "red",
  "Environments": "cyan",
};

// Titles whose body is currently collapsed. Module-scoped so the
// state survives renderSettingsForm() rebuilds within a session.
let collapsedSections = new Set();

function _svgFromBiHtml(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  return tpl.content.firstElementChild;
}

function makeSection(title) {
  const sec = document.createElement("div");
  sec.className = "cfg-section";
  const tint = SECTION_TINT[title];
  if (tint) sec.classList.add("tint-" + tint);
  const isCollapsed = collapsedSections.has(title);
  if (isCollapsed) sec.classList.add("collapsed");

  const h = document.createElement("h3");
  // Title carried as a data attribute so the delegated click handler
  // on #settings-form (see _wireSectionToggle below) can identify
  // which section was clicked without needing per-section closures.
  h.setAttribute("data-section-title", title);

  // Insert the icon SVG directly as a flex child so the h3's
  // align-items:center positions it on the same cross-axis line as
  // the title text and the trailing caret. Wrapping the SVG in a
  // span added an extra layout layer where the SVG and the text
  // ended up vertically offset by ~1-2px in some browsers.
  const iconName = SECTION_ICONS[title];
  if (iconName && window.biIcon) {
    const svg = _svgFromBiHtml(window.biIcon(iconName, 14));
    if (svg) h.appendChild(svg);
  }

  const titleSpan = document.createElement("span");
  titleSpan.className = "section-title";
  titleSpan.textContent = title;
  h.appendChild(titleSpan);

  if (window.biIcon) {
    const caret = _svgFromBiHtml(
      window.biIcon(isCollapsed ? "chevron-right" : "chevron-down", 12),
    );
    if (caret) {
      caret.classList.add("section-caret");
      h.appendChild(caret);
    }
  }

  const body = document.createElement("div");
  body.className = "body";
  sec.appendChild(h); sec.appendChild(body);
  return sec;
}

// One-shot delegated click handler for cfg-section h3s. Bound once
// to #settings-form, which keeps its identity across the
// `settingsForm.innerHTML = ""` wipes that renderSettingsForm()
// performs every render. Per-h3 addEventListener calls were
// dropping clicks intermittently — likely a browser-specific quirk
// where clicks on the inline <svg> child weren't bubbling reliably
// to the h3. Delegation sidesteps the bubble path entirely.
let _sectionToggleWired = false;
function _wireSectionToggle() {
  if (_sectionToggleWired) return;
  _sectionToggleWired = true;
  settingsForm.addEventListener("click", (ev) => {
    const target = ev.target;
    if (!target || !target.closest) return;
    const h = target.closest("h3[data-section-title]");
    if (!h || !settingsForm.contains(h)) return;
    const title = h.getAttribute("data-section-title");
    if (collapsedSections.has(title)) collapsedSections.delete(title);
    else collapsedSections.add(title);
    renderSettingsForm();
  });
}
_wireSectionToggle();

function renderGenericSection(key, title, target, schema) {
  const sec = makeSection(title);
  const body = sec.querySelector(".body");
  for (const [field, type] of schema) {
    body.appendChild(renderRow(field, target, field, type, `${key}.${field}`));
  }
  return sec;
}

// Generic chip-list editor for ``string[]`` config values. Each
// existing entry is a removable chip; the trailing input adds new
// entries on Enter / blur. Used for ``allowed_plugins`` and
// ``allowlist_domains`` under Environments.
function _renderStringList(arr, opts) {
  if (!Array.isArray(arr)) {
    // Caller passed a non-array; coerce to empty so the UI still
    // renders rather than throwing on .filter() / .push().
    arr = [];
  }
  const wrap = document.createElement("div");
  wrap.className = "cap-multi";  // re-use existing chip-strip styling
  function repaint() {
    wrap.innerHTML = "";
    for (const v of arr) {
      const chip = document.createElement("span");
      chip.className = "cap-chip";
      chip.appendChild(document.createTextNode(v));
      const x = document.createElement("span");
      x.className = "x";
      x.textContent = "×";
      x.addEventListener("click", () => {
        const idx = arr.indexOf(v);
        if (idx !== -1) arr.splice(idx, 1);
        repaint();
      });
      chip.appendChild(x);
      wrap.appendChild(chip);
    }
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = (opts && opts.placeholder) || "+ add…";
    input.style.cssText =
      "border:none;background:transparent;color:var(--text);" +
      "font-family:inherit;font-size:11px;flex:1;min-width:80px;outline:none";
    function commit() {
      const v = input.value.trim();
      if (!v) return;
      if (!arr.includes(v)) arr.push(v);
      input.value = "";
      repaint();
      // Re-focus the new input so the user can keep typing names.
      const newInput = wrap.querySelector("input");
      if (newInput) newInput.focus();
    }
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); commit(); }
    });
    input.addEventListener("blur", commit);
    wrap.appendChild(input);
  }
  repaint();
  return wrap;
}

function _renderListRow(label, arr, helpPath, opts) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = label;
  if (helpPath && HELP[helpPath]) lab.title = HELP[helpPath];
  row.appendChild(lab);
  row.appendChild(_renderStringList(arr, opts));
  return row;
}

function renderEnvironmentsSection(envs) {
  // Top-level `environments:` block. Maps to runtime's
  // krakey.models.config.environments.EnvironmentsSection — `local`
  // is always-on with just an allow-list; `sandbox` is optional and
  // gated by an enable toggle.
  const sec = makeSection("Environments");
  const body = sec.querySelector(".body");

  // Local sub-block.
  if (!envs.local) envs.local = { allowed_plugins: [] };
  if (!Array.isArray(envs.local.allowed_plugins)) {
    envs.local.allowed_plugins = [];
  }
  const localBlock = document.createElement("div");
  localBlock.className = "subblock";
  const localH = document.createElement("h4");
  localH.appendChild(document.createTextNode("Local"));
  localBlock.appendChild(localH);
  localBlock.appendChild(_renderListRow(
    "allowed_plugins", envs.local.allowed_plugins,
    "environments.local.allowed_plugins",
    { placeholder: "plugin name + Enter" },
  ));
  body.appendChild(localBlock);

  // Sandbox sub-block — togglable. When the toggle is off, the
  // saved config has ``environments.sandbox: null`` (runtime does
  // not register the sandbox env at all). Flipping it on hydrates
  // a sandbox object with sensible defaults so all sub-rows
  // render and the user can edit in place.
  const sbBlock = document.createElement("div");
  sbBlock.className = "subblock";
  const sbHead = document.createElement("h4");
  sbHead.style.display = "flex";
  sbHead.style.alignItems = "center";
  sbHead.style.gap = "8px";
  sbHead.appendChild(document.createTextNode("Sandbox VM"));
  const enabled = envs.sandbox != null;
  const toggle = document.createElement("span");
  toggle.className = "toggle" + (enabled ? " on" : "");
  toggle.title = "register a sandbox execution environment";
  toggle.addEventListener("click", () => {
    if (envs.sandbox == null) {
      envs.sandbox = {
        allowed_plugins: [],
        guest_os: "",
        provider: "qemu",
        vm_name: "",
        display: "headed",
        resources: { cpu: 2, memory_mb: 4096, disk_gb: 40 },
        agent: { url: "", token: "" },
        network_mode: "nat_allowlist",
        allowlist_domains: [],
      };
    } else {
      envs.sandbox = null;
    }
    renderSettingsForm();
  });
  sbHead.appendChild(toggle);
  sbBlock.appendChild(sbHead);

  if (envs.sandbox != null) {
    const sb = envs.sandbox;
    if (!sb.resources) sb.resources = { cpu: 2, memory_mb: 4096, disk_gb: 40 };
    if (!sb.agent) sb.agent = { url: "", token: "" };
    if (!Array.isArray(sb.allowed_plugins)) sb.allowed_plugins = [];
    if (!Array.isArray(sb.allowlist_domains)) sb.allowlist_domains = [];

    sbBlock.appendChild(_renderListRow(
      "allowed_plugins", sb.allowed_plugins,
      "environments.sandbox.allowed_plugins",
      { placeholder: "plugin name + Enter" },
    ));
    for (const [f, t] of SCHEMAS.env_sandbox_scalars) {
      sbBlock.appendChild(renderRow(
        f, sb, f, t, `environments.sandbox.${f}`,
      ));
    }
    sbBlock.appendChild(_renderListRow(
      "allowlist_domains", sb.allowlist_domains,
      "environments.sandbox.allowlist_domains",
      { placeholder: "domain + Enter" },
    ));

    const resBlock = document.createElement("div");
    resBlock.className = "subblock";
    const resH = document.createElement("h4");
    resH.textContent = "resources";
    resBlock.appendChild(resH);
    for (const [f, t] of SCHEMAS.env_sandbox_resources) {
      resBlock.appendChild(renderRow(
        f, sb.resources, f, t, `environments.sandbox.resources.${f}`,
      ));
    }
    sbBlock.appendChild(resBlock);

    const agentBlock = document.createElement("div");
    agentBlock.className = "subblock";
    const agH = document.createElement("h4");
    agH.textContent = "agent";
    agentBlock.appendChild(agH);
    for (const [f, t] of SCHEMAS.env_sandbox_agent) {
      agentBlock.appendChild(renderRow(
        f, sb.agent, f, t, `environments.sandbox.agent.${f}`,
      ));
    }
    sbBlock.appendChild(agentBlock);
  }
  body.appendChild(sbBlock);

  return sec;
}

const SAFETY_CONFIRMS = {};

function renderRow(label, target, key, type, helpPath) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = label;
  if (helpPath && HELP[helpPath]) {
    lab.title = HELP[helpPath];
  }
  row.appendChild(lab);

  let widget;
  const val = target[key];
  if (type === "bool") {
    widget = document.createElement("span");
    widget.className = "toggle" + (val ? " on" : "");
    if (helpPath && HELP[helpPath]) widget.title = HELP[helpPath];
    widget.addEventListener("click", () => {
      const wasOn = !!target[key];
      const willBeOn = !wasOn;
      if (wasOn && !willBeOn && helpPath && SAFETY_CONFIRMS[helpPath]) {
        if (!confirm(SAFETY_CONFIRMS[helpPath])) return;
      }
      target[key] = willBeOn;
      widget.classList.toggle("on", willBeOn);
    });
  } else if (type === "number" || type === "number_float") {
    widget = document.createElement("input");
    widget.type = "number";
    if (type === "number_float") widget.step = "any";
    widget.value = val == null ? "" : val;
    widget.addEventListener("input", () => {
      const v = widget.value;
      if (v === "") { delete target[key]; return; }
      target[key] = type === "number_float" ? parseFloat(v) : parseInt(v, 10);
    });
  } else if (type === "password") {
    widget = document.createElement("input");
    widget.type = "password";
    widget.value = val == null ? "" : val;
    widget.addEventListener("input", () => { target[key] = widget.value; });
  } else if (type === "enum") {
    // Dropdown whose option list is passed via the row's 5th arg
    // (choices). Handled by renderEnumRow — renderRow itself falls
    // back to text if no choices are wired up.
    widget = document.createElement("input");
    widget.type = "text";
    widget.value = val == null ? "" : val;
    widget.addEventListener("input", () => { target[key] = widget.value; });
  } else if (type === "list") {
    // Comma-separated list for stop_sequences / retry_on_status.
    // Parsed back into array on save; empty → undefined (drops field).
    widget = document.createElement("input");
    widget.type = "text";
    widget.placeholder = "comma, separated, values";
    widget.value = Array.isArray(val) ? val.join(", ") : (val == null ? "" : val);
    widget.addEventListener("input", () => {
      const v = widget.value.trim();
      if (v === "") { delete target[key]; return; }
      const parts = v.split(",").map(s => s.trim()).filter(s => s);
      // If every part parses as an integer, store as ints (retry codes
      // are ints in the dataclass). Otherwise keep strings.
      const allInts = parts.every(s => /^-?\d+$/.test(s));
      target[key] = allInts ? parts.map(s => parseInt(s, 10)) : parts;
    });
  } else {
    widget = document.createElement("input");
    widget.type = "text";
    widget.value = val == null ? "" : val;
    widget.addEventListener("input", () => { target[key] = widget.value; });
  }
  if (helpPath && HELP[helpPath] && type !== "bool") widget.title = HELP[helpPath];
  row.appendChild(widget);
  return row;
}

// Enum row — dropdown <select> with a given choice list. Used for
// reasoning_mode / response_format. Kept separate from the generic
// renderRow so the <select> can be populated without hacking the
// main dispatch table.
function renderEnumRow(label, target, key, choices, helpPath) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = label;
  if (helpPath && HELP[helpPath]) lab.title = HELP[helpPath];
  row.appendChild(lab);
  const sel = document.createElement("select");
  for (const c of choices) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c === "" ? "(unset)" : c;
    sel.appendChild(opt);
  }
  const cur = target[key];
  sel.value = cur == null ? "" : cur;
  sel.addEventListener("change", () => {
    if (sel.value === "") {
      delete target[key];
    } else {
      target[key] = sel.value;
    }
  });
  if (helpPath && HELP[helpPath]) sel.title = HELP[helpPath];
  row.appendChild(sel);
  return row;
}

function renderFatigueThresholds(fatigue) {
  const block = document.createElement("div");
  block.className = "subblock";
  const h = document.createElement("h4");
  h.appendChild(document.createTextNode("thresholds (% → hint)"));
  const actions = document.createElement("span");
  actions.className = "actions";
  const addBtn = mkBtn("+ add", () => {
    let key = 0;
    while (key in fatigue.thresholds) key += 25;
    fatigue.thresholds[key] = "";
    redraw();
  });
  actions.appendChild(addBtn);
  h.appendChild(actions);
  block.appendChild(h);

  function redraw() {
    [...block.querySelectorAll(".cfg-row")].forEach((r) => r.remove());
    for (const k of Object.keys(fatigue.thresholds).sort((a, b) => +a - +b)) {
      const row = document.createElement("div");
      row.className = "cfg-row";
      const keyIn = document.createElement("input");
      keyIn.type = "number"; keyIn.value = k; keyIn.style.maxWidth = "80px";
      const valIn = document.createElement("input");
      valIn.type = "text"; valIn.value = fatigue.thresholds[k];
      const del = mkBtn("×", () => { delete fatigue.thresholds[k]; redraw(); }, "btn-x");
      keyIn.addEventListener("change", () => {
        const newK = parseInt(keyIn.value, 10);
        if (Number.isNaN(newK) || String(newK) === k) return;
        fatigue.thresholds[newK] = fatigue.thresholds[k];
        delete fatigue.thresholds[k];
        redraw();
      });
      valIn.addEventListener("input", () => { fatigue.thresholds[k] = valIn.value; });
      const wrap = document.createElement("div");
      wrap.style.display = "grid";
      wrap.style.gridTemplateColumns = "80px 1fr auto";
      wrap.style.gap = "6px";
      wrap.appendChild(keyIn); wrap.appendChild(valIn); wrap.appendChild(del);
      row.appendChild(document.createElement("label"));
      row.appendChild(wrap);
      block.appendChild(row);
    }
  }
  redraw();
  return block;
}

function mkBtn(text, onClick, cls = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn-mini" + (cls ? " " + cls : "");
  b.textContent = text;
  b.addEventListener("click", onClick);
  return b;
}

// ---------------- LLM section ----------------

// Pending inline-rename target. Set by "+ add provider/tag/purpose"
// to {scope, name}; the matching editable-key span auto-switches to
// input mode at render time so the user types the real name in
// place instead of seeing a prompt() dialog. Cleared after the
// auto-edit fires so it only triggers once.
let _pendingNewKey = null;

// Build a span that displays a key (provider name, tag name, etc.)
// and turns into an inline <input> when clicked. On commit (Enter or
// blur) it calls onRename(newName); validate(newName) returning a
// truthy string blocks commit and surfaces the message.
function _renderEditableKey(currentName, opts) {
  const span = document.createElement("span");
  span.className = "editable-key";
  span.textContent = currentName;
  span.title = "click to rename";

  function _enterEdit() {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "editable-key-input";
    input.value = currentName;
    if (opts.placeholder) input.placeholder = opts.placeholder;
    let committing = false;
    function _commit() {
      if (committing) return;
      committing = true;
      const nv = (input.value || "").trim();
      if (!nv || nv === currentName) { _revert(); return; }
      const err = opts.validate ? opts.validate(nv) : null;
      if (err) { alert(err); committing = false; input.focus(); return; }
      opts.onRename(nv);
    }
    function _revert() {
      span.textContent = currentName;
      if (input.parentNode) input.replaceWith(span);
    }
    input.addEventListener("blur", _commit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); _commit(); }
      else if (e.key === "Escape") { e.preventDefault(); _revert(); }
    });
    span.replaceWith(input);
    input.focus();
    input.select();
  }
  span.addEventListener("click", _enterEdit);

  // If this span is the freshly-added "+ add ___" target, auto-enter
  // edit mode after the DOM is in place (focus needs the element to
  // be attached). The flag is consumed so a subsequent re-render
  // doesn't re-trigger.
  if (_pendingNewKey && _pendingNewKey.scope === opts.scope &&
      _pendingNewKey.name === currentName) {
    _pendingNewKey = null;
    setTimeout(_enterEdit, 0);
  }
  return span;
}

// Find a unique placeholder name for a "+ add" action — keeps the
// suffix incrementing until the dict has no entry with that key.
function _uniqueDraftName(dict, base) {
  let n = 1;
  while (dict[`${base}_${n}`]) n++;
  return `${base}_${n}`;
}

// Shape (post tag-based refactor 2026-04-26):
//   llm.providers     : dict of provider connections (with API keys)
//   llm.tags          : dict of named (provider/model + params)
//   llm.core_purposes : dict purpose_name → tag_name
//   llm.embedding     : tag_name (string) — required for vec_search
//   llm.reranker      : tag_name (string, optional)
function renderLLMSection(llm) {
  // Migration nudge: old `llm.roles:` shape would still be present
  // in cfgState if a user opens the page on a deprecated config —
  // the loader will reject it on next restart, but until then we
  // hide the field to avoid editing a dead structure.
  if (llm && "roles" in llm && !("tags" in llm)) {
    llm.tags = llm.tags || {};
    llm.core_purposes = llm.core_purposes || {};
  }

  const sec = makeSection("LLM");
  const body = sec.querySelector(".body");

  // Providers
  const provHead = document.createElement("h4");
  provHead.style.cssText = "color:var(--text);font-weight:bold;font-size:11px;margin:0 0 6px";
  provHead.appendChild(document.createTextNode("Providers"));
  const addProv = mkBtn("+ add provider", () => {
    // Add directly with a placeholder name; the heading enters
    // edit-mode on render so the user types the real name inline.
    const name = _uniqueDraftName(llm.providers, "provider");
    llm.providers[name] = {
      type: "openai_compatible", base_url: "", api_key: "", models: [],
    };
    _pendingNewKey = { scope: "provider", name };
    renderSettingsForm();
  });
  const headWrap = document.createElement("div");
  headWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:6px";
  headWrap.appendChild(provHead); headWrap.appendChild(addProv);
  body.appendChild(headWrap);

  for (const [pname, prov] of Object.entries(llm.providers || {})) {
    body.appendChild(renderProviderBlock(pname, prov, llm));
  }

  // Tags
  llm.tags = llm.tags || {};
  const tagsHead = document.createElement("h4");
  tagsHead.style.cssText = "color:var(--text);font-weight:bold;font-size:11px;margin:12px 0 6px";
  tagsHead.appendChild(document.createTextNode("Tags"));
  const addTag = mkBtn("+ add tag", () => {
    const provNames = Object.keys(llm.providers || {});
    if (!provNames.length) { alert("add a provider first"); return; }
    const firstProv = llm.providers[provNames[0]];
    const firstModel = (firstProv.models && firstProv.models[0]
                          && firstProv.models[0].name) || "";
    const name = _uniqueDraftName(llm.tags, "tag");
    llm.tags[name] = {
      provider: `${provNames[0]}/${firstModel}`,
      params: {},
    };
    _pendingNewKey = { scope: "tag", name };
    renderSettingsForm();
  });
  const tagsHeadWrap = document.createElement("div");
  tagsHeadWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:12px 0 6px";
  tagsHeadWrap.appendChild(tagsHead); tagsHeadWrap.appendChild(addTag);
  body.appendChild(tagsHeadWrap);

  for (const tname of Object.keys(llm.tags)) {
    body.appendChild(renderTagRow(tname, llm.tags, llm.providers));
  }

  // Core purposes (chat use cases — Self / compact / classifier ...)
  body.appendChild(renderCorePurposesBlock(llm));
  // Plugin LLM purposes — aggregate of every enabled plugin's
  // ``llm_purposes`` declarations (per meta.yaml). Renders here so
  // the user has a single "all the components that can take an
  // LLM" pane in the LLM section instead of having to hunt through
  // expanded plugin rows in the Plugins panel.
  body.appendChild(renderPluginPurposesBlock(llm));
  // Embedding + reranker (model-type slots, not purposes)
  body.appendChild(renderModelSlotBlock(
    llm, "embedding",
    "GM auto-recall + vec_search use this. Required.",
  ));
  body.appendChild(renderModelSlotBlock(
    llm, "reranker",
    "Optional — leave empty to skip reranking in recall.",
  ));

  return sec;
}

function renderProviderBlock(pname, prov, llm) {
  const block = document.createElement("div");
  block.className = "subblock";
  const h = document.createElement("h4");
  // Inline-editable name. Click the heading → it turns into an
  // input. Replaces the old "rename" prompt() flow.
  h.appendChild(_renderEditableKey(pname, {
    scope: "provider",
    placeholder: "provider name",
    validate: (nv) => llm.providers[nv] ? "name exists" : null,
    onRename: (nv) => {
      llm.providers[nv] = llm.providers[pname];
      delete llm.providers[pname];
      // Cascade: tags whose provider field starts with "<pname>/"
      // need their prefix substituted so they keep pointing at the
      // same provider object.
      const prefix = pname + "/";
      for (const t of Object.values(llm.tags || {})) {
        if (typeof t.provider === "string" && t.provider.startsWith(prefix)) {
          t.provider = nv + "/" + t.provider.slice(prefix.length);
        }
      }
      renderSettingsForm();
    },
  }));
  const actions = document.createElement("span");
  actions.className = "actions";
  const delBtn = mkBtn("delete", () => {
    if (!confirm(`delete provider "${pname}"?`)) return;
    delete llm.providers[pname];
    renderSettingsForm();
  }, "danger");
  actions.appendChild(delBtn);
  h.appendChild(actions);
  block.appendChild(h);

  block.appendChild(renderRow("type", prov, "type", "text", "provider.type"));
  block.appendChild(renderRow("base_url", prov, "base_url", "text", "provider.base_url"));
  block.appendChild(renderRow("api_key", prov, "api_key", "password", "provider.api_key"));

  // Models list
  const modBlock = document.createElement("div");
  modBlock.style.cssText = "margin-top:6px";
  const modHead = document.createElement("div");
  modHead.style.cssText = "display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);margin-bottom:4px";
  modHead.appendChild(document.createTextNode("models"));
  const addModel = mkBtn("+ add model", () => {
    if (!Array.isArray(prov.models)) prov.models = [];
    prov.models.push({ name: "", capabilities: ["chat"] });
    renderSettingsForm();
  });
  modHead.appendChild(addModel);
  modBlock.appendChild(modHead);
  for (let i = 0; i < (prov.models || []).length; i++) {
    modBlock.appendChild(renderModelRow(prov, i));
  }
  block.appendChild(modBlock);
  return block;
}

const KNOWN_CAPABILITIES = ["chat", "embedding", "rerank", "vision", "tool_use"];

function renderModelRow(prov, idx) {
  const m = prov.models[idx];
  const row = document.createElement("div");
  row.className = "model-row";
  const nameIn = document.createElement("input");
  nameIn.type = "text"; nameIn.value = m.name || "";
  nameIn.placeholder = "model name";
  nameIn.addEventListener("input", () => { m.name = nameIn.value; });
  row.appendChild(nameIn);
  row.appendChild(renderCapabilitiesMulti(m));
  const del = mkBtn("×", () => { prov.models.splice(idx, 1); renderSettingsForm(); }, "btn-x");
  row.appendChild(del);
  return row;
}

function renderCapabilitiesMulti(model) {
  if (!Array.isArray(model.capabilities)) model.capabilities = [];
  const wrap = document.createElement("div");
  wrap.className = "cap-multi";

  function repaint() {
    wrap.innerHTML = "";
    for (const cap of model.capabilities) {
      const chip = document.createElement("span");
      chip.className = "cap-chip";
      chip.appendChild(document.createTextNode(cap));
      const x = document.createElement("span");
      x.className = "x"; x.textContent = "×";
      x.addEventListener("click", () => {
        model.capabilities = model.capabilities.filter((c) => c !== cap);
        repaint();
      });
      chip.appendChild(x);
      wrap.appendChild(chip);
    }
    const taken = new Set(model.capabilities);
    const items = KNOWN_CAPABILITIES
      .filter((c) => !taken.has(c))
      .map((c) => ({ value: c, label: c }));
    items.push({ value: "__custom__", label: "+ custom…", custom: true });
    wrap.appendChild(mkDropdown("+ add…", items, (v) => {
      let chosen = v;
      if (chosen === "__custom__") {
        chosen = (prompt("custom capability:") || "").trim();
        if (!chosen || taken.has(chosen)) return;
      }
      model.capabilities.push(chosen);
      repaint();
    }));
  }
  repaint();
  return wrap;
}

// ---------------- Custom dropdown widget ----------------

let _ddOpen = null;
document.addEventListener("click", (ev) => {
  if (_ddOpen && !_ddOpen.contains(ev.target)) {
    _ddOpen.querySelector(".dd-menu").classList.add("hidden");
    _ddOpen = null;
  }
});

function mkDropdown(triggerLabel, items, onPick) {
  const dd = document.createElement("div");
  dd.className = "dd";
  const trig = document.createElement("button");
  trig.type = "button"; trig.className = "dd-trigger"; trig.textContent = triggerLabel;
  const menu = document.createElement("div");
  menu.className = "dd-menu hidden";
  for (const it of items) {
    const el = document.createElement("div");
    el.className = "dd-item" + (it.custom ? " custom" : "");
    el.textContent = it.label;
    el.addEventListener("click", () => {
      menu.classList.add("hidden"); _ddOpen = null;
      onPick(it.value);
    });
    menu.appendChild(el);
  }
  trig.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (_ddOpen && _ddOpen !== dd) {
      _ddOpen.querySelector(".dd-menu").classList.add("hidden");
    }
    menu.classList.toggle("hidden");
    _ddOpen = menu.classList.contains("hidden") ? null : dd;
  });
  dd.appendChild(trig); dd.appendChild(menu);
  return dd;
}

function renderTagRow(tname, tags, providers) {
  // Wrapper so the params <details> sits directly below the tag row
  // (sharing a container keeps delete/rename behavior correct).
  const container = document.createElement("div");

  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  // Inline-editable tag name. Click → input → commit on blur/Enter.
  lab.appendChild(_renderEditableKey(tname, {
    scope: "tag",
    placeholder: "tag name",
    validate: (nv) => tags[nv] ? "name exists" : null,
    onRename: (nv) => {
      tags[nv] = tags[tname];
      delete tags[tname];
      // Cascade: any core_purpose / embedding / reranker that
      // referenced the old name needs to follow.
      const llm = cfgState.llm || {};
      for (const [purp, t] of Object.entries(llm.core_purposes || {})) {
        if (t === tname) llm.core_purposes[purp] = nv;
      }
      if (llm.embedding === tname) llm.embedding = nv;
      if (llm.reranker === tname) llm.reranker = nv;
      renderSettingsForm();
    },
  }));
  row.appendChild(lab);

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:grid;grid-template-columns:1fr 1fr auto;gap:6px";

  // Tag's `provider:` field is "<provider>/<model>". Render two
  // dropdowns; serialize on change.
  const tag = tags[tname];
  function splitProviderField() {
    const v = tag.provider || "";
    const idx = v.indexOf("/");
    if (idx < 0) return ["", ""];
    return [v.slice(0, idx), v.slice(idx + 1)];
  }
  let [provName, modelName] = splitProviderField();

  const provSel = document.createElement("select");
  for (const pname of Object.keys(providers || {})) {
    const opt = document.createElement("option");
    opt.value = pname; opt.textContent = pname;
    provSel.appendChild(opt);
  }
  if (provName) provSel.value = provName;

  const modSel = document.createElement("select");
  function refreshModels() {
    modSel.innerHTML = "";
    const prov = providers[provSel.value];
    const models = (prov && prov.models) || [];
    if (!models.length) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = "(no models)"; modSel.appendChild(opt);
    } else {
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.name; opt.textContent = m.name;
        modSel.appendChild(opt);
      }
    }
    // Preserve existing model name if it's still valid for this provider
    if (modelName && [...modSel.options].some(o => o.value === modelName)) {
      modSel.value = modelName;
    } else {
      modelName = modSel.value;
    }
    tag.provider = `${provSel.value}/${modelName}`;
  }
  provSel.addEventListener("change", () => {
    provName = provSel.value;
    refreshModels();
  });
  modSel.addEventListener("change", () => {
    modelName = modSel.value;
    tag.provider = `${provSel.value}/${modelName}`;
  });
  refreshModels();

  const del = mkBtn("×", () => {
    if (!confirm(`delete tag "${tname}"?`)) return;
    delete tags[tname];
    // Also clear any core_purpose mapping that referenced this tag —
    // leaving a stale tag name behind would silently break runtime.
    const llm = cfgState.llm || {};
    for (const [purp, t] of Object.entries(llm.core_purposes || {})) {
      if (t === tname) delete llm.core_purposes[purp];
    }
    if (llm.embedding === tname) llm.embedding = "";
    if (llm.reranker === tname) llm.reranker = "";
    renderSettingsForm();
  }, "btn-x");

  wrap.appendChild(provSel); wrap.appendChild(modSel); wrap.appendChild(del);
  row.appendChild(wrap);
  container.appendChild(row);

  // Params sub-block (driven by /api/config/schema). Collapsed by
  // default so the LLM section stays scannable; user opens only the
  // tag they're tuning.
  container.appendChild(renderTagParamsBlock(tname, tags));
  return container;
}


// Tag → params editor (collapsed <details>). Each LLMParams field
// is rendered using the schema descriptors served by /api/config/schema.
function renderTagParamsBlock(tname, tags) {
  const details = document.createElement("details");
  details.className = "tag-params";
  details.style.cssText = "margin:4px 0 12px 0;padding:6px 10px;" +
    "border:1px solid var(--border,#e2e8f0);border-radius:4px;" +
    "background:rgba(0,0,0,0.015)";
  const summary = document.createElement("summary");
  summary.style.cssText = "cursor:pointer;font-size:11px;color:var(--muted);" +
    "user-select:none";
  summary.textContent = `params (${tname})`;
  details.appendChild(summary);

  if (tags[tname].params == null) tags[tname].params = {};
  const target = tags[tname].params;

  const body = document.createElement("div");
  body.style.cssText = "padding:4px 0";
  const schema = configSchema.llm_params || [];
  for (const fdef of schema) {
    const helpPath = `llm.tag.${tname}.params.${fdef.field}`;
    if (fdef.help) HELP[helpPath] = fdef.help;
    let r;
    if (fdef.type === "enum" && fdef.choices) {
      r = renderEnumRow(fdef.field, target, fdef.field, fdef.choices,
                         helpPath);
    } else {
      r = renderRow(fdef.field, target, fdef.field, fdef.type, helpPath);
    }
    body.appendChild(r);
  }
  details.appendChild(body);
  return details;
}


// Render the core_purposes mapping as `purpose: tag` rows. Users
// can add custom purposes (e.g. for future Modifiers), but the
// well-known core purposes (self_thinking required; compact /
// classifier optional) are always shown so people know they exist.
const KNOWN_CORE_PURPOSES = [
  ["self_thinking", "required — Self's per-beat heartbeat LLM"],
  ["compact", "sliding-window history → GM compaction LLM (also drives sleep clustering + KB index rebuild)"],
  ["classifier", "node category classifier (extractor + classifier; falls back to compact then self_thinking)"],
  ["hypothalamus", "legacy compat — modern setups bind translator via the Hypothalamus plugin's per-plugin llm_purposes instead. Leave (unbound) unless you need the deprecated central path."],
];

function renderCorePurposesBlock(llm) {
  llm.core_purposes = llm.core_purposes || {};
  const sub = document.createElement("div");
  sub.className = "subblock";
  const head = document.createElement("h4");
  head.appendChild(document.createTextNode("Core Purposes"));
  sub.appendChild(head);

  const tagNames = Object.keys(llm.tags || {});

  // Always show the known purposes (with help) — even if not yet bound.
  const seen = new Set();
  for (const [purp, helpText] of KNOWN_CORE_PURPOSES) {
    seen.add(purp);
    sub.appendChild(_purposeRow(llm, purp, tagNames, helpText));
  }
  // Then any user-added purposes that aren't in the well-known set
  for (const purp of Object.keys(llm.core_purposes)) {
    if (seen.has(purp)) continue;
    sub.appendChild(_purposeRow(llm, purp, tagNames, ""));
  }

  const addBtn = mkBtn("+ add purpose", () => {
    const name = _uniqueDraftName(llm.core_purposes, "purpose");
    llm.core_purposes[name] = "";
    _pendingNewKey = { scope: "purpose", name };
    renderSettingsForm();
  });
  sub.appendChild(addBtn);
  return sub;
}

// Aggregated view of every enabled plugin's ``llm_purposes`` (from
// each plugin's meta.yaml). One row per "<plugin>.<purpose>" with a
// tag-binding dropdown. Edits feed straight into the per-plugin
// config (``modifierConfigEdits[plugin].llm_purposes[purpose]``) —
// the same backing store the per-plugin row in the Plugins panel
// uses, so both views stay in sync without a separate save path.
//
// Rendered in the LLM section so users have a single "what
// component uses what LLM?" pane; a long-standing complaint was
// that these knobs were buried inside expanded plugin rows.
function renderPluginPurposesBlock(llm) {
  const sub = document.createElement("div");
  sub.className = "subblock";
  const head = document.createElement("h4");
  head.appendChild(document.createTextNode("Plugin Purposes"));
  sub.appendChild(head);

  // Filter to enabled plugins that actually declare an LLM purpose;
  // empty list ⇒ render a placeholder explaining what would show up.
  const enabledNames = new Set([
    ...((cfgState.modifiers || [])),
    ...((cfgState.plugins || [])),
  ]);
  const candidates = (availableModifiers || []).filter((p) =>
    enabledNames.has(p.name)
    && Array.isArray(p.llm_purposes) && p.llm_purposes.length,
  );

  if (!candidates.length) {
    const hint = document.createElement("p");
    hint.className = "section-hint";
    hint.style.borderLeftColor = "var(--muted)";
    hint.textContent =
      "no enabled plugin declares an llm_purposes block. Plugins "
      + "that need an LLM (e.g. hypothalamus.translator) surface "
      + "their bindings here once enabled.";
    sub.appendChild(hint);
    return sub;
  }

  const tagNames = Object.keys(llm.tags || {});
  for (const plugin of candidates) {
    // Lazy-load the plugin's per-folder config the first time we
    // need it. The per-plugin row uses the same cache, so editing
    // here updates the row's view too on the next render.
    if (!modifierConfigEdits[plugin.name]) {
      modifierConfigEdits[plugin.name] = {};
      fetch(`/api/modifiers/${encodeURIComponent(plugin.name)}/config`)
        .then((r) => (r.ok ? r.json() : { config: {} }))
        .then((body) => {
          modifierConfigEdits[plugin.name] = body.config || {};
          renderSettingsForm();
        })
        .catch(() => {});
    }
    const cfg = modifierConfigEdits[plugin.name];
    cfg.llm_purposes = cfg.llm_purposes || {};

    for (const purpose of plugin.llm_purposes) {
      const row = document.createElement("div");
      row.className = "cfg-row";
      const lab = document.createElement("label");
      lab.textContent = `${plugin.name}.${purpose.name}`;
      if (purpose.description) lab.title = purpose.description;
      row.appendChild(lab);

      const sel = document.createElement("select");
      const blank = document.createElement("option");
      blank.value = ""; blank.textContent = "(unbound)";
      sel.appendChild(blank);
      for (const t of tagNames) {
        const opt = document.createElement("option");
        opt.value = t; opt.textContent = t;
        sel.appendChild(opt);
      }
      sel.value = cfg.llm_purposes[purpose.name] || "";
      sel.addEventListener("change", () => {
        cfg.llm_purposes = cfg.llm_purposes || {};
        if (sel.value) cfg.llm_purposes[purpose.name] = sel.value;
        else delete cfg.llm_purposes[purpose.name];
      });
      row.appendChild(sel);
      sub.appendChild(row);
    }
  }
  return sub;
}

function _purposeRow(llm, purp, tagNames, helpText) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  // Well-known purposes (self_thinking / compact / classifier) drive
  // runtime behaviour by name — renaming would silently break
  // dispatch — so they stay as plain text. User-added purposes are
  // free-form labels and get inline-rename like providers + tags.
  const isKnown = KNOWN_CORE_PURPOSES.some(([p]) => p === purp);
  if (isKnown) {
    lab.textContent = purp;
  } else {
    lab.appendChild(_renderEditableKey(purp, {
      scope: "purpose",
      placeholder: "purpose name",
      validate: (nv) => llm.core_purposes[nv] != null ? "name exists" : null,
      onRename: (nv) => {
        llm.core_purposes[nv] = llm.core_purposes[purp];
        delete llm.core_purposes[purp];
        renderSettingsForm();
      },
    }));
  }
  if (helpText) lab.title = helpText;
  row.appendChild(lab);

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:6px";

  const sel = document.createElement("select");
  const blank = document.createElement("option");
  blank.value = ""; blank.textContent = "(unbound)";
  sel.appendChild(blank);
  for (const t of tagNames) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  }
  sel.value = llm.core_purposes[purp] || "";
  sel.addEventListener("change", () => {
    if (sel.value) llm.core_purposes[purp] = sel.value;
    else delete llm.core_purposes[purp];
  });
  wrap.appendChild(sel);

  // Custom (non-known) purposes get a delete button; well-known ones
  // are persistent. (`isKnown` was already computed above to gate
  // the editable-name path — reuse instead of redeclaring.)
  if (!isKnown) {
    const del = mkBtn("×", () => {
      delete llm.core_purposes[purp]; renderSettingsForm();
    }, "btn-x");
    wrap.appendChild(del);
  } else {
    wrap.appendChild(document.createElement("span"));
  }

  row.appendChild(wrap);
  return row;
}


// embedding / reranker are model-type slots — single tag name.
function renderModelSlotBlock(llm, fieldName, helpText) {
  const sub = document.createElement("div");
  sub.className = "subblock";
  const head = document.createElement("h4");
  head.appendChild(document.createTextNode(fieldName));
  sub.appendChild(head);

  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = "tag";
  if (helpText) lab.title = helpText;
  row.appendChild(lab);

  const sel = document.createElement("select");
  const blank = document.createElement("option");
  blank.value = ""; blank.textContent = "(unbound)";
  sel.appendChild(blank);
  for (const t of Object.keys(llm.tags || {})) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  }
  sel.value = llm[fieldName] || "";
  sel.addEventListener("change", () => {
    llm[fieldName] = sel.value || null;
  });
  row.appendChild(sel);
  sub.appendChild(row);
  return sub;
}

// (renderRoleParamsBlock removed in 2026-04-26 tag refactor — its
// replacement, renderTagParamsBlock, lives near renderTagRow above.
// Reset-to-defaults dropped because tags have no per-purpose defaults
// any more; LLMParams field defaults are the only baseline.)

// ---------------- Unified Plugins section ----------------
//
// One section, one row per plugin (modifiers + tools + channels are
// just component kinds inside a plugin's meta.yaml). Source of truth
// for the row list is the disk catalogue endpoint
// /api/modifiers/available — that endpoint returns plugin metadata
// (description + config_schema + components[] with kind, plus any
// llm_purposes), already grouped by plugin folder.
//
// Live load status (loaded ✓ / error ✗) comes from /api/plugins,
// which is the runtime-observation snapshot and is keyed by component
// project — we aggregate per plugin name for the badge.
//
// Enable/disable is owned by central config.yaml's two flat lists:
//   * cfgState.modifiers — ordered list (order = heartbeat chain
//     execution order); a plugin's name lands here when it has at
//     least one component of kind="modifier".
//   * cfgState.plugins   — set-style list; a plugin's name lands here
//     when it has at least one tool/channel component.
// A plugin with both kinds shows up in both lists.

// Plugin metadata snapshot: [{name, description, config_schema,
// components:[{kind,role}], llm_purposes}]. Rebuilt each
// loadSettings() call via /api/modifiers/available.
let availableModifiers = [];

// Per-plugin config-edit cache (mirrors workspace/plugins/<name>/
// config.yaml). Keyed by plugin name. Loaded lazily when the user
// expands a plugin row; the global #settings-save handler POSTs each
// dirty entry to /api/modifiers/<name>/config. (The endpoint name
// retains the legacy "modifiers" prefix from before the unified
// plugin model — see MEMORY.md.)
let modifierConfigEdits = {};

// Names of plugins whose row is currently expanded in the unified
// panel. Default-collapsed so the section stays scannable; the user
// expands a row to edit its config or LLM-purpose bindings. Survives
// renderSettingsForm() rebuilds (the Set is module-scoped).
// Seeded with "dashboard" so the running dashboard's host/port/
// history_path are visible at first paint — this IS the dashboard's
// own settings page, so its own row is the natural entry point.
let pluginExpanded = new Set(["dashboard"]);

async function loadAvailableModifiers() {
  try {
    const r = await fetch("/api/modifiers/available");
    if (r.ok) {
      const body = await r.json();
      availableModifiers = body.modifiers || [];
    } else {
      availableModifiers = [];
    }
  } catch (e) {
    availableModifiers = [];
  }
}

function _pluginKinds(plugin) {
  const out = new Set();
  for (const c of (plugin.components || [])) {
    if (c && c.kind) out.add(c.kind);
  }
  return out;
}

function _hasModifierKind(plugin) {
  return _pluginKinds(plugin).has("modifier");
}

function _hasServiceKind(plugin) {
  const kinds = _pluginKinds(plugin);
  return kinds.has("tool") || kinds.has("channel");
}

// Live load status aggregated per plugin name from /api/plugins
// (which lists tools + channels separately). Returns
// {loaded:boolean, error:string|null} or undefined when the plugin
// isn't loaded yet.
function _liveStatusByName() {
  const out = {};
  for (const kind of ["tools", "channels"]) {
    const items = (pluginReport && pluginReport[kind]) || [];
    for (const entry of items) {
      const proj = entry.project || entry.name;
      if (!proj) continue;
      const cur = out[proj] || { loaded: true, error: null };
      cur.loaded = cur.loaded && !!entry.loaded;
      if (entry.error && !cur.error) cur.error = entry.error;
      out[proj] = cur;
    }
  }
  return out;
}

function _setPluginEnabled(plugin, enabled) {
  cfgState.modifiers = cfgState.modifiers || [];
  cfgState.plugins = cfgState.plugins || [];
  const name = plugin.name;
  const inMods = cfgState.modifiers.includes(name);
  const inPlugs = cfgState.plugins.includes(name);
  if (enabled) {
    if (_hasModifierKind(plugin) && !inMods) {
      cfgState.modifiers.push(name);
    }
    if (_hasServiceKind(plugin) && !inPlugs) {
      cfgState.plugins.push(name);
    }
  } else {
    if (inMods) {
      cfgState.modifiers = cfgState.modifiers.filter(n => n !== name);
    }
    if (inPlugs) {
      cfgState.plugins = cfgState.plugins.filter(n => n !== name);
    }
  }
}

function _isPluginEnabled(plugin) {
  const name = plugin.name;
  const mods = cfgState.modifiers || [];
  const plugs = cfgState.plugins || [];
  return mods.includes(name) || plugs.includes(name);
}

function renderPluginsSection() {
  const sec = makeSection("Plugins");
  const body = sec.querySelector(".body");

  const intro = document.createElement("p");
  intro.className = "hint";
  intro.style.margin = "0 0 8px";
  intro.innerHTML =
    "Every plugin discovered in <code>workspace/plugins/</code> and " +
    "in-tree builtins. A plugin bundles any combination of " +
    "<em>modifiers</em> (heartbeat hooks), <em>tools</em> (action " +
    "verbs), and <em>channels</em> (stimulus sources) — one row each. " +
    "Tick to enable; expand to edit per-plugin config and bind LLM " +
    "purposes to tags. Order among modifier-bearing plugins = " +
    "heartbeat chain order.";
  body.appendChild(intro);

  cfgState.modifiers = cfgState.modifiers || [];
  cfgState.plugins = cfgState.plugins || [];

  const live = _liveStatusByName();
  const byName = new Map(availableModifiers.map(p => [p.name, p]));

  // Render order:
  //   1. enabled modifier-bearing plugins, in cfgState.modifiers order
  //      (so the user-visible reorder ↑↓ matches heartbeat chain order)
  //   2. other enabled plugins (tool/channel only), name-sorted
  //   3. disabled-but-available plugins, name-sorted
  const modifierOrder = cfgState.modifiers.slice();
  const renderedModifierPlugins = [];
  const seenInModifierBlock = new Set();
  for (const n of modifierOrder) {
    const plugin = byName.get(n);
    if (!plugin || !_hasModifierKind(plugin)) continue;
    renderedModifierPlugins.push(plugin);
    seenInModifierBlock.add(n);
  }

  const otherEnabled = availableModifiers
    .filter(p => !seenInModifierBlock.has(p.name) && _isPluginEnabled(p))
    .sort((a, b) => a.name.localeCompare(b.name));
  const disabled = availableModifiers
    .filter(p => !seenInModifierBlock.has(p.name) && !_isPluginEnabled(p))
    .sort((a, b) => a.name.localeCompare(b.name));

  for (let i = 0; i < renderedModifierPlugins.length; i++) {
    body.appendChild(_renderPluginCard(
      renderedModifierPlugins[i], true, live, i, modifierOrder.length,
    ));
  }
  for (const p of otherEnabled) {
    body.appendChild(_renderPluginCard(p, true, live, -1, 0));
  }
  if (disabled.length) {
    const head = document.createElement("h4");
    head.style.cssText = "color:var(--muted);font-size:11px;margin:12px 0 6px";
    head.textContent = "Available (disabled)";
    body.appendChild(head);
    for (const p of disabled) {
      body.appendChild(_renderPluginCard(p, false, live, -1, 0));
    }
  }

  // Orphans: enable lists that mention a plugin not on disk.
  const known = new Set(availableModifiers.map(p => p.name));
  const seenOrphan = new Set();
  const orphans = [];
  for (const n of cfgState.modifiers.concat(cfgState.plugins)) {
    if (!known.has(n) && !seenOrphan.has(n)) {
      seenOrphan.add(n);
      orphans.push(n);
    }
  }
  if (orphans.length) {
    const warn = document.createElement("div");
    warn.style.cssText = "color:var(--red);font-size:11px;margin-top:8px";
    warn.textContent =
      "Unknown plugin names in config: " + orphans.join(", ") +
      " — these will be skipped at startup with a warning.";
    body.appendChild(warn);
  }
  return sec;
}

function _renderKindBadge(kind) {
  const span = document.createElement("span");
  span.style.cssText =
    "font-size:10px;padding:1px 6px;border-radius:3px;" +
    "border:1px solid var(--border);color:var(--text);";
  span.textContent = kind;
  return span;
}

function _renderStatusBadge(status) {
  const span = document.createElement("span");
  span.style.cssText =
    "font-size:10px;padding:1px 6px;border-radius:3px;" +
    "border:1px solid var(--border);color:var(--muted);";
  if (!status) {
    span.textContent = "not loaded";
  } else if (status.error) {
    span.textContent = "error";
    span.style.fontWeight = "bold";
    span.style.color = "var(--text)";
  } else if (status.loaded) {
    span.textContent = "loaded";
    span.style.color = "var(--text)";
  } else {
    span.textContent = "pending";
  }
  return span;
}

function _renderPluginCard(plugin, enabled, liveByName, modIdx, modCount) {
  const card = document.createElement("div");
  card.className = enabled ? "plugin-card" : "plugin-card disabled";

  const isExpanded = enabled && pluginExpanded.has(plugin.name);

  const head = document.createElement("div");
  head.style.cssText =
    "display:flex;align-items:center;gap:8px;flex-wrap:wrap;user-select:none";
  // Disabled rows have no expandable body, so they get neither a
  // caret nor a click-to-expand handler — clicking the head just
  // selects the row (default browser behaviour).
  if (enabled) {
    head.style.cursor = "pointer";
    // Click anywhere on the head toggles expansion, except over inputs
    // (checkbox) and buttons (reorder ↑↓) — those have their own
    // handlers. We can't read the row at click time so we filter on
    // the event target's tag.
    head.addEventListener("click", (ev) => {
      const tag = ev.target && ev.target.tagName;
      if (tag === "INPUT" || tag === "BUTTON") return;
      if (pluginExpanded.has(plugin.name)) {
        pluginExpanded.delete(plugin.name);
      } else {
        pluginExpanded.add(plugin.name);
      }
      renderSettingsForm();
    });
    const caret = document.createElement("span");
    caret.style.cssText = "color:var(--muted);display:inline-flex;align-items:center";
    caret.innerHTML = window.biIcon(
      isExpanded ? "chevron-down" : "chevron-right", 12,
    );
    head.appendChild(caret);
  }

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = enabled;
  checkbox.addEventListener("change", () => {
    _setPluginEnabled(plugin, checkbox.checked);
    renderSettingsForm();
  });
  head.appendChild(checkbox);

  const title = document.createElement("strong");
  title.textContent = plugin.name;
  if (plugin.description) title.title = plugin.description;
  head.appendChild(title);

  for (const kind of _pluginKinds(plugin)) {
    head.appendChild(_renderKindBadge(kind));
  }

  // Live load status: only meaningful for enabled plugins (disabled
  // ones aren't expected to appear in the runtime registry).
  if (enabled) {
    head.appendChild(_renderStatusBadge(liveByName[plugin.name]));
  }

  // Reorder ↑↓ — only relevant when the plugin participates in the
  // ordered modifiers list. modIdx === -1 means "not in modifier
  // block" (tool/channel-only plugin or disabled).
  if (enabled && _hasModifierKind(plugin) && modIdx >= 0 && modCount > 1) {
    const upBtn = mkBtn("↑", () => _reorderEnabled(modIdx, -1));
    const dnBtn = mkBtn("↓", () => _reorderEnabled(modIdx, +1));
    upBtn.disabled = (modIdx === 0);
    dnBtn.disabled = (modIdx === modCount - 1);
    head.appendChild(upBtn);
    head.appendChild(dnBtn);
  }

  card.appendChild(head);

  if (!isExpanded) return card;

  // Live error pre-block, when /api/plugins surfaced one for this
  // plugin. The header status badge already says "error"; this shows
  // the message body for diagnosis.
  const status = liveByName[plugin.name];
  if (status && status.error) {
    const err = document.createElement("pre");
    err.style.cssText =
      "color:var(--red);font-size:10px;background:var(--bg);" +
      "padding:4px 6px;border-radius:3px;max-height:120px;" +
      "overflow:auto;margin:4px 0";
    err.textContent = status.error;
    card.appendChild(err);
  }

  if (!enabled) return card;

  // Lazy-load this plugin's per-folder config the first time we
  // render it expanded. Subsequent renders use the cached/edited copy.
  if (!modifierConfigEdits[plugin.name]) {
    modifierConfigEdits[plugin.name] = {};
    fetch(`/api/modifiers/${encodeURIComponent(plugin.name)}/config`)
      .then(r => r.ok ? r.json() : { config: {} })
      .then(body => {
        modifierConfigEdits[plugin.name] = body.config || {};
        renderSettingsForm();
      })
      .catch(() => {});
  }
  const cfg = modifierConfigEdits[plugin.name];

  if (plugin.config_schema && plugin.config_schema.length) {
    const cfgBlock = document.createElement("div");
    cfgBlock.style.cssText = "margin:8px 0 4px";
    const cfgHead = document.createElement("div");
    cfgHead.style.cssText =
      "font-size:11px;color:var(--muted);margin-bottom:4px";
    cfgHead.textContent = "Config";
    cfgBlock.appendChild(cfgHead);
    for (const fdef of plugin.config_schema) {
      const fname = fdef.field;
      const type = fdef.type || "text";
      const helpPath = `plugin.${plugin.name}.${fname}`;
      if (fdef.help) HELP[helpPath] = fdef.help;
      if (cfg[fname] == null && fdef.default != null) {
        cfg[fname] = fdef.default;
      }
      cfgBlock.appendChild(renderRow(fname, cfg, fname, type, helpPath));
    }
    card.appendChild(cfgBlock);
  }

  if (plugin.llm_purposes && plugin.llm_purposes.length) {
    card.appendChild(_renderLLMPurposesEditor(plugin, cfg));
  }

  return card;
}

function _reorderEnabled(index, delta) {
  const list = cfgState.modifiers;
  if (!Array.isArray(list)) return;
  const target = index + delta;
  if (target < 0 || target >= list.length) return;
  const tmp = list[index];
  list[index] = list[target];
  list[target] = tmp;
  renderSettingsForm();
}

function _renderLLMPurposesEditor(plugin, cfg) {
  cfg.llm_purposes = cfg.llm_purposes || {};
  const block = document.createElement("div");
  block.style.cssText =
    "margin:8px 0 4px;padding:6px;background:rgba(0,0,0,0.02)";
  const head = document.createElement("div");
  head.style.cssText =
    "font-size:11px;color:var(--muted);margin-bottom:4px";
  head.textContent = "LLM purpose bindings (tag picker)";
  block.appendChild(head);

  const tagNames = Object.keys((cfgState.llm || {}).tags || {});
  for (const purpose of plugin.llm_purposes) {
    const row = document.createElement("div");
    row.className = "cfg-row";
    const lab = document.createElement("label");
    lab.textContent = purpose.name;
    if (purpose.description) lab.title = purpose.description;
    row.appendChild(lab);

    const sel = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = ""; blank.textContent = "(unbound)";
    sel.appendChild(blank);
    for (const t of tagNames) {
      const opt = document.createElement("option");
      opt.value = t; opt.textContent = t;
      sel.appendChild(opt);
    }
    sel.value = (cfg.llm_purposes || {})[purpose.name] || "";
    sel.addEventListener("change", () => {
      cfg.llm_purposes = cfg.llm_purposes || {};
      if (sel.value) cfg.llm_purposes[purpose.name] = sel.value;
      else delete cfg.llm_purposes[purpose.name];
    });
    row.appendChild(sel);
    block.appendChild(row);
  }
  return block;
}

// Map toast level → Bootstrap icon. Re-using the existing icons
// keeps the visual vocabulary consistent with the rest of the SPA
// (status bar, tabs, etc.).
const _TOAST_ICON = {
  ok: "check-circle-fill",
  err: "x-circle-fill",
  warn: "exclamation-circle-fill",
  info: "info-circle-fill",
  pending: "arrow-clockwise",
};

function showToast(text, level = "ok") {
  const icon = window.biIcon(_TOAST_ICON[level] || _TOAST_ICON.info, 13);
  const t = document.createElement("span");
  t.className = "toast-text";
  t.textContent = text;
  settingsToast.innerHTML = "";
  settingsToast.insertAdjacentHTML("beforeend", icon);
  settingsToast.appendChild(t);
  settingsToast.className = "toast " + level;
  setTimeout(() => {
    settingsToast.textContent = "";
    settingsToast.className = "";
  }, 5000);
}

$("#settings-save").addEventListener("click", async () => {
  if (cfgState == null) { showToast("nothing to save", "err"); return; }
  try {
    // Central config.yaml carries the two enable lists
    // (cfgState.modifiers, cfgState.plugins); the unified panel's
    // checkboxes mutate them in-place during the session. Per-plugin
    // config_schema values live in their own files, written below
    // via /api/modifiers/<name>/config — they don't ride along here.
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parsed: cfgState }),
    });
    const body = await r.json();
    if (!r.ok) {
      showToast(`save failed: ${body.detail || r.statusText}`, "err");
      return;
    }

    // Persist each dirty plugin's config to workspace/plugins/<name>/
    // config.yaml. Errors are collected but don't abort — the core
    // save already succeeded by this point.
    const pluginErrs = [];
    for (const [name, cfg] of Object.entries(modifierConfigEdits)) {
      try {
        const pr = await fetch(
          `/api/modifiers/${encodeURIComponent(name)}/config`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config: cfg }),
          },
        );
        if (!pr.ok) {
          const pb = await pr.json().catch(() => ({}));
          pluginErrs.push(`${name}: ${pb.detail || pr.statusText}`);
        }
      } catch (e) {
        pluginErrs.push(`${name}: ${e}`);
      }
    }

    if (pluginErrs.length) {
      showToast(`plugin saves failed: ${pluginErrs.join(", ")}`, "err");
    } else {
      showToast(`saved (backup: ${body.backup || "n/a"}). Restart for changes to take effect.`, "ok");
    }
  } catch (e) {
    showToast("network: " + e, "err");
  }
});

$("#settings-restart").addEventListener("click", async () => {
  if (!confirm("Restart Krakey? The web UI will briefly disconnect.")) return;
  try {
    const r = await fetch("/api/restart", { method: "POST" });
    if (r.ok) {
      showToast("restarting...", "pending");
    } else {
      const body = await r.json();
      showToast(`restart failed: ${body.detail || r.statusText}`, "err");
    }
  } catch (e) {
    // Network error is expected during restart
    showToast("restarting (lost connection)...", "pending");
  }
});
