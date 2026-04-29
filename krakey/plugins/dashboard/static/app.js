// Krakey Dashboard SPA — vanilla JS, no build step.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

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
    }
  });
});

// ============== STATUS BAR ==============

const statusBar = $("#status-bar");
let lastStats = {};
function setStatus() {
  const parts = [];
  if (lastStats.heartbeat_id != null) parts.push(`HB #${lastStats.heartbeat_id}`);
  if (lastStats.node_count != null) parts.push(`gm=${lastStats.node_count}n/${lastStats.edge_count}e`);
  if (lastStats.fatigue_pct != null) parts.push(`fatigue=${lastStats.fatigue_pct}%`);
  parts.push(eventsWS && eventsWS.readyState === 1 ? "events✓" : "events✗");
  parts.push(chatWS && chatWS.readyState === 1 ? "chat✓" : "chat✗");
  statusBar.textContent = parts.join("  |  ");
  renderStatusPanel();
}

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
const hypoEl = $("#hypo-stream");
const stimList = $("#stim-list");
const statusPanel = $("#status-panel");
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

function renderStimuli(stims) {
  stimList.innerHTML = "";
  if (!stims.length) {
    const li = document.createElement("li");
    li.textContent = "(empty)";
    li.style.color = "var(--muted)";
    stimList.appendChild(li);
    return;
  }
  for (const s of stims) {
    const li = document.createElement("li");
    if (s.adrenalin) li.classList.add("adrenalin");
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = `[${s.type}] ${s.source}`;
    li.appendChild(src);
    li.appendChild(document.createTextNode(s.content.slice(0, 200)));
    stimList.appendChild(li);
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
      renderStimuli(e.stimuli);
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
    case "hypothalamus":
      appendEntry(hypoEl, e.heartbeat_id,
        `tool_calls=${e.tool_calls_count} writes=${e.memory_writes_count}` +
        ` updates=${e.memory_updates_count} sleep=${e.sleep_requested}`);
      break;
    case "dispatch":
      appendEntry(hypoEl, e.heartbeat_id,
        `→ ${e.tool} : ${e.intent}${e.adrenalin ? " (adrenalin)" : ""}`);
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
      appendEntry(hypoEl, "—", "💤 sleep started: " + e.reason);
      lastStats.mode = "sleeping";
      setStatus();
      break;
    case "sleep_done":
      appendEntry(hypoEl, "—", "🌅 sleep done: " + JSON.stringify(e.stats));
      lastStats.mode = "normal";
      lastStats.last_sleep = new Date().toISOString();
      setStatus();
      break;
    case "hibernate":
      // could render but it's noisy; skip in UI
      break;
  }
}

function connectEvents() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  eventsWS = new WebSocket(`${proto}//${location.host}/ws/events`);
  eventsWS.onopen = setStatus;
  eventsWS.onclose = () => { setStatus(); setTimeout(connectEvents, 2000); };
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

function connectChat() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  chatWS = new WebSocket(`${proto}//${location.host}/ws/chat`);
  chatWS.onopen = () => { chatMeta.textContent = "connected"; setStatus(); };
  chatWS.onclose = () => {
    chatMeta.textContent = "disconnected — reconnecting...";
    setStatus();
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
  if (!files.length) return;
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
});

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

let currentMemView = "stats";

$$(".mem-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".mem-btn").forEach((b) => b.classList.toggle("active", b === btn));
    currentMemView = btn.dataset.mem;
    loadMemory(currentMemView);
  });
});

async function loadMemory(view) {
  const target = $("#mem-content");
  target.textContent = "loading...";
  try {
    if (view === "stats") {
      const r = await fetch("/api/gm/stats").then((r) => r.json());
      target.innerHTML = renderStats(r);
    } else if (view === "nodes") {
      const r = await fetch("/api/gm/nodes?limit=500").then((r) => r.json());
      target.innerHTML = renderNodes(r);
    } else if (view === "edges") {
      const r = await fetch("/api/gm/edges?limit=500").then((r) => r.json());
      target.innerHTML = renderEdges(r);
    } else if (view === "kbs") {
      const r = await fetch("/api/kbs").then((r) => r.json());
      target.innerHTML = renderKBs(r);
      $$(".kb-card button").forEach((btn) => {
        btn.addEventListener("click", () => loadKBEntries(btn.dataset.kbid));
      });
    }
  } catch (e) {
    target.textContent = "error: " + e;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderStats(s) {
  const cats = Object.entries(s.by_category || {})
    .map(([k, v]) => `<span class="cat-${k}">${k}=${v}</span>`).join("&nbsp;&nbsp;");
  const srcs = Object.entries(s.by_source || {})
    .map(([k, v]) => `${k}=${v}`).join("&nbsp;&nbsp;");
  return `<h3>Graph Memory Stats</h3>
    <p>Total: <b>${s.total_nodes}</b> nodes, <b>${s.total_edges}</b> edges</p>
    <p>By category: ${cats}</p>
    <p>By source: ${srcs}</p>`;
}

function renderNodes(r) {
  if (!r.nodes.length) return "<p>no nodes</p>";
  const rows = r.nodes.map((n) => {
    const cls = n.classified ? "✓" : "";
    return `<tr>
      <td>${n.id}</td>
      <td><span class="cat-${n.category}">${n.category}</span></td>
      <td>${n.source_type}</td>
      <td>${n.importance.toFixed(1)}</td>
      <td>${escapeHtml(n.name)}</td>
      <td class="muted">${escapeHtml((n.description || "").slice(0, 80))}</td>
    </tr>`;
  }).join("");
  return `<h3>${r.count} nodes</h3>
    <table class="mem-table">
      <thead><tr><th>id</th><th>cat</th><th>src</th><th>imp</th><th>name</th><th>desc</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderEdges(r) {
  if (!r.edges.length) return "<p>no edges</p>";
  const rows = r.edges.map((e) => `<tr>
    <td>${escapeHtml(e.source)}</td>
    <td><b>${e.predicate}</b></td>
    <td>${escapeHtml(e.target)}</td>
  </tr>`).join("");
  return `<h3>${r.count} edges</h3>
    <table class="mem-table"><tbody>${rows}</tbody></table>`;
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

loadMemory("stats");

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
  hibernate: { min_interval: 2, max_interval: 300, default_interval: 10 },
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
  sleep: { max_duration_seconds: 7200 },
  safety: { gm_node_hard_limit: 1200, max_consecutive_no_action: 100 },
  dashboard: { enabled: true, host: "127.0.0.1", port: 8765, prompt_log_size: 20 },
  sandbox: {
    guest_os: "", provider: "qemu", vm_name: "",
    display: "headed",
    network_mode: "nat_allowlist",
  },
};

// Hover tooltip text per "section.field" key.
const HELP = {
  "hibernate.min_interval": "Minimum hibernate interval (seconds). Self uses [HIBERNATE] N to control each beat, but it will never go below this value.",
  "hibernate.max_interval": "Maximum hibernate interval (seconds). Even if Self requests a longer hibernate, it will not exceed this value.",
  "hibernate.default_interval": "Default hibernate interval (seconds) when Self does not specify one.",
  "fatigue.gm_node_soft_limit": "Soft upper bound on GM nodes. fatigue% = nodes / soft_limit * 100. Self uses fatigue% to decide whether to sleep proactively.",
  "fatigue.force_sleep_threshold": "Force-sleep threshold (fatigue%). Above this, runtime enters sleep without waiting for Self's consent.",
  "graph_memory.db_path": "Path to the GM SQLite file.",
  "graph_memory.auto_ingest_similarity_threshold": "Similarity threshold (0-1) for stimulus auto_ingest. Below this, the stimulus is treated as a new GM node.",
  "graph_memory.recall_per_stimulus_k": "Hard cap on per-stimulus vec_search top_k. The actual top_k is computed dynamically from recall_screening_token_multiplier and never exceeds this.",
  "graph_memory.recall_screening_token_multiplier": "Token multiplier for the screening pool: each stimulus tries to surface multiplier × recall_token_budget tokens of candidates so the final budget cut has a rich pool to choose from. 1.0 = no over-sampling; default 3.0.",
  "graph_memory.neighbor_expand_depth": "Neighbor-expansion depth at recall time (how many edges to traverse).",
  "knowledge_base.dir": "Directory for KB SQLite files; sleep migration writes here.",
  "sleep.max_duration_seconds": "Maximum allowed duration for a single sleep (seconds), to prevent sleep from hanging.",
  "safety.gm_node_hard_limit": "Hard upper bound on GM nodes. Above this, sleep refuses to add more nodes (prevents runaway growth).",
  "safety.max_consecutive_no_action": "After this many consecutive 'No action' beats, runtime considers Self stuck and triggers a self-rescue sleep.",
  "dashboard.enabled": "Master switch for the web UI. Off = next launch has no browser UI, only logs.",
  "dashboard.host": "Listening address. 127.0.0.1 = local only; 0.0.0.0 = LAN-accessible (insecure).",
  "dashboard.port": "Listening port.",
  "dashboard.prompt_log_size": "The Prompts tab keeps the last N fully-built heartbeat prompts. In-memory ring buffer, not persisted, cleared on restart. Default 20.",
  "provider.type": "Provider implementation type. Currently only openai_compatible is supported.",
  "provider.base_url": "API root URL (without trailing /v1 etc.; LLMClient appends it automatically).",
  "provider.api_key": "API key. Supports a ${ENV_VAR} placeholder to read from environment variables.",
  "model.name": "Model ID, matching the provider's API.",
  "model.capabilities": "Capability tags for the model. Currently informational only; not strictly validated.",
  "role.provider": "Pick a provider for this role.",
  "role.model": "Pick a model under the chosen provider.",
  "channel.enabled": "Whether this channel is enabled.",
  "channel.default_adrenalin": "Whether stimuli pushed by this channel default to adrenalin=true (interrupting hibernate).",
  "tool.enabled": "Whether to register this tool for Hypothalamus to use.",
  "tool.max_results": "Maximum number of search results.",
  "tool.sandbox_dir": "Working directory for code / file operations.",
  "tool.timeout_seconds": "Subprocess timeout (seconds).",
  "tool.max_output_chars": "Truncation length for stdout/stderr.",
  "tool.screenshot_dir": "Directory for GUI screenshots.",
  "tool.history_path": "JSONL persistence path for web chat.",
  "tool.sandbox": "Whether this tool's non-idempotent operations are confined to the sandbox VM. Default true — turning off is dangerous (code / GUI runs on your host).",
  "sandbox.guest_os": "Sandbox guest OS: linux / macos / windows. Required before any sandboxed tool can be enabled.",
  "sandbox.provider": "VM manager: qemu (recommended) / virtualbox / utm.",
  "sandbox.vm_name": "VM instance name (must be pre-provisioned).",
  "sandbox.display": "headed = VM desktop shown in a window so you can watch / intervene; headless = VM hidden, only the agent interacts. Choose by your usage preference.",
  "sandbox.resources.cpu": "vCPU count assigned to the VM.",
  "sandbox.resources.memory_mb": "RAM (MB) assigned to the VM.",
  "sandbox.resources.disk_gb": "VM disk size (GB).",
  "sandbox.agent.url": "HTTP URL of the in-VM guest agent, e.g. http://10.0.2.10:8765. Must be on the host-only subnet.",
  "sandbox.agent.token": "Shared token between host and agent. Use ${ENV_VAR} to read from the environment.",
  "sandbox.network_mode": "VM network policy: nat_allowlist (egress allow-list) / host_only (no internet) / isolated (no network).",
};

// Fixed numeric/string dataclass schemas — drives generic renderer.
const SCHEMAS = {
  hibernate: [
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
  ],
  safety: [
    ["gm_node_hard_limit", "number"],
    ["max_consecutive_no_action", "number"],
  ],
  dashboard: [
    ["enabled", "bool"],
    ["host", "text"],
    ["port", "number"],
    ["prompt_log_size", "number"],
  ],
  sandbox_scalars: [
    ["guest_os", "text"],
    ["provider", "text"],
    ["vm_name", "text"],
    ["display", "text"],
    ["network_mode", "text"],
  ],
  sandbox_resources: [
    ["cpu", "number"],
    ["memory_mb", "number"],
    ["disk_gb", "number"],
  ],
  sandbox_agent: [
    ["url", "text"],
    ["token", "password"],
  ],
};

let pluginReport = { tools: [], channels: [] };
// Dirty-tracking target for plugin edits. One entry per project, shape:
//   { enabled: bool, ...schemaFields }
// Populated on each render from /api/plugins `values`; re-built from
// scratch so stale projects fall off. Saved via
// POST /api/plugins/<project>/config (NOT /api/settings — plugin
// configs live in their own per-plugin YAML files now).
let pluginEdits = {};

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

  // Modifiers — list available + enable/order/configure
  settingsForm.appendChild(renderModifiersSection());

  // Generic sections (each seeded from SECTION_DEFAULTS so missing fields
  // pre-populate to runtime defaults instead of looking "off"/empty)
  ensureSection("hibernate");
  settingsForm.appendChild(renderGenericSection("hibernate", "Hibernate",
    cfgState.hibernate, SCHEMAS.hibernate));

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

  // Plugins — one card per component (tool / channel) known to
  // the runtime RIGHT NOW. Components from the same project share one
  // config_schema; edits land in pluginEdits[<project>] and are
  // persisted per-project via POST /api/plugins/<project>/config.
  pluginEdits = {};  // rebuild each render so removed projects fall off
  settingsForm.appendChild(renderPluginsSection());

  ensureSection("sleep");
  settingsForm.appendChild(renderGenericSection("sleep", "Sleep",
    cfgState.sleep, SCHEMAS.sleep));
  ensureSection("safety");
  settingsForm.appendChild(renderGenericSection("safety", "Safety",
    cfgState.safety, SCHEMAS.safety));
  ensureSection("dashboard");
  settingsForm.appendChild(renderGenericSection("dashboard", "Dashboard",
    cfgState.dashboard, SCHEMAS.dashboard));

  // Sandbox — composite (scalars + resources sub-block + agent sub-block).
  ensureSection("sandbox");
  const sb = cfgState.sandbox;
  if (sb.resources == null) sb.resources = { cpu: 2, memory_mb: 4096, disk_gb: 40 };
  if (sb.agent == null) sb.agent = { url: "", token: "" };
  const sbSec = renderGenericSection("sandbox", "Sandbox VM",
    sb, SCHEMAS.sandbox_scalars);
  const body = sbSec.querySelector(".body");
  const resBlock = document.createElement("div");
  resBlock.className = "subblock";
  const resH = document.createElement("h4");
  resH.textContent = "resources";
  resBlock.appendChild(resH);
  for (const [f, t] of SCHEMAS.sandbox_resources) {
    resBlock.appendChild(renderRow(f, sb.resources, f, t,
      `sandbox.resources.${f}`));
  }
  body.appendChild(resBlock);
  const agentBlock = document.createElement("div");
  agentBlock.className = "subblock";
  const agH = document.createElement("h4");
  agH.textContent = "agent";
  agentBlock.appendChild(agH);
  for (const [f, t] of SCHEMAS.sandbox_agent) {
    agentBlock.appendChild(renderRow(f, sb.agent, f, t,
      `sandbox.agent.${f}`));
  }
  body.appendChild(agentBlock);
  settingsForm.appendChild(sbSec);
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

function makeSection(title) {
  const sec = document.createElement("div");
  sec.className = "cfg-section";
  const h = document.createElement("h3");
  h.textContent = title;
  const body = document.createElement("div");
  body.className = "body";
  sec.appendChild(h); sec.appendChild(body);
  return sec;
}

function renderGenericSection(key, title, target, schema) {
  const sec = makeSection(title);
  const body = sec.querySelector(".body");
  for (const [field, type] of schema) {
    body.appendChild(renderRow(field, target, field, type, `${key}.${field}`));
  }
  return sec;
}

const SAFETY_CONFIRMS = {
  "dashboard.enabled":
    "Turning this off means no web UI on next restart — only logs. Continue?",
};

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
      const del = mkBtn("×", () => { delete fatigue.thresholds[k]; redraw(); }, "danger");
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
  provHead.style.cssText = "color:var(--magenta);font-size:11px;margin:0 0 6px";
  provHead.appendChild(document.createTextNode("Providers"));
  const addProv = mkBtn("+ add provider", () => {
    let name = prompt("Provider name (unique key):");
    if (!name) return;
    name = name.trim();
    if (!name || llm.providers[name]) { alert("invalid or exists"); return; }
    llm.providers[name] = {
      type: "openai_compatible", base_url: "", api_key: "", models: [],
    };
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
  tagsHead.style.cssText = "color:var(--magenta);font-size:11px;margin:12px 0 6px";
  tagsHead.appendChild(document.createTextNode("Tags"));
  const addTag = mkBtn("+ add tag", () => {
    const name = prompt("Tag name (e.g. fast_generation):");
    if (!name) return;
    if (llm.tags[name]) { alert("exists"); return; }
    const provNames = Object.keys(llm.providers || {});
    if (!provNames.length) { alert("add a provider first"); return; }
    // First model of first provider as a starting point
    const firstProv = llm.providers[provNames[0]];
    const firstModel = (firstProv.models && firstProv.models[0]
                          && firstProv.models[0].name) || "";
    llm.tags[name] = {
      provider: `${provNames[0]}/${firstModel}`,
      params: {},
    };
    renderSettingsForm();
  });
  const tagsHeadWrap = document.createElement("div");
  tagsHeadWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:12px 0 6px";
  tagsHeadWrap.appendChild(tagsHead); tagsHeadWrap.appendChild(addTag);
  body.appendChild(tagsHeadWrap);

  for (const tname of Object.keys(llm.tags)) {
    body.appendChild(renderTagRow(tname, llm.tags, llm.providers));
  }

  // Core purposes (chat use cases — Self / compact / classifier)
  body.appendChild(renderCorePurposesBlock(llm));
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
  h.appendChild(document.createTextNode(pname));
  const actions = document.createElement("span");
  actions.className = "actions";
  const renameBtn = mkBtn("rename", () => {
    const nn = prompt("New name:", pname);
    if (!nn || nn === pname) return;
    if (llm.providers[nn]) { alert("exists"); return; }
    llm.providers[nn] = llm.providers[pname];
    delete llm.providers[pname];
    // Update roles referencing old name
    for (const r of Object.values(llm.roles || {})) {
      if (r.provider === pname) r.provider = nn;
    }
    renderSettingsForm();
  });
  const delBtn = mkBtn("delete", () => {
    if (!confirm(`delete provider "${pname}"?`)) return;
    delete llm.providers[pname];
    renderSettingsForm();
  }, "danger");
  actions.appendChild(renameBtn); actions.appendChild(delBtn);
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
  const del = mkBtn("×", () => { prov.models.splice(idx, 1); renderSettingsForm(); }, "danger");
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
  lab.textContent = tname;
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
  }, "danger");

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
  ["compact", "sliding-window history → GM compaction LLM"],
  ["classifier", "node category classifier (often same as compact)"],
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
    const name = prompt("Custom core purpose name:");
    if (!name || llm.core_purposes[name]) return;
    llm.core_purposes[name] = "";
    renderSettingsForm();
  });
  sub.appendChild(addBtn);
  return sub;
}

function _purposeRow(llm, purp, tagNames, helpText) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = purp;
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
  // are persistent.
  const isKnown = KNOWN_CORE_PURPOSES.some(([p]) => p === purp);
  if (!isKnown) {
    const del = mkBtn("×", () => {
      delete llm.core_purposes[purp]; renderSettingsForm();
    }, "danger");
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

  if (helpText) {
    const desc = document.createElement("div");
    desc.style.cssText = "font-size:11px;color:var(--muted);margin-top:4px";
    desc.textContent = helpText;
    sub.appendChild(desc);
  }
  return sub;
}

// (renderRoleParamsBlock removed in 2026-04-26 tag refactor — its
// replacement, renderTagParamsBlock, lives near renderTagRow above.
// Reset-to-defaults dropped because tags have no per-purpose defaults
// any more; LLMParams field defaults are the only baseline.)

// ---------------- Modifiers section ----------------

// Cache of {name → metadata} from /api/modifiers/available. Populated
// alongside configSchema during loadSettings(). Drives the
// "Available Modifiers" UI.
let availableModifiers = [];

// Cache of per-Modifier config (workspace/modifiers/<name>/config.yaml).
// Keyed by Modifier name. Loaded lazily when the user expands a
// Modifier's row, persisted via POST /api/modifiers/<name>/config.
let modifierConfigEdits = {};

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

function renderModifiersSection() {
  const sec = makeSection("Modifiers");
  const body = sec.querySelector(".body");

  // Top: explainer
  const intro = document.createElement("div");
  intro.style.cssText = "font-size:11px;color:var(--muted);margin-bottom:8px";
  intro.textContent =
    "Modifiers are deeper-than-tool plugins that hook into the " +
    "heartbeat. The list below is what's installed on disk; check the " +
    "ones you want loaded. Order in the list = chain execution order.";
  body.appendChild(intro);

  cfgState.modifiers = cfgState.modifiers || [];
  // Only treat as "user explicitly disabled" when the field is an
  // empty array; null/undefined means "unconfigured" — render with
  // the available list so the user can opt in.
  const enabledList = Array.isArray(cfgState.modifiers)
    ? cfgState.modifiers.slice() : [];

  // Modifiers available on disk that the user hasn't enabled yet.
  const enabledSet = new Set(enabledList);
  const knownNames = new Set(availableModifiers.map(m => m.name));

  // Render the user's enabled list first, in the user-declared order.
  // Each enabled Modifier can be expanded for purpose mapping.
  for (let i = 0; i < enabledList.length; i++) {
    const name = enabledList[i];
    const meta = availableModifiers.find(m => m.name === name);
    body.appendChild(renderModifierCard(name, meta, true, i, enabledList));
  }

  // Then disabled (available but not in user list)
  const disabled = availableModifiers.filter(m => !enabledSet.has(m.name));
  if (disabled.length) {
    const head = document.createElement("h4");
    head.style.cssText = "color:var(--muted);font-size:11px;margin:12px 0 6px";
    head.textContent = "Available (disabled)";
    body.appendChild(head);
    for (const meta of disabled) {
      body.appendChild(renderModifierCard(meta.name, meta, false, -1,
                                          enabledList));
    }
  }

  // Detect orphan names — in user's modifiers list but not on disk.
  const orphans = enabledList.filter(n => !knownNames.has(n));
  if (orphans.length) {
    const warn = document.createElement("div");
    warn.style.cssText = "color:var(--red);font-size:11px;margin-top:8px";
    warn.textContent =
      `Unknown modifier names in config: ${orphans.join(", ")} — these will ` +
      "be skipped at startup with a warning.";
    body.appendChild(warn);
  }
  return sec;
}

function renderModifierCard(name, meta, enabled, index, enabledList) {
  const card = document.createElement("div");
  card.className = "subblock";
  card.style.cssText = "margin:6px 0";

  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:8px";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = enabled;
  checkbox.addEventListener("change", () => {
    cfgState.modifiers = cfgState.modifiers || [];
    if (checkbox.checked) {
      if (!cfgState.modifiers.includes(name)) cfgState.modifiers.push(name);
    } else {
      cfgState.modifiers = cfgState.modifiers.filter(n => n !== name);
    }
    renderSettingsForm();
  });
  head.appendChild(checkbox);

  const title = document.createElement("strong");
  title.textContent = name;
  head.appendChild(title);

  if (meta) {
    const kindBadge = document.createElement("span");
    kindBadge.style.cssText =
      "font-size:10px;color:var(--muted);background:var(--panel-2);" +
      "padding:1px 6px;border-radius:3px";
    kindBadge.textContent = meta.kind;
    head.appendChild(kindBadge);
  }

  // Reorder buttons (only show when enabled and there's something to swap with)
  if (enabled && enabledList.length > 1) {
    const upBtn = mkBtn("↑", () => _reorderEnabled(index, -1));
    const dnBtn = mkBtn("↓", () => _reorderEnabled(index, +1));
    upBtn.disabled = (index === 0);
    dnBtn.disabled = (index === enabledList.length - 1);
    head.appendChild(upBtn);
    head.appendChild(dnBtn);
  }

  card.appendChild(head);

  if (meta && meta.description) {
    const desc = document.createElement("div");
    desc.style.cssText = "font-size:11px;color:var(--muted);margin:4px 0";
    desc.textContent = meta.description;
    card.appendChild(desc);
  }

  // LLM purposes editor — only if the plugin declares any
  if (meta && enabled && meta.llm_purposes && meta.llm_purposes.length) {
    card.appendChild(renderModifierLLMPurposes(name, meta));
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

function renderModifierLLMPurposes(name, meta) {
  // Lazy-load this plugin's per-folder config.yaml the first time
  // we need it; subsequent renders use the cached/edited copy.
  if (!modifierConfigEdits[name]) {
    modifierConfigEdits[name] = { llm_purposes: {} };
    fetch(`/api/modifiers/${encodeURIComponent(name)}/config`)
      .then(r => r.ok ? r.json() : { config: {} })
      .then(body => {
        modifierConfigEdits[name] = body.config || {};
        if (!modifierConfigEdits[name].llm_purposes)
          modifierConfigEdits[name].llm_purposes = {};
        renderSettingsForm();
      })
      .catch(() => {});
  }
  const cfg = modifierConfigEdits[name];

  const block = document.createElement("div");
  block.style.cssText = "margin:8px 0 4px;padding:6px;background:rgba(0,0,0,0.02)";
  const head = document.createElement("div");
  head.style.cssText = "font-size:11px;color:var(--muted);margin-bottom:4px";
  head.textContent = "LLM purpose bindings (tag picker)";
  block.appendChild(head);

  const tagNames = Object.keys((cfgState.llm || {}).tags || {});
  for (const purpose of meta.llm_purposes) {
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

  // Save button — writes this plugin's config to its folder
  const saveBtn = mkBtn("save plugin config", async () => {
    try {
      const r = await fetch(`/api/modifiers/${encodeURIComponent(name)}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: cfg }),
      });
      if (!r.ok) {
        const err = await r.text();
        alert(`save failed: ${err}`);
        return;
      }
      alert(`${name} config saved (restart required to apply)`);
    } catch (e) {
      alert(`save failed: ${e}`);
    }
  });
  block.appendChild(saveBtn);
  return block;
}

// ---------------- Generic dict section (channel / tool) ----------------

// ---------------- Plugins section (auto-discovered) ----------------

// Per-render guard so a multi-component project's config_schema
// appears on only the first component rendered, not duplicated on
// every member.
let _pluginSchemaSeen = null;

function renderPluginsSection() {
  const sec = makeSection("Plugins (auto-discovered)");
  const body = sec.querySelector(".body");
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.style.margin = "0 0 8px";
  hint.innerHTML =
    "Tools + channels registered this run — built-ins plus anything " +
    "dropped into <code>workspace/plugins/</code>. A project can carry " +
    "multiple components (channel + tool) that share config. See " +
    "<code>PLUGINS.md</code> for the contract.";
  body.appendChild(hint);

  _pluginSchemaSeen = new Set();
  body.appendChild(_renderPluginGroup("Tools",
    pluginReport.tools || [], "tool"));
  body.appendChild(_renderPluginGroup("Channels",
    pluginReport.channels || [], "channel"));
  return sec;
}

function _renderPluginGroup(title, items, kindKey) {
  const block = document.createElement("div");
  block.className = "subblock";
  const h = document.createElement("h4");
  h.textContent = `${title} (${items.length})`;
  block.appendChild(h);
  if (!items.length) {
    const empty = document.createElement("div");
    empty.style.cssText = "color:var(--muted);font-size:11px;padding:4px 0";
    empty.textContent = "(none)";
    block.appendChild(empty);
    return block;
  }
  for (const p of items) {
    block.appendChild(_renderPluginCard(p, kindKey));
  }
  return block;
}

function _renderPluginCard(p, kindKey) {
  const card = document.createElement("div");
  card.className = "subblock";
  card.style.margin = "6px 0";
  const header = document.createElement("h4");
  header.style.fontSize = "11px";
  const nameSpan = document.createElement("span");
  nameSpan.textContent = p.name;
  const badge = document.createElement("span");
  badge.style.cssText = "margin-left:8px;font-size:10px;padding:1px 6px;" +
    "border-radius:3px;border:1px solid;";
  if (p.source === "builtin") {
    badge.textContent = "builtin";
    badge.style.borderColor = "var(--cyan)";
    badge.style.color = "var(--cyan)";
  } else {
    badge.textContent = p.loaded ? "plugin ✓" : (p.error ? "plugin ✗" : "plugin ○");
    badge.style.borderColor = p.error ? "var(--red)" : "var(--magenta)";
    badge.style.color = p.error ? "var(--red)" : "var(--magenta)";
  }
  header.appendChild(nameSpan);
  header.appendChild(badge);
  if (p.is_internal) {
    const int = document.createElement("span");
    int.style.cssText = "margin-left:6px;font-size:10px;color:var(--muted)";
    int.textContent = "internal";
    header.appendChild(int);
  }
  card.appendChild(header);

  if (p.description) {
    const d = document.createElement("div");
    d.style.cssText = "color:var(--muted);font-size:11px;margin:2px 0 6px";
    d.textContent = p.description;
    card.appendChild(d);
  }
  if (p.error) {
    const err = document.createElement("pre");
    err.style.cssText = "color:var(--red);font-size:10px;background:var(--bg);" +
      "padding:4px 6px;border-radius:3px;max-height:120px;overflow:auto;margin:4px 0";
    err.textContent = p.error;
    card.appendChild(err);
    return card;
  }

  // Config is keyed by project name (one project = one config dict,
  // shared across its components). Render the `enabled` toggle and the
  // schema on the FIRST component of each project only; siblings get
  // a "shared" note.
  const projectKey = p.project || p.name;
  const firstOfProject = _pluginSchemaSeen && !_pluginSchemaSeen.has(projectKey);
  if (_pluginSchemaSeen) _pluginSchemaSeen.add(projectKey);

  if (p.source === "core") {
    // Core items are always on; no enabled toggle, no schema.
    return card;
  }

  if (!firstOfProject) {
    const note = document.createElement("div");
    note.style.cssText = "color:var(--muted);font-size:10px;font-style:italic";
    note.textContent = `(config shared with project ${projectKey} — edit there)`;
    card.appendChild(note);
    return card;
  }

  // Flat edit target: {enabled, ...schemaFields}. Seeded from the
  // authoritative values reported by /api/plugins (which come from
  // the per-plugin YAML file). renderRow mutates this object in-place
  // as the user edits, and the Save handler splits it back into
  // {enabled, values} before POSTing.
  const edit = ensure(pluginEdits, projectKey, () => ({
    enabled: !!p.enabled,
    ...(p.values || {}),
  }));

  // Loader-owned `enabled` toggle — rendered separately from the
  // plugin's own config_schema. Default: false. The loader strips
  // any user-declared "enabled" field from config_schema so a plugin
  // author can't override the default or the widget.
  const enabledHelpPath = `plugin.${projectKey}.enabled`;
  HELP[enabledHelpPath] = "Master switch for this plugin project. " +
    "Default OFF — the factory never runs until you set this to true.";
  card.appendChild(renderRow("enabled", edit, "enabled", "bool",
                                 enabledHelpPath));

  if (p.config_schema && p.config_schema.length) {
    for (const fdef of p.config_schema) {
      const type = fdef.type || "text";
      const helpPath = `plugin.${projectKey}.${fdef.field}`;
      if (fdef.help) HELP[helpPath] = fdef.help;
      if (edit[fdef.field] == null && fdef.default != null) {
        edit[fdef.field] = fdef.default;
      }
      card.appendChild(renderRow(fdef.field, edit, fdef.field, type, helpPath));
    }
  } else {
    const note = document.createElement("div");
    note.style.cssText = "color:var(--muted);font-size:10px;font-style:italic";
    note.textContent = "(no configurable parameters — only the toggle above)";
    card.appendChild(note);
  }
  return card;
}

function showToast(text, ok = true) {
  settingsToast.textContent = text;
  settingsToast.className = ok ? "ok" : "err";
  setTimeout(() => { settingsToast.textContent = ""; settingsToast.className = ""; }, 5000);
}

$("#settings-save").addEventListener("click", async () => {
  if (cfgState == null) { showToast("✗ nothing to save", false); return; }
  try {
    // Plugin configs now live in their own per-plugin YAML files —
    // strip `plugins` from the core config.yaml payload so it stops
    // accumulating stale state. The backend still tolerates it for
    // backwards compat, but new saves shouldn't touch it.
    const corePayload = { ...cfgState };
    delete corePayload.plugins;

    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parsed: corePayload }),
    });
    const body = await r.json();
    if (!r.ok) {
      showToast(`✗ save failed: ${body.detail || r.statusText}`, false);
      return;
    }

    // Persist each dirty plugin project to its own file. Errors are
    // collected but don't abort — we still report the core save.
    const pluginErrs = [];
    for (const [project, edit] of Object.entries(pluginEdits)) {
      const enabled = !!edit.enabled;
      const values = {};
      for (const [k, v] of Object.entries(edit)) {
        if (k !== "enabled") values[k] = v;
      }
      try {
        const pr = await fetch(`/api/plugins/${encodeURIComponent(project)}/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled, values }),
        });
        if (!pr.ok) {
          const pb = await pr.json().catch(() => ({}));
          pluginErrs.push(`${project}: ${pb.detail || pr.statusText}`);
        }
      } catch (e) {
        pluginErrs.push(`${project}: ${e}`);
      }
    }

    if (pluginErrs.length) {
      showToast(`✗ plugin saves failed: ${pluginErrs.join(", ")}`, false);
    } else {
      showToast(`✓ saved (backup: ${body.backup || "n/a"}). Restart for changes to take effect.`);
    }
  } catch (e) {
    showToast("✗ network: " + e, false);
  }
});

$("#settings-restart").addEventListener("click", async () => {
  if (!confirm("Restart Krakey? The web UI will briefly disconnect.")) return;
  try {
    const r = await fetch("/api/restart", { method: "POST" });
    if (r.ok) {
      showToast("⏳ restarting...");
    } else {
      const body = await r.json();
      showToast(`✗ restart failed: ${body.detail || r.statusText}`, false);
    }
  } catch (e) {
    // Network error is expected during restart
    showToast("⏳ restarting (lost connection)...");
  }
});
