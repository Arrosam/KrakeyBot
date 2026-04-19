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
}

// ============== INNER THOUGHTS — /ws/events ==============

let eventsWS = null;
const thinkingEl = $("#thinking-stream");
const decisionEl = $("#decision-stream");
const hypoEl = $("#hypo-stream");
const stimList = $("#stim-list");
const promptHb = $("#prompt-hb");
const latestPrompt = $("#latest-prompt");

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
        `tentacle_calls=${e.tentacle_calls_count} writes=${e.memory_writes_count}` +
        ` updates=${e.memory_updates_count} sleep=${e.sleep_requested}`);
      break;
    case "dispatch":
      appendEntry(hypoEl, e.heartbeat_id,
        `→ ${e.tentacle} : ${e.intent}${e.adrenalin ? " (adrenalin)" : ""}`);
      break;
    case "prompt_built":
      promptHb.textContent = e.heartbeat_id;
      latestPrompt.textContent = e.layers.full_prompt || "(empty)";
      break;
    case "sleep_start":
      appendEntry(hypoEl, "—", "💤 sleep started: " + e.reason);
      break;
    case "sleep_done":
      appendEntry(hypoEl, "—", "🌅 sleep done: " + JSON.stringify(e.stats));
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
  div.appendChild(document.createTextNode(msg.content));
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = fmtTime(msg.ts);
  div.appendChild(ts);
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;
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

chatForm.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = chatInput.value.trim();
  if (!text || !chatWS || chatWS.readyState !== 1) return;
  chatWS.send(JSON.stringify({ text }));
  chatInput.value = "";
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

// ============== SETTINGS ==============

const settingsText = $("#settings-text");
const settingsToast = $("#settings-toast");

async function loadSettings() {
  settingsToast.textContent = "";
  try {
    const r = await fetch("/api/settings");
    if (r.status === 503) {
      settingsText.value = "(settings endpoint not wired — config_path not provided to dashboard)";
      return;
    }
    const data = await r.json();
    settingsText.value = data.raw;
  } catch (e) {
    settingsText.value = "error loading: " + e;
  }
}

function showToast(text, ok = true) {
  settingsToast.textContent = text;
  settingsToast.className = ok ? "ok" : "err";
  setTimeout(() => { settingsToast.textContent = ""; settingsToast.className = ""; }, 5000);
}

$("#settings-save").addEventListener("click", async () => {
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw: settingsText.value }),
    });
    const body = await r.json();
    if (r.ok) {
      showToast(`✓ saved (backup: ${body.backup || "n/a"}). Click Restart for changes to take effect.`);
    } else {
      showToast(`✗ save failed: ${body.detail || r.statusText}`, false);
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
