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
}

// ============== INNER THOUGHTS — /ws/events ==============

let eventsWS = null;
const thinkingEl = $("#thinking-stream");
const decisionEl = $("#decision-stream");
const hypoEl = $("#hypo-stream");
const stimList = $("#stim-list");
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
      out.push({ title: "DNA / 系统提示", body: p });
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
        `tentacle_calls=${e.tentacle_calls_count} writes=${e.memory_writes_count}` +
        ` updates=${e.memory_updates_count} sleep=${e.sleep_requested}`);
      break;
    case "dispatch":
      appendEntry(hypoEl, e.heartbeat_id,
        `→ ${e.tentacle} : ${e.intent}${e.adrenalin ? " (adrenalin)" : ""}`);
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
  sliding_window: { max_tokens: 4096 },
  graph_memory: {
    db_path: "workspace/data/graph_memory.sqlite",
    auto_ingest_similarity_threshold: 0.92,
    recall_per_stimulus_k: 5, max_recall_nodes: 20, neighbor_expand_depth: 1,
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
  "hibernate.min_interval": "睡眠最短间隔（秒）。Self 用 [HIBERNATE] N 控制每跳间隔, 但不会低于这个值。",
  "hibernate.max_interval": "睡眠最长间隔（秒）。即使 Self 要求 hibernate 更久, 也不会超过这个值。",
  "hibernate.default_interval": "Self 没指定时的默认 hibernate 间隔（秒）。",
  "fatigue.gm_node_soft_limit": "GM 节点软上限。fatigue% = nodes / soft_limit * 100。Self 看到 fatigue% 决定是否主动睡眠。",
  "fatigue.force_sleep_threshold": "强制睡眠阈值（fatigue%）。超过这个值, runtime 不等 Self 同意直接进 sleep。",
  "sliding_window.max_tokens": "滑动上下文窗口最大 token 数。超过则压缩旧 round 为 summary。",
  "graph_memory.db_path": "GM SQLite 文件路径。",
  "graph_memory.auto_ingest_similarity_threshold": "stimulus auto_ingest 的相似度阈值 (0-1)。低于则当作新节点入 GM。",
  "graph_memory.recall_per_stimulus_k": "每条 stimulus 召回的 top-K 节点数。",
  "graph_memory.max_recall_nodes": "单次 prompt 中召回节点的总数上限。",
  "graph_memory.neighbor_expand_depth": "召回时邻居展开深度（沿 edges 走几跳）。",
  "knowledge_base.dir": "KB SQLite 文件目录, sleep migration 会写到这里。",
  "sleep.max_duration_seconds": "单次 sleep 最长允许时长（秒）, 防 sleep 卡死。",
  "safety.gm_node_hard_limit": "GM 节点硬上限。超过则 sleep 拒绝继续添加节点（防爆炸）。",
  "safety.max_consecutive_no_action": "Self 连续 'No action' 多少次后视为僵死, runtime 触发 sleep 自救。",
  "dashboard.enabled": "Web UI 总开关。关掉这个 = 下次启动后没有浏览器界面, 只剩日志。",
  "dashboard.host": "监听地址。127.0.0.1 = 仅本机; 0.0.0.0 = 局域网可访问（不安全）。",
  "dashboard.port": "监听端口。",
  "dashboard.prompt_log_size": "Prompts 标签页保留最近 N 次心跳构造的完整 prompt。运行时环形缓冲, 不落盘, 重启清零。默认 20。",
  "provider.type": "Provider 实现类型。目前只支持 openai_compatible。",
  "provider.base_url": "API 根 URL（不含 /v1 等后缀, LLMClient 会自动加）。",
  "provider.api_key": "API 密钥。可填 ${ENV_VAR} 占位符从环境变量读取。",
  "model.name": "模型 ID, 与 provider 的 API 一致。",
  "model.capabilities": "模型能力标签。仅供后续路由参考, 当前不强制校验。",
  "role.provider": "为该 role 选一个 provider。",
  "role.model": "在选定 provider 下选一个 model。",
  "sensory.enabled": "是否启用此 sensory 通道。",
  "sensory.default_adrenalin": "该 sensory 推送的 stimulus 默认是否激活肾上腺素 (打断 hibernate)。",
  "tentacle.enabled": "是否注册此 tentacle 给 Hypothalamus 使用。",
  "tentacle.max_results": "搜索结果数上限。",
  "tentacle.sandbox_dir": "代码 / 文件操作的工作目录。",
  "tentacle.timeout_seconds": "子进程超时（秒）。",
  "tentacle.max_output_chars": "stdout/stderr 截断字符数。",
  "tentacle.screenshot_dir": "GUI 截图保存目录。",
  "tentacle.history_path": "Web chat 持久化 JSONL 路径。",
  "tentacle.sandbox": "该 tentacle 的非幂等操作是否只在沙盒 VM 内发生。默认 true — 关掉 = 危险 (代码/GUI 直接跑在你的机器)。",
  "sandbox.guest_os": "沙盒客机操作系统: linux / macos / windows。启用任何 sandboxed tentacle 必须先填。",
  "sandbox.provider": "虚拟机管理器: qemu (推荐) / virtualbox / utm。",
  "sandbox.vm_name": "VM 实例名 (预先 provision 好)。",
  "sandbox.display": "headed = VM 桌面显示一个窗口, 你能看能介入; headless = VM 完全不显示, 只通过 agent 交互。由你按使用偏好选。",
  "sandbox.resources.cpu": "分配给 VM 的 vCPU 数。",
  "sandbox.resources.memory_mb": "分配给 VM 的内存 (MB)。",
  "sandbox.resources.disk_gb": "VM 磁盘容量 (GB)。",
  "sandbox.agent.url": "VM 内 guest agent 的 HTTP 地址, e.g. http://10.0.2.10:8765。必须在 host-only 子网上。",
  "sandbox.agent.token": "host ↔ agent 共享 token。放 ${ENV_VAR} 从环境变量读。",
  "sandbox.network_mode": "VM 网络策略: nat_allowlist (出互联网白名单) / host_only (无外网) / isolated (全断网)。",
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
  sliding_window: [
    ["max_tokens", "number"],
  ],
  graph_memory: [
    ["db_path", "text"],
    ["auto_ingest_similarity_threshold", "number_float"],
    ["recall_per_stimulus_k", "number"],
    ["max_recall_nodes", "number"],
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

let pluginReport = { tentacles: [], sensories: [] };

async function loadSettings() {
  settingsToast.textContent = "";
  settingsForm.innerHTML = "loading...";
  try {
    // Load config + plugin discovery in parallel
    const [cfgRes, pluginRes] = await Promise.all([
      fetch("/api/settings"),
      fetch("/api/plugins").catch(() => null),
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
      pluginReport = { tentacles: [], sensories: [] };
    }
    renderSettingsForm();
  } catch (e) {
    settingsForm.innerHTML = "error loading: " + escapeHtml(String(e));
  }
}

function renderSettingsForm() {
  settingsForm.innerHTML = "";
  // LLM
  const llm = ensure(cfgState, "llm", () => ({ providers: {}, roles: {} }));
  ensure(llm, "providers", () => ({}));
  ensure(llm, "roles", () => ({}));
  settingsForm.appendChild(renderLLMSection(llm));

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

  ensureSection("sliding_window");
  settingsForm.appendChild(renderGenericSection("sliding_window", "Sliding Window",
    cfgState.sliding_window, SCHEMAS.sliding_window));
  ensureSection("graph_memory");
  settingsForm.appendChild(renderGenericSection("graph_memory", "Graph Memory",
    cfgState.graph_memory, SCHEMAS.graph_memory));
  ensureSection("knowledge_base");
  settingsForm.appendChild(renderGenericSection("knowledge_base", "Knowledge Base",
    cfgState.knowledge_base, SCHEMAS.knowledge_base));

  // Plugins — one card per component (tentacle / sensory) known to
  // the runtime RIGHT NOW. Components from the same project share one
  // config_schema; edits land in cfgState.plugins[<project>].
  settingsForm.appendChild(renderPluginsSection());

  // Raw plugins editor — advanced override for anything not declared
  // in a plugin's config_schema. Edits go to cfgState.plugins.*.
  settingsForm.appendChild(renderDictSection("plugins", "Plugins (raw)",
    ensure(cfgState, "plugins", () => ({})),
    { defaultEntry: () => ({ enabled: true }) }));

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
    "关掉这个 = 下次重启后没有 Web UI, 只剩日志。确定？",
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

function renderLLMSection(llm) {
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

  // Roles
  const rolesHead = document.createElement("h4");
  rolesHead.style.cssText = "color:var(--magenta);font-size:11px;margin:12px 0 6px";
  rolesHead.appendChild(document.createTextNode("Roles"));
  const addRole = mkBtn("+ add role", () => {
    const name = prompt("Role name (e.g. self / hypothalamus):");
    if (!name) return;
    if (llm.roles[name]) { alert("exists"); return; }
    const provNames = Object.keys(llm.providers);
    if (!provNames.length) { alert("add a provider first"); return; }
    llm.roles[name] = { provider: provNames[0], model: "" };
    renderSettingsForm();
  });
  const rolesHeadWrap = document.createElement("div");
  rolesHeadWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:12px 0 6px";
  rolesHeadWrap.appendChild(rolesHead); rolesHeadWrap.appendChild(addRole);
  body.appendChild(rolesHeadWrap);

  for (const rname of Object.keys(llm.roles || {})) {
    body.appendChild(renderRoleRow(rname, llm.roles, llm.providers));
  }

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

function renderRoleRow(rname, roles, providers) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const lab = document.createElement("label");
  lab.textContent = rname;
  row.appendChild(lab);

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:grid;grid-template-columns:1fr 1fr auto;gap:6px";

  const provSel = document.createElement("select");
  for (const pname of Object.keys(providers || {})) {
    const opt = document.createElement("option");
    opt.value = pname; opt.textContent = pname;
    provSel.appendChild(opt);
  }
  provSel.value = roles[rname].provider || "";

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
    if (roles[rname].model && [...modSel.options].some(o => o.value === roles[rname].model)) {
      modSel.value = roles[rname].model;
    } else {
      roles[rname].model = modSel.value;
    }
  }
  provSel.addEventListener("change", () => {
    roles[rname].provider = provSel.value;
    refreshModels();
  });
  modSel.addEventListener("change", () => { roles[rname].model = modSel.value; });
  refreshModels();

  const del = mkBtn("×", () => { delete roles[rname]; renderSettingsForm(); }, "danger");

  wrap.appendChild(provSel); wrap.appendChild(modSel); wrap.appendChild(del);
  row.appendChild(wrap);
  return row;
}

// ---------------- Generic dict section (sensory / tentacle) ----------------

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
    "Tentacles + sensories registered this run — built-ins plus anything " +
    "dropped into <code>workspace/plugins/</code>. A project can carry " +
    "multiple components (sensory + tentacle) that share config. See " +
    "<code>PLUGINS.md</code> for the contract.";
  body.appendChild(hint);

  _pluginSchemaSeen = new Set();
  body.appendChild(_renderPluginGroup("Tentacles",
    pluginReport.tentacles || [], "tentacle"));
  body.appendChild(_renderPluginGroup("Sensories",
    pluginReport.sensories || [], "sensory"));
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

  const plugins = ensure(cfgState, "plugins", () => ({}));
  const entry = ensure(plugins, projectKey, () => ({}));

  // Loader-owned `enabled` toggle — rendered separately from the
  // plugin's own config_schema. Default: false. The loader strips
  // any user-declared "enabled" field from config_schema so a plugin
  // author can't override the default or the widget.
  if (entry.enabled == null) entry.enabled = false;
  const enabledHelpPath = `plugin.${projectKey}.enabled`;
  HELP[enabledHelpPath] = "Master switch for this plugin project. " +
    "Default OFF — the factory never runs until you set this to true.";
  card.appendChild(renderRow("enabled", entry, "enabled", "bool",
                                 enabledHelpPath));

  if (p.config_schema && p.config_schema.length) {
    for (const fdef of p.config_schema) {
      const type = fdef.type || "text";
      const helpPath = `plugin.${projectKey}.${fdef.field}`;
      if (fdef.help) HELP[helpPath] = fdef.help;
      if (entry[fdef.field] == null && fdef.default != null) {
        entry[fdef.field] = fdef.default;
      }
      card.appendChild(renderRow(fdef.field, entry, fdef.field, type, helpPath));
    }
  } else {
    const note = document.createElement("div");
    note.style.cssText = "color:var(--muted);font-size:10px;font-style:italic";
    note.textContent = "(no additional config_schema declared — toggle above is the only switch)";
    card.appendChild(note);
  }
  return card;
}

function renderDictSection(key, title, target, opts) {
  const sec = makeSection(title);
  const body = sec.querySelector(".body");
  const headWrap = document.createElement("div");
  headWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:6px";
  const addBtn = mkBtn("+ add entry", () => {
    const name = prompt(`${title} entry key:`);
    if (!name) return;
    if (target[name]) { alert("exists"); return; }
    target[name] = opts.defaultEntry();
    renderSettingsForm();
  });
  headWrap.appendChild(addBtn);
  body.appendChild(headWrap);

  for (const [name, entry] of Object.entries(target)) {
    body.appendChild(renderDictEntry(name, entry, target));
  }
  return sec;
}

function renderDictEntry(name, entry, parent) {
  const block = document.createElement("div");
  block.className = "subblock";
  const h = document.createElement("h4");
  h.appendChild(document.createTextNode(name));
  const actions = document.createElement("span");
  actions.className = "actions";
  const addField = mkBtn("+ field", () => {
    const fname = prompt("field name:");
    if (!fname || fname in entry) return;
    entry[fname] = "";
    renderSettingsForm();
  });
  const del = mkBtn("delete", () => {
    if (!confirm(`delete "${name}"?`)) return;
    delete parent[name];
    renderSettingsForm();
  }, "danger");
  actions.appendChild(addField); actions.appendChild(del);
  h.appendChild(actions);
  block.appendChild(h);

  for (const k of Object.keys(entry)) {
    const v = entry[k];
    let type;
    if (typeof v === "boolean") type = "bool";
    else if (typeof v === "number") type = Number.isInteger(v) ? "number" : "number_float";
    else type = "text";
    // Generic per-section help: e.g. tentacle.<key> applies regardless of entry name
    const sectionKey = parent === cfgState.sensory ? "sensory" : "tentacle";
    block.appendChild(renderRow(k, entry, k, type, `${sectionKey}.${k}`));
  }
  return block;
}

function showToast(text, ok = true) {
  settingsToast.textContent = text;
  settingsToast.className = ok ? "ok" : "err";
  setTimeout(() => { settingsToast.textContent = ""; settingsToast.className = ""; }, 5000);
}

$("#settings-save").addEventListener("click", async () => {
  if (cfgState == null) { showToast("✗ nothing to save", false); return; }
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parsed: cfgState }),
    });
    const body = await r.json();
    if (r.ok) {
      showToast(`✓ saved (backup: ${body.backup || "n/a"}). Restart for changes to take effect.`);
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
