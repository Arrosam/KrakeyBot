// Krakey i18n core — mirrors the theme-toggle localStorage pattern.
// A locale key is stored in 'krakey-lang'; on load the stored value
// (or 'en') becomes the active locale. setLocale() swaps the active
// locale and persists the choice; the app.js IIFE is responsible for
// updating <html lang> and re-rendering strings via applyLocale().
//
//   t(key)           -> resolved string (active locale, then en, then key)
//   t(key, {n: 3})   -> same, with {n} placeholders substituted
//
// Adding a language is a pure data drop-in: register another entry on
// LOCALES (e.g. LOCALES.zh = {...}) and the #lang-toggle activates
// automatically (it stays hidden while only one locale is registered).
(function () {
  window.LOCALES = {
    en: {
      // ── Tabs (header nav) ──────────────────────────────────────────
      tab_thoughts: "Inner Thoughts",
      tab_chat: "Chat",
      tab_memory: "Memory",
      tab_prompts: "Prompts",
      tab_log: "Log",
      tab_settings: "Settings",

      // ── Header control aria-labels ─────────────────────────────────
      aria_runtime_toggle: "Pause/resume heartbeat",
      aria_lang_toggle: "Switch language",
      aria_theme_toggle: "Toggle light/dark mode",

      // ── Runtime pause/resume toggle ────────────────────────────────
      runtime_resume: "Resume",
      runtime_pause: "Pause",
      runtime_resume_title: "Resume heartbeat",
      runtime_pause_title: "Pause heartbeat",

      // ── Sleep banners ──────────────────────────────────────────────
      sleep_reason_default: "compacting memory",
      sleep_banner: "Krakey is sleeping ({reason}) — Memory tab is paused until sleep finishes.",
      memory_sleeping: "Krakey is sleeping — Memory will load automatically when sleep finishes.",

      // ── WebSocket / async status ───────────────────────────────────
      ws_connected: "connected",
      ws_disconnected: "disconnected — reconnecting...",
      ws_error: "error",
      loading: "loading...",
      error_prefix: "error: ",
      error_loading_prefix: "error loading: ",

      // ── Prompts view ───────────────────────────────────────────────
      prompts_empty: "No prompts yet — the first will appear in ~{secs}s at the next heartbeat.",
      prompts_paused: "paused — toggle live to resume",
      prompts_pending: "{count} new prompt{plural} since paused",

      // ── Memory view ────────────────────────────────────────────────
      memory_graph_hint: "drag to pan · scroll to zoom · drag a node to move it",

      // ── Settings: section titles (keyed by stable section key) ─────
      section_llm: "LLM",
      section_plugins: "Plugins",
      section_idle: "Idle",
      section_fatigue: "Fatigue",
      section_sliding_window: "Working Memory",
      // Shortened variant for the compact settings jump-rail (the full
      // title is too wide for the rail column).
      section_sliding_window_short: "Working Memory",
      section_graph_memory: "Graph Memory",
      section_knowledge_base: "Knowledge Base",
      section_sleep: "Sleep",
      section_safety: "Safety",
      section_environments: "Environments",
      section_core_implementations: "Engine Overrides",

      // ── Settings: misc labels & buttons ────────────────────────────
      safety_advisory: "advisory only — runtime does not yet enforce these limits",
      opt_custom_path: "Custom path…",
      available_disabled: "Available (disabled)",
      config_label: "Config",
      llm_purpose_bindings: "LLM purpose bindings (tag picker)",
      btn_add: "+ add",
      btn_add_provider: "+ add provider",
      btn_add_tag: "+ add tag",
      btn_add_model: "+ add model",
      btn_add_purpose: "+ add purpose",
      btn_save: "Save",
      btn_apply_changes: "Apply changes",
      btn_restart: "Restart Krakey",
      btn_send: "Send",
      btn_clear: "clear",

      // ── View panels & static chrome (view.html / index.template) ───
      brand_tag: "digital being",
      panel_thinking: "Thinking (inner monologue)",
      panel_decision: "Decision",
      panel_tool_usage: "Tool Usage",
      panel_stimulus: "Stimulus",
      panel_status: "Status (runtime state)",
      mem_graph: "GM Graph",
      mem_kbs: "KBs",
      log_autoscroll: "auto-scroll",
      prompts_hint: "The full prompt built for each heartbeat (last 50 beats, in-memory ring buffer; cleared on restart).",
      prompts_live: "Live updates",
      chat_placeholder: "Say something to Krakey... (Enter to send, Shift+Enter for newline)",
      chat_attach_title: "Attach image / file",

      // ── Confirm dialogs ────────────────────────────────────────────
      confirm_restart: "Restart Krakey? The web UI will briefly disconnect.",

      // ── Chat message delivery / read-receipt status ────────────────
      msg_status_delivered: "delivered",
      msg_status_read: "read",
      msg_status_failed: "send failed — agent offline",
      msg_resend: "Resend",
    },
  };

  // Native display names for the language picker — shown in each
  // locale's own language regardless of the active UI locale. A locale
  // without an entry here falls back to its raw code.
  window.LOCALE_NAMES = {
    en: "English",
    "zh-CN": "简体中文",
  };

  var _locale = localStorage.getItem('krakey-lang') || 'en';

  window.getLocale = function () { return _locale; };

  window.availableLocales = function () { return Object.keys(window.LOCALES); };

  window.localeName = function (code) {
    return (window.LOCALE_NAMES && window.LOCALE_NAMES[code]) || code;
  };

  // Resolve a key in the active locale, falling back to en, then to the
  // raw key. Optional `params` substitutes {name} placeholders.
  window.t = function (key, params) {
    var s = (window.LOCALES[_locale] && window.LOCALES[_locale][key]);
    if (s == null) s = (window.LOCALES.en && window.LOCALES.en[key]);
    if (s == null) s = key;
    if (params) {
      s = s.replace(/\{(\w+)\}/g, function (m, name) {
        return (params[name] != null) ? params[name] : m;
      });
    }
    return s;
  };

  // NO-OP for unregistered langs. Does NOT touch <html lang> or any
  // button — that responsibility belongs to the app.js IIFE.
  window.setLocale = function (lang) {
    if (!window.LOCALES[lang]) return;
    _locale = lang;
    localStorage.setItem('krakey-lang', lang);
  };
})();
