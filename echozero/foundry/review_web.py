"""EZ Review web page: Teenage Engineering-leaning pocket review workspace.
Exists because fast manual verification should feel tactile, focused, and self-contained.
Connects review-session APIs to one static HTML page with live queue filters and revisit navigation.
"""

from __future__ import annotations


def build_review_page() -> str:
    """Return the single-screen EZ Review page."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>EZ Review</title>
  <meta name="apple-mobile-web-app-capable" content="yes">
  <style>
    :root {
      color-scheme: dark;
      --bg: #11100c;
      --shell: #e8dfcf;
      --shell-strong: #f5efdf;
      --panel: rgba(232, 223, 207, 0.96);
      --panel-soft: rgba(213, 203, 187, 0.94);
      --ink: #171510;
      --muted: #655f54;
      --line: rgba(23, 21, 16, 0.14);
      --line-strong: rgba(23, 21, 16, 0.22);
      --accent: #ff6a13;
      --accent-soft: rgba(255, 106, 19, 0.18);
      --success: #17b86c;
      --danger: #ff4d3f;
      --shadow: 0 24px 64px rgba(0, 0, 0, 0.38);
      --radius-lg: 30px;
      --radius-md: 22px;
      --radius-sm: 16px;
      font-family: "Arial Rounded MT Bold", "Trebuchet MS", "Avenir Next", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      background:
        radial-gradient(circle at top, rgba(255, 106, 19, 0.18), transparent 28%),
        linear-gradient(180deg, #1f1b16, var(--bg));
      color: var(--ink);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
      background-size: 24px 24px;
      opacity: 0.16;
      pointer-events: none;
    }
    main {
      position: relative;
      z-index: 1;
      width: min(100%, 560px);
      min-height: 100dvh;
      margin: 0 auto;
      padding:
        max(16px, env(safe-area-inset-top))
        16px
        calc(18px + env(safe-area-inset-bottom))
        16px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.22);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }
    .topbar {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .title-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .brand {
      display: grid;
      gap: 6px;
    }
    .brand-mark {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .brand-dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: inset 0 0 0 2px rgba(255,255,255,0.28);
    }
    .brand-title {
      margin: 0;
      font-family: "Arial Black", "Arial Narrow", sans-serif;
      font-size: clamp(2rem, 9vw, 3.4rem);
      line-height: 0.9;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .brand-subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .counter {
      min-width: 96px;
      padding: 12px 14px;
      border-radius: 22px;
      background: var(--ink);
      color: var(--shell-strong);
      text-align: right;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 1.12rem;
      font-weight: 800;
      letter-spacing: 0.08em;
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    select {
      width: 100%;
      min-height: 48px;
      padding: 0 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.46);
      color: var(--ink);
      font: inherit;
      font-weight: 700;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.32);
    }
    .meter-strip {
      display: grid;
      gap: 8px;
    }
    .meter-copy {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .progress-rail {
      height: 10px;
      border-radius: 999px;
      background: rgba(23, 21, 16, 0.12);
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #ffca72);
      transition: width 140ms ease;
    }
    .workspace {
      min-height: 0;
      padding: 16px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      background:
        linear-gradient(180deg, var(--panel), var(--panel-soft)),
        radial-gradient(circle at top right, rgba(255, 106, 19, 0.1), transparent 32%);
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }
    .meta-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.4);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.73rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .pill strong {
      font-size: 0.78rem;
    }
    .pill.mode-history {
      border-color: rgba(255, 106, 19, 0.38);
      background: var(--accent-soft);
    }
    .app-session-meta {
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      text-align: right;
    }
    .center {
      min-height: 0;
      display: grid;
      align-content: center;
      gap: 16px;
    }
    .hero {
      padding: 20px;
      border-radius: 28px;
      border: 1px solid var(--line);
      background: var(--shell-strong);
      display: grid;
      gap: 10px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.36);
    }
    .hero-ribbon {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      background: rgba(23, 21, 16, 0.08);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .prediction {
      margin: 0;
      font-family: "Arial Black", "Arial Narrow", sans-serif;
      font-size: clamp(2.4rem, 12vw, 4.8rem);
      line-height: 0.88;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .file-name,
    .status-line {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .context-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .context-card {
      padding: 14px;
      border-radius: 22px;
      background: rgba(23, 21, 16, 0.08);
      border: 1px solid var(--line);
      display: grid;
      gap: 6px;
    }
    .context-card span {
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .context-card strong {
      font-size: 0.98rem;
      line-height: 1.2;
    }
    .play-shell {
      display: grid;
      place-items: center;
      gap: 10px;
      padding-top: 2px;
    }
    .play-button {
      width: min(44vw, 156px);
      aspect-ratio: 1;
      border-radius: 42px;
      border: 1px solid rgba(23, 21, 16, 0.18);
      background:
        radial-gradient(circle at 30% 30%, rgba(255,255,255,0.72), transparent 28%),
        linear-gradient(180deg, #fff4d9, #ffcb66 58%, #ff9f43);
      color: var(--ink);
      font: inherit;
      box-shadow:
        inset 0 3px 0 rgba(255,255,255,0.38),
        inset 0 -10px 24px rgba(0, 0, 0, 0.08),
        0 20px 34px rgba(0, 0, 0, 0.2);
    }
    .play-glyph {
      display: block;
      font-size: 3.1rem;
      line-height: 1;
      transform: translateX(4px);
    }
    .play-caption {
      display: block;
      margin-top: 10px;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.74rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .empty-state {
      display: grid;
      gap: 10px;
      place-items: center;
      padding: 28px 20px;
      text-align: center;
      color: var(--muted);
      font-size: 0.98rem;
    }
    .empty-mark {
      width: 72px;
      height: 72px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: rgba(23, 21, 16, 0.08);
      display: grid;
      place-items: center;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 1rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .status-line {
      min-height: 1.2em;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .bottom-bar {
      padding: 10px;
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      background: rgba(17, 16, 12, 0.96);
      color: var(--shell);
    }
    .action {
      min-height: 78px;
      border-radius: 24px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: inherit;
      font: inherit;
      font-size: 2rem;
      font-weight: 900;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .action.correct { color: var(--success); }
    .action.incorrect { color: var(--danger); }
    .action.reclass { color: #ffd15c; }
    .action.nav {
      display: grid;
      place-items: center;
      gap: 4px;
      font-size: 1.4rem;
      color: rgba(245, 239, 223, 0.92);
    }
    .action-label {
      display: block;
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.64rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .action:disabled {
      opacity: 0.34;
      box-shadow: none;
    }
    .sheet-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(8, 7, 5, 0.44);
      opacity: 0;
      pointer-events: none;
      transition: opacity 120ms ease;
    }
    .sheet-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }
    .sheet {
      position: fixed;
      left: 50%;
      bottom: max(10px, env(safe-area-inset-bottom));
      transform: translate(-50%, calc(100% + 24px));
      width: min(calc(100% - 16px), 560px);
      max-height: 62dvh;
      overflow: auto;
      padding: 14px;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.2);
      box-shadow: var(--shadow);
      transition: transform 160ms ease;
      z-index: 3;
    }
    .sheet.open { transform: translate(-50%, 0); }
    .sheet-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }
    .sheet-title {
      margin: 0;
      font-family: "Arial Black", "Arial Narrow", sans-serif;
      font-size: 1.5rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .sheet-close {
      width: 42px;
      height: 42px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(23, 21, 16, 0.08);
      color: var(--ink);
      font: inherit;
      font-size: 1.1rem;
    }
    .class-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .class-chip {
      min-height: 40px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.42);
      color: var(--ink);
      font: inherit;
      font-size: 0.9rem;
      font-weight: 700;
    }
    .class-chip.selected {
      border-color: rgba(255, 106, 19, 0.36);
      background: var(--accent-soft);
    }
    .sheet-field {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", "Menlo", monospace;
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .sheet-field input,
    .sheet-field textarea {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.5);
      color: var(--ink);
      font: inherit;
      padding: 12px;
    }
    .sheet-field textarea {
      min-height: 92px;
      resize: vertical;
    }
    .sheet-submit {
      width: 100%;
      min-height: 54px;
      border-radius: 18px;
      border: 1px solid rgba(255, 106, 19, 0.26);
      background: linear-gradient(180deg, #ffd78c, #ffb347 58%, #ff8f3a);
      color: var(--ink);
      font: inherit;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    audio { display: none; }
    @media (max-width: 420px) {
      .control-grid,
      .context-grid {
        grid-template-columns: 1fr;
      }
      .counter {
        min-width: 86px;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel topbar">
      <div class="title-row">
        <div class="brand">
          <span class="brand-mark"><span class="brand-dot"></span> pocket verifier</span>
          <h1 class="brand-title">EZ Review</h1>
          <p class="brand-subtitle" id="session-name">Loading batch…</p>
        </div>
        <div class="counter" id="counter">0 / 0</div>
      </div>
      <div class="control-grid">
        <label>Batch<select id="session-select"></select></label>
        <label>Song<select id="song-select"></select></label>
        <label>Layer<select id="layer-select"></select></label>
        <label>Lane<select id="class-filter"></select></label>
      </div>
      <div class="meter-strip">
        <div class="meter-copy">
          <span id="session-meta">Waiting for queue…</span>
          <span id="history-meta">Queue mode</span>
        </div>
        <div class="progress-rail"><div class="progress-fill" id="progress-fill"></div></div>
      </div>
    </section>

    <section class="panel workspace">
      <div class="meta-row">
        <div class="meta-pills">
          <span class="pill" id="mode-pill"><strong>Queue</strong></span>
          <span class="pill" id="sample-pill">Pending</span>
        </div>
        <div class="app-session-meta" id="app-session-meta"></div>
      </div>
      <div class="center" id="center"></div>
      <p class="status-line" id="status-line">Switching sample…</p>
    </section>

    <section class="panel bottom-bar">
      <button class="action nav" id="prev-btn" title="Previous">
        <span>←</span>
        <span class="action-label">Prev</span>
      </button>
      <button class="action incorrect" id="incorrect-btn" title="Incorrect">✕</button>
      <button class="action reclass" id="reclass-btn" title="Reclassify">✦</button>
      <button class="action correct" id="correct-btn" title="Correct">✓</button>
      <button class="action nav" id="next-btn" title="Next">
        <span>→</span>
        <span class="action-label">Next</span>
      </button>
    </section>
  </main>

  <div class="sheet-backdrop" id="sheet-backdrop"></div>
  <section class="sheet" id="reclass-sheet">
    <div class="sheet-head">
      <h2 class="sheet-title">Reclassify</h2>
      <button class="sheet-close" id="sheet-close" title="Close">✕</button>
    </div>
    <div class="class-grid" id="class-grid"></div>
    <label class="sheet-field">Pick or Type a Class
      <input id="reclass-label" type="text" placeholder="kick, snare, shaker…">
    </label>
    <label class="sheet-field">Describe It
      <textarea id="reclass-note" placeholder="Plain English note for later cleanup or LLM relabeling."></textarea>
    </label>
    <button class="sheet-submit" id="sheet-submit">Save + Next</button>
  </section>

  <audio id="player" preload="auto"></audio>

  <script>
    const query = new URLSearchParams(window.location.search);
    const state = {
      sessionId: query.get('sessionId') || '',
      songRef: query.get('songRef') || 'all',
      layerRef: query.get('layerRef') || 'all',
      targetClass: query.get('targetClass') || 'all',
      cursor: 0,
      snapshot: null,
      sessions: [],
      selectedClassChip: '',
      historyBack: [],
      historyForward: [],
    };

    const player = document.getElementById('player');

    function selectedSessionId() {
      state.sessionId = state.sessionId || state.sessions[0]?.id || '';
      return state.sessionId;
    }

    function buildSessionParams({ cursor = state.cursor, itemId = '' } = {}) {
      const params = new URLSearchParams({
        sessionId: selectedSessionId(),
        cursor: String(Math.max(0, Number(cursor) || 0)),
        outcome: 'pending',
        targetClass: state.targetClass || 'all',
        songRef: state.songRef || 'all',
        layerRef: state.layerRef || 'all',
      });
      if (itemId) params.set('itemId', itemId);
      return params;
    }

    function updateUrl() {
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set('sessionId', selectedSessionId());
      if (state.songRef && state.songRef !== 'all') nextUrl.searchParams.set('songRef', state.songRef);
      else nextUrl.searchParams.delete('songRef');
      if (state.layerRef && state.layerRef !== 'all') nextUrl.searchParams.set('layerRef', state.layerRef);
      else nextUrl.searchParams.delete('layerRef');
      if (state.targetClass && state.targetClass !== 'all') nextUrl.searchParams.set('targetClass', state.targetClass);
      else nextUrl.searchParams.delete('targetClass');
      window.history.replaceState({}, '', nextUrl);
    }

    function resetHistory() {
      state.historyBack = [];
      state.historyForward = [];
    }

    function currentHistoryEntry() {
      const itemId = state.snapshot?.currentItem?.itemId;
      return itemId ? { itemId } : null;
    }

    function pushHistoryEntry(stack, entry) {
      if (!entry) return;
      if (stack[stack.length - 1]?.itemId === entry.itemId) return;
      stack.push(entry);
      if (stack.length > 64) stack.shift();
    }

    async function fetchJson(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return payload;
    }

    async function loadSessions() {
      const payload = await fetchJson('/api/sessions');
      state.sessions = payload.items || [];
      if (!state.sessionId) {
        state.sessionId = payload.defaultSessionId || state.sessions[0]?.id || '';
      }
      renderSessionSelect();
    }

    async function loadLiveCursor(cursor = 0) {
      state.cursor = Math.max(0, Number(cursor) || 0);
      const payload = await fetchJson('/api/session?' + buildSessionParams({ cursor: state.cursor }).toString());
      applySnapshot(payload);
    }

    async function loadFocusedItem(itemId) {
      const payload = await fetchJson('/api/session?' + buildSessionParams({ itemId }).toString());
      applySnapshot(payload);
    }

    function applySnapshot(payload) {
      state.snapshot = payload;
      state.cursor = payload.navigation?.cursor ?? 0;
      state.songRef = payload.filters?.songRef || 'all';
      state.layerRef = payload.filters?.layerRef || 'all';
      state.targetClass = payload.filters?.targetClass || 'all';
      render(payload);
    }

    async function postReview(outcome, extra = {}) {
      const item = state.snapshot?.currentItem;
      if (!item) return;
      pushHistoryEntry(state.historyBack, currentHistoryEntry());
      state.historyForward = [];
      const response = await fetchJson('/api/review?' + buildSessionParams().toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ itemId: item.itemId, outcome, ...extra }),
      });
      applySnapshot(response);
    }

    async function navigateBack() {
      const previousEntry = state.historyBack.pop();
      if (previousEntry) {
        pushHistoryEntry(state.historyForward, currentHistoryEntry());
        await loadFocusedItem(previousEntry.itemId);
        return;
      }
      const previousCursor = state.snapshot?.navigation?.previousCursor;
      if (previousCursor === null || previousCursor === undefined) return;
      pushHistoryEntry(state.historyBack, currentHistoryEntry());
      state.historyForward = [];
      await loadLiveCursor(previousCursor);
    }

    async function navigateForward() {
      const nextHistoryEntry = state.historyForward.pop();
      if (nextHistoryEntry) {
        pushHistoryEntry(state.historyBack, currentHistoryEntry());
        await loadFocusedItem(nextHistoryEntry.itemId);
        return;
      }
      const nextCursor = state.snapshot?.navigation?.nextCursor;
      if (nextCursor === null || nextCursor === undefined) return;
      pushHistoryEntry(state.historyBack, currentHistoryEntry());
      state.historyForward = [];
      await loadLiveCursor(nextCursor);
    }

    function renderSessionSelect() {
      const select = document.getElementById('session-select');
      select.innerHTML = state.sessions
        .map((session) => `<option value="${session.id}">${session.name}</option>`)
        .join('');
      if (state.sessionId) {
        select.value = state.sessionId;
      }
    }

    function renderSelect(selectId, items, allLabel, currentValue) {
      const select = document.getElementById(selectId);
      const options = [`<option value="all">${allLabel}</option>`].concat(
        (items || []).map((item) => `<option value="${item.value}">${item.label}</option>`)
      );
      select.innerHTML = options.join('');
      const validValues = new Set(['all'].concat((items || []).map((item) => item.value)));
      const nextValue = validValues.has(currentValue) ? currentValue : 'all';
      select.value = nextValue;
      return nextValue;
    }

    function renderClassFilter(classMap) {
      const options = (classMap || []).map((name) => ({ value: name, label: name }));
      state.targetClass = renderSelect('class-filter', options, 'All lanes', state.targetClass);
    }

    function renderSongFilter(scopeOptions) {
      state.songRef = renderSelect('song-select', scopeOptions?.songs || [], 'All songs', state.songRef);
    }

    function renderLayerFilter(scopeOptions) {
      state.layerRef = renderSelect('layer-select', scopeOptions?.layers || [], 'All layers', state.layerRef);
    }

    function syncNavigationButtons(navigation) {
      document.getElementById('prev-btn').disabled =
        state.historyBack.length === 0 && !navigation?.hasPrevious;
      document.getElementById('next-btn').disabled =
        state.historyForward.length === 0 && !navigation?.hasNext;
    }

    function syncActionButtons(isEnabled) {
      document.getElementById('incorrect-btn').disabled = !isEnabled;
      document.getElementById('reclass-btn').disabled = !isEnabled;
      document.getElementById('correct-btn').disabled = !isEnabled;
    }

    function render(payload) {
      const session = payload.session || {};
      const item = payload.currentItem;
      const navigation = payload.navigation || {};
      const viewMode = navigation.viewMode || 'queue';
      const reviewed = session.reviewedCount || 0;
      const total = session.totalItems || 0;
      const pending = session.pendingCount || 0;

      document.getElementById('session-name').textContent = session.name || 'Review batch';
      document.getElementById('counter').textContent = `${reviewed} / ${total}`;
      document.getElementById('session-meta').textContent = navigation.filteredCount
        ? `${navigation.currentItemNumber} of ${navigation.filteredCount} live · ${pending} pending · ${formatReviewMode(session.reviewMode)}`
        : `${pending} pending · ${formatReviewMode(session.reviewMode)}`;
      document.getElementById('history-meta').textContent = viewMode === 'history'
        ? `History ${state.historyBack.length} back · ${state.historyForward.length} fwd`
        : `Queue ${state.historyBack.length} back · ${state.historyForward.length} fwd`;
      document.getElementById('progress-fill').style.width =
        `${Math.max(0, Math.min(100, (session.completionRatio || 0) * 100))}%`;
      document.getElementById('app-session-meta').textContent =
        formatApplicationSession(session.applicationSession);

      renderSessionSelect();
      renderSongFilter(session.scopeOptions);
      renderLayerFilter(session.scopeOptions);
      renderClassFilter(session.classMap || []);
      renderClassGrid(session.classMap || []);
      updateUrl();

      const modePill = document.getElementById('mode-pill');
      modePill.classList.toggle('mode-history', viewMode === 'history');
      modePill.innerHTML = `<strong>${viewMode === 'history' ? 'History' : 'Queue'}</strong>`;
      document.getElementById('sample-pill').textContent =
        item ? `${(item.reviewOutcome || 'pending').toUpperCase()} · ${item.polarity || 'sample'}` : 'Queue clear';
      syncNavigationButtons(navigation);

      const center = document.getElementById('center');
      if (!item) {
        center.innerHTML = `
          <div class="empty-state">
            <div class="empty-mark">clear</div>
            <div>Queue clear for this batch, song, layer, and lane.</div>
          </div>
        `;
        document.getElementById('status-line').textContent = 'Select another slice or batch.';
        player.pause();
        player.removeAttribute('src');
        player.load();
        syncActionButtons(false);
        return;
      }

      syncActionButtons(true);
      center.innerHTML = `
        <section class="hero">
          <span class="hero-ribbon">${displaySong(item)} / ${displayLayer(item)}</span>
          <h2 class="prediction">${item.predictedLabel}</h2>
          <p class="file-name">${item.fileName}</p>
          <div class="context-grid">
            <div class="context-card">
              <span>Song</span>
              <strong>${displaySong(item)}</strong>
            </div>
            <div class="context-card">
              <span>Layer</span>
              <strong>${displayLayer(item)}</strong>
            </div>
            <div class="context-card">
              <span>Mode</span>
              <strong>${formatOutcome(item.reviewOutcome)}</strong>
            </div>
            <div class="context-card">
              <span>Score</span>
              <strong>${formatScore(item.score)}</strong>
            </div>
          </div>
        </section>
        <div class="play-shell">
          <button class="play-button" id="play-button" title="Play sample">
            <span class="play-glyph">▶</span>
            <span class="play-caption">Play</span>
          </button>
        </div>
      `;
      document.getElementById('play-button').addEventListener('click', () => replayCurrentItem().catch(showStatus));
      document.getElementById('status-line').textContent = describeStatusLine(item, viewMode);
      switchToItem(item).catch(showStatus);
    }

    function formatOutcome(outcome) {
      return String(outcome || 'pending').replaceAll('_', ' ');
    }

    function formatReviewMode(reviewMode) {
      return String(reviewMode || 'all_events').replaceAll('_', ' ');
    }

    function formatApplicationSession(applicationSession) {
      if (!applicationSession) return '';
      const parts = [];
      if (applicationSession.activeSongId) parts.push(`Song ${applicationSession.activeSongId}`);
      if (applicationSession.activeSongVersionId) parts.push(`Version ${applicationSession.activeSongVersionId}`);
      if (applicationSession.activeTimelineId) parts.push(`Timeline ${applicationSession.activeTimelineId}`);
      if (applicationSession.sessionId) parts.push(`App ${applicationSession.sessionId}`);
      return parts.join(' · ');
    }

    function formatScore(score) {
      if (score === null || score === undefined || Number.isNaN(Number(score))) return 'n/a';
      return `${Math.round(Number(score) * 100)}%`;
    }

    function readableRef(value) {
      if (!value) return 'unknown';
      const parts = String(value).split(':');
      return (parts[1] || parts[0] || 'unknown').replaceAll('_', ' ');
    }

    function displaySong(item) {
      return item.songTitle || readableRef(item.songRef);
    }

    function displayLayer(item) {
      return item.layerName || item.targetClass || readableRef(item.layerRef);
    }

    function describeStatusLine(item, viewMode) {
      if (item.correctedLabel) return `Relabeled as ${item.correctedLabel}`;
      if (viewMode === 'history') return `History view · ${formatOutcome(item.reviewOutcome)}`;
      return 'Loading audio…';
    }

    async function switchToItem(item) {
      player.pause();
      player.src = item.audioUrl;
      player.load();
      await attemptPlayback();
    }

    async function attemptPlayback() {
      try {
        player.currentTime = 0;
        await player.play();
        document.getElementById('status-line').textContent = 'Playing';
      } catch (error) {
        document.getElementById('status-line').textContent = 'Tap Play';
      }
    }

    async function replayCurrentItem() {
      if (!state.snapshot?.currentItem) return;
      await attemptPlayback();
    }

    function showStatus(error) {
      document.getElementById('status-line').textContent = error.message || String(error);
    }

    function openSheet() {
      const item = state.snapshot?.currentItem;
      if (!item) return;
      state.selectedClassChip = item.correctedLabel || '';
      document.getElementById('reclass-label').value = item.correctedLabel || '';
      document.getElementById('reclass-note').value = item.reviewNote || '';
      syncSelectedClassChip();
      document.getElementById('sheet-backdrop').classList.add('open');
      document.getElementById('reclass-sheet').classList.add('open');
    }

    function closeSheet() {
      document.getElementById('sheet-backdrop').classList.remove('open');
      document.getElementById('reclass-sheet').classList.remove('open');
    }

    function renderClassGrid(classMap) {
      const grid = document.getElementById('class-grid');
      grid.innerHTML = (classMap || [])
        .map((name) => `<button class="class-chip" data-class-chip="${name}">${name}</button>`)
        .join('');
      grid.querySelectorAll('[data-class-chip]').forEach((button) => {
        button.addEventListener('click', () => {
          state.selectedClassChip = button.dataset.classChip || '';
          document.getElementById('reclass-label').value = state.selectedClassChip;
          syncSelectedClassChip();
        });
      });
      syncSelectedClassChip();
    }

    function syncSelectedClassChip() {
      document.querySelectorAll('[data-class-chip]').forEach((button) => {
        button.classList.toggle('selected', button.dataset.classChip === state.selectedClassChip);
      });
    }

    async function markIncorrect() {
      await postReview('incorrect');
    }

    async function markCorrect() {
      await postReview('correct');
    }

    async function submitReclass() {
      const correctedLabel = document.getElementById('reclass-label').value.trim();
      const reviewNote = document.getElementById('reclass-note').value.trim();
      if (!correctedLabel && !reviewNote) {
        showStatus(new Error('Pick a class or describe the sound.'));
        return;
      }
      closeSheet();
      await postReview('incorrect', { correctedLabel, reviewNote });
    }

    function refreshScopeFromControls() {
      updateUrl();
      resetHistory();
      state.cursor = 0;
      loadLiveCursor(0).catch(showStatus);
    }

    document.getElementById('session-select').addEventListener('change', (event) => {
      state.sessionId = event.target.value;
      state.songRef = 'all';
      state.layerRef = 'all';
      state.targetClass = 'all';
      refreshScopeFromControls();
    });
    document.getElementById('song-select').addEventListener('change', (event) => {
      state.songRef = event.target.value || 'all';
      state.layerRef = 'all';
      refreshScopeFromControls();
    });
    document.getElementById('layer-select').addEventListener('change', (event) => {
      state.layerRef = event.target.value || 'all';
      refreshScopeFromControls();
    });
    document.getElementById('class-filter').addEventListener('change', (event) => {
      state.targetClass = event.target.value || 'all';
      refreshScopeFromControls();
    });
    document.getElementById('prev-btn').addEventListener('click', () => navigateBack().catch(showStatus));
    document.getElementById('next-btn').addEventListener('click', () => navigateForward().catch(showStatus));
    document.getElementById('incorrect-btn').addEventListener('click', () => markIncorrect().catch(showStatus));
    document.getElementById('correct-btn').addEventListener('click', () => markCorrect().catch(showStatus));
    document.getElementById('reclass-btn').addEventListener('click', openSheet);
    document.getElementById('sheet-backdrop').addEventListener('click', closeSheet);
    document.getElementById('sheet-close').addEventListener('click', closeSheet);
    document.getElementById('sheet-submit').addEventListener('click', () => submitReclass().catch(showStatus));
    document.getElementById('reclass-label').addEventListener('input', (event) => {
      state.selectedClassChip = event.target.value.trim();
      syncSelectedClassChip();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft') markIncorrect().catch(showStatus);
      if (event.key === 'ArrowRight') markCorrect().catch(showStatus);
      if (event.key.toLowerCase() === 'p') navigateBack().catch(showStatus);
      if (event.key.toLowerCase() === 'n') navigateForward().catch(showStatus);
      if (event.key.toLowerCase() === 'r') openSheet();
      if (event.key === 'Escape') closeSheet();
      if (event.key === ' ') {
        event.preventDefault();
        replayCurrentItem().catch(showStatus);
      }
    });

    loadSessions()
      .then(() => loadLiveCursor(0))
      .catch(showStatus);
  </script>
</body>
</html>"""
