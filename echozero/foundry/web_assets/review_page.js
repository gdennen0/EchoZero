const state = {
  songRef: 'all',
  layerRef: 'all',
  cursor: 0,
  snapshot: null,
  posting: false,
};

const player = document.getElementById('player');

async function fetchJson(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function buildParams({ cursor = state.cursor, itemId = '' } = {}) {
  const params = new URLSearchParams({
    cursor: String(Math.max(0, Number(cursor) || 0)),
    outcome: 'all',
    songRef: state.songRef || 'all',
    layerRef: state.layerRef || 'all',
  });
  if (itemId) params.set('itemId', itemId);
  return params;
}

function showStatus(message) {
  const line = document.getElementById('status-line');
  line.textContent = String(message || 'Ready');
}

function readableRef(value) {
  const text = String(value || '').trim();
  if (!text) return 'Unknown';
  const index = text.indexOf(':');
  return index >= 0 ? text.slice(index + 1) : text;
}

function formatOutcome(value) {
  const text = String(value || 'pending').toLowerCase();
  if (text === 'correct') return 'Promoted';
  if (text === 'incorrect') return 'Demoted';
  return 'Pending';
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(2);
}

function formatClipRange(item) {
  const startMs = item?.windowStartMs;
  const endMs = item?.windowEndMs;
  if (startMs === null || startMs === undefined || endMs === null || endMs === undefined) {
    return 'Clip range unavailable';
  }
  const start = (Number(startMs) / 1000).toFixed(2);
  const end = (Number(endMs) / 1000).toFixed(2);
  return `${start}s - ${end}s`;
}

function describePosition(navigation) {
  const current = Number(navigation?.currentScopeItemNumber || 0);
  const total = Number(navigation?.scopeCount || 0);
  if (!total) return 'No events in scope';
  return `Event ${current || 1} of ${total}`;
}

function applySelectOptions(elementId, options, currentValue, fallbackLabel) {
  const select = document.getElementById(elementId);
  const rows = Array.isArray(options) ? options : [];
  const values = new Set(['all']);
  const optionHtml = [`<option value="all">${fallbackLabel}</option>`];
  rows.forEach((row) => {
    const value = String(row?.value || '').trim();
    if (!value) return;
    values.add(value);
    const count = row?.itemCount !== undefined ? ` (${row.itemCount})` : '';
    optionHtml.push(`<option value="${value}">${row.label || value}${count}</option>`);
  });
  select.innerHTML = optionHtml.join('');
  const nextValue = values.has(currentValue) ? currentValue : 'all';
  select.value = nextValue;
  return nextValue;
}

function renderProgress(progress) {
  const container = document.getElementById('progress-boxes');
  const items = Array.isArray(progress?.items) ? progress.items : [];
  if (!items.length) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = items
    .map((item, index) => (
      `<button class="progress-box ${item.reviewOutcome || 'pending'} ${item.isCurrent ? 'current' : ''}" ` +
      `data-item-id="${item.itemId}" title="Item ${index + 1}: ${formatOutcome(item.reviewOutcome)}"></button>`
    ))
    .join('');
  container.querySelectorAll('[data-item-id]').forEach((button) => {
    button.addEventListener('click', () => {
      const itemId = String(button.dataset.itemId || '').trim();
      if (!itemId) return;
      loadSnapshot({ itemId }).catch((error) => showStatus(error.message));
    });
  });
  container.querySelector('.progress-box.current')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

function syncButtons(navigation, hasItem) {
  const prev = document.getElementById('prev-btn');
  const next = document.getElementById('next-btn');
  prev.disabled = !navigation?.hasPrevious;
  next.disabled = !navigation?.hasNext;

  const demote = document.getElementById('incorrect-btn');
  const reclass = document.getElementById('reclass-btn');
  const promote = document.getElementById('correct-btn');
  demote.disabled = !hasItem || state.posting;
  reclass.disabled = !hasItem || state.posting;
  promote.disabled = !hasItem || state.posting;
}

async function replayCurrentItem() {
  const item = state.snapshot?.currentItem;
  if (!item?.audioUrl) return;
  if (player.src !== item.audioUrl) {
    player.src = item.audioUrl;
  }
  player.currentTime = 0;
  await player.play();
}

function renderCurrentItem(center, item, navigation) {
  const songLabel = item.songTitle || readableRef(item.songRef);
  const layerLabel = item.layerName || item.targetClass || readableRef(item.layerRef);
  center.innerHTML = `
    <section class="hero">
      <div class="hero-head">
        <div class="hero-copy">
          <div class="hero-context">${songLabel} · ${layerLabel}</div>
          <div class="hero-summary">
            <span class="hero-label">${item.predictedLabel || item.targetClass || 'event'}</span>
            <p class="hero-meta">${formatClipRange(item)} · score ${formatScore(item.score)}</p>
          </div>
        </div>
      </div>
      <div class="wave-shell">
        <div class="wave-meta">
          <span>${item.targetClass || 'event'}</span>
          <div class="wave-actions">
            <span>${item.polarity || 'sample'}</span>
            <button class="replay-button" id="play-button" title="Replay clip">Replay</button>
          </div>
        </div>
      </div>
      <div class="hero-stats">
        <span class="hero-stat">${describePosition(navigation)}</span>
        <span class="hero-stat">${formatOutcome(item.reviewOutcome)}</span>
      </div>
    </section>
  `;
  document.getElementById('play-button')?.addEventListener('click', () => {
    replayCurrentItem().catch((error) => showStatus(error.message));
  });
}

function renderEmptyState(center, navigation) {
  center.innerHTML = `
    <div class="empty-state">
      <div class="empty-mark">LIVE</div>
      <h2 class="empty-title">No Events In Current Scope</h2>
      <p class="empty-copy">Adjust Song or Event Layer to continue reviewing.</p>
    </div>
  `;
  const scopeCount = Number(navigation?.scopeCount || 0);
  if (!scopeCount) {
    showStatus('No events available for current runtime context.');
  } else {
    showStatus('No matching events for current filters.');
  }
}

function render(payload) {
  state.snapshot = payload;
  const session = payload.session || {};
  const project = payload.sessions?.project || {};
  const navigation = payload.navigation || {};
  const progress = payload.progress || {};
  const item = payload.currentItem || null;

  const projectName =
    session.applicationSession?.projectName
    || project.projectName
    || 'Current project';
  document.getElementById('project-name').textContent = projectName;
  document.getElementById('session-name').textContent = session.name || 'Live review';

  const reviewed = Number(session.reviewedCount || 0);
  const total = Number(session.totalItems || 0);
  const pending = Number(session.pendingCount || 0);
  document.getElementById('counter').textContent = `${reviewed} / ${total}`;
  document.getElementById('counter-detail').textContent = `Pending ${pending}`;
  document.getElementById('scope-summary').textContent = describePosition(navigation);
  document.getElementById('history-meta').textContent = 'Live';

  state.songRef = applySelectOptions(
    'song-select',
    session.scopeOptions?.songs || [],
    state.songRef,
    'All songs'
  );
  state.layerRef = applySelectOptions(
    'layer-select',
    session.scopeOptions?.layers || [],
    state.layerRef,
    'All layers'
  );

  renderProgress(progress);
  syncButtons(navigation, Boolean(item));

  const center = document.getElementById('center');
  if (item) {
    renderCurrentItem(center, item, navigation);
    showStatus('Live controller ready.');
  } else {
    renderEmptyState(center, navigation);
  }
}

async function loadSnapshot({ cursor = state.cursor, itemId = '' } = {}) {
  const nextCursor = Math.max(0, Number(cursor) || 0);
  state.cursor = nextCursor;
  const payload = await fetchJson('/api/session?' + buildParams({ cursor: nextCursor, itemId }).toString());
  const resolvedCursor = payload?.navigation?.cursor;
  state.cursor = Number.isInteger(resolvedCursor) ? resolvedCursor : nextCursor;
  render(payload);
}

async function postDecision(outcome, extra = {}) {
  const item = state.snapshot?.currentItem;
  if (!item || state.posting) return;
  state.posting = true;
  syncButtons(state.snapshot?.navigation, true);
  try {
    const payload = await fetchJson('/api/review?' + buildParams().toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ itemId: item.itemId, outcome, ...extra }),
    });
    render(payload);
    const nextCursor = payload?.navigation?.nextCursor;
    if (Number.isInteger(nextCursor)) {
      await loadSnapshot({ cursor: nextCursor });
    }
  } finally {
    state.posting = false;
    syncButtons(state.snapshot?.navigation, Boolean(state.snapshot?.currentItem));
  }
}

function openReclassSheet() {
  const sheet = document.getElementById('reclass-sheet');
  const backdrop = document.getElementById('sheet-backdrop');
  sheet.classList.add('is-open');
  backdrop.classList.add('is-open');
  const predicted = state.snapshot?.currentItem?.predictedLabel || '';
  document.getElementById('reclass-label').value = predicted;
  document.getElementById('reclass-note').value = '';
  document.getElementById('reclass-label').focus();
}

function closeReclassSheet() {
  document.getElementById('reclass-sheet').classList.remove('is-open');
  document.getElementById('sheet-backdrop').classList.remove('is-open');
}

function renderClassGrid() {
  const classMap = state.snapshot?.session?.classMap || [];
  const grid = document.getElementById('class-grid');
  grid.innerHTML = classMap
    .map((name) => `<button class="class-chip" data-class-name="${name}">${name}</button>`)
    .join('');
  grid.querySelectorAll('[data-class-name]').forEach((button) => {
    button.addEventListener('click', () => {
      document.getElementById('reclass-label').value = String(button.dataset.className || '').trim();
    });
  });
}

function bindEvents() {
  document.getElementById('song-select').addEventListener('change', (event) => {
    state.songRef = String(event.target.value || 'all');
    state.layerRef = 'all';
    loadSnapshot({ cursor: 0 }).catch((error) => showStatus(error.message));
  });

  document.getElementById('layer-select').addEventListener('change', (event) => {
    state.layerRef = String(event.target.value || 'all');
    loadSnapshot({ cursor: 0 }).catch((error) => showStatus(error.message));
  });

  document.getElementById('prev-btn').addEventListener('click', () => {
    const previous = state.snapshot?.navigation?.previousCursor;
    if (!Number.isInteger(previous)) return;
    loadSnapshot({ cursor: previous }).catch((error) => showStatus(error.message));
  });

  document.getElementById('next-btn').addEventListener('click', () => {
    const next = state.snapshot?.navigation?.nextCursor;
    if (!Number.isInteger(next)) return;
    loadSnapshot({ cursor: next }).catch((error) => showStatus(error.message));
  });

  document.getElementById('incorrect-btn').addEventListener('click', () => {
    postDecision('incorrect').catch((error) => showStatus(error.message));
  });

  document.getElementById('correct-btn').addEventListener('click', () => {
    postDecision('correct').catch((error) => showStatus(error.message));
  });

  document.getElementById('reclass-btn').addEventListener('click', () => {
    renderClassGrid();
    openReclassSheet();
  });

  document.getElementById('sheet-close').addEventListener('click', closeReclassSheet);
  document.getElementById('sheet-backdrop').addEventListener('click', closeReclassSheet);

  document.getElementById('sheet-submit').addEventListener('click', () => {
    const correctedLabel = String(document.getElementById('reclass-label').value || '').trim();
    const reviewNote = String(document.getElementById('reclass-note').value || '').trim();
    if (!correctedLabel) {
      showStatus('Choose a target class for reclassify.');
      return;
    }
    closeReclassSheet();
    postDecision('incorrect', { correctedLabel, reviewNote }).catch((error) => showStatus(error.message));
  });
}

async function boot() {
  bindEvents();
  await loadSnapshot({ cursor: 0 });
}

boot().catch((error) => {
  showStatus(error.message || 'Review controller failed to load.');
});
