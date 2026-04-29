
    const initialQuery = new URLSearchParams(window.location.search);
    const state = {
      explicitSessionQuery: initialQuery.has('sessionId'),
      defaultSessionId: '',
      sessionId: initialQuery.get('sessionId') || '',
      projectContext: {},
      outcome: initialQuery.get('outcome') || 'pending',
      songRef: initialQuery.get('songRef') || 'all',
      layerRef: initialQuery.get('layerRef') || 'all',
      targetClass: initialQuery.get('targetClass') || 'all',
      cursor: 0,
      snapshot: null,
      sessions: [],
      selectedClassChip: '',
      historyBack: [],
      historyForward: [],
      didSeedScope: false,
      waveformToken: 0,
      audioContext: null,
      currentAudioUrl: '',
      audioClipCache: new Map(),
      audioClipLoads: new Map(),
      waveformCache: new Map(),
      waveformLoads: new Map(),
      prefetchLoads: new Map(),
      sessionsSignature: '',
      snapshotSignature: '',
      sessionsRequestId: 0,
      snapshotRequestId: 0,
    };
    const player = document.getElementById('player');
    const MEDIA_CACHE_LIMIT = 8;
    const REFRESH_INTERVAL_MS = 4000;

    function currentSearchParams() {
      return new URLSearchParams(window.location.search);
    }

    function selectedSessionId() {
      state.sessionId = state.sessionId || state.defaultSessionId || '';
      return state.sessionId;
    }

    function selectedSessionSummary() {
      const sessionId = selectedSessionId();
      return state.sessions.find((session) => session.id === sessionId) || null;
    }

    function buildSessionParams({ cursor = state.cursor, itemId = '' } = {}) {
      const params = new URLSearchParams({
        sessionId: selectedSessionId(),
        cursor: String(Math.max(0, Number(cursor) || 0)),
        outcome: state.outcome || 'pending',
        targetClass: state.targetClass || 'all',
        songRef: state.songRef || 'all',
        layerRef: state.layerRef || 'all',
      });
      if (itemId) params.set('itemId', itemId);
      return params;
    }

    function updateUrl() {
      const nextUrl = new URL(window.location.href);
      const sessionId = selectedSessionId();
      if (sessionId && (
        state.explicitSessionQuery
        || !state.defaultSessionId
        || sessionId !== state.defaultSessionId
      )) {
        nextUrl.searchParams.set('sessionId', sessionId);
      } else {
        nextUrl.searchParams.delete('sessionId');
      }
      if (state.outcome && state.outcome !== 'pending') nextUrl.searchParams.set('outcome', state.outcome);
      else nextUrl.searchParams.delete('outcome');
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

    function clearMediaCaches() {
      state.audioClipCache.forEach((clip) => {
        if (clip?.blobUrl) URL.revokeObjectURL(clip.blobUrl);
      });
      state.audioClipCache.clear();
      state.audioClipLoads.clear();
      state.waveformCache.clear();
      state.waveformLoads.clear();
      state.prefetchLoads.clear();
      state.waveformToken += 1;
    }

    function resetReviewState({ preserveSessionId = true } = {}) {
      resetHistory();
      clearMediaCaches();
      state.cursor = 0;
      state.snapshot = null;
      state.sessionsSignature = '';
      state.snapshotSignature = '';
      state.didSeedScope = false;
      state.selectedClassChip = '';
      state.currentAudioUrl = '';
      if (!preserveSessionId) state.sessionId = '';
      player.pause();
      player.removeAttribute('src');
      player.load();
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

    async function applyNavigationTransition(run, { historyStack = null, clearForward = false } = {}) {
      const currentEntry = currentHistoryEntry();
      await run();
      if (historyStack) pushHistoryEntry(historyStack, currentEntry);
      if (clearForward) state.historyForward = [];
      syncNavigationButtons(state.snapshot?.navigation);
    }

    function isMissingHistoryItemError(error) {
      return String(error?.message || error || '').includes('ReviewItem not found');
    }

    function rememberCacheEntry(cache, key, value) {
      if (!key) return value;
      if (cache.has(key)) cache.delete(key);
      cache.set(key, value);
      while (cache.size > MEDIA_CACHE_LIMIT) {
        const oldestKey = cache.keys().next().value;
        const oldestValue = cache.get(oldestKey);
        cache.delete(oldestKey);
        if (oldestValue?.blobUrl) URL.revokeObjectURL(oldestValue.blobUrl);
      }
      return value;
    }

    function cachedAudioClip(audioUrl) {
      if (!audioUrl) return null;
      const cached = state.audioClipCache.get(audioUrl) || null;
      if (cached) rememberCacheEntry(state.audioClipCache, audioUrl, cached);
      return cached;
    }

    async function ensureAudioClip(item) {
      const audioUrl = item?.audioUrl;
      if (!audioUrl) return null;
      const cached = cachedAudioClip(audioUrl);
      if (cached) return cached;
      const pending = state.audioClipLoads.get(audioUrl);
      if (pending) return pending;
      const load = (async () => {
        const response = await fetch(audioUrl);
        if (!response.ok) throw new Error(`Audio request failed: ${response.status}`);
        const bytes = await response.arrayBuffer();
        const mimeType = response.headers.get('Content-Type') || 'audio/wav';
        const clip = {
          bytes,
          mimeType,
          blobUrl: URL.createObjectURL(new Blob([bytes], { type: mimeType })),
        };
        return rememberCacheEntry(state.audioClipCache, audioUrl, clip);
      })();
      state.audioClipLoads.set(audioUrl, load);
      try {
        return await load;
      } finally {
        if (state.audioClipLoads.get(audioUrl) === load) state.audioClipLoads.delete(audioUrl);
      }
    }

    async function ensureWaveformPeaks(item) {
      const audioUrl = item?.audioUrl;
      if (!audioUrl) return [];
      const cached = state.waveformCache.get(audioUrl);
      if (cached) {
        rememberCacheEntry(state.waveformCache, audioUrl, cached);
        return cached;
      }
      const pending = state.waveformLoads.get(audioUrl);
      if (pending) return pending;
      const load = (async () => {
        const clip = await ensureAudioClip(item);
        if (!clip?.bytes) return [];
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) return [];
        state.audioContext = state.audioContext || new AudioContextCtor();
        const buffer = await state.audioContext.decodeAudioData(clip.bytes.slice(0));
        const peaks = computeWaveformPeaks(buffer);
        rememberCacheEntry(state.waveformCache, audioUrl, peaks);
        return peaks;
      })();
      state.waveformLoads.set(audioUrl, load);
      try {
        return await load;
      } finally {
        if (state.waveformLoads.get(audioUrl) === load) state.waveformLoads.delete(audioUrl);
      }
    }

    function computeWaveformPeaks(buffer) {
      const samples = buffer.length;
      const channelCount = Math.max(1, buffer.numberOfChannels || 1);
      const bars = 120;
      const step = Math.max(1, Math.floor(samples / bars));
      const peaks = [];
      for (let index = 0; index < bars; index += 1) {
        let peak = 0;
        const start = index * step;
        const end = Math.min(samples, start + step);
        for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
          const channel = buffer.getChannelData(channelIndex);
          for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
            const value = Math.abs(channel[sampleIndex] || 0);
            if (value > peak) peak = value;
          }
        }
        peaks.push(peak);
      }
      return peaks;
    }

    function drawWaveformPeaks(peaks) {
      const canvas = document.getElementById('waveform');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#101419';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const mid = canvas.height / 2;
      ctx.strokeStyle = '#29313a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(canvas.width, mid);
      ctx.stroke();
      const normalizedPeaks = Array.isArray(peaks) && peaks.length ? peaks : [0];
      const barWidth = canvas.width / normalizedPeaks.length;
      ctx.fillStyle = '#8fb3ff';
      normalizedPeaks.forEach((peak, index) => {
        const height = Math.max(3, Number(peak || 0) * (canvas.height * 0.42));
        const x = index * barWidth;
        ctx.fillRect(x, mid - (height / 2), Math.max(2, barWidth - 2), height);
      });
    }

    function prefetchAdjacentItem(cursor) {
      if (cursor === null || cursor === undefined) return Promise.resolve();
      const key = buildSessionParams({ cursor }).toString();
      const pending = state.prefetchLoads.get(key);
      if (pending) return pending;
      const load = (async () => {
        const payload = await fetchJson('/api/session?' + buildSessionParams({ cursor }).toString());
        const item = payload?.currentItem;
        if (!item?.audioUrl) return;
        await ensureAudioClip(item);
        await ensureWaveformPeaks(item);
      })().catch(() => undefined);
      state.prefetchLoads.set(key, load);
      return load.finally(() => {
        if (state.prefetchLoads.get(key) === load) state.prefetchLoads.delete(key);
      });
    }

    function prefetchNeighborMedia(navigation) {
      prefetchAdjacentItem(navigation?.nextCursor);
      prefetchAdjacentItem(navigation?.previousCursor);
    }

    async function fetchJson(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return payload;
    }

    function sessionIndexSignature(payload) {
      const items = payload?.items || [];
      const project = payload?.project || {};
      return [
        payload?.stateRevision || 0,
        payload?.defaultSessionId || '',
        project.projectRef || '',
        project.projectName || '',
        project.projectRoot || '',
      ]
        .concat(items.map((session) => (
          [
            session.id || '',
            session.updatedAt || '',
            session.pendingCount || 0,
            session.reviewedCount || 0,
          ].join(':')
        )))
        .join('|');
    }

    function snapshotSignature(payload) {
      const session = payload?.session || {};
      const navigation = payload?.navigation || {};
      const currentItem = payload?.currentItem || {};
      const progressItems = (payload?.progress?.items || [])
        .map((item) => `${item.itemId || ''}:${item.reviewOutcome || 'pending'}:${item.isCurrent ? 1 : 0}`)
        .join('|');
      return [
        payload?.stateRevision || 0,
        session.id || '',
        session.updatedAt || '',
        session.reviewedCount || 0,
        session.pendingCount || 0,
        session.totalItems || 0,
        navigation.cursor ?? '',
        navigation.viewMode || '',
        payload?.filteredCount ?? 0,
        currentItem.itemId || '',
        currentItem.reviewOutcome || '',
        currentItem.correctedLabel || '',
        progressItems,
      ].join('||');
    }

    async function loadSessions() {
      const requestId = ++state.sessionsRequestId;
      const payload = await fetchJson('/api/sessions');
      if (requestId !== state.sessionsRequestId) {
        return { changed: false, sessionChanged: false };
      }
      const previousSignature = state.sessionsSignature;
      const previousDefaultSessionId = state.defaultSessionId;
      const previousSessionId = state.sessionId;
      const nextSignature = sessionIndexSignature(payload);
      state.sessions = payload.items || [];
      state.projectContext = payload.project || {};
      state.defaultSessionId = payload.defaultSessionId || '';
      const sessionIds = new Set(state.sessions.map((session) => session.id));
      if (!state.sessionId) {
        state.sessionId = state.defaultSessionId;
      } else if (!sessionIds.has(state.sessionId)) {
        state.sessionId = state.defaultSessionId;
      } else if (
        !state.explicitSessionQuery
        && previousSessionId
        && previousSessionId === previousDefaultSessionId
        && state.defaultSessionId
      ) {
        state.sessionId = state.defaultSessionId;
      }
      const sessionChanged = previousSessionId !== state.sessionId;
      if (sessionChanged) {
        resetReviewState();
        state.outcome = 'pending';
        state.songRef = 'all';
        state.layerRef = 'all';
        state.targetClass = 'all';
      }
      state.sessionsSignature = nextSignature;
      renderSessionSelect();
      return {
        changed: nextSignature !== previousSignature,
        sessionChanged,
      };
    }

    async function requestSnapshot({ cursor = state.cursor, itemId = '' } = {}) {
      const requestId = ++state.snapshotRequestId;
      const payload = await fetchJson('/api/session?' + buildSessionParams({ cursor, itemId }).toString());
      if (requestId !== state.snapshotRequestId) {
        return null;
      }
      return { payload, requestId };
    }

    async function loadLiveCursor(cursor = 0) {
      const nextCursor = Math.max(0, Number(cursor) || 0);
      const result = await requestSnapshot({ cursor: nextCursor });
      if (result) {
        applySnapshot(result.payload, { requestId: result.requestId });
      }
    }

    async function loadFocusedItem(itemId) {
      const result = await requestSnapshot({ itemId });
      if (result) {
        applySnapshot(result.payload, { requestId: result.requestId });
      }
    }

    function hasExplicitScopeQuery() {
      const liveQuery = currentSearchParams();
      return liveQuery.has('songRef') || liveQuery.has('layerRef') || liveQuery.has('targetClass');
    }

    function maybeSeedPreferredScope(payload) {
      if (state.didSeedScope || hasExplicitScopeQuery()) return false;
      const session = payload.session || {};
      const scopeOptions = session.scopeOptions || {};
      const songs = scopeOptions.songs || [];
      const preferredSongRef = preferredSongRefFromSession(session, songs);
      state.didSeedScope = true;
      if (state.songRef === 'all' && preferredSongRef && songs.some((song) => song.value === preferredSongRef)) {
        state.songRef = preferredSongRef;
        state.layerRef = 'all';
        state.cursor = 0;
        loadLiveCursor(0).catch(showStatus);
        return true;
      }
      return false;
    }

    function preferredSongRefFromSession(session, songs) {
      const applicationSession = session.applicationSession || {};
      if (applicationSession.activeSongRef) return applicationSession.activeSongRef;
      if (applicationSession.activeSongId) return `song:${applicationSession.activeSongId}`;
      if ((songs || []).length === 1) return songs[0].value;
      return '';
    }

    function applySnapshot(payload, { requestId = null, forceRender = false } = {}) {
      if (requestId !== null && requestId !== state.snapshotRequestId) return;
      const previousItemId = state.snapshot?.currentItem?.itemId || '';
      const previousAudioUrl = state.snapshot?.currentItem?.audioUrl || '';
      const nextSignature = snapshotSignature(payload);
      const snapshotChanged = nextSignature !== state.snapshotSignature;
      state.snapshot = payload;
      state.cursor = payload.navigation?.cursor ?? 0;
      state.outcome = payload.filters?.outcome || 'pending';
      state.songRef = payload.filters?.songRef || 'all';
      state.layerRef = payload.filters?.layerRef || 'all';
      state.targetClass = payload.filters?.targetClass || 'all';
      if (maybeSeedPreferredScope(payload)) return;
      if (!snapshotChanged && !forceRender) return;
      state.snapshotSignature = nextSignature;
      const currentItem = payload.currentItem || {};
      render(payload, {
        replayCurrentItem: (
          previousItemId !== (currentItem.itemId || '')
          || previousAudioUrl !== (currentItem.audioUrl || '')
        ),
      });
    }

    async function postReview(outcome, extra = {}) {
      const item = state.snapshot?.currentItem;
      if (!item) return;
      await applyNavigationTransition(async () => {
        const requestId = ++state.snapshotRequestId;
        const response = await fetchJson('/api/review?' + buildSessionParams().toString(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ itemId: item.itemId, outcome, ...extra }),
        });
        applySnapshot(response, { requestId, forceRender: true });
      }, { historyStack: state.historyBack, clearForward: true });
    }

    async function focusItemFromCurrentView(itemId) {
      const targetItemId = String(itemId || '').trim();
      if (!targetItemId || targetItemId === state.snapshot?.currentItem?.itemId) return;
      await applyNavigationTransition(
        () => loadFocusedItem(targetItemId),
        { historyStack: state.historyBack, clearForward: true },
      );
    }

    async function navigateBack() {
      while (state.historyBack.length > 0) {
        const previousEntry = state.historyBack.pop();
        const previousItemId = String(previousEntry?.itemId || '').trim();
        if (!previousItemId || previousItemId === state.snapshot?.currentItem?.itemId) continue;
        try {
          await applyNavigationTransition(
            () => loadFocusedItem(previousItemId),
            { historyStack: state.historyForward },
          );
          return;
        } catch (error) {
          if (isMissingHistoryItemError(error)) continue;
          throw error;
        }
      }
      const previousCursor = state.snapshot?.navigation?.previousCursor;
      if (previousCursor === null || previousCursor === undefined) return;
      await applyNavigationTransition(
        () => loadLiveCursor(previousCursor),
        { historyStack: state.historyForward },
      );
    }

    async function navigateForward() {
      while (state.historyForward.length > 0) {
        const nextHistoryEntry = state.historyForward.pop();
        const nextItemId = String(nextHistoryEntry?.itemId || '').trim();
        if (!nextItemId || nextItemId === state.snapshot?.currentItem?.itemId) continue;
        try {
          await applyNavigationTransition(
            () => loadFocusedItem(nextItemId),
            { historyStack: state.historyBack },
          );
          return;
        } catch (error) {
          if (isMissingHistoryItemError(error)) continue;
          throw error;
        }
      }
      const nextCursor = state.snapshot?.navigation?.nextCursor;
      if (nextCursor === null || nextCursor === undefined) return;
      await applyNavigationTransition(
        () => loadLiveCursor(nextCursor),
        { historyStack: state.historyBack, clearForward: true },
      );
    }

    function renderSessionSelect() {
      const select = document.getElementById('session-select');
      select.innerHTML = state.sessions
        .map((session) => {
          const status = session.pendingCount ? `${session.pendingCount} pending` : 'complete';
          const contextLabel = session.context?.sourceLabel || session.name;
          return `<option value="${session.id}">${session.name} · ${contextLabel} · ${status}</option>`;
        })
        .join('');
      if (state.sessionId) select.value = state.sessionId;
      const drawer = document.getElementById('session-drawer');
      drawer.hidden = state.sessions.length === 0 || (
        state.sessions.length <= 1 && Boolean(state.defaultSessionId)
      );
      if (drawer.hidden) drawer.open = false;
      document.getElementById('session-drawer-title').textContent =
        selectedSessionSummary()?.name || (state.defaultSessionId ? 'Current run' : 'Archived runs');
      document.getElementById('session-drawer-meta').textContent = `${state.sessions.length} loaded`;
    }

    function renderSelect(selectId, items, allLabel, currentValue) {
      const select = document.getElementById(selectId);
      const options = [`<option value="all">${allLabel}</option>`].concat(
        (items || []).map((item) => `<option value="${item.value}">${buildOptionLabel(item)}</option>`)
      );
      select.innerHTML = options.join('');
      const validValues = new Set(['all'].concat((items || []).map((item) => item.value)));
      const nextValue = validValues.has(currentValue) ? currentValue : 'all';
      select.value = nextValue;
      return nextValue;
    }

    function buildOptionLabel(item) {
      const count = item.itemCount !== undefined ? ` (${item.itemCount})` : '';
      return `${item.label}${count}`;
    }

    function renderOutcomeFilter() {
      const select = document.getElementById('outcome-filter');
      const options = [
        { value: 'pending', label: 'Pending' },
        { value: 'all', label: 'All' },
        { value: 'incorrect', label: 'Wrong' },
        { value: 'correct', label: 'Correct' },
      ];
      select.innerHTML = options
        .map((option) => `<option value="${option.value}">${option.label}</option>`)
        .join('');
      const allowedValues = new Set(options.map((option) => option.value));
      if (!allowedValues.has(state.outcome)) state.outcome = 'pending';
      select.value = state.outcome;
    }

    function renderSongFilter(scopeOptions) {
      state.songRef = renderSelect('song-select', scopeOptions?.songs || [], 'All songs', state.songRef);
    }

    function renderLayerFilter(scopeOptions) {
      state.layerRef = renderSelect('layer-select', scopeOptions?.layers || [], 'All layers', state.layerRef);
    }

    function renderClassFilter(classMap) {
      const options = (classMap || []).map((name) => ({ value: name, label: name }));
      state.targetClass = renderSelect('class-filter', options, 'All classes', state.targetClass);
    }

    function renderProgressBoxes(progress) {
      const container = document.getElementById('progress-boxes');
      const items = progress?.items || [];
      if (!items.length) {
        container.innerHTML = '';
        return;
      }
      container.innerHTML = items
        .map((item, index) => (
          `<button class="progress-box ${item.reviewOutcome || 'pending'} ${item.isCurrent ? 'current' : ''}" ` +
          `data-progress-item="${item.itemId}" title="Item ${index + 1}: ${formatOutcome(item.reviewOutcome)}"></button>`
        ))
        .join('');
      container.querySelectorAll('[data-progress-item]').forEach((button) => {
        button.addEventListener('click', () => focusItemFromCurrentView(button.dataset.progressItem).catch(showStatus));
      });
      container.querySelector('.progress-box.current')?.scrollIntoView({ block: 'nearest', inline: 'center' });
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

    function render(payload, { replayCurrentItem = true } = {}) {
      const session = payload.session || {};
      const sessionSummary = selectedSessionSummary() || {};
      const item = payload.currentItem;
      const navigation = payload.navigation || {};
      const context = session.context || {};
      const applicationSession = session.applicationSession || {};
      const projectContext = state.projectContext || {};
      const viewMode = navigation.viewMode || 'queue';
      const reviewed = session.reviewedCount || 0;
      const total = session.totalItems || 0;
      const pending = session.pendingCount || 0;

      document.getElementById('project-name').textContent =
        applicationSession.projectName
        || context.projectName
        || projectContext.projectName
        || context.sourceLabel
        || 'Review session';
      document.getElementById('session-name').textContent = session.name || 'Waiting for run';
      document.getElementById('counter').textContent = `${reviewed} / ${total}`;
      document.getElementById('counter-detail').textContent = `Pending ${pending}`;
      document.getElementById('scope-summary').textContent = buildScopeSummary(session, navigation, pending);
      document.getElementById('history-meta').textContent = viewMode === 'history'
        ? 'Revisit'
        : 'In order';
      renderProgressBoxes(payload.progress);
      document.getElementById('active-context').textContent =
        buildActiveContext(applicationSession, item, context);
      document.getElementById('source-context').textContent =
        buildSourceContext(context, sessionSummary, applicationSession);
      document.getElementById('archive-status').textContent =
        buildArchiveStatus(sessionSummary);

      renderOutcomeFilter();
      renderSessionSelect();
      renderSongFilter(session.scopeOptions);
      renderLayerFilter(session.scopeOptions);
      renderClassFilter(session.classMap || []);
      renderClassGrid(session.classMap || []);
      updateUrl();

      const modePill = document.getElementById('mode-pill');
      modePill.classList.toggle('mode-history', viewMode === 'history');
      modePill.innerHTML = `<strong>${viewMode === 'history' ? 'History' : 'Queue'}</strong>`;
      document.getElementById('kind-pill').textContent = formatSourceKind(context.sourceKind);
      document.getElementById('sample-pill').textContent =
        item ? `${formatOutcome(state.outcome)} · ${(item.polarity || 'sample').toUpperCase()}` : `${formatOutcome(state.outcome)} · idle`;
      syncNavigationButtons(navigation);

      const center = document.getElementById('center');
      if (!item) {
        renderEmptyState(center, session, context);
        state.currentAudioUrl = '';
        player.pause();
        player.removeAttribute('src');
        player.load();
        syncActionButtons(false);
        return;
      }

      syncActionButtons(true);
      center.innerHTML = `
        <section class="hero">
          <div class="hero-head">
            <div class="hero-copy">
              <div class="hero-context">${displaySong(item)} · ${displayLayer(item)} · ${item.versionLabel || readableRef(item.versionRef)}</div>
              <div class="hero-summary">
                <span class="hero-label">${item.predictedLabel}</span>
                <p class="hero-meta">${formatClipRange(item)} · score ${formatScore(item.score)}</p>
              </div>
            </div>
          </div>
          <div class="wave-shell">
            <canvas class="waveform" id="waveform" width="960" height="160"></canvas>
            <div class="wave-meta">
              <span>${formatSourceKind(context.sourceKind)}</span>
              <div class="wave-actions">
                <span>${item.polarity || 'sample'}</span>
                <button class="replay-button" id="play-button" title="Replay clip">Replay</button>
              </div>
            </div>
          </div>
          <div class="hero-stats">
            <span class="hero-stat">${describeSessionPosition(item, navigation)}</span>
            <span class="hero-stat">${item.targetClass || item.predictedLabel}</span>
            <span class="hero-stat">${formatOutcome(item.reviewOutcome)}</span>
          </div>
        </section>
      `;
      document.getElementById('play-button').addEventListener('click', () => replayCurrentItem().catch(showStatus));
      document.getElementById('status-line').textContent = describeStatusLine(item, viewMode);
      syncCurrentItemMedia(item, { replay: replayCurrentItem }).catch(showStatus);
    }

    function renderEmptyState(center, session, context) {
      const empty = describeEmptyState(session);
      center.innerHTML = `
        <div class="empty-state">
          <div class="empty-mark">${empty.badge}</div>
          <h2 class="empty-title">${empty.title}</h2>
          <p class="empty-copy">${empty.copy}</p>
          ${empty.buttonLabel ? `<button class="empty-action" id="empty-action">${empty.buttonLabel}</button>` : ''}
        </div>
      `;
      document.getElementById('status-line').textContent = empty.status;
      const actionButton = document.getElementById('empty-action');
      if (actionButton) {
        actionButton.addEventListener('click', () => handleEmptyAction(empty.action).catch(showStatus));
      }
    }

    function renderIdleProjectState() {
      const projectContext = state.projectContext || {};
      const projectName = projectContext.projectName || 'Current project';
      const idlePayload = {
        session: {
          name: 'Waiting for review run',
          totalItems: 0,
          pendingCount: 0,
          reviewedCount: 0,
          classMap: [],
          reviewMode: 'all_events',
          scopeOptions: { songs: [], layers: [] },
          context: {
            sourceKind: 'project',
            projectName,
            projectRef: projectContext.projectRef || '',
            sourceLabel: projectName,
            sourceRef: projectContext.projectRoot || '',
          },
          applicationSession: projectContext.applicationSession || {
            projectName,
            projectRef: projectContext.projectRef || '',
          },
        },
        currentItem: null,
        filteredCount: 0,
        progress: { items: [] },
        navigation: {
          viewMode: 'queue',
          scopeCount: 0,
          scopePendingCount: 0,
          scopeReviewedCount: 0,
        },
      };
      state.snapshot = idlePayload;
      render(idlePayload, { replayCurrentItem: false });
      document.getElementById('archive-status').textContent = buildArchiveStatus(null);
    }

    function describeEmptyState(session) {
      const filteredCount = state.snapshot?.filteredCount || 0;
      const pending = session.pendingCount || 0;
      const reviewed = session.reviewedCount || 0;
      if (state.outcome === 'pending' && pending === 0 && reviewed > 0) {
        return {
          badge: 'done',
          title: 'Run Complete',
          copy: 'This review run has no pending hits left. Switch to reviewed history or open a different archived run.',
          buttonLabel: 'Show Reviewed',
          action: 'showReviewed',
          status: 'No pending items remain in this run.',
        };
      }
      if (filteredCount === 0 && (session.totalItems || 0) > 0) {
        return {
          badge: 'slice',
          title: 'No Hits In This Slice',
          copy: 'The current song, layer, class, or view filter excludes every item in this run.',
          buttonLabel: 'Reset Filters',
          action: 'resetFilters',
          status: 'No review items match the current filter slice.',
        };
      }
      return {
        badge: formatSourceKind(session.context?.sourceKind).slice(0, 4).toLowerCase() || 'idle',
        title: 'No Review Items',
        copy: `Nothing is available yet for ${session.context?.sourceLabel || 'this source'}.`,
        buttonLabel: '',
        action: '',
        status: 'No review items are available.',
      };
    }

    async function handleEmptyAction(action) {
      if (action === 'showReviewed') {
        state.outcome = 'all';
        refreshScopeFromControls();
        return;
      }
      if (action === 'resetFilters') {
        state.songRef = preferredSongRefFromSession(state.snapshot?.session || {}, state.snapshot?.session?.scopeOptions?.songs || []) || 'all';
        state.layerRef = 'all';
        state.targetClass = 'all';
        refreshScopeFromControls();
      }
    }

    function buildScopeSummary(session, navigation, pending) {
      const scopeCount = navigation.scopeCount || session.totalItems || 0;
      const currentScopeItemNumber = navigation.currentScopeItemNumber || 0;
      const scopePendingCount = navigation.scopePendingCount ?? pending;
      const reviewMode = formatReviewMode(session.reviewMode);
      const view = formatOutcome(state.outcome);
      if (scopeCount && currentScopeItemNumber) {
        return `${currentScopeItemNumber} / ${scopeCount} · ${view} · ${scopePendingCount} pending`;
      }
      if (scopeCount) {
        return `${view} · ${scopePendingCount} pending · ${scopeCount} in scope`;
      }
      return `${view} · ${pending} pending · ${reviewMode}`;
    }

    function buildActiveContext(applicationSession, item, context) {
      const parts = [];
      if (applicationSession.activeSongTitle) parts.push(applicationSession.activeSongTitle);
      else if (item?.songTitle) parts.push(item.songTitle);
      if (applicationSession.activeSongVersionLabel) parts.push(applicationSession.activeSongVersionLabel);
      else if (item?.versionLabel) parts.push(item.versionLabel);
      if (!parts.length && context.projectName) parts.push(context.projectName);
      return parts.join(' · ') || 'Detached review session';
    }

    function buildSourceContext(context, sessionSummary, applicationSession) {
      const parts = [];
      if (formatSourceKind(context.sourceKind)) parts.push(formatSourceKind(context.sourceKind));
      if (sessionSummary.pendingCount !== undefined) {
        parts.push(sessionSummary.pendingCount ? `${sessionSummary.pendingCount} pending` : 'complete');
      }
      if (applicationSession.sessionId) parts.push(`App ${applicationSession.sessionId}`);
      return parts.join(' · ') || context.sourceRef || 'No source';
    }

    function buildArchiveStatus(sessionSummary) {
      const summary = sessionSummary || state.sessions[0] || null;
      if (!summary) return 'No archived review runs loaded.';
      const pending = summary.pendingCount || 0;
      const reviewed = summary.reviewedCount || 0;
      const sourceLabel = summary.context?.sourceLabel || summary.name || 'Review run';
      return `${sourceLabel} · ${pending ? `${pending} pending` : `${reviewed} reviewed`}`;
    }

    function describeSessionPosition(item, navigation) {
      const itemNumber = navigation.currentScopeItemNumber || 0;
      const scopeCount = navigation.scopeCount || 0;
      if (!scopeCount || !itemNumber) return item.reviewOutcome || 'pending';
      return `Item ${itemNumber} of ${scopeCount} · ${formatOutcome(item.reviewOutcome)}`;
    }

    function formatOutcome(outcome) {
      const labels = {
        pending: 'Pending',
        all: 'All',
        correct: 'Correct',
        incorrect: 'Incorrect',
      };
      return labels[String(outcome || 'pending')] || String(outcome || 'pending');
    }

    function formatReviewMode(reviewMode) {
      return String(reviewMode || 'all_events').replaceAll('_', ' ');
    }

    function formatSourceKind(sourceKind) {
      const labels = {
        ez_project: 'Project',
        project: 'Project',
        folder: 'Folder',
        json: 'JSON',
        jsonl: 'JSONL',
        session: 'Session',
      };
      return labels[String(sourceKind || 'session')] || readableRef(sourceKind || 'session');
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
      if (item.correctedLabel) return `Relabeled to ${item.correctedLabel}`;
      if (viewMode === 'history') return `History view · ${formatOutcome(item.reviewOutcome)}`;
      return 'Ready';
    }

    async function syncCurrentItemMedia(item, { replay = true } = {}) {
      const audioUrl = item?.audioUrl || '';
      const shouldReloadPlayer = state.currentAudioUrl !== audioUrl || !player.getAttribute('src');
      if (shouldReloadPlayer) {
        player.pause();
        const cachedClip = cachedAudioClip(audioUrl);
        player.src = cachedClip?.blobUrl || audioUrl;
        player.load();
        state.currentAudioUrl = audioUrl;
      }
      const cachedPeaks = state.waveformCache.get(item.audioUrl);
      if (cachedPeaks) drawWaveformPeaks(cachedPeaks);
      else drawWaveformPlaceholder('Loading waveform…');
      renderWaveform(item).catch(() => drawWaveformPlaceholder('Waveform unavailable'));
      prefetchNeighborMedia(state.snapshot?.navigation);
      if (replay || shouldReloadPlayer) {
        await attemptPlayback();
      }
    }

    async function refreshSessionState({ forceRender = false, resetCursor = false } = {}) {
      const sessionsResult = await loadSessions();
      if (!selectedSessionId()) {
        renderIdleProjectState();
        return;
      }
      const shouldResetCursor = resetCursor || sessionsResult.sessionChanged;
      const navigation = state.snapshot?.navigation || {};
      const focusedHistoryItemId = !shouldResetCursor && navigation.viewMode === 'history'
        ? String(navigation.focusedItemId || state.snapshot?.currentItem?.itemId || '').trim()
        : '';
      const snapshotResult = await requestSnapshot(
        focusedHistoryItemId
          ? { itemId: focusedHistoryItemId }
          : { cursor: shouldResetCursor ? 0 : state.cursor }
      );
      if (snapshotResult) {
        applySnapshot(snapshotResult.payload, {
          requestId: snapshotResult.requestId,
          forceRender: forceRender || Boolean(sessionsResult.changed),
        });
      }
    }

    async function manualReloadStatus() {
      document.getElementById('status-line').textContent = 'Reloading review status…';
      clearMediaCaches();
      state.currentAudioUrl = '';
      player.pause();
      player.removeAttribute('src');
      player.load();
      await refreshSessionState({ forceRender: true });
    }

    function scheduleAutoRefresh() {
      window.setInterval(() => {
        if (document.visibilityState !== 'visible') return;
        refreshSessionState().catch(() => undefined);
      }, REFRESH_INTERVAL_MS);
    }

    async function attemptPlayback() {
      try {
        player.currentTime = 0;
        await player.play();
        document.getElementById('status-line').textContent = 'Playing';
      } catch (error) {
        document.getElementById('status-line').textContent = 'Tap Replay Clip';
      }
    }

    async function replayCurrentItem() {
      if (!state.snapshot?.currentItem) return;
      await attemptPlayback();
    }

    function formatClipRange(item) {
      const startMs = Number(item?.sourceProvenance?.current_start_ms);
      const endMs = Number(item?.sourceProvenance?.current_end_ms);
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return 'Clip';
      return `${(startMs / 1000).toFixed(2)}s - ${(endMs / 1000).toFixed(2)}s`;
    }

    function drawWaveformPlaceholder(message) {
      const canvas = document.getElementById('waveform');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#101419';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#2d3641';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      ctx.fillStyle = '#a7b0ba';
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(message, canvas.width / 2, (canvas.height / 2) + 8);
    }

    async function renderWaveform(item) {
      const token = ++state.waveformToken;
      const peaks = await ensureWaveformPeaks(item);
      if (token !== state.waveformToken) return;
      if (!peaks.length) {
        drawWaveformPlaceholder('Waveform unavailable');
        return;
      }
      drawWaveformPeaks(peaks);
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
        showStatus(new Error('Pick a class or add an operator note.'));
        return;
      }
      closeSheet();
      await postReview('incorrect', { correctedLabel, reviewNote });
    }

    function refreshScopeFromControls() {
      updateUrl();
      resetHistory();
      state.cursor = 0;
      if (!selectedSessionId()) {
        renderIdleProjectState();
        return;
      }
      loadLiveCursor(0).catch(showStatus);
    }

    document.getElementById('session-select').addEventListener('change', (event) => {
      state.sessionId = event.target.value;
      state.songRef = 'all';
      state.layerRef = 'all';
      state.targetClass = 'all';
      state.outcome = 'pending';
      state.didSeedScope = false;
      refreshScopeFromControls();
    });
    document.getElementById('outcome-filter').addEventListener('change', (event) => {
      state.outcome = event.target.value || 'pending';
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
    document.getElementById('reload-status-btn').addEventListener('click', () => {
      manualReloadStatus().catch(showStatus);
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

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        refreshSessionState().catch(() => undefined);
      }
    });
    window.addEventListener('focus', () => {
      refreshSessionState().catch(() => undefined);
    });

    scheduleAutoRefresh();
    refreshSessionState()
      .catch(showStatus);
  