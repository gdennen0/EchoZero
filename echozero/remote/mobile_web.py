"""
Mobile web page: Tiny phone-first UI for private EchoZero remote control.
Exists because the first remote surface should be a thin web client, not a native app.
Connects a browser over Tailscale to the wrapper API with lightweight polling and transport controls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MobileWebConfig:
    """Config values for the phone-facing remote web page."""

    app_title: str = "EchoZero Remote"
    poll_interval_ms: int = 1500


def build_mobile_web_page(*, config: MobileWebConfig | None = None) -> str:
    """Return the phone-facing HTML page served by the remote wrapper."""
    resolved = config or MobileWebConfig()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{resolved.app_title}</title>
  <meta name="apple-mobile-web-app-capable" content="yes">
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif;
      --bg: #efe8dd;
      --card: #fffaf3;
      --ink: #1f1a15;
      --muted: #675b4f;
      --accent: #1b5e56;
      --danger: #a9442e;
      --border: rgba(31, 26, 21, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top, #fbf6ee, var(--bg)); color: var(--ink); }}
    main {{ max-width: 640px; margin: 0 auto; padding: 16px 14px 40px; }}
    .card {{
      background: var(--card);
      border-radius: 18px;
      box-shadow: 0 12px 28px rgba(31, 26, 21, 0.08);
      border: 1px solid var(--border);
      padding: 14px;
      margin-bottom: 12px;
    }}
    h1, h2, p, pre {{ margin: 0; }}
    h1 {{ font-size: 1.4rem; }}
    h2 {{ font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px; }}
    .meta {{ margin-top: 8px; color: var(--muted); font-size: 0.92rem; }}
    .pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: #ece4d8; color: var(--muted); font-size: 0.84rem; }}
    .row {{ display: grid; gap: 10px; }}
    .transport {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .stat {{
      padding: 12px;
      border-radius: 14px;
      background: #f4ede3;
      min-height: 72px;
    }}
    .stat-label {{
      display: block;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat-value {{
      display: block;
      font-size: 1rem;
      font-weight: 700;
      line-height: 1.25;
    }}
    button {{
      border: 0;
      border-radius: 14px;
      padding: 14px 12px;
      font-size: 1rem;
      font-weight: 700;
      background: #ece4d8;
      color: var(--ink);
    }}
    button.primary {{ background: var(--accent); color: white; }}
    button.danger {{ background: var(--danger); color: white; }}
    button:disabled {{ opacity: 0.5; }}
    a.shortcut-link {{
      display: block;
      padding: 12px 14px;
      border-radius: 14px;
      background: #f4ede3;
      color: var(--ink);
      font-weight: 700;
      text-decoration: none;
    }}
    img {{
      width: 100%;
      border-radius: 14px;
      background: #ddd1c0;
      min-height: 160px;
      object-fit: contain;
    }}
    audio {{ width: 100%; margin-top: 8px; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      line-height: 1.4;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>{resolved.app_title}</h1>
      <p class="meta" id="status">Disconnected</p>
      <p class="meta">This is a thin private wrapper over the localhost app bridge.</p>
    </section>
    <section class="card row">
      <h2>Session</h2>
      <div class="row">
        <button class="primary" id="connect">Start Session</button>
        <button id="disconnect">End Session</button>
      </div>
      <p class="meta" id="session-meta">No session token.</p>
    </section>
    <section class="card row">
      <h2>Shortcuts</h2>
      <p class="meta" id="shortcut-meta">Start one session to generate bookmarkable transport links.</p>
      <div class="row">
        <a class="shortcut-link" id="shortcut-play" href="#" rel="noreferrer">Play Shortcut</a>
        <a class="shortcut-link" id="shortcut-pause" href="#" rel="noreferrer">Pause Shortcut</a>
        <a class="shortcut-link" id="shortcut-stop" href="#" rel="noreferrer">Stop Shortcut</a>
      </div>
    </section>
    <section class="card">
      <h2>Transport</h2>
      <div class="transport">
        <button class="primary" data-transport="play">Play</button>
        <button data-transport="pause">Pause</button>
        <button class="danger" data-transport="stop">Stop</button>
      </div>
      <p class="meta" id="transport-state">Stopped</p>
    </section>
    <section class="card">
      <h2>Now Active</h2>
      <div class="stats">
        <div class="stat">
          <span class="stat-label">Project</span>
          <span class="stat-value" id="project-title">No project loaded</span>
        </div>
        <div class="stat">
          <span class="stat-label">Song</span>
          <span class="stat-value" id="song-title">No song selected</span>
        </div>
        <div class="stat">
          <span class="stat-label">Version</span>
          <span class="stat-value" id="version-title">No active version</span>
        </div>
        <div class="stat">
          <span class="stat-label">Sync</span>
          <span class="stat-value" id="sync-state">off</span>
        </div>
      </div>
    </section>
    <section class="card">
      <h2>Snapshot</h2>
      <pre id="snapshot-summary">No snapshot yet.</pre>
    </section>
    <section class="card">
      <h2>Audio Monitor</h2>
      <p class="meta" id="audio-meta">No active playback source.</p>
      <audio id="audio-monitor" controls preload="metadata"></audio>
    </section>
    <section class="card">
      <h2>Screenshot</h2>
      <img id="screenshot" alt="EchoZero screenshot preview">
    </section>
  </main>
  <script>
    const pollIntervalMs = {resolved.poll_interval_ms};
    const tokenKey = 'echozero_remote_token';
    let pollHandle = null;

    function readToken() {{
      return window.localStorage.getItem(tokenKey);
    }}

    function writeToken(token) {{
      if (token) {{
        window.localStorage.setItem(tokenKey, token);
      }} else {{
        window.localStorage.removeItem(tokenKey);
      }}
    }}

    function authHeaders() {{
      const token = readToken();
      return token ? {{ Authorization: `Bearer ${{token}}` }} : {{}};
    }}

    async function requestJson(method, path, payload) {{
      const response = await fetch(path, {{
        method,
        headers: {{ 'Content-Type': 'application/json', ...authHeaders() }},
        body: payload ? JSON.stringify(payload) : undefined,
      }});
      const data = await response.json();
      if (!response.ok) {{
        throw new Error(data.error || `Request failed: ${{response.status}}`);
      }}
      return data;
    }}

    function setStatus(text) {{
      document.getElementById('status').textContent = text;
    }}

    function buildShortcutUrl(actionName) {{
      const token = readToken();
      if (!token) {{
        return '#';
      }}
      const path = `/api/transport/${{actionName}}?session_token=${{encodeURIComponent(token)}}`;
      return new URL(path, window.location.href).toString();
    }}

    function updateShortcutLinks() {{
      const token = readToken();
      const linkActions = ['play', 'pause', 'stop'];
      linkActions.forEach((actionName) => {{
        const link = document.getElementById(`shortcut-${{actionName}}`);
        link.href = buildShortcutUrl(actionName);
        link.toggleAttribute('aria-disabled', !token);
      }});
      document.getElementById('shortcut-meta').textContent = token
        ? 'Long-press one shortcut to copy or share it into Telegram or iOS Shortcuts.'
        : 'Start one session to generate bookmarkable transport links.';
    }}

    function clearAudioMonitor() {{
      const audio = document.getElementById('audio-monitor');
      audio.pause();
      audio.removeAttribute('src');
      delete audio.dataset.sourceRef;
      audio.load();
      document.getElementById('audio-meta').textContent = 'No active playback source.';
    }}

    function updateAudioMonitor(sourceRef) {{
      const token = readToken();
      const audio = document.getElementById('audio-monitor');
      if (!token || !sourceRef) {{
        clearAudioMonitor();
        return;
      }}
      const nextSrc = `/api/audio/current?session_token=${{encodeURIComponent(token)}}&source_ref=${{encodeURIComponent(sourceRef)}}`;
      if (audio.dataset.sourceRef !== sourceRef) {{
        audio.dataset.sourceRef = sourceRef;
        audio.src = nextSrc;
        audio.load();
      }}
      document.getElementById('audio-meta').textContent = sourceRef;
    }}

    function renderSnapshot(snapshot) {{
      const artifacts = snapshot?.artifacts || {{}};
      const projectTitle = artifacts.project_title || 'EchoZero';
      const songTitle = artifacts.active_song_title || 'No song selected';
      const activeVersion = (artifacts.song_versions || []).find((version) => version.is_active);
      const transport = artifacts.transport || {{}};
      const playback = artifacts.playback || {{}};
      const activeSource = (playback.active_sources || []).find((source) => source.source_ref) || null;
      const selection = (snapshot?.selection || []).join(', ') || 'none';
      const focused = snapshot?.focused_target_id || 'none';
      const actionCount = (snapshot?.actions || []).length;
      const syncMode = snapshot?.sync?.mode || 'off';
      const syncConnected = snapshot?.sync?.connected ? 'connected' : 'offline';
      document.getElementById('project-title').textContent = projectTitle;
      document.getElementById('song-title').textContent = songTitle;
      document.getElementById('version-title').textContent = activeVersion?.label || 'No active version';
      document.getElementById('sync-state').textContent = `${{syncMode}} · ${{syncConnected}}`;
      document.getElementById('transport-state').textContent =
        `${{transport.is_playing ? 'Playing' : 'Stopped'}} · ${{transport.current_time_label || '00:00.00'}}`;
      document.getElementById('snapshot-summary').textContent =
        `selection: ${{selection}}\\nfocused: ${{focused}}\\nactions: ${{actionCount}}\\nbackend: ${{playback.backend_name || 'unknown'}}\\nlatency_ms: ${{playback.latency_ms ?? 'n/a'}}`;
      updateAudioMonitor(activeSource?.source_ref || '');
    }}

    async function refreshScreenshot() {{
      const payload = await requestJson('POST', '/api/screenshot', {{}});
      document.getElementById('screenshot').src = `data:image/png;base64,${{payload.png_base64}}`;
    }}

    async function refreshState() {{
      if (!readToken()) {{
        setStatus('Disconnected');
        return;
      }}
      const health = await requestJson('GET', '/api/health');
      const snapshot = await requestJson('GET', '/api/snapshot');
      setStatus(`Connected to bridge at ${{health.bridge.address.host}}:${{health.bridge.address.port}}`);
      renderSnapshot(snapshot);
      await refreshScreenshot();
    }}

    async function startSession() {{
      const payload = await requestJson('POST', '/api/session/start', {{}});
      writeToken(payload.token);
      updateShortcutLinks();
      document.getElementById('session-meta').textContent =
        `Session expires at ${{new Date(payload.expires_at * 1000).toLocaleTimeString()}}`;
      await refreshState();
    }}

    async function stopSession() {{
      const token = readToken();
      if (token) {{
        await requestJson('POST', '/api/session/stop', {{}});
      }}
      writeToken(null);
      updateShortcutLinks();
      document.getElementById('session-meta').textContent = 'No session token.';
      document.getElementById('screenshot').removeAttribute('src');
      clearAudioMonitor();
      renderSnapshot(null);
      setStatus('Disconnected');
    }}

    async function invokeTransport(actionName) {{
      await requestJson('POST', `/api/transport/${{actionName}}`, {{}});
      await refreshState();
    }}

    document.getElementById('connect').addEventListener('click', () => startSession().catch((error) => setStatus(error.message)));
    document.getElementById('disconnect').addEventListener('click', () => stopSession().catch((error) => setStatus(error.message)));
    document.querySelectorAll('[data-transport]').forEach((button) => {{
      button.addEventListener('click', () => invokeTransport(button.dataset.transport).catch((error) => setStatus(error.message)));
    }});

    updateShortcutLinks();
    if (readToken()) {{
      refreshState().catch((error) => setStatus(error.message));
    }}
    pollHandle = window.setInterval(() => {{
      if (readToken()) {{
        refreshState().catch((error) => setStatus(error.message));
      }}
    }}, pollIntervalMs);
  </script>
</body>
</html>
"""
