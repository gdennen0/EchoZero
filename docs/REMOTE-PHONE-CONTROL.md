# Remote Phone Control

Use this document for the first private phone-control path for EchoZero.

## Posture

- Keep the canonical app automation bridge on `127.0.0.1`.
- Run a thin remote wrapper on the same host.
- Share the wrapper privately through Tailscale.
- Do not expose the raw bridge or a public internet endpoint.

## Host Flow

1. Launch EchoZero with the canonical automation bridge:

   ```bash
   .venv/bin/python run_echozero.py --automation-port 0
   ```

2. Read the emitted bridge URL, for example:

   ```text
   automation_bridge=http://127.0.0.1:43210
   ```

3. Start the remote wrapper against that bridge URL:

   ```bash
   .venv/bin/python -m echozero.remote.server \
     --bridge-url http://127.0.0.1:43210 \
     --host 100.x.y.z \
     --port 8765
   ```

4. Open the wrapper privately from the phone over Tailscale at:

   ```text
   http://100.x.y.z:8765
   ```

   Bind the wrapper directly to the host's tailnet IP when you want the
   simplest private path and do not need Tailscale Serve/Funnel.

5. If you want bookmarkable one-action transport links, start the wrapper with
   a longer session TTL, open the page, tap `Start Session`, and copy the
   generated `Play`, `Pause`, and `Stop` shortcut links.

6. If you want wrapper-up/down alerts in Telegram through OpenClaw, run the
   health monitor alongside the wrapper:

   ```bash
   .venv/bin/python -m echozero.remote.monitor \
     --health-url http://100.x.y.z:8765/api/health \
     --notify-openclaw
   ```

7. If you also want the full OpenClaw Control UI in Safari, prefer Tailscale
   Serve so the dashboard can load over HTTPS. Do not widen the EchoZero
   wrapper just to mimic the full admin surface.

## iPhone Flow

1. Install and sign into Tailscale on the iPhone.
2. Connect the iPhone to the same tailnet as the EchoZero host.
3. Open the private wrapper URL in Safari.
4. Tap `Start Session`.
5. Use the transport controls, phone-sized status cards, screenshot view, and
   current audio monitor from the phone page.
6. Long-press the shortcut links if you want to save `Play`, `Pause`, or
   `Stop` into Telegram, Safari bookmarks, or iOS Shortcuts.

## Current Limits

- This is `v0`, not full remote editing parity.
- Session issuance is short-lived but not yet paired to an explicit desktop
  approval flow.
- Direct transport shortcut links are only as long-lived as the session token
  you generated for them.
- The safe action surface is transport-only by default.
- The audio monitor streams the current playback source file through the same
  private wrapper. It is not a low-latency live mix bus.
