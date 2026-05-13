<!--
MA3 connection operating model for EchoZero public-release readiness.
Exists because MA3 OSC setup and troubleshooting currently span multiple mismatched surfaces.
Connects operator workflow, app diagnostics, and MA3 Lua responsibilities into one contract.
-->

# MA3 Connection Operating Model

Status: active baseline implemented
Last updated: 2026-05-13

## Goal

Make MA3 <> EchoZero connection setup feel like one simple system:

- one setup path
- one health story
- one troubleshooting ladder
- one shared vocabulary between EZ and MA3

The operator should always be able to answer three questions quickly:

1. Can EchoZero bind its local receive listener?
2. Can EchoZero send commands to MA3?
3. Can the MA3 plugin receive the command and send a reply back to EchoZero?

The baseline implementation now routes the main app surfaces through one shared connection-check
contract. The remaining public-release gap is live hardware proof, not the automated app/simulator
baseline.

## Original Problems Addressed

### 1. "Status" and "Ping" mean different things

Before this implementation, behavior was split:

- `Check Status` in [echozero/ui/qt/osc_settings_panel.py](/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/osc_settings_panel.py:292) verifies:
  - EZ can bind the configured receive host/port
  - EZ can send one OSC packet to MA3
- `Ping` in that same panel verifies:
  - EZ can temporarily listen
  - EZ can send `EZ.SetTarget(...)`
  - MA3 can receive `EZ.Ping()`
  - MA3 can send a reply back to EZ

Those are not the same level of proof.

### 2. The MA3 connection HUD uses weaker checks than the OSC settings panel

[echozero/ui/qt/ma3_connection_hud.py](/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/ma3_connection_hud.py:160) only tests:

- receive bind
- raw send call

It does not prove:

- MA3 plugin loaded
- MA3 received the command
- MA3 can reply to EZ
- callback target is correct

This creates conflicting operator experiences depending on which surface they opened.

### 3. The plugin status payload is too shallow for public troubleshooting

`EZ.Status()` in [MA3/plugins/echozero.lua](/Users/march/Documents/GitHub/EchoZero/MA3/plugins/echozero.lua:953) reports:

- target ip/port
- socket available
- debug
- command mode
- hook count
- version

That helps, but it does not clearly answer:

- is OSC output enabled in MA3 show settings?
- is OSC input enabled in MA3 show settings?
- which MA3 OSC line should the operator edit?
- when did EZ last set the callback target?
- when did MA3 last successfully send a reply?
- is the loaded plugin stale versus local bundle?

### 4. Port errors are treated like low-level transport problems instead of operator actions

"Port already in use" is a common operator issue, but current UI mostly reports the raw socket
failure. The operator needs a direct action:

- what owns this port?
- should I change the EZ receive port or stop the other process?
- did EZ intentionally choose port `0` before and now bind a different real port?

### 5. We do not yet present a single proven operating recipe

The repo has good harness tools and plugin health checks, but the operator-facing path is still
closer to "toolbox" than "guided system."

## Canonical Mental Model

Treat the MA3 <> EZ lane as four stacked layers:

1. `Local bind`
   EchoZero can open its receive listener on the configured host/port.
2. `Outbound command`
   EchoZero can send OSC to the MA3 command endpoint.
3. `Plugin path`
   MA3 received the command and the EZ plugin handled it.
4. `Round trip`
   MA3 successfully sent a reply back to EchoZero's listener.

Every diagnostic must name which layer passed and which layer failed.

Do not collapse these into one ambiguous "status" label.

## Proposed Operator Contract

### One primary action: `Run Connection Check`

Replace the current ambiguous split with one operator-first action that runs a fixed ladder:

1. Validate EZ receive config
2. Bind EZ receive listener
3. Validate EZ send config
4. Send `EZ.SetTarget(...)`
5. Send `EZ.Ping()`
6. Request `EZ.Status()`
7. Request `EZ.Version()`
8. Request `EZ.GetPluginHealth()`
9. Summarize results in one report

The user sees one result card with explicit stages:

- `Receive Listener`
- `Command Send`
- `MA3 Reply`
- `Plugin Version`
- `Plugin Health`

### Health states

Use explicit connection states:

- `Not Configured`
- `Bind Failed`
- `Send Failed`
- `Reply Failed`
- `Plugin Missing`
- `Plugin Stale`
- `Connected`
- `Connected With Warnings`

These should be user-facing and consistent in:

- OSC settings panel
- MA3 connection HUD
- harness summary output

### Always show the real callback endpoint

After a successful check, surface:

- configured EZ receive host
- configured EZ receive port
- actual bound receive host
- actual bound receive port
- MA3 send target host
- MA3 send target port

This is especially important when EZ receive port is `0`, because the operator must know the real
ephemeral port that was bound during the check.

## Troubleshooting Ladder

### Stage 1: EZ local bind

Fail examples:

- port already in use
- host not available
- OS permission or socket error

Operator output should say:

- `EchoZero could not start its receive listener on 127.0.0.1:9001`
- `Reason: address already in use`
- `Fix: close the other listener or change EZ Receive Port`

### Stage 2: EZ outbound command

Fail examples:

- send port empty
- invalid host
- OS send error

Operator output should say:

- `EchoZero could not send commands to MA3 at 192.168.1.50:8000`
- `Fix: verify the MA3 OSC destination line and command port`

### Stage 3: MA3 reply path

Fail examples:

- MA3 OSC output disabled
- callback target wrong
- plugin not loaded
- plugin loaded but outbound socket broken
- network route issue

Operator output should say:

- `MA3 did not reply to EZ.Ping()`
- `This means EchoZero could send a command, but MA3 did not complete the round trip`
- `Check MA3 OSC input/output enablement, plugin load state, and callback target`

### Stage 4: Plugin health

Pass with warnings examples:

- plugin loads but HitMaker support is partial
- plugin version/build is older than local bundle

Operator output should say:

- `MA3 replied, but the loaded EZ plugin is older than the local plugin bundle`
- `Fix: reload plugins in MA3 with RP`

## MA3 Lua Upgrade Plan

Yes, the MA3 Lua side should be upgraded.

### Add one richer diagnostic payload

Keep `EZ.Ping()` minimal, but add a dedicated diagnostic call such as `EZ.ConnectionReport()`
that returns:

- `ez_version`
- `ez_build`
- `target_ip`
- `target_port`
- `socket_ok`
- `hooks`
- `cmd_mode`
- `hitmaker_loaded`
- `hitmaker_version`
- `hitmaker_build`
- `last_target_set_at`
- `last_ping_received_at`
- `last_send_ok_at`
- `last_send_error`
- `osc_module_loaded`

This gives EZ a richer end-to-end diagnostic without overloading `EZ.Status()`.

### Track send outcomes inside the plugin

The plugin should record:

- last successful send timestamp
- last failed send timestamp
- last send error string
- current configured callback target

That turns "ping failed" from a mystery into evidence.

### Add a first-run operator entrypoint

Provide one MA3-side command the operator can run safely:

- `EZ.ConfigureTarget`
- then `EZ.ConnectionReport()`

If possible, expose a compact plugin UI action for:

- set EZ target
- test reply
- print result in MA3 feedback

## EZ App Upgrade Plan

### 1. Unify the two diagnostics surfaces

The MA3 connection HUD should stop using its weaker custom probes and instead share the same
connection-check service as the OSC settings panel.

One implementation, two entry surfaces.

### 2. Replace "Check Status" label

Rename it to something explicit:

- `Run Connection Check`

`Check Status` implies a passive state read, but the current flow actively binds, sends, sets
target, and waits for a reply.

### 3. Produce an operator-first result summary

Example:

- `Receive Listener: OK on 127.0.0.1:9001`
- `Command Send: OK to 192.168.1.50:8000`
- `MA3 Reply: FAILED`
- `Plugin Version: not confirmed`
- `Most likely fix: enable MA3 OSC output and reload the EZ plugin`

### 4. Add port-conflict diagnosis

When bind fails:

- detect `address already in use`
- show the configured conflicting endpoint
- suggest using a different EZ receive port
- optionally add a helper button to pick the next free port

### 5. Save a short diagnostic transcript

After every connection check, keep a small transcript the user can copy into a bug report:

- config used
- actual listener endpoint
- commands sent
- replies received
- final classified failure stage

## Network Discovery

Network discovery is useful, but it should be treated as a secondary assist, not the core proof.

### Good uses

- find likely local IPv4 addresses for EZ
- suggest candidate MA3 hosts on the local subnet
- show whether EZ and MA3 appear to be on the same subnet

### Limits

- a discovered host is not proof that the MA3 OSC plugin path works
- host discovery cannot verify MA3 OSC show settings
- host discovery cannot prove the callback target is correct

Recommended approach:

- add a lightweight `Network Hints` section later
- do not block the main connection check on subnet scanning

## Auto-injecting the plugin through OSC

This is worth exploring, but it is secondary.

The preferred order is:

1. make connection proof rock solid
2. make plugin version/load state explicit
3. then explore automated plugin reload or install helpers

Possible targets:

- remote `RP` trigger through the command path
- remote commands that validate the expected plugin bundle is loaded
- guided MA3-side install/reload workflow, if full injection is not safe or portable

For public release, reliable detection is more important than clever installation.

## Recommended Release Order

### Wave 1

- one shared EZ connection-check service
- rename `Check Status` to `Run Connection Check`
- classify failures by stage
- unify the HUD and settings panel

### Wave 2

- richer MA3 plugin diagnostic payload
- stale plugin detection in the app UI
- port-conflict guidance and suggested fixes

### Wave 3

- network hints
- guided MA3 operator checklist
- optional remote plugin reload helpers

## Success Bar

This lane is ready for public release when:

- a first-time operator can connect without reading deep docs
- a failed connection always identifies the failed layer
- EZ and MA3 report the same target and plugin version story
- port conflicts produce a direct fix, not just a raw socket error
- all operator-facing connection surfaces use the same proof logic
