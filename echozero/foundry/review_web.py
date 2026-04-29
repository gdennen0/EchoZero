"""EZ Review web page: Retro phone-first project review workspace.
Exists because operators need a clear current-project review lane, not a batch maze.
Connects review-session APIs to one static mobile UI with scoped filters and revisit navigation.
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
      --bg: #090909;
      --bg-grid: rgba(67, 255, 169, 0.08);
      --panel: rgba(14, 20, 18, 0.94);
      --panel-strong: rgba(18, 26, 23, 0.98);
      --panel-soft: rgba(24, 34, 30, 0.94);
      --ink: #f5f6d6;
      --muted: #9bb29f;
      --line: rgba(103, 141, 122, 0.62);
      --line-strong: rgba(163, 214, 187, 0.82);
      --accent: #43ffa9;
      --accent-strong: #fff06a;
      --danger: #ff5c8a;
      --success: #5eff6c;
      --shadow: 0 18px 48px rgba(0, 0, 0, 0.44);
      --pixel: 4px;
      font-family: "Courier New", "Lucida Console", monospace;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      background:
        linear-gradient(var(--bg-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px),
        radial-gradient(circle at top, rgba(67, 255, 169, 0.12), transparent 36%),
        linear-gradient(180deg, #101814, var(--bg));
      background-size: 22px 22px, 22px 22px, auto, auto;
      color: var(--ink);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 12%, transparent 88%, rgba(255,255,255,0.05));
      pointer-events: none;
      mix-blend-mode: screen;
    }
    main {
      position: relative;
      z-index: 1;
      width: min(100%, 560px);
      min-height: 100dvh;
      margin: 0 auto;
      padding:
        max(16px, env(safe-area-inset-top))
        14px
        calc(18px + env(safe-area-inset-bottom))
        14px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
    }
    .panel {
      border: var(--pixel) solid var(--line-strong);
      border-radius: 18px;
      background:
        linear-gradient(180deg, var(--panel-strong), var(--panel)),
        linear-gradient(90deg, rgba(67, 255, 169, 0.08), transparent 40%);
      box-shadow: var(--shadow);
    }
    .panel-inset {
      border: 2px solid rgba(67, 255, 169, 0.26);
      border-radius: 12px;
      background: rgba(7, 11, 10, 0.34);
    }
    .topbar {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .title-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: start;
    }
    .eyebrow,
    .microcopy,
    .control-label,
    .section-label {
      font-size: 0.7rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .brand-stack {
      display: grid;
      gap: 7px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .eyebrow::before {
      content: "";
      width: 12px;
      height: 12px;
      background: var(--accent);
      box-shadow: 0 0 0 2px rgba(255, 240, 106, 0.28);
      border-radius: 2px;
    }
    .brand-title {
      margin: 0;
      font-size: clamp(2rem, 10vw, 3.35rem);
      line-height: 0.9;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-strong);
      text-shadow: 0 0 14px rgba(255, 240, 106, 0.18);
    }
    .title-meta {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px 10px;
    }
    .project-name {
      margin: 0;
      font-size: 1rem;
      line-height: 1.15;
    }
    .session-name {
      margin: 0;
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 2px solid rgba(103, 141, 122, 0.52);
      background: rgba(7, 10, 9, 0.56);
      color: var(--muted);
      font-size: 0.84rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .counter {
      min-width: 112px;
      padding: 12px 14px;
      border-radius: 12px;
      border: 2px solid rgba(255, 240, 106, 0.55);
      background: rgba(5, 7, 6, 0.92);
      text-align: right;
      display: grid;
      gap: 5px;
    }
    .counter strong {
      font-size: 1.18rem;
      color: var(--accent-strong);
    }
    .meter-strip {
      padding: 10px 12px;
      display: grid;
      gap: 8px;
    }
    .meter-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .meter-actions {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .icon-button {
      min-width: 34px;
      min-height: 34px;
      padding: 0 10px;
      border-radius: 999px;
      border: 2px solid rgba(255, 240, 106, 0.38);
      background: rgba(255, 240, 106, 0.1);
      color: var(--accent-strong);
      cursor: pointer;
      font: inherit;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      transition: transform 120ms ease, filter 120ms ease;
    }
    .icon-button:hover {
      transform: translateY(-1px);
      filter: brightness(1.06);
    }
    .icon-button:active {
      transform: translateY(1px);
    }
    .progress-rail {
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      border: 2px solid rgba(67, 255, 169, 0.22);
      background: rgba(67, 255, 169, 0.08);
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-strong));
      transition: width 140ms ease;
    }
    .scope-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: grid;
      gap: 6px;
    }
    .control-label {
      padding-left: 2px;
    }
    select,
    button,
    input,
    textarea {
      font: inherit;
    }
    select,
    input,
    textarea {
      width: 100%;
      min-height: 46px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 2px solid rgba(103, 141, 122, 0.75);
      background: rgba(7, 10, 9, 0.84);
      color: var(--ink);
      box-shadow: inset 0 0 0 1px rgba(67, 255, 169, 0.08);
    }
    textarea {
      min-height: 110px;
      resize: vertical;
    }
    .session-drawer {
      padding: 10px 12px;
    }
    .session-drawer summary {
      list-style: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .session-drawer summary::-webkit-details-marker {
      display: none;
    }
    .session-drawer-copy {
      display: grid;
      gap: 5px;
    }
    .session-drawer strong {
      font-size: 0.92rem;
      color: var(--accent);
    }
    .session-drawer-body {
      padding-top: 12px;
      display: grid;
      gap: 10px;
    }
    .workspace {
      min-height: 0;
      padding: 14px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      background:
        linear-gradient(180deg, rgba(20, 29, 26, 0.98), rgba(13, 18, 17, 0.96)),
        linear-gradient(90deg, rgba(255, 240, 106, 0.06), transparent 32%);
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .meta-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      min-height: 34px;
      padding: 0 12px;
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 2px solid rgba(103, 141, 122, 0.75);
      background: rgba(5, 8, 7, 0.76);
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
    }
    .pill strong {
      color: var(--ink);
    }
    .pill.mode-history {
      border-color: rgba(255, 240, 106, 0.7);
      color: var(--accent-strong);
    }
    .pill.kind-project {
      border-color: rgba(67, 255, 169, 0.62);
      color: var(--accent);
    }
    .context-copy {
      max-width: 240px;
      text-align: right;
      display: grid;
      gap: 4px;
    }
    .context-copy strong {
      font-size: 0.88rem;
      color: var(--ink);
    }
    .center {
      min-height: 0;
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .hero {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .hero-ribbon {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 2px solid rgba(255, 240, 106, 0.36);
      background: rgba(255, 240, 106, 0.08);
      color: var(--accent-strong);
      font-size: 0.72rem;
      letter-spacing: 0.09em;
      text-transform: uppercase;
    }
    .prediction {
      margin: 0;
      font-size: clamp(2.4rem, 13vw, 4.9rem);
      line-height: 0.88;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      text-shadow: 0 0 18px rgba(67, 255, 169, 0.16);
      word-break: break-word;
    }
    .subprediction {
      margin: 0;
      display: grid;
      gap: 3px;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.35;
    }
    .context-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .context-card {
      padding: 12px;
      border-radius: 12px;
      border: 2px solid rgba(103, 141, 122, 0.5);
      background: rgba(7, 10, 9, 0.74);
      display: grid;
      gap: 6px;
    }
    .context-card span {
      font-size: 0.68rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .context-card strong {
      font-size: 0.98rem;
      line-height: 1.25;
      color: var(--ink);
    }
    .play-shell {
      display: grid;
      gap: 12px;
      align-items: center;
      justify-items: center;
    }
    .play-button,
    .replay-button,
    .empty-action,
    .sheet-submit,
    .sheet-close,
    .action {
      border: 0;
      cursor: pointer;
      transition: transform 120ms ease, filter 120ms ease, opacity 120ms ease;
    }
    .play-button:hover,
    .replay-button:hover,
    .empty-action:hover,
    .sheet-submit:hover,
    .sheet-close:hover,
    .action:hover {
      transform: translateY(-1px);
      filter: brightness(1.04);
    }
    .play-button:active,
    .replay-button:active,
    .empty-action:active,
    .sheet-submit:active,
    .sheet-close:active,
    .action:active {
      transform: translateY(1px);
    }
    .play-button {
      min-width: min(78vw, 230px);
      min-height: 120px;
      padding: 18px;
      border-radius: 16px;
      background:
        linear-gradient(180deg, rgba(255, 240, 106, 0.96), rgba(255, 188, 67, 0.96));
      color: #111;
      box-shadow: inset 0 -10px 0 rgba(0, 0, 0, 0.12), 0 18px 26px rgba(0, 0, 0, 0.3);
      display: grid;
      gap: 8px;
      place-items: center;
    }
    .play-button strong {
      font-size: 2.8rem;
      line-height: 1;
      transform: translateX(4px);
    }
    .play-button span {
      font-size: 0.82rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .empty-state {
      min-height: 100%;
      padding: 24px 16px;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 12px;
      text-align: center;
    }
    .empty-mark {
      width: 76px;
      height: 76px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      border: 3px solid rgba(255, 240, 106, 0.68);
      background: rgba(255, 240, 106, 0.12);
      color: var(--accent-strong);
      font-size: 0.76rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .empty-title {
      margin: 0;
      font-size: 1.12rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--ink);
    }
    .empty-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .empty-action {
      min-height: 46px;
      padding: 0 16px;
      border-radius: 12px;
      border: 2px solid rgba(67, 255, 169, 0.6);
      background: rgba(67, 255, 169, 0.12);
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .status-line {
      margin: 0;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.35;
      min-height: 1.2em;
    }
    .bottom-bar {
      padding: 10px;
      display: grid;
      gap: 8px;
      background:
        linear-gradient(180deg, rgba(12, 17, 16, 0.98), rgba(7, 10, 9, 0.98));
    }
    .action {
      min-height: 58px;
      border-radius: 12px;
      color: #101110;
      font-size: 1.35rem;
      font-weight: 700;
      display: grid;
      place-items: center;
      background: rgba(103, 141, 122, 0.32);
    }
    .action.nav {
      background: rgba(67, 255, 169, 0.16);
      color: var(--accent);
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .action.correct {
      background: var(--success);
    }
    .action.incorrect {
      background: var(--danger);
    }
    .action.reclass {
      background: var(--accent-strong);
    }
    .action[disabled],
    .empty-action[disabled] {
      opacity: 0.4;
      cursor: default;
      transform: none;
      filter: none;
    }
    .sheet-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.54);
      opacity: 0;
      pointer-events: none;
      transition: opacity 140ms ease;
    }
    .sheet {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 16px;
      border-top-left-radius: 18px;
      border-top-right-radius: 18px;
      border: var(--pixel) solid var(--line-strong);
      border-bottom: 0;
      background: linear-gradient(180deg, rgba(18, 26, 23, 0.99), rgba(10, 14, 12, 0.98));
      transform: translateY(102%);
      transition: transform 160ms ease;
      box-shadow: 0 -22px 50px rgba(0, 0, 0, 0.4);
      z-index: 10;
    }
    .sheet.open,
    .sheet-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }
    .sheet.open {
      transform: translateY(0%);
    }
    .sheet-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .sheet-title {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-strong);
    }
    .sheet-close {
      min-width: 42px;
      min-height: 42px;
      border-radius: 10px;
      background: rgba(255, 92, 138, 0.14);
      color: var(--danger);
    }
    .class-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .class-chip {
      min-height: 42px;
      padding: 0 10px;
      border-radius: 10px;
      border: 2px solid rgba(103, 141, 122, 0.75);
      background: rgba(7, 10, 9, 0.84);
      color: var(--ink);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .class-chip.selected {
      border-color: rgba(255, 240, 106, 0.72);
      background: rgba(255, 240, 106, 0.12);
      color: var(--accent-strong);
    }
    .sheet-field {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .sheet-submit {
      width: 100%;
      min-height: 48px;
      border-radius: 12px;
      background: linear-gradient(180deg, var(--accent), #24cc81);
      color: #101110;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    audio {
      display: none;
    }
    @media (max-width: 420px) {
      .scope-grid,
      .context-grid,
      .class-grid {
        grid-template-columns: 1fr;
      }
      .context-copy {
        max-width: none;
        text-align: left;
      }
    }
    :root {
      color-scheme: dark;
      --bg: #15181d;
      --panel: #20252c;
      --panel-strong: #252b33;
      --panel-soft: #1b2027;
      --ink: #f1f3f5;
      --muted: #a7b0ba;
      --line: #363f4a;
      --line-strong: #4a5663;
      --accent: #78a8ff;
      --accent-strong: #9bbcff;
      --danger: #d97070;
      --success: #76b27a;
      --warning: #c9a45d;
      --shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
      --pixel: 1px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    html, body {
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
    }
    body::before,
    .eyebrow,
    .meta-pills,
    .progress-rail {
      display: none;
    }
    main {
      width: min(100%, 520px);
      height: 100dvh;
      min-height: 100dvh;
      padding: max(8px, env(safe-area-inset-top)) 8px calc(8px + env(safe-area-inset-bottom)) 8px;
      gap: 8px;
      overflow: hidden;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .panel-inset {
      border: 1px solid var(--line);
      background: var(--panel-soft);
      border-radius: 10px;
    }
    .topbar,
    .workspace {
      padding: 8px;
      gap: 8px;
    }
    .title-row {
      align-items: center;
      gap: 10px;
    }
    .brand-title {
      font-size: 1rem;
      line-height: 1;
      letter-spacing: 0.02em;
      text-transform: none;
      color: var(--ink);
      text-shadow: none;
    }
    .brand-stack,
    .counter {
      gap: 2px;
    }
    .title-meta {
      gap: 4px 6px;
    }
    .project-name {
      font-size: 0.9rem;
      font-weight: 600;
    }
    .session-name,
    .microcopy,
    .control-label,
    .section-label {
      font-size: 0.62rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .session-name {
      min-height: 22px;
      padding: 0 7px;
      border-width: 1px;
    }
    .counter {
      min-width: 82px;
      padding: 8px;
      border: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .counter strong {
      font-size: 0.92rem;
      color: var(--ink);
    }
    .meter-strip {
      padding: 8px;
      gap: 6px;
    }
    .meter-row {
      font-size: 0.66rem;
      letter-spacing: 0.03em;
    }
    .progress-boxes {
      display: flex;
      gap: 4px;
      align-items: center;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 2px 0 4px;
      overscroll-behavior-x: contain;
      scrollbar-width: none;
      -ms-overflow-style: none;
    }
    .progress-boxes::-webkit-scrollbar {
      display: none;
    }
    .progress-box {
      appearance: none;
      border: 0;
      border-radius: 999px;
      width: 9px;
      min-width: 9px;
      height: 9px;
      padding: 0;
      flex: 0 0 9px;
      background: #39414c;
      opacity: 0.85;
    }
    .progress-box.pending { background: #39414c; }
    .progress-box.correct { background: var(--success); }
    .progress-box.incorrect { background: var(--danger); }
    .progress-box.current {
      outline: 2px solid var(--accent-strong);
      outline-offset: 1px;
    }
    .scope-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    select,
    input,
    textarea {
      min-height: 36px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #171b20;
      box-shadow: none;
      font-size: 0.9rem;
    }
    textarea {
      min-height: 88px;
    }
    .session-drawer {
      padding: 6px 8px;
    }
    .session-drawer-body {
      gap: 6px;
    }
    .workspace {
      grid-template-rows: auto 1fr auto;
      min-height: 0;
      background: var(--panel);
    }
    .meta-row {
      display: none;
    }
    .context-copy {
      width: 100%;
      text-align: left;
      gap: 2px;
    }
    .context-copy strong {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--ink);
    }
    .center {
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 2px;
    }
    .hero {
      min-height: 100%;
      padding: 10px;
      display: grid;
      grid-template-rows: auto auto auto auto;
      gap: 8px;
      align-content: start;
    }
    .hero-head,
    .hero-copy,
    .hero-summary {
      display: grid;
      gap: 4px;
    }
    .hero-context {
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .hero-label {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #171b20;
      color: var(--ink);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .hero-meta {
      margin: 0;
      color: var(--muted);
      font-size: 0.8rem;
      line-height: 1.3;
    }
    .hero-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .hero-stat {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #171b20;
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .wave-shell {
      min-height: 0;
      padding: 8px;
      display: grid;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #171b20;
    }
    .waveform {
      width: 100%;
      height: 76px;
      display: block;
      border-radius: 8px;
      background: #101419;
    }
    .wave-meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      font-size: 0.66rem;
      color: var(--muted);
      text-transform: uppercase;
    }
    .wave-actions {
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      gap: 6px;
      flex-wrap: wrap;
    }
    .replay-button {
      min-height: 28px;
      padding: 0 10px;
      border-radius: 8px;
      border: 1px solid var(--line-strong);
      background: #1a2027;
      color: var(--ink);
      font-weight: 600;
      font-size: 0.72rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .status-line {
      min-height: 1.1rem;
      margin: 0;
      font-size: 0.76rem;
      color: var(--muted);
    }
    .bottom-bar {
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .decision-row,
    .nav-row {
      display: grid;
      gap: 8px;
      width: 100%;
    }
    .decision-row {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .nav-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .action {
      min-height: 42px;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 0 6px;
      font-size: 0.88rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      box-shadow: none;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .action.nav,
    .action.reclass {
      background: #222830;
      color: var(--ink);
    }
    .action.correct {
      background: rgba(118, 178, 122, 0.18);
      color: #d8f0db;
      border-color: rgba(118, 178, 122, 0.45);
    }
    .action.incorrect {
      background: rgba(217, 112, 112, 0.16);
      color: #f3d3d3;
      border-color: rgba(217, 112, 112, 0.4);
    }
    .action:disabled,
    .progress-box:disabled {
      opacity: 0.45;
    }
    .empty-state {
      height: 100%;
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 10px;
      background: #171b20;
    }
    .sheet-backdrop {
      background: rgba(0, 0, 0, 0.45);
    }
    .sheet {
      border-top-left-radius: 16px;
      border-top-right-radius: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    @media (max-width: 430px) {
      .waveform {
        height: 72px;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel topbar">
      <div class="title-row">
        <div class="brand-stack">
          <span class="eyebrow">Pocket Review Loop</span>
          <h1 class="brand-title">EZ Review</h1>
          <div class="title-meta">
            <p class="project-name" id="project-name">Loading current project…</p>
            <p class="session-name" id="session-name">Waiting for review run…</p>
          </div>
        </div>
        <div class="counter panel-inset">
          <span class="microcopy">Reviewed</span>
          <strong id="counter">0 / 0</strong>
          <span class="microcopy" id="counter-detail">Pending 0</span>
        </div>
      </div>

      <div class="panel-inset meter-strip">
        <div class="meter-row">
          <span id="scope-summary">Locating queue…</span>
          <div class="meter-actions">
            <span id="history-meta">Queue</span>
            <button class="icon-button" id="reload-status-btn" title="Reload review status" aria-label="Reload review status">&#8635;</button>
          </div>
        </div>
        <div class="progress-boxes" id="progress-boxes"></div>
      </div>

      <div class="scope-grid">
        <label>
          <span class="control-label">View</span>
          <select id="outcome-filter"></select>
        </label>
        <label>
          <span class="control-label">Song</span>
          <select id="song-select"></select>
        </label>
        <label>
          <span class="control-label">Layer</span>
          <select id="layer-select"></select>
        </label>
        <label>
          <span class="control-label">Class</span>
          <select id="class-filter"></select>
        </label>
      </div>

      <details class="panel-inset session-drawer" id="session-drawer">
        <summary>
          <div class="session-drawer-copy">
            <span class="section-label">Runs</span>
            <strong id="session-drawer-title">Current run</strong>
          </div>
          <span class="microcopy" id="session-drawer-meta">0 loaded</span>
        </summary>
        <div class="session-drawer-body">
          <label>
            <span class="control-label">Run Archive</span>
            <select id="session-select"></select>
          </label>
          <p class="status-line" id="archive-status"></p>
        </div>
      </details>
    </section>

    <section class="panel workspace">
      <div class="meta-row">
        <div class="meta-pills">
          <span class="pill" id="mode-pill"><strong>Queue</strong></span>
          <span class="pill kind-project" id="kind-pill">Session</span>
          <span class="pill" id="sample-pill">Pending</span>
        </div>
        <div class="context-copy">
          <strong id="active-context">Waiting for app context…</strong>
          <span class="microcopy" id="source-context">No project source yet</span>
        </div>
      </div>
      <div class="center" id="center"></div>
      <p class="status-line" id="status-line">Switching clip…</p>
    </section>

    <section class="panel bottom-bar">
      <div class="decision-row">
        <button class="action incorrect" id="incorrect-btn" title="Incorrect">Incorrect</button>
        <button class="action reclass" id="reclass-btn" title="Reclassify">Reclassify</button>
        <button class="action correct" id="correct-btn" title="Correct">Correct</button>
      </div>
      <div class="nav-row">
        <button class="action nav" id="prev-btn" title="Back">Back</button>
        <button class="action nav" id="next-btn" title="Forward">Forward</button>
      </div>
    </section>
  </main>

  <div class="sheet-backdrop" id="sheet-backdrop"></div>
  <section class="sheet" id="reclass-sheet">
    <div class="sheet-head">
      <h2 class="sheet-title">Reclassify Hit</h2>
      <button class="sheet-close" id="sheet-close" title="Close">✕</button>
    </div>
    <div class="class-grid" id="class-grid"></div>
    <label class="sheet-field">Pick or Type a Class
      <input id="reclass-label" type="text" placeholder="kick, snare, shaker">
    </label>
    <label class="sheet-field">Operator Note
      <textarea id="reclass-note" placeholder="What was wrong with this hit?"></textarea>
    </label>
    <button class="sheet-submit" id="sheet-submit">Save + Advance</button>
  </section>

  <audio id="player" preload="auto"></audio>

  <script>
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
  </script>
</body>
</html>"""
