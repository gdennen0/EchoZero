from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_FIXTURE = Path(r"C:\Users\griff\Desktop\Doechii_NissanAltima_117bpm_SPMTE_v02 [chan 1].wav")
DEFAULT_PROOF_ROOT = Path(r"C:\Users\griff\.openclaw\workspace\tmp\phase3-daw-proof")
DEFAULT_WALKTHROUGH_SOURCE = Path(r"C:\Users\griff\.openclaw\workspace\tmp\object-info-demo")
DEFAULT_MA3_REPLAY_FIXTURE = Path("tests/fixtures/ma3/reconnect_replay_v1.json")
REAL_DATA_CAPTURE_SCRIPT = REPO_ROOT / "run_timeline_real_data_capture.py"


@dataclass(slots=True)
class ArtifactState:
    status: str
    path: Path | None = None
    detail: str | None = None
    external_bootstrap: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the Phase 3 LD-01 evidence pack.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date used in the evidence folder name.")
    parser.add_argument("--operator", default="codex", help="Operator name recorded in run-notes.")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_FIXTURE), help="Canonical LD-01 audio fixture path.")
    parser.add_argument(
        "--proof-root",
        default=str(DEFAULT_PROOF_ROOT),
        help="External root for Phase 3 evidence packs.",
    )
    parser.add_argument(
        "--walkthrough-source-dir",
        default=str(DEFAULT_WALKTHROUGH_SOURCE),
        help="Existing external object-info walkthrough output directory.",
    )
    parser.add_argument(
        "--ma3-replay-fixture",
        default=str(DEFAULT_MA3_REPLAY_FIXTURE),
        help="Repo-relative MA3 replay fixture used for the LD-01 sync proof path.",
    )
    parser.add_argument(
        "--divergence-shot",
        default="",
        help="Optional explicit divergence screenshot to copy into the evidence pack.",
    )
    parser.add_argument(
        "--resolution-shot",
        default="",
        help="Optional explicit post-resolution/sync screenshot to copy into the evidence pack.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without creating artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proof_root = Path(args.proof_root)
    evidence_dir = proof_root / f"{args.date}-ld-01"
    raw_capture_dir = evidence_dir / "raw" / "timeline-real-data"
    work_root = evidence_dir / "work" / "real-data"
    walkthrough_source_dir = Path(args.walkthrough_source_dir)
    audio_path = Path(args.audio)
    divergence_shot = Path(args.divergence_shot) if args.divergence_shot else None
    resolution_shot = Path(args.resolution_shot) if args.resolution_shot else None
    ma3_replay_fixture = Path(args.ma3_replay_fixture)
    if not ma3_replay_fixture.is_absolute():
        ma3_replay_fixture = (REPO_ROOT / ma3_replay_fixture).resolve()

    print(f"EVIDENCE_DIR={evidence_dir}")
    print(f"RAW_CAPTURE_DIR={raw_capture_dir}")
    print(f"WORK_ROOT={work_root}")
    print(f"AUDIO_FIXTURE={audio_path}")
    print(f"MA3_REPLAY_FIXTURE={ma3_replay_fixture}")
    print(f"WALKTHROUGH_SOURCE_DIR={walkthrough_source_dir}")
    print(f"DIVERGENCE_SHOT={divergence_shot if divergence_shot else '<none>'}")
    print(f"RESOLUTION_SHOT={resolution_shot if resolution_shot else '<none>'}")

    capture_cmd = [
        sys.executable,
        str(REAL_DATA_CAPTURE_SCRIPT),
        "--audio",
        str(audio_path),
        "--output-dir",
        str(raw_capture_dir),
        "--working-root",
        str(work_root),
    ]

    if args.dry_run:
        print("DRY_RUN=1")
        print("CAPTURE_COMMAND=" + subprocess.list2cmdline(capture_cmd))
        return 0

    try:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        raw_capture_dir.mkdir(parents=True, exist_ok=True)
        work_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR=failed to prepare evidence directories: {exc}", file=sys.stderr)
        return 2

    capture_result = subprocess.run(
        capture_cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    print("REAL_DATA_CAPTURE_EXIT_CODE=" + str(capture_result.returncode))
    if capture_result.stdout:
        print(capture_result.stdout.rstrip())
    if capture_result.stderr:
        print(capture_result.stderr.rstrip(), file=sys.stderr)

    commit_hash = git_head()
    timestamp = datetime.now(timezone.utc).isoformat()

    initial_artifact = mark_external_bootstrap(
        copy_if_exists(
            walkthrough_source_dir / "object_info_empty.png",
            evidence_dir / "01-initial-state.png",
            detail="Seeded from the current external object-info walkthrough bootstrap.",
        )
    )
    post_extract_artifact = copy_if_exists(
        raw_capture_dir / "timeline_real_default.png",
        evidence_dir / "02-post-extract-all.png",
        detail="Generated by run_timeline_real_data_capture.py.",
    )
    divergence_artifact = resolve_optional_artifact(
        source=divergence_shot,
        destination=evidence_dir / "03-divergence-visible.png",
        note_path=evidence_dir / "03-divergence-visible.note.md",
        success_detail_prefix="Bootstrap copied explicit external divergence capture",
        missing_message="Divergence capture is not automated in the bootstrap helper yet; no explicit --divergence-shot input was provided.",
    )
    resolution_artifact = resolve_optional_artifact(
        source=resolution_shot,
        destination=evidence_dir / "04-post-resolution-or-sync.png",
        note_path=evidence_dir / "04-post-resolution-or-sync.note.md",
        success_detail_prefix="Bootstrap copied explicit external resolution/sync capture",
        missing_message="Resolution/sync capture is not automated in the bootstrap helper yet; no explicit --resolution-shot input was provided.",
    )
    walkthrough_artifact = resolve_walkthrough_video(
        walkthrough_source_dir=walkthrough_source_dir,
        destination=evidence_dir / "phase3-ld-01-walkthrough.mp4",
    )

    artifact_states = {
        "01-initial-state.png": initial_artifact,
        "02-post-extract-all.png": post_extract_artifact,
        "03-divergence-visible.png": divergence_artifact,
        "04-post-resolution-or-sync.png": resolution_artifact,
        "phase3-ld-01-walkthrough.mp4": walkthrough_artifact,
    }
    run_status, signoff_ready, run_outcome_note = evaluate_run_status(
        capture_exit_code=capture_result.returncode,
        artifacts=artifact_states,
    )

    run_notes = build_run_notes(
        timestamp=timestamp,
        operator=args.operator,
        commit_hash=commit_hash,
        run_status=run_status,
        signoff_ready=signoff_ready,
        run_outcome_note=run_outcome_note,
        audio_path=audio_path,
        ma3_replay_fixture=ma3_replay_fixture,
        evidence_dir=evidence_dir,
        raw_capture_dir=raw_capture_dir,
        work_root=work_root,
        walkthrough_source_dir=walkthrough_source_dir,
        capture_result=capture_result,
        artifacts=artifact_states,
    )
    run_notes_path = evidence_dir / "run-notes.md"
    run_notes_path.write_text(run_notes, encoding="utf-8")

    print(f"RUN_STATUS={run_status}")
    print(f"RUN_NOTES={run_notes_path}")
    return 0 if capture_result.returncode == 0 else capture_result.returncode


def copy_if_exists(source: Path, destination: Path, *, detail: str) -> ArtifactState:
    if not source.exists():
        return ArtifactState(status="MISSING", detail=f"Source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return ArtifactState(status="PASS", path=destination, detail=detail)


def missing_artifact(note_path: Path, message: str) -> ArtifactState:
    note_path.write_text(message + "\n", encoding="utf-8")
    return ArtifactState(status="INCOMPLETE", path=note_path, detail=message)


def resolve_optional_artifact(
    *,
    source: Path | None,
    destination: Path,
    note_path: Path,
    success_detail_prefix: str,
    missing_message: str,
) -> ArtifactState:
    if source is not None and source.exists():
        if note_path.exists():
            note_path.unlink()
        return mark_external_bootstrap(
            copy_if_exists(
                source,
                destination,
                detail=f"{success_detail_prefix} from {source}.",
            )
        )
    return missing_artifact(note_path, missing_message)


def resolve_walkthrough_video(*, walkthrough_source_dir: Path, destination: Path) -> ArtifactState:
    summary_path = walkthrough_source_dir / "run-summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ArtifactState(status="MISSING", detail=f"Invalid walkthrough summary JSON: {exc}")
        video_path = Path(str(summary.get("video", "")))
        if video_path.exists():
            shutil.copy2(video_path, destination)
            return ArtifactState(
                status="PASS",
                path=destination,
                detail=f"Copied existing object-info walkthrough video from {video_path}.",
                external_bootstrap=True,
            )
    direct_video = walkthrough_source_dir / "object_info_demo.mp4"
    if direct_video.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(direct_video, destination)
        return ArtifactState(
            status="PASS",
            path=destination,
            detail=f"Copied existing walkthrough video from {direct_video}.",
            external_bootstrap=True,
        )
    return ArtifactState(status="MISSING", detail=f"Walkthrough video not found under {walkthrough_source_dir}")


def mark_external_bootstrap(state: ArtifactState) -> ArtifactState:
    return ArtifactState(
        status=state.status,
        path=state.path,
        detail=state.detail,
        external_bootstrap=True,
    )


def evaluate_run_status(*, capture_exit_code: int, artifacts: dict[str, ArtifactState]) -> tuple[str, bool, str]:
    required_missing = [
        name
        for name, state in artifacts.items()
        if state.status != "PASS"
    ]
    if capture_exit_code != 0:
        return (
            "INCOMPLETE",
            False,
            f"bootstrap-only: real-data capture failed with exit code {capture_exit_code}, so the pack is not signoff-ready.",
        )
    if required_missing:
        return (
            "INCOMPLETE",
            False,
            "bootstrap-only: required artifacts are missing or incomplete, so the pack is not signoff-ready.",
        )
    if any(state.external_bootstrap for state in artifacts.values()):
        return (
            "BOOTSTRAP_PASS",
            False,
            "bootstrap-only: all required artifacts are present, but at least one required artifact depends on an explicit external bootstrap source.",
        )
    return (
        "PASS",
        True,
        "signoff-ready: all required artifacts were produced by this run without explicit external bootstrap dependencies.",
    )


def git_head() -> str:
    result = subprocess.run(
        ["git", "-c", "safe.directory=C:/Users/griff/EchoZero", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "UNKNOWN"


def build_run_notes(
    *,
    timestamp: str,
    operator: str,
    commit_hash: str,
    run_status: str,
    signoff_ready: bool,
    run_outcome_note: str,
    audio_path: Path,
    ma3_replay_fixture: Path,
    evidence_dir: Path,
    raw_capture_dir: Path,
    work_root: Path,
    walkthrough_source_dir: Path,
    capture_result: subprocess.CompletedProcess[str],
    artifacts: dict[str, ArtifactState],
) -> str:
    lines = [
        "# Phase 3 LD-01 Evidence Pack",
        "",
        f"- status: {run_status}",
        f"- signoff_ready: {'yes' if signoff_ready else 'no'}",
        f"- executed_at_utc: {timestamp}",
        f"- operator: {operator}",
        f"- commit: {commit_hash}",
        f"- audio_fixture: {audio_path}",
        f"- ma3_replay_fixture: {ma3_replay_fixture}",
        f"- evidence_dir: {evidence_dir}",
        f"- raw_capture_dir: {raw_capture_dir}",
        f"- work_root: {work_root}",
        f"- walkthrough_source_dir: {walkthrough_source_dir}",
        f"- real_data_capture_exit_code: {capture_result.returncode}",
        f"- run_classification: {run_outcome_note}",
        "",
        "## Artifact status",
        "",
    ]
    for name, state in artifacts.items():
        target = state.path if state.path is not None else "not-created"
        detail = state.detail or ""
        lines.append(f"- {name}: {state.status} ({target})")
        if detail:
            lines.append(f"  - {detail}")
        lines.append(f"  - external_bootstrap: {'yes' if state.external_bootstrap else 'no'}")
    lines.extend(
        [
            "",
            "## Signoff",
            "",
            run_outcome_note,
            "",
            "## Capture command",
            "",
            f"`{subprocess.list2cmdline(capture_result.args) if isinstance(capture_result.args, list) else str(capture_result.args)}`",
            "",
            "## Capture stdout",
            "",
            "```text",
            capture_result.stdout.rstrip() or "<empty>",
            "```",
            "",
            "## Capture stderr",
            "",
            "```text",
            capture_result.stderr.rstrip() or "<empty>",
            "```",
            "",
            "Bootstrap result: the external proof folder is created, the canonical fixture references are recorded, the real-data screenshot lane is automated, and artifact provenance is explicit for both generated captures and any externally supplied bootstrap screenshots.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
