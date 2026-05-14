#!/usr/bin/env python3
"""
macOS release-artifact verifier for EchoZero zip uploads.
Exists because release gates must inspect the downloaded zip, not a mutable dist app.
Connects archive integrity, bundle signing, launch smoke, and asset parity to one command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smoke_packaged_app import run_packaged_smoke

DEFAULT_REPORT_PATH = Path("dist") / "macos-release-verification.json"
DEFAULT_SMOKE_TIMEOUT_SECONDS = 45.0
DEFAULT_SMOKE_EXIT_SECONDS = 6.0
_RUNTIME_CONFIG_DIR = Path("Contents") / "MacOS" / "config"


@dataclass(frozen=True)
class VerificationFailure:
    """Describes one failed release-artifact check with an operator action."""

    check: str
    message: str
    action: str


@dataclass
class VerificationReport:
    """Collects release-artifact verification evidence for JSON output."""

    archive: str
    status: str = "failed"
    failures: list[VerificationFailure] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)

    def add_failure(self, check: str, message: str, action: str) -> None:
        """Add a failure while preserving a stable check identifier."""
        self.failures.append(VerificationFailure(check=check, message=message, action=action))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report payload."""
        return {
            "schema": "echozero.macos_release_verification.v1",
            "archive": self.archive,
            "status": self.status,
            "platform": platform.platform(),
            "checks": self.checks,
            "failures": [failure.__dict__ for failure in self.failures],
        }


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a release artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a zip archive after rejecting path traversal entries."""
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"zip entry escapes extraction root: {member.filename}")
    ditto = shutil.which("ditto")
    if ditto:
        result = subprocess.run(
            [ditto, "-x", "-k", str(zip_path), str(destination)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise OSError(result.stderr.strip() or "ditto extraction failed")
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def find_single_app_bundle(root: Path, app_name: str = "EchoZero.app") -> Path:
    """Find the single EchoZero .app bundle extracted from a release zip."""
    matches = sorted(path for path in root.rglob(app_name) if path.is_dir())
    if len(matches) != 1:
        detail = "none found" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(f"expected exactly one {app_name}; found {detail}")
    return matches[0]


def resolve_bundle_executable(app_bundle: Path) -> Path:
    """Resolve the primary executable inside an EchoZero macOS app bundle."""
    executable = app_bundle / "Contents" / "MacOS" / app_bundle.stem
    if not executable.is_file():
        raise FileNotFoundError(f"app executable not found: {executable}")
    return executable


def list_runtime_config_files(app_bundle: Path) -> list[str]:
    """List mutable runtime config files that must not exist inside the app bundle."""
    config_dir = app_bundle / _RUNTIME_CONFIG_DIR
    if not config_dir.exists():
        return []
    return sorted(
        str(path.relative_to(app_bundle))
        for path in config_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def extract_binary_uuids(executable: Path) -> list[str]:
    """Read Mach-O UUIDs from the packaged executable using dwarfdump."""
    result = subprocess.run(
        ["dwarfdump", "--uuid", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "dwarfdump returned a non-zero exit code"
        raise RuntimeError(stderr)
    return re.findall(r"UUID:\s*([0-9A-Fa-f-]{36})", result.stdout)


def verify_codesign_strict(app_bundle: Path) -> str:
    """Run strict macOS codesign verification for the extracted app bundle."""
    result = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app_bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(output or "codesign strict verification failed")
    return output


def build_tree_manifest(app_bundle: Path) -> dict[str, str]:
    """Build a path-to-SHA256 manifest for app bundle asset equivalence checks."""
    manifest: dict[str, str] = {}
    for path in sorted(item for item in app_bundle.rglob("*") if item.is_file()):
        manifest[str(path.relative_to(app_bundle))] = compute_sha256(path)
    return manifest


def compare_app_assets(primary_app: Path, comparison_zip: Path) -> dict[str, object]:
    """Compare an extracted app bundle against a second release zip's bundle contents."""
    with tempfile.TemporaryDirectory(prefix="echozero-compare-") as temp_dir:
        compare_root = Path(temp_dir) / "zip"
        compare_root.mkdir(parents=True)
        safe_extract_zip(comparison_zip, compare_root)
        comparison_app = find_single_app_bundle(compare_root)
        primary_manifest = build_tree_manifest(primary_app)
        comparison_manifest = build_tree_manifest(comparison_app)
    missing = sorted(set(primary_manifest) - set(comparison_manifest))
    extra = sorted(set(comparison_manifest) - set(primary_manifest))
    changed = sorted(
        path
        for path in set(primary_manifest) & set(comparison_manifest)
        if primary_manifest[path] != comparison_manifest[path]
    )
    return {
        "comparison_zip": str(comparison_zip),
        "matched": not missing and not extra and not changed,
        "file_count": len(primary_manifest),
        "missing": missing[:20],
        "extra": extra[:20],
        "changed": changed[:20],
        "truncated": len(missing) > 20 or len(extra) > 20 or len(changed) > 20,
    }


@dataclass(frozen=True)
class VerificationOptions:
    """CLI-configurable macOS release artifact verification options."""

    archive: Path
    expected_sha256: str | None = None
    expected_binary_uuid: str | None = None
    compare_zip: Path | None = None
    report_path: Path = DEFAULT_REPORT_PATH
    smoke_timeout_seconds: float = DEFAULT_SMOKE_TIMEOUT_SECONDS
    smoke_exit_seconds: float = DEFAULT_SMOKE_EXIT_SECONDS


def verify_macos_release_artifact(options: VerificationOptions) -> VerificationReport:
    """Verify a macOS EchoZero release zip from first principles."""
    report = VerificationReport(archive=str(options.archive))
    if not options.archive.is_file():
        report.add_failure(
            "archive_exists",
            f"not found: {options.archive}",
            "Download the GitHub release asset zip first.",
        )
        return report

    actual_sha256 = compute_sha256(options.archive)
    report.checks["sha256"] = {"actual": actual_sha256, "expected": options.expected_sha256}
    if options.expected_sha256 and actual_sha256.lower() != options.expected_sha256.lower():
        report.add_failure(
            "sha256",
            f"expected {options.expected_sha256}, got {actual_sha256}",
            "Delete the zip and redownload the GitHub release asset; do not test this artifact.",
        )
        return report

    with tempfile.TemporaryDirectory(prefix="echozero-release-") as temp_dir:
        extract_root = Path(temp_dir) / "zip"
        extract_root.mkdir(parents=True)
        try:
            safe_extract_zip(options.archive, extract_root)
            app_bundle = find_single_app_bundle(extract_root)
            executable = resolve_bundle_executable(app_bundle)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            report.add_failure(
                "extract_app",
                str(exc),
                "Rebuild and re-upload a zip containing exactly one EchoZero.app bundle.",
            )
            return report
        report.checks["app_bundle"] = str(app_bundle.relative_to(extract_root))
        report.checks["executable"] = str(executable.relative_to(app_bundle))

        _verify_no_runtime_config(report, app_bundle, phase="pre_smoke")
        _verify_binary_uuid(report, executable, options.expected_binary_uuid)
        _verify_codesign(report, app_bundle)
        if options.compare_zip is not None:
            _verify_asset_equivalence(report, app_bundle, options.compare_zip)
        if not _has_failures(report, "runtime_config_pre_smoke", "binary_uuid", "codesign_strict"):
            _verify_packaged_smoke(report, app_bundle, options)
            _verify_no_runtime_config(report, app_bundle, phase="post_smoke")
        else:
            report.checks["packaged_smoke"] = {"status": "skipped", "reason": "pre_launch_failure"}

    report.status = "passed" if not report.failures else "failed"
    return report


def _has_failures(report: VerificationReport, *checks: str) -> bool:
    return any(failure.check in checks for failure in report.failures)


def _verify_no_runtime_config(report: VerificationReport, app_bundle: Path, *, phase: str) -> None:
    files = list_runtime_config_files(app_bundle)
    report.checks[f"runtime_config_{phase}"] = {"files": files}
    if files:
        report.add_failure(
            f"runtime_config_{phase}",
            f"bundle contains mutable config files under {_RUNTIME_CONFIG_DIR}: {files}",
            "Rebuild from a clean app bundle; frozen settings must live in the user profile, "
            "not Contents/MacOS/config.",
        )


def _verify_binary_uuid(
    report: VerificationReport, executable: Path, expected_binary_uuid: str | None
) -> None:
    try:
        uuids = extract_binary_uuids(executable)
    except (OSError, RuntimeError) as exc:
        report.add_failure(
            "binary_uuid",
            str(exc),
            "Run on macOS with Xcode Command Line Tools installed, then rebuild if UUID "
            "extraction still fails.",
        )
        return
    report.checks["binary_uuid"] = {"actual": uuids, "expected": expected_binary_uuid}
    if not uuids:
        report.add_failure(
            "binary_uuid",
            "no Mach-O UUID found",
            "Rebuild the macOS app and verify the packaged executable is a valid Mach-O binary.",
        )
    elif expected_binary_uuid and expected_binary_uuid.upper() not in {
        uuid.upper() for uuid in uuids
    }:
        report.add_failure(
            "binary_uuid",
            f"expected {expected_binary_uuid}, got {uuids}",
            "Stop testing this zip; it is not the approved app binary.",
        )


def _verify_codesign(report: VerificationReport, app_bundle: Path) -> None:
    try:
        output = verify_codesign_strict(app_bundle)
    except (OSError, RuntimeError) as exc:
        report.add_failure(
            "codesign_strict",
            str(exc),
            "Re-sign the final app bundle before zipping; never mutate a signed bundle "
            "after signing.",
        )
        return
    report.checks["codesign_strict"] = {"passed": True, "output": output}


def _verify_asset_equivalence(
    report: VerificationReport, app_bundle: Path, compare_zip: Path
) -> None:
    try:
        comparison = compare_app_assets(app_bundle, compare_zip)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        report.add_failure(
            "asset_equivalence",
            str(exc),
            "Provide the second GitHub release zip and ensure it contains exactly one "
            "EchoZero.app bundle.",
        )
        return
    report.checks["asset_equivalence"] = comparison
    if not comparison["matched"]:
        report.add_failure(
            "asset_equivalence",
            json.dumps(comparison, sort_keys=True),
            "Do not publish two differently named macOS zips unless their extracted app "
            "payloads are byte-equivalent.",
        )


def _verify_packaged_smoke(
    report: VerificationReport, app_bundle: Path, options: VerificationOptions
) -> None:
    smoke_working_root = app_bundle.parent / "smoke-working"
    smoke_log_dir = app_bundle.parent / "smoke-logs"
    try:
        smoke_report = run_packaged_smoke(
            app_bundle,
            timeout_seconds=options.smoke_timeout_seconds,
            smoke_exit_seconds=options.smoke_exit_seconds,
            working_dir_root=smoke_working_root,
            log_dir=smoke_log_dir,
        )
    except Exception as exc:  # smoke_packaged_app converts expected launch failures to reports
        smoke_report = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    report.checks["packaged_smoke"] = smoke_report
    if smoke_report.get("status") != "passed":
        report.add_failure(
            "packaged_smoke",
            json.dumps(smoke_report, sort_keys=True),
            "Fix the packaged launch crash/failure before asking anyone to test the release.",
        )


def write_report(report: VerificationReport, report_path: Path) -> None:
    """Write a stable JSON verification report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the macOS release verifier."""
    parser = argparse.ArgumentParser(description="Verify a macOS EchoZero release zip.")
    parser.add_argument("archive", type=Path, help="Downloaded EchoZero macOS release zip.")
    parser.add_argument("--expected-sha256", help="Expected SHA-256 digest for the zip.")
    parser.add_argument(
        "--expected-binary-uuid",
        help="Expected Mach-O UUID for Contents/MacOS/EchoZero.",
    )
    parser.add_argument(
        "--compare-zip",
        type=Path,
        help="Second macOS zip that must contain a byte-equivalent EchoZero.app payload.",
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=float,
        default=DEFAULT_SMOKE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--smoke-exit-seconds", type=float, default=DEFAULT_SMOKE_EXIT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the canonical macOS release artifact verifier."""
    parsed = build_parser().parse_args(argv)
    options = VerificationOptions(
        archive=parsed.archive,
        expected_sha256=parsed.expected_sha256,
        expected_binary_uuid=parsed.expected_binary_uuid,
        compare_zip=parsed.compare_zip,
        report_path=parsed.report_path,
        smoke_timeout_seconds=parsed.smoke_timeout_seconds,
        smoke_exit_seconds=parsed.smoke_exit_seconds,
    )
    report = verify_macos_release_artifact(options)
    write_report(report, options.report_path)
    print(f"report={options.report_path}")
    print(f"status={report.status}")
    for failure in report.failures:
        print(f"FAIL {failure.check}: {failure.message}")
        print(f"ACTION {failure.check}: {failure.action}")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
