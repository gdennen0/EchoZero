"""Docs freshness and integrity audit.

Enforces repository documentation guardrails:
- required status/date metadata
- no machine-local absolute markdown links
- no broken local markdown links
- no archive promotion in docs front doors/nav
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT_CORE_DOCS = ("README.md", "AGENTS.md", "GLOSSARY.md", "STYLE.md")
ALLOWED_STATUS = {"active", "historical", "reference", "draft"}
CANONICAL_DOCS = {
    "README.md",
    "AGENTS.md",
    "GLOSSARY.md",
    "STYLE.md",
    "docs/index.md",
    "docs/STATUS.md",
    "docs/AGENT-CONTEXT.md",
    "docs/TESTING.md",
    "docs/ARCHITECTURE.md",
    "docs/UNIFIED-IMPLEMENTATION-PLAN.md",
    "docs/APP-DELIVERY-PLAN.md",
    "docs/UI-STANDARD.md",
    "docs/SONG-IMPORT-BATCH-LTC-WORKFLOW.md",
}

STATUS_RE = re.compile(r"^\s*Status:\s*([A-Za-z\- ]+)\s*$", re.I)
LAST_VERIFIED_RE = re.compile(r"^\s*Last verified:\s*(.+)\s*$", re.I)
LAST_REVIEWED_RE = re.compile(r"^\s*Last reviewed:\s*(.+)\s*$", re.I)
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
LOCAL_ABS_RE = re.compile(r"\]\((/Users/|[A-Za-z]:\\Users\\)")
LINE_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]+:\d+$")


def collect_docs(repo_root: Path) -> list[Path]:
    docs = sorted((repo_root / "docs").rglob("*.md"))
    for doc in ROOT_CORE_DOCS:
        p = repo_root / doc
        if p.exists():
            docs.append(p)
    return docs


def read_head(path: Path, max_lines: int = 40) -> list[str]:
    return path.read_text(errors="ignore").splitlines()[:max_lines]


def extract_status_and_dates(head_lines: list[str]) -> tuple[str | None, bool, bool]:
    status: str | None = None
    has_verified = False
    has_reviewed = False
    for line in head_lines:
        if status is None:
            m = STATUS_RE.match(line)
            if m:
                status = m.group(1).strip().lower()
        if LAST_VERIFIED_RE.match(line):
            has_verified = True
        if LAST_REVIEWED_RE.match(line):
            has_reviewed = True
    return status, has_verified, has_reviewed


def is_external_link(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def strip_fragment_and_line_suffix(target: str) -> str:
    base = target.split("#", 1)[0]
    if LINE_SUFFIX_RE.search(base):
        base = base.rsplit(":", 1)[0]
    return base


def classify_bucket(rel: str, status: str | None) -> str:
    if rel.startswith("docs/archive/") or status == "historical":
        return "historical"
    if rel in CANONICAL_DOCS:
        return "canonical"
    upper = rel.upper()
    if any(token in upper for token in ("PLAN", "SPEC", "TASK-BOARD", "BOARD")):
        return "active-plan"
    return "unknown"


def validate_archive_promotion(repo_root: Path) -> list[str]:
    errors: list[str] = []
    mkdocs = repo_root / "mkdocs.yml"
    if mkdocs.exists():
        for idx, line in enumerate(mkdocs.read_text(errors="ignore").splitlines(), start=1):
            if "archive/" in line and "archive/README.md" not in line:
                errors.append(
                    f"{mkdocs}:{idx}: archive file promoted in nav (only archive/README.md is allowed)"
                )
    index = repo_root / "docs" / "index.md"
    if index.exists():
        for idx, line in enumerate(index.read_text(errors="ignore").splitlines(), start=1):
            if "](archive/" in line and "](archive/README.md)" not in line:
                errors.append(
                    f"{index}:{idx}: archive file promoted in front door (only archive/README.md is allowed)"
                )
    return errors


def audit(repo_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []
    inventory: list[dict[str, str]] = []

    for path in collect_docs(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        head = read_head(path)
        status, has_verified, has_reviewed = extract_status_and_dates(head)

        if status is None:
            errors.append(f"{rel}: missing `Status:` in top section")
        elif status not in ALLOWED_STATUS:
            errors.append(f"{rel}: invalid status `{status}` (allowed: {sorted(ALLOWED_STATUS)})")

        in_archive = rel.startswith("docs/archive/")
        if in_archive:
            if status != "historical":
                errors.append(f"{rel}: archived docs must have `Status: historical`")
            if not has_reviewed:
                errors.append(f"{rel}: archived docs must include `Last reviewed:`")
        else:
            if status == "historical":
                errors.append(f"{rel}: non-archive docs cannot use `Status: historical`")
            if rel in CANONICAL_DOCS:
                if not has_verified:
                    errors.append(f"{rel}: canonical docs must include `Last verified:`")
            elif not (has_verified or has_reviewed):
                errors.append(f"{rel}: docs must include `Last verified:` or `Last reviewed:`")

        text = path.read_text(errors="ignore")
        if LOCAL_ABS_RE.search(text):
            errors.append(f"{rel}: contains machine-local absolute markdown link")

        for link in MD_LINK_RE.findall(text):
            if is_external_link(link):
                continue
            base = strip_fragment_and_line_suffix(link)
            if not base:
                continue
            target = (path.parent / base).resolve() if not base.startswith("/") else Path(base)
            if not target.exists():
                errors.append(f"{rel}: broken local link `{link}`")

        inventory.append(
            {
                "path": rel,
                "status": status or "missing",
                "bucket": classify_bucket(rel, status),
            }
        )

    errors.extend(validate_archive_promotion(repo_root))
    return errors, inventory


def write_inventory(inventory: list[dict[str, str]], json_path: Path | None, csv_path: Path | None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(inventory, indent=2) + "\n")
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "status", "bucket"])
            writer.writeheader()
            writer.writerows(inventory)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit docs metadata, links, and archive promotion.")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--inventory-json", help="Optional JSON inventory output path")
    parser.add_argument("--inventory-csv", help="Optional CSV inventory output path")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    errors, inventory = audit(repo_root)
    write_inventory(
        inventory,
        Path(args.inventory_json).resolve() if args.inventory_json else None,
        Path(args.inventory_csv).resolve() if args.inventory_csv else None,
    )

    if errors:
        print("docs_audit: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"docs_audit: PASS ({len(inventory)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
