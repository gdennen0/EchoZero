"""
Extract golden test files from EchoZero v1 .ez project files.
Reads the project.json inside a .ez ZIP, extracts event data as clean golden JSON files.
Output goes to tests/fixtures/golden/ for use with assert_matches_golden().

Usage:
    python scripts/extract_golden.py path/to/project.ez
    python scripts/extract_golden.py path/to/project.ez --list       # list available data items
    python scripts/extract_golden.py path/to/project.ez --item NAME  # extract specific item
    python scripts/extract_golden.py path/to/project.ez --all        # extract all event items
"""

import json
import sys
import zipfile
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "golden"


def load_project(ez_path: str) -> dict:
    """Load project.json from a .ez file (ZIP or plain JSON)."""
    path = Path(ez_path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("project.json"))
    else:
        return json.loads(path.read_text())


def list_items(project: dict) -> None:
    """Print all data items with event counts."""
    print(f"\nProject: {project['name']}")
    print(f"{'#':<4} {'Name':<60} {'Type':<8} {'Events':<8} {'Layers':<6}")
    print("-" * 90)
    for i, di in enumerate(project.get("data_items", [])):
        layers = di.get("layers", [])
        total_events = sum(len(layer.get("events", [])) for layer in layers)
        print(f"{i:<4} {di['name']:<60} {di['type']:<8} {total_events:<8} {len(layers):<6}")


def extract_item(project: dict, item_name: str, prefix: str = "") -> Path:
    """Extract a single data item's events to a golden JSON file."""
    for di in project.get("data_items", []):
        if di["name"] == item_name:
            golden = {
                "source_project": project["name"],
                "source_item": di["name"],
                "source_type": di["type"],
                "layers": [],
            }

            for layer in di.get("layers", []):
                clean_events = []
                for event in layer.get("events", []):
                    clean_events.append({
                        "time": event["time"],
                        "duration": event.get("duration", 0.0),
                        "classification": event.get("classification", ""),
                        "source": event.get("metadata", {}).get("source", "unknown"),
                    })

                # Sort by time for consistent comparison
                clean_events.sort(key=lambda e: e["time"])

                golden["layers"].append({
                    "name": layer.get("name", "unnamed"),
                    "event_count": len(clean_events),
                    "events": clean_events,
                })

            # Generate filename
            safe_name = item_name.replace(" ", "_").replace("/", "_")[:80]
            if prefix:
                safe_name = f"{prefix}_{safe_name}"
            out_path = GOLDEN_DIR / f"{safe_name}_expected.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(golden, indent=2))
            total = sum(l["event_count"] for l in golden["layers"])
            print(f"  OK {out_path.name} ({total} events)")
            return out_path

    print(f"  MISSING Item '{item_name}' not found")
    return Path()


def extract_all(project: dict, prefix: str = "") -> list[Path]:
    """Extract all Event-type data items."""
    paths = []
    for di in project.get("data_items", []):
        if di["type"] == "Event":
            layers = di.get("layers", [])
            total = sum(len(l.get("events", [])) for l in layers)
            if total > 0:
                p = extract_item(project, di["name"], prefix)
                if p.exists():
                    paths.append(p)
    return paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_golden.py <path.ez> [--list|--all|--item NAME]")
        sys.exit(1)

    ez_path = sys.argv[1]
    project = load_project(ez_path)
    prefix = project["name"].replace(" ", "_")

    if "--list" in sys.argv:
        list_items(project)
    elif "--all" in sys.argv:
        print(f"\nExtracting all events from '{project['name']}'...")
        paths = extract_all(project, prefix)
        print(f"\nDone: {len(paths)} golden files written to {GOLDEN_DIR}/")
    elif "--item" in sys.argv:
        idx = sys.argv.index("--item")
        if idx + 1 < len(sys.argv):
            extract_item(project, sys.argv[idx + 1], prefix)
        else:
            print("Error: --item requires a name")
    else:
        # Default: list items
        list_items(project)
        print(f"\nUse --all to extract, or --item NAME for a specific one.")
