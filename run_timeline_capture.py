from pathlib import Path

from echozero.ui.qt.timeline.test_harness import capture_demo_variants


if __name__ == "__main__":
    output_dir = Path("artifacts") / "timeline-demo"
    paths = capture_demo_variants(output_dir)
    for path in paths:
        print(path)
