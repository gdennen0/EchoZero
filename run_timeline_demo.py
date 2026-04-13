import sys

from run_echozero import main as _run_echozero_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--use-demo-fixture" not in args:
        args.append("--use-demo-fixture")
    return _run_echozero_main(args)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
