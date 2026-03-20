# Test Fixtures

## Directory Structure

- `audio/` — Sample audio files for processor tests (WAV, click tracks, drum loops)
- `golden/` — Golden file snapshots for regression tests (JSON expected outputs)
- `projects/` — Template `.ez` project files for save/load round-trip tests

## Conventions

- Golden files are named `<test_name>_expected.json`
- Audio fixtures are generated programmatically when possible (see conftest.py)
- Real audio samples go here only when synthetic audio isn't sufficient
- Keep fixtures small — test files should be seconds, not minutes
