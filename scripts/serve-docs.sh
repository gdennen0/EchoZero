#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python -m mkdocs serve --open
