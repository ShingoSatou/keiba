#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmd="${1:-check}"

case "$cmd" in
  lint)
    uv run ruff check .
    ;;
  fmt-check)
    uv run ruff format --check .
    ;;
  test)
    uv run pytest -q
    ;;
  check)
    "$0" lint
    "$0" fmt-check
    "$0" test
    ;;
  fix)
    uv run ruff format .
    "$0" lint
    "$0" test
    ;;
  *)
    echo "Usage: $0 {lint|fmt-check|test|check|fix}" >&2
    exit 1
    ;;
esac
