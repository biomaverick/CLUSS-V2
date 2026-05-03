#!/usr/bin/env bash
set -euo pipefail

echo "Installing CLUSS+ ..."
pip install -e ".[dev]"
echo ""
echo "Done. Run: cluss+ --help"
