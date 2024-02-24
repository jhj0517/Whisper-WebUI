#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/venv/Scripts/activate"

PYTHON="$SCRIPT_DIR/venv/Scripts/python.exe"
echo "venv ${PYTHON}"
echo ""

python app.py "$@"

deactivate
