#!/bin/bash
source "./venv/Scripts/activate.bat"

PYTHON="./venv/Scripts/python.exe"
echo "venv ${PYTHON}"
echo ""

echo "Executing app.py..."
"${PYTHON}" app.py

deactivate
read -p "Press enter to continue"