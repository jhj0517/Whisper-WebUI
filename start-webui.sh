#!/bin/bash

source venv/bin/activate
echo "activate venv"

PYTHON="venv/bin/python"
echo "venv ${PYTHON}"
echo ""

python app.py

echo "launching the app"

deactivate
