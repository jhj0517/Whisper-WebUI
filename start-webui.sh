#!/bin/bash

PYTHON="./venv/bin/python"
echo "venv ${PYTHON}"
echo ""

$PYTHON ./app.py

echo "launching the app"

deactivate
