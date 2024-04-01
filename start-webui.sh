#!/bin/bash

echo "activate venv"
source venv/bin/activate

PYTHON="venv/bin/python"
echo "venv ${PYTHON}"
echo ""

python app.py $*

echo "launching the app"

deactivate
