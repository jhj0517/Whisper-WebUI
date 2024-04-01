#!/bin/bash

source venv/bin/activate

PYTHON="venv/bin/python"
echo "venv ${PYTHON}"
echo ""

$PYTHON app.py $*

deactivate
