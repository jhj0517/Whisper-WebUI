#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "Requirements installation failed. please remove venv folder and run install.sh again."
    deactivate
    exit 1
fi

echo ""
echo "Requirements installed successfully."

deactivate