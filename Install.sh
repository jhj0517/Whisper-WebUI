#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

python -m pip install -U pip wheel
pip install --no-build-isolation -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    deactivate
    exit 1
}

deactivate
