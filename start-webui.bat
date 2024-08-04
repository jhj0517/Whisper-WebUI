@echo off

call venv\scripts\activate
python app.py %*

echo "launching the app"
pause
