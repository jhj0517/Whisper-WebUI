@echo off

call venv\scripts\activate
python app.py  --whisper_type whisper

echo "launching the app"
pause
