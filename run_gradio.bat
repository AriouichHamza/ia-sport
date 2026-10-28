@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
python ui/app_gradio.py
pause
