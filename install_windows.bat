@echo off
cd /d "%~dp0"
py -3.10 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-run.txt
pause
