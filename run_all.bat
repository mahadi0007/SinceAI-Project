@echo off
rem One command: environment -> data check -> pipeline -> viewer. Options: see run_all.py
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 run_all.py %*) else (python run_all.py %*)
