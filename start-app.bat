@echo off
rem Studiepodcast - start the web app. Double-click this file.
rem It checks that the OpenCode server (your writing model) is reachable,
rem starts the web app on http://127.0.0.1:8000 and opens your browser.
setlocal
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8000"
set "STUDIE=%~dp0.venv\Scripts\studiepodcast.exe"

if not exist "%STUDIE%" (
    echo [studiepodcast] No virtualenv found at .venv. Create it first:
    echo     uv venv --python 3.12 .venv
    echo     uv pip install -e ".[dev,cuda]" --no-build-isolation-package pkuseg
    echo.
    pause
    exit /b 1
)

echo [studiepodcast] Checking the OpenCode server (the model that writes the scripts)...
"%~dp0.venv\Scripts\python.exe" -c "from pipeline.config import Settings; from pipeline.opencode_llm import OpenCodeLLM; ok, msg = OpenCodeLLM(Settings.from_env()).check(); print('  ok: ' + msg if ok else '  FAIL: ' + msg); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo The OpenCode server is not reachable, so the pipeline cannot write any text.
    echo Start the OpenCode desktop app, or run "opencode serve" in a terminal,
    echo and start this file again.
    echo.
    pause
    exit /b 1
)

echo [studiepodcast] Web app: http://%HOST%:%PORT%  -  close this window to stop it.
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://%HOST%:%PORT%"
"%STUDIE%" serve --host %HOST% --port %PORT%
pause
