@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 (
        echo Could not create .venv. Install Python 3.10 or newer and try again.
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, jinja2, dotenv, google.genai, huggingface_hub, PIL, fpdf" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m uvicorn app.main:app --reload