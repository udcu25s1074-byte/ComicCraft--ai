# ComicCraft AI

ComicCraft turns a premise into a six-panel comic, illustrated preview, and downloadable PDF. It runs locally on Windows and does not require PowerShell script activation.

## Start on Windows

Double-click `run.bat`. The first run creates `.venv`, installs the project dependencies, and starts the app. Open <http://127.0.0.1:8000> in your browser. Stop the server with `Ctrl+C`.

You can also run it explicitly from PowerShell without activating the environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

This direct-Python workflow avoids the execution-policy error from `Activate.ps1`.

## AI providers

The app works without API keys using its local story and illustration engines. To enable Gemini story generation, set `GEMINI_API_KEY` in `.env`. The `GEMINI_FLASH_MODEL` and `GEMINI_PRO_MODEL` settings are optional; the Flash default is `gemini-3.1-flash-lite`, with automatic Flash-model fallback if a configured model is retired or temporarily unavailable.

To enable optional generated panel artwork, set `HF_API_KEY` in `.env` with Inference Providers permission, then select **Generate AI artwork** in the form. The default `IMAGE_MODEL_ID` is `black-forest-labs/FLUX.1-schnell`; the official Hugging Face client automatically selects a supported provider. Failed image requests fall back to local illustrations.

Generated images and PDFs are saved under `static/generated/`, which is excluded from version control.

## Endpoints

- `GET /` opens the comic studio.
- `POST /generate` creates a comic from the web form.
- `POST /api/comics` accepts the same fields as JSON.
- `GET /download/{filename}` downloads a generated PDF.
- `GET /health` reports app readiness and Gemini configuration.