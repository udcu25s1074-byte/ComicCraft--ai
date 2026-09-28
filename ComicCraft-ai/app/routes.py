import logging
import os
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.comic import OUTPUT_DIR, create_comic


logger = logging.getLogger(__name__)
router = APIRouter()


class ComicRequest(BaseModel):
    story_prompt: str = Field(min_length=8, max_length=1000)
    character_name: str = Field(min_length=1, max_length=60)
    setting: str = Field(min_length=1, max_length=100)
    tone: str = Field(min_length=1, max_length=40)
    art_style: str = Field(min_length=1, max_length=60)
    use_ai_art: bool = False


def _form_values(
    story_prompt: str = "",
    character_name: str = "",
    setting: str = "",
    tone: str = "adventure",
    art_style: str = "graphic novel",
) -> dict[str, str]:
    return {
        "story_prompt": story_prompt,
        "character_name": character_name,
        "setting": setting,
        "tone": tone,
        "art_style": art_style,
    }


@router.get("/")
def home(request: Request):
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "values": _form_values(),
            "hf_enabled": bool(os.getenv("HF_API_KEY", "").strip()),
            "error": None,
        },
    )


@router.get("/health")
def health():
    return {"status": "ok", "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip())}


@router.post("/generate")
def generate(
    request: Request,
    story_prompt: str = Form(..., min_length=8, max_length=1000),
    character_name: str = Form(..., min_length=1, max_length=60),
    setting: str = Form(..., min_length=1, max_length=100),
    tone: str = Form(..., min_length=1, max_length=40),
    art_style: str = Form(..., min_length=1, max_length=60),
    use_ai_art: bool = Form(False),
):
    values = _form_values(story_prompt, character_name, setting, tone, art_style)
    try:
        comic = create_comic(**values, use_ai_art=use_ai_art)
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="comic_preview.html",
            context={"request": request, "comic": comic},
        )
    except Exception:
        logger.exception("Comic generation failed")
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "values": values,
                "hf_enabled": bool(os.getenv("HF_API_KEY", "").strip()),
                "error": "The comic could not be completed. Check the server log and try again.",
            },
            status_code=500,
        )


@router.post("/api/comics")
def generate_json(payload: ComicRequest):
    try:
        comic = create_comic(**payload.model_dump())
        return JSONResponse({
            "success": True,
            "id": comic["id"],
            "title": comic["title"],
            "panels": [
                {key: value for key, value in panel.items() if key != "image_path"}
                for panel in comic["panels"]
            ],
            "pdf_url": f"/download/{comic['pdf_filename']}",
            "story_source": comic["story_source"],
            "art_source": comic["art_source"],
            "notices": comic["notices"],
        })
    except Exception as error:
        logger.exception("JSON comic generation failed")
        raise HTTPException(status_code=500, detail="Comic generation failed.") from error


@router.get("/download/{filename}")
def download(filename: str):
    if Path(filename).name != filename or not filename.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Comic not found.")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Comic not found.")
    return FileResponse(path, filename=filename, media_type="application/pdf")