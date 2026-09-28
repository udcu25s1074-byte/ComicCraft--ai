import os
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageOps

from app.ai.gemini_flash import generate_outline, generate_story


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "static" / "generated"
PANEL_COUNT = 6


def _local_story(
    story_prompt: str,
    character_name: str,
    setting: str,
) -> dict[str, Any]:
    hero = character_name.strip() or "Alex"
    place = setting.strip() or "a hidden city"
    premise = story_prompt.strip().rstrip(".")
    beats = [
        (f"A strange clue appears in {place}.", f"{hero}: This has to mean something."),
        (f"{hero} follows the clue and discovers the trouble: {premise}.", "The whole place is counting on me."),
        (f"A small ally arrives with one useful idea.", "Ally: Try looking at it another way."),
        (f"The first plan goes sideways, right when it matters most.", f"{hero}: Okay. New plan."),
        (f"{hero} combines courage and the ally's idea to turn things around.", "Ally: You figured it out!"),
        (f"By sunset, {place} is safe and a new adventure is already calling.", f"{hero}: Let's see where this goes."),
    ]
    return {
        "title": f"{hero} and the {place.title()} Mystery",
        "panels": [
            {"caption": caption, "dialogue": dialogue, "scene": caption}
            for caption, dialogue in beats
        ],
    }


def _hex_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _local_illustration(path: Path, panel: dict[str, Any], index: int, style: str) -> None:
    width, height = 960, 600
    palettes = {
        "manga": ("#e4e5df", "#8fa5a1", "#151e25", "#ee6b55"),
        "watercolor": ("#a9d7d0", "#e4d1a4", "#255b58", "#e47762"),
        "retro": ("#e7bb62", "#e78358", "#34585c", "#db4f40"),
        "graphic novel": ("#8eb8bd", "#d79b70", "#193a42", "#ec684e"),
        "storybook": ("#91c6b7", "#d9c58a", "#345548", "#ec765e"),
    }
    key = next((name for name in palettes if name in style.lower()), "graphic novel")
    sky, horizon, dark, accent = (_hex_color(color) for color in palettes[key])
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        fraction = y / height
        color = tuple(int(sky[c] * (1 - fraction) + horizon[c] * fraction) for c in range(3))
        for x in range(width):
            pixels[x, y] = color

    draw = ImageDraw.Draw(image)
    seed = index * 127 + sum(ord(char) for char in panel.get("scene", ""))
    rng = random.Random(seed)
    sun_x = 740 if index % 2 else 220
    draw.ellipse((sun_x - 85, 80, sun_x + 85, 250), fill=(245, 211, 123))
    draw.polygon([(0, 410), (180, 235), (350, 415), (520, 210), (760, 420), (900, 270), (960, 340), (960, 600), (0, 600)], fill=dark)
    for building in range(11):
        x = building * 100 - 20
        top = rng.randint(300, 425)
        draw.rectangle((x, top, x + rng.randint(48, 86), 530), fill=dark)
        for wx in range(x + 10, x + 70, 23):
            for wy in range(top + 18, 510, 30):
                if rng.random() > 0.3:
                    draw.rectangle((wx, wy, wx + 7, wy + 11), fill=(231, 197, 116))
    draw.polygon([(0, 515), (220, 470), (430, 535), (700, 465), (960, 510), (960, 600), (0, 600)], fill=accent)

    hero_x = 478 + (index % 3 - 1) * 75
    hero_y = 350 + (index % 2) * 15
    draw.ellipse((hero_x - 61, hero_y - 138, hero_x + 61, hero_y - 16), fill=(244, 204, 156), outline=dark, width=8)
    draw.polygon([(hero_x - 61, hero_y - 24), (hero_x + 61, hero_y - 24), (hero_x + 103, 555), (hero_x - 97, 555)], fill=dark)
    draw.polygon([(hero_x - 54, hero_y + 8), (hero_x - 180, 510), (hero_x - 118, 530), (hero_x - 20, hero_y + 88)], fill=accent)
    draw.line((hero_x - 14, hero_y - 55, hero_x + 38, hero_y - 46), fill=dark, width=8)
    draw.ellipse((hero_x + 21, hero_y - 70, hero_x + 34, hero_y - 57), fill=dark)
    draw.line((hero_x + 82, hero_y + 50, hero_x + 174, hero_y - 20), fill=(244, 204, 156), width=20)

    if index in (1, 3, 4):
        points = [(735, 240), (755, 282), (800, 287), (766, 317), (776, 362), (735, 338), (696, 362), (704, 317), (670, 287), (715, 282)]
        draw.polygon(points, fill=(245, 211, 123), outline=dark)
    draw.rectangle((10, 10, width - 11, height - 11), outline=(248, 241, 222), width=8)
    image.save(path, format="PNG", optimize=True)


def _huggingface_illustration(path: Path, prompt: str) -> None:
    token = os.getenv("HF_API_KEY", "").strip()
    if not token:
        raise RuntimeError("HF_API_KEY is not configured.")
    from huggingface_hub import InferenceClient

    model = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.1-schnell").strip()
    generated = InferenceClient(api_key=token, timeout=60).text_to_image(
        prompt=f"{prompt}, comic book illustration, no text, no lettering, no watermark",
        model=model,
    )
    ImageOps.fit(
        generated.convert("RGB"),
        (960, 600),
        method=Image.Resampling.LANCZOS,
    ).save(path, "PNG", optimize=True)


def _art_fallback_notice(error: Exception) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in (401, 403):
        return "Hugging Face token lacks Inference Providers permission; local artwork was used. Update the token permission to enable AI artwork."
    if status_code == 404:
        return "The configured Hugging Face image model is unavailable; local artwork was used. Check IMAGE_MODEL_ID."
    if status_code == 429:
        return "Hugging Face rate limit reached; local artwork was used."
    detail = str(error)
    if "Inference Providers permission" in detail:
        return "Hugging Face token lacks Inference Providers permission; local artwork was used. Update the token permission to enable AI artwork."
    if "image model is unavailable" in detail:
        return "The configured Hugging Face image model is unavailable; local artwork was used. Check IMAGE_MODEL_ID."
    if "quota or rate limit" in detail:
        return "Hugging Face rate limit reached; local artwork was used."
    return "Hugging Face image generation failed; local artwork was used. Check the token, model, and network connection."


def _story_fallback_notice(error: Exception) -> str:
    status_code = getattr(error, "code", None)
    if status_code in (429, 503):
        return "Gemini is temporarily at capacity; the local story engine was used. Try again later for AI story generation."
    if status_code == 404:
        return "No configured Gemini model was available; the local story engine was used. Check the model settings."
    if status_code in (401, 403):
        return "Gemini rejected the API key or model access; the local story engine was used. Check GEMINI_API_KEY permissions."
    return "Gemini could not generate the story; the local story engine was used. Check the key and model settings."


def _make_pdf(title: str, panels: list[dict[str, Any]], filename: str) -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_text_color(24, 43, 42)
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 12, _pdf_text(title[:90]))
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(75, 99, 93)
    pdf.cell(0, 8, "A six-panel story made with ComicCraft", new_x="LMARGIN", new_y="NEXT")

    panel_width = (pdf.w - 32) / 2
    image_height = panel_width * 0.625
    for index, panel in enumerate(panels):
        column = index % 2
        row = index // 2
        x = 12 + column * (panel_width + 8)
        y = 38 + row * 78
        pdf.set_draw_color(24, 43, 42)
        pdf.rect(x, y, panel_width, 69)
        pdf.image(str(panel["image_path"]), x=x + 1, y=y + 1, w=panel_width - 2, h=image_height)
        pdf.set_xy(x + 3, y + image_height + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(24, 43, 42)
        pdf.multi_cell(panel_width - 6, 4, _pdf_text(str(panel.get("caption", ""))[:115]))
        pdf.set_x(x + 3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(75, 99, 93)
        pdf.multi_cell(panel_width - 6, 4, _pdf_text(str(panel.get("dialogue", ""))[:105]))
    pdf.output(filename)


def _pdf_text(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').encode("latin-1", "replace").decode("latin-1")


def create_comic(
    story_prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
    use_ai_art: bool = False,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    story_source = "Local story engine"
    generation_note = ""
    if os.getenv("GEMINI_API_KEY", "").strip():
        try:
            outline = generate_outline(story_prompt, character_name, setting, tone, art_style)
            script = generate_story(outline, character_name, tone)
            raw_panels = [
                {**outline["panels"][index], **script["panels"][index]}
                for index in range(PANEL_COUNT)
            ]
            title = script["title"]
            story_source = "Gemini"
        except Exception as error:
            script = _local_story(story_prompt, character_name, setting)
            raw_panels = script["panels"]
            title = script["title"]
            generation_note = _story_fallback_notice(error)
    else:
        script = _local_story(story_prompt, character_name, setting)
        raw_panels = script["panels"]
        title = script["title"]

    comic_id = uuid.uuid4().hex[:12]
    warnings: list[str] = []
    remote_disabled = Event()
    warning_lock = Lock()

    def create_panel_art(item: tuple[int, dict[str, Any]]) -> tuple[int, Path, bool]:
        index, panel = item
        path = OUTPUT_DIR / f"{comic_id}-panel-{index + 1}.png"
        remote_used = False
        if use_ai_art and not remote_disabled.is_set():
            try:
                _huggingface_illustration(path, str(panel.get("image_prompt", panel.get("scene", ""))))
                remote_used = True
            except Exception as error:
                with warning_lock:
                    if not remote_disabled.is_set():
                        remote_disabled.set()
                        warnings.append(_art_fallback_notice(error))
        if not remote_used:
            _local_illustration(path, panel, index, art_style)
        return index, path, remote_used

    with ThreadPoolExecutor(max_workers=3) as pool:
        art_results = list(pool.map(create_panel_art, enumerate(raw_panels)))

    panels = []
    for index, path, remote_used in sorted(art_results):
        raw = raw_panels[index]
        panels.append({
            "number": index + 1,
            "caption": str(raw.get("caption", raw.get("scene", "")))[:180],
            "dialogue": str(raw.get("dialogue", ""))[:140],
            "image_url": f"/static/generated/{path.name}",
            "image_path": path,
            "ai_art": remote_used,
        })

    pdf_filename = f"{comic_id}.pdf"
    _make_pdf(title, panels, str(OUTPUT_DIR / pdf_filename))
    notices = [note for note in (generation_note, *warnings) if note]
    return {
        "id": comic_id,
        "title": title,
        "panels": panels,
        "pdf_filename": pdf_filename,
        "story_source": story_source,
        "art_source": "Hugging Face + local fallback" if use_ai_art else "Local illustration engine",
        "notices": notices,
    }