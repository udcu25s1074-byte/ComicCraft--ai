import json
import os
import re
from typing import Any


def _generate_json(
	model: str,
	prompt: str,
	fallback_models: tuple[str, ...] = (),
) -> dict[str, Any]:
	from google import genai
	from google.genai import types

	client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
	models = tuple(dict.fromkeys((model, *fallback_models)))
	for index, candidate in enumerate(models):
		try:
			response = client.models.generate_content(
				model=candidate,
				contents=prompt,
				config=types.GenerateContentConfig(
					response_mime_type="application/json",
					temperature=0.85,
				),
			)
			break
		except Exception as error:
			if getattr(error, "code", None) not in {404, 429, 500, 502, 503, 504} or index == len(models) - 1:
				raise
	text = response.text or ""
	text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text)
	result = json.loads(text)
	if not isinstance(result, dict):
		raise ValueError("The story model returned an unexpected response.")
	return result


def generate_outline(
	story_prompt: str,
	character_name: str,
	setting: str,
	tone: str,
	art_style: str,
) -> dict[str, Any]:
	prompt = f"""Create a concise original six-panel comic outline as JSON.
Return exactly this shape: {{"title": "...", "panels": [{{"scene": "...", "image_prompt": "..."}}]}}.
Make every image_prompt a specific visual description with consistent characters and the requested art style. Do not ask for lettering inside the art. Keep the story self-contained and suitable for a general audience.
Premise: {story_prompt}
Lead character: {character_name}
Setting: {setting}
Tone: {tone}
Art style: {art_style}"""
	result = _generate_json(
		os.getenv("GEMINI_FLASH_MODEL", "gemini-3.1-flash-lite"),
		prompt,
		fallback_models=("gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"),
	)
	panels = result.get("panels")
	if not isinstance(panels, list) or len(panels) < 6:
		raise ValueError("The outline model did not return six panels.")
	return {"title": str(result.get("title", "Untitled Comic")), "panels": panels[:6]}


def generate_story(
	outline: dict[str, Any],
	character_name: str,
	tone: str,
) -> dict[str, Any]:
	prompt = f"""Turn this outline into a polished six-panel comic script as JSON.
Return exactly this shape: {{"title": "...", "panels": [{{"caption": "...", "dialogue": "..."}}]}}.
Use short captions and one short, natural line of dialogue per panel. Keep character names and events consistent. Tone: {tone}. Lead character: {character_name}.
Outline: {json.dumps(outline, ensure_ascii=True)}"""
	result = _generate_json(
		os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-flash-lite"),
		prompt,
		fallback_models=(
			os.getenv("GEMINI_FLASH_MODEL", "gemini-3.1-flash-lite"),
			"gemini-3.1-flash-lite",
			"gemini-3.6-flash",
			"gemini-3.5-flash",
		),
	)
	panels = result.get("panels")
	if not isinstance(panels, list) or len(panels) < 6:
		raise ValueError("The script model did not return six panels.")
	return {"title": str(result.get("title", outline["title"])), "panels": panels[:6]}
