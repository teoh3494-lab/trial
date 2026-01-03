from __future__ import annotations

from typing import Iterable

from google import genai


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_prompt(context: Iterable[dict], style: str, language: str, count: int) -> str:
    context_lines = []
    for item in context:
        context_lines.append(
            f"- {item['title']} | {item['channel']} | {item['views']} views | {item['duration']}s"
        )
    context_block = "\n".join(context_lines) if context_lines else "(no context)"
    return f"""
You are a YouTube trends idea generator. Use this context of top videos:
{context_block}

Generate {count} ideas in {language} with style: {style}.
For each idea, return JSON with keys:
- hook (<= 8 words)
- premise (1 sentence)
- cta (yes/no or comment)
- titles (3 alternatives)
- thumbnail (1 sentence)

Return a JSON array only.
"""


def generate_ideas(
    api_key: str,
    context: Iterable[dict],
    style: str,
    language: str,
    count: int,
) -> str:
    client = build_client(api_key)
    prompt = build_prompt(context, style, language, count)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    return response.text or ""
