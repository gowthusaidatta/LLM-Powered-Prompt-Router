"""
Routes classified intent to the correct expert system prompt
and generates a response using Groq llama-3.3-70b-versatile.
"""
import re
from groq import Groq
from src.config import config
from src.prompts import PROMPTS

_client = Groq(api_key=config.GROQ_API_KEY)


def _sanitize_user_message(message: str) -> str:
    stripped = message.strip()
    return re.sub(r"^@\w+\s+", "", stripped).strip() or stripped


def route_and_respond(message: str, intent: dict) -> str:
    """Select expert prompt and generate final response.

    Args:
        message: Original user message.
        intent: Dict with 'intent' and 'confidence' keys.

    Returns:
        Generated response string.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    label = intent.get("intent", "unclear")
    if label not in PROMPTS:
        label = "unclear"

    system_prompt = PROMPTS[label]
    clean = _sanitize_user_message(message)

    try:
        response = _client.chat.completions.create(
            model=config.RESPONSE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Response generation failed for intent '{label}': {exc}") from exc
