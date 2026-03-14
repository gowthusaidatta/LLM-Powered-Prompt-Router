"""
Routes classified intent to the correct expert system prompt
and generates a response using Groq llama-3.3-70b-versatile.
"""
import re
from groq import Groq
from src.config import config
from src.prompts import PROMPTS

_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None


def _sanitize_user_message(message: str) -> str:
    stripped = message.strip()
    return re.sub(r"^@\w+\s+", "", stripped).strip() or stripped


def _offline_response(label: str, message: str) -> str:
    if label == "code":
        return (
            "I could not reach the LLM provider, so here is a safe offline response.\n"
            "For your coding request, share your current code and expected output, and I will help fix it step-by-step."
        )
    if label == "data":
        return (
            "I could not reach the LLM provider, so here is a safe offline response.\n"
            "For data analysis, please share sample rows or summary stats. I can help with averages, distributions, and chart suggestions."
        )
    if label == "writing":
        return (
            "I could not reach the LLM provider, so here is a safe offline response.\n"
            "Please paste your text and I will give targeted feedback on clarity, tone, and structure without rewriting it for you."
        )
    if label == "career":
        return (
            "I could not reach the LLM provider, so here is a safe offline response.\n"
            "What role are you targeting and what is your current experience level? I can then suggest concrete next steps."
        )
    return "Are you asking for help with coding, data analysis, writing improvement, or career advice?"


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

    if _client is None:
        return _offline_response(label, clean)

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
        final_text = (response.choices[0].message.content or "").strip()
        if final_text:
            return final_text
        return "Could you clarify if you need help with coding, data analysis, writing, or career advice?"
    except Exception as exc:
        if config.OFFLINE_FALLBACK:
            return _offline_response(label, clean)
        raise RuntimeError(f"Response generation failed for intent '{label}': {exc}") from exc
