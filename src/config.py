"""
Centralized application configuration loaded from environment variables.
Validated at import time so startup fails fast with a clear message.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your values."
        )
    return value


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default)


def _float_in_range(key: str, default: str, min_value: float, max_value: float) -> float:
    raw = _optional(key, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise EnvironmentError(f"Environment variable '{key}' must be a float.") from exc
    if value < min_value or value > max_value:
        raise EnvironmentError(
            f"Environment variable '{key}' must be between {min_value} and {max_value}."
        )
    return value


class Config:
    GROQ_API_KEY: str = _require("GROQ_API_KEY")
    CLASSIFIER_MODEL: str = _optional("CLASSIFIER_MODEL", "llama-3.1-8b-instant")
    RESPONSE_MODEL: str = _optional("RESPONSE_MODEL", "llama-3.3-70b-versatile")
    CONFIDENCE_THRESHOLD: float = _float_in_range("CONFIDENCE_THRESHOLD", "0.7", 0.0, 1.0)
    APP_PORT: int = int(_optional("APP_PORT", "8000"))
    LOG_FILE: str = "route_log.jsonl"
    PROMPTS_FILE: str = "prompts/prompts.json"
    VALID_INTENTS: tuple = ("code", "data", "writing", "career", "unclear")


config = Config()
