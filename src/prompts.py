"""
Loads and validates expert system prompts from the JSON configuration file.
"""
import json
import os
from src.config import config


def load_prompts() -> dict:
    if not os.path.exists(config.PROMPTS_FILE):
        raise FileNotFoundError(f"Prompts file not found at '{config.PROMPTS_FILE}'.")
    with open(config.PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    missing = set(config.VALID_INTENTS) - set(prompts.keys())
    if missing:
        raise ValueError(f"Prompts file missing required keys: {missing}")
    return prompts


PROMPTS: dict = load_prompts()
