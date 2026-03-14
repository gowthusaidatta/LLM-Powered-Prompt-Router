"""
Appends one JSON Lines entry per request to route_log.jsonl.
Log failures print to stderr but never interrupt application flow.
"""
import json
import sys
from datetime import datetime, timezone
from src.config import config


def log_route(user_message: str, intent: str, confidence: float, final_response: str) -> None:
    """Append a routing decision to the log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "confidence": round(confidence, 4),
        "user_message": user_message,
        "final_response": final_response,
    }
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[WARNING] Failed to write log: {exc}", file=sys.stderr)
