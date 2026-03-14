"""
Intent classification using Groq llama-3.1-8b-instant.
Returns structured intent dict. Never raises — always falls back to unclear.
"""
import json
import re
from groq import Groq
from src.config import config

_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

_SYSTEM = """You are a strict intent classification engine for a prompt routing system.

The five intents and their EXACT meanings:

code — User wants help writing, fixing, debugging, explaining, or reviewing CODE or programming concepts. Includes SQL queries, algorithms, functions, bugs, any software task.

data — User is asking about a DATASET or wants statistical analysis: averages, distributions, correlations, pivot tables, chart recommendations. Numbers or data must be present or clearly implied.

writing — User wants feedback on TEXT THEY HAVE ALREADY WRITTEN. They must have existing prose to be critiqued. Does NOT include creating new content.

career — User wants professional life advice: jobs, interviews, resumes, cover letters, promotions, career decisions, workplace situations.

unclear — Use for: greetings, gibberish, requests to CREATE new writing (poems, stories, essays from scratch), arithmetic with no dataset context, or anything not fitting the above.

CRITICAL EXAMPLES — memorize these:
"Can you write me a poem about clouds?" -> unclear (creating new content)
"explain this sql query for me" -> code (SQL is programming)
"what is a pivot table" -> data (data analysis concept)
"This paragraph sounds awkward, can you help?" -> writing (has existing text)
"My boss says my writing is too verbose" -> writing (feedback on their existing writing)
"How do I structure a cover letter?" -> career (professional advice)
"I'm preparing for a job interview, any tips?" -> career
"I need to write a function that takes a user id" -> code (programming task)
"Rewrite this sentence to be more professional" -> writing (existing text)
"hey" -> unclear
"asdfjkl;" -> unclear
"what is 2 + 2" -> unclear (no dataset)
"Help me make this better" -> unclear (too vague, no context)

Return ONLY this JSON, no other text:
{"intent": "<label>", "confidence": <float 0.0-1.0>}"""

_SAFE: dict = {"intent": "unclear", "confidence": 0.0}


def _safe() -> dict:
    return dict(_SAFE)


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()
    return cleaned


def _detect_override(message: str) -> dict | None:
    """Check for manual @intent override prefix.

    Args:
        message: Raw user message.

    Returns:
        Dict with intent and confidence 1.0 if valid override, else None.
    """
    match = re.match(r"^@(\w+)\s", message.strip())
    if match:
        label = match.group(1).lower()
        if label in config.VALID_INTENTS and label != "unclear":
            return {"intent": label, "confidence": 1.0}
    return None


def _parse(raw: str) -> dict:
    """Parse LLM response into validated intent dict. Never raises."""
    try:
        cleaned = _strip_markdown_fence(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return _safe()
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            return _safe()
        intent = str(parsed.get("intent", "")).lower().strip()
        confidence = parsed.get("confidence", 0.0)
        if intent not in config.VALID_INTENTS:
            return _safe()
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        return {"intent": intent, "confidence": round(max(0.0, min(1.0, float(confidence))), 4)}
    except Exception:
        return _safe()


def _classify_offline(message: str) -> dict:
    """Heuristic fallback used when LLM is unavailable."""
    text = message.lower().strip()
    if not text:
        return _safe()

    if text in {"hi", "hey", "hello", "yo", "sup"} or len(text) < 3:
        return {"intent": "unclear", "confidence": 0.95}

    score = {"code": 0, "data": 0, "writing": 0, "career": 0}

    code_hits = ["python", "sql", "bug", "function", "api", "code", "debug", "query", "loop"]
    data_hits = ["average", "mean", "median", "dataset", "data", "pivot", "distribution", "chart", "bimodal"]
    writing_hits = ["paragraph", "sentence", "rewrite", "writing", "verbose", "tone", "clarity", "awkward"]
    career_hits = ["resume", "career", "interview", "cover letter", "job", "promotion", "salary"]

    for token in code_hits:
        if token in text:
            score["code"] += 1
    for token in data_hits:
        if token in text:
            score["data"] += 1
    for token in writing_hits:
        if token in text:
            score["writing"] += 1
    for token in career_hits:
        if token in text:
            score["career"] += 1

    best_intent = max(score, key=score.get)
    best_score = score[best_intent]
    if best_score == 0:
        return {"intent": "unclear", "confidence": 0.0}

    confidence = min(0.95, 0.65 + best_score * 0.08)
    result = {"intent": best_intent, "confidence": round(confidence, 4)}
    if result["confidence"] < config.CONFIDENCE_THRESHOLD:
        return {"intent": "unclear", "confidence": result["confidence"]}
    return result


def classify_intent(message: str) -> dict:
    """Classify the intent of a user message.

    Checks for manual @intent override prefix first. Otherwise calls the
    Groq classifier LLM and parses the structured JSON response.

    Never raises. Any failure defaults to unclear with confidence 0.0.

    Args:
        message: The raw user message string.

    Returns:
        A dict: {"intent": str, "confidence": float}
    """
    if not message or not message.strip():
        return _safe()

    override = _detect_override(message)
    if override:
        return override

    clipped_message = message.strip()[: config.MAX_CLASSIFIER_CHARS]

    if _client is None:
        return _classify_offline(clipped_message)

    try:
        response = _client.chat.completions.create(
            model=config.CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Classify this message:\n\n{clipped_message}"},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content or ""
        result = _parse(raw)
        if result["confidence"] < config.CONFIDENCE_THRESHOLD:
            return {"intent": "unclear", "confidence": result["confidence"]}
        return result
    except Exception:
        if config.OFFLINE_FALLBACK:
            return _classify_offline(clipped_message)
        return _safe()
