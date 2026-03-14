# LLM-Powered Prompt Router for Intent Classification

A Python service that classifies user intent and routes requests to specialized AI personas. Built with FastAPI and Groq.

---

## The Idea

Instead of one large generic prompt, this project uses a two-step flow:

1. A lightweight model classifies the user message into one of: `code`, `data`, `writing`, `career`, `unclear`.
2. A larger model generates the final answer using a persona prompt matched to the detected intent.

This improves response quality, cost-efficiency, and control.

---

## How Routing Works

```text
User message
   |
   v
Optional manual override (@code, @data, @writing, @career)
   |
   v
Intent classification (small model)
   |
   v
Confidence threshold check (default: 0.7)
   |
   v
Persona prompt selection from prompts/prompts.json
   |
   v
Final response generation (larger model)
   |
   v
Append route + output to route_log.jsonl
```

---

## Project Layout

```text
llm-prompt-router/
|- src/
|  |- __init__.py
|  |- config.py
|  |- prompts.py
|  |- classifier.py
|  |- router.py
|  |- logger.py
|  |- main.py
|- prompts/
|  |- prompts.json
|- tests/
|  |- __init__.py
|  |- run_tests.py
|- route_log.jsonl
|- .env.example
|- Dockerfile
|- docker-compose.yml
|- .dockerignore
|- .gitignore
|- requirements.txt
`- README.md
```

---

## Expert Prompts

Prompts are configured in `prompts/prompts.json` and validated at startup.

| Intent | Persona | Core Behavior |
|---|---|---|
| code | Senior programmer | Gives technical, implementation-focused help |
| data | Data analyst | Uses statistical framing and visualization suggestions |
| writing | Writing coach | Gives feedback without rewriting for the user |
| career | Career advisor | Gives concrete, actionable career guidance |
| unclear | Intake assistant | Asks for clarification instead of guessing |

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| GROQ_API_KEY | Yes | - | Groq API key |
| CLASSIFIER_MODEL | No | llama-3.1-8b-instant | Classification model |
| RESPONSE_MODEL | No | llama-3.3-70b-versatile | Response model |
| CONFIDENCE_THRESHOLD | No | 0.7 | Low-confidence fallback to `unclear` |
| APP_PORT | No | 8000 | FastAPI server port |
| LOG_FILE | No | route_log.jsonl | Route log output file |
| PROMPTS_FILE | No | prompts/prompts.json | Prompt configuration source |
| MAX_CLASSIFIER_CHARS | No | 4000 | Max message size sent to classifier |
| OFFLINE_FALLBACK | No | 1 | Allows local fallback when API key is missing/unavailable |

---

## Local Setup

Requirements: Python 3.11+ and a Groq API key.

```bash
git clone https://github.com/gowthusaidatta/LLM-Powered-Prompt-Router.git
cd LLM-Powered-Prompt-Router
cp .env.example .env
# Set GROQ_API_KEY in .env
pip install -r requirements.txt
python -m src.main
```

On Windows PowerShell (if `python` is unavailable), use:

```powershell
py -m src.main
```

Open `http://localhost:8000`.

For CLI mode:

```bash
python -m src.main --cli
```

---

## Docker

With compose:

```bash
docker compose up --build
```

The compose setup includes practical hardening defaults (`init`, `no-new-privileges`, read-only root filesystem with `/tmp` tmpfs) and JSON-file log rotation.

Build and run directly:

```bash
docker build -t llm-prompt-router:latest .
docker run --env-file .env -p 8000:8000 llm-prompt-router:latest
```

---

## API

| Method | Path | Description |
|---|---|---|
| GET | / | Web UI |
| POST | /api/chat | Classify + route + respond |
| GET | /api/logs | Recent log entries |
| GET | /health | Health status |

Note: `/api/logs?limit=` is clamped between 1 and 200 to prevent invalid or excessive payload requests.

Example request:

```json
{ "message": "how do i sort a list of objects in python?" }
```

Example response:

```json
{
  "intent": "code",
  "confidence": 0.96,
  "response": "..."
}
```

---

## Running Tests

```bash
python -m tests.run_tests
```

Windows PowerShell alternative:

```powershell
py -m tests.run_tests
```

The test runner executes required sample inputs and logs each run.

---

## Route Log Format

Each request appends one JSON object per line to `route_log.jsonl`.

```json
{
  "timestamp": "2026-03-13T11:17:11.110845+00:00",
  "intent": "code",
  "confidence": 1.0,
  "user_message": "how do i sort a list of objects in python?",
  "final_response": "..."
}
```

Required keys: `intent`, `confidence`, `user_message`, `final_response`.

---

## Troubleshooting

- Error: `Required environment variable 'GROQ_API_KEY' is not set`
   - Fix: set `GROQ_API_KEY` in `.env` (or system environment), or keep `OFFLINE_FALLBACK=1` for local fallback mode.
- Error: `python is not recognized`
  - Fix: use `py` instead of `python` on Windows.

---

## License

MIT
