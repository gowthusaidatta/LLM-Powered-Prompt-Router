"""
FastAPI application and CLI entry point for the LLM Prompt Router.

Endpoints:
  GET  /         Serves the single-page web interface.
  POST /api/chat Classifies intent and returns expert response.
  GET  /api/logs Returns recent log entries.
  GET  /health   Health check for container probes.
"""
import argparse
import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
from src.classifier import classify_intent
from src.config import config
from src.logger import log_route
from src.router import route_and_respond

app = FastAPI(
    title="LLM Prompt Router",
    description="Intent-based routing to specialized AI expert personas using Groq.",
    version="1.0.0",
)


def _sanitize_limit(limit: int) -> int:
    if limit < 1:
        return 1
    if limit > 200:
        return 200
    return limit


class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message must not be empty.")
        return v.strip()


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>LLM Prompt Router</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f13;color:#e8e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem 1rem}
header{text-align:center;margin-bottom:2rem}
header h1{font-size:1.8rem;font-weight:700;color:#fff}
header p{margin-top:.4rem;font-size:.9rem;color:#888}
.chat-container{width:100%;max-width:760px;display:flex;flex-direction:column;gap:1rem}
.messages{display:flex;flex-direction:column;gap:1rem;min-height:300px;max-height:520px;overflow-y:auto;padding:1rem;background:#16161d;border-radius:12px;border:1px solid #2a2a38}
.message{display:flex;flex-direction:column;gap:.25rem}
.message.user .bubble{align-self:flex-end;background:#2563eb;color:#fff;border-radius:12px 12px 2px 12px;padding:.7rem 1rem;max-width:80%;font-size:.95rem;line-height:1.5}
.message.assistant .bubble{align-self:flex-start;background:#1e1e2e;border:1px solid #2a2a38;border-radius:12px 12px 12px 2px;padding:.7rem 1rem;max-width:90%;font-size:.95rem;line-height:1.6;white-space:pre-wrap}
.intent-badge{align-self:flex-start;font-size:.72rem;padding:.2rem .55rem;border-radius:999px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-top:.25rem}
.badge-code{background:#1e3a5f;color:#60a5fa}
.badge-data{background:#1e3f2a;color:#4ade80}
.badge-writing{background:#3d2a1e;color:#fb923c}
.badge-career{background:#2d1e3f;color:#c084fc}
.badge-unclear{background:#2a2a2a;color:#9ca3af}
.input-row{display:flex;gap:.5rem}
textarea{flex:1;background:#16161d;border:1px solid #2a2a38;border-radius:10px;color:#e8e8f0;font-size:.95rem;padding:.75rem 1rem;resize:none;height:72px;line-height:1.5;transition:border-color .2s}
textarea:focus{outline:none;border-color:#2563eb}
button{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:0 1.4rem;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
button:hover{background:#1d4ed8}
button:disabled{background:#334;cursor:not-allowed}
.hint{font-size:.78rem;color:#555;text-align:center}
.confidence-bar{height:4px;border-radius:999px;background:#2a2a38;flex:1;overflow:hidden;margin-top:.3rem}
.confidence-fill{height:100%;border-radius:999px;background:#2563eb;transition:width .4s ease}
.empty-state{color:#444;font-size:.9rem;text-align:center;margin:auto}
</style>
</head>
<body>
<header>
<h1>LLM Prompt Router</h1>
<p>Your message is classified and routed to a specialized AI expert in real time.</p>
<p style="margin-top:.2rem;font-size:.78rem;color:#555">Tip: prefix with @code, @data, @writing, or @career to manually override routing.</p>
</header>
<div class="chat-container">
<div class="messages" id="messages"><div class="empty-state" id="empty">Send a message to get started.</div></div>
<div class="input-row">
<textarea id="input" placeholder="Ask anything..."></textarea>
<button id="sendBtn" onclick="sendMessage()">Send</button>
</div>
<p class="hint">Press Shift+Enter for a new line. Enter to send.</p>
</div>
<script>
const input=document.getElementById('input');
const sendBtn=document.getElementById('sendBtn');
const messages=document.getElementById('messages');
let emptyEl=document.getElementById('empty');
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});
const badges={code:'badge-code',data:'badge-data',writing:'badge-writing',career:'badge-career',unclear:'badge-unclear'};
function append(role,text,intent,conf){
  if(emptyEl){emptyEl.remove();emptyEl=null;}
  const w=document.createElement('div');w.className='message '+role;
  const b=document.createElement('div');b.className='bubble';b.textContent=text;w.appendChild(b);
  if(role==='assistant'&&intent){
    const badge=document.createElement('span');
    badge.className='intent-badge '+(badges[intent]||'badge-unclear');
    badge.textContent=intent+' ('+(conf*100).toFixed(0)+'% confidence)';
    w.appendChild(badge);
    const bar=document.createElement('div');bar.className='confidence-bar';
    const fill=document.createElement('div');fill.className='confidence-fill';fill.style.width=(conf*100)+'%';
    bar.appendChild(fill);w.appendChild(bar);
  }
  messages.appendChild(w);messages.scrollTop=messages.scrollHeight;
}
async function sendMessage(){
  const text=input.value.trim();if(!text)return;
  input.value='';sendBtn.disabled=true;
  append('user',text,null,null);
  const t=document.createElement('div');t.className='message assistant';
  t.innerHTML='<div class="bubble" style="color:#555">Thinking...</div>';
  messages.appendChild(t);messages.scrollTop=messages.scrollHeight;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
    const data=await res.json();t.remove();
    if(res.ok)append('assistant',data.response,data.intent,data.confidence);
    else append('assistant','Error: '+(data.detail||'Unknown error'),'unclear',0);
  }catch(err){t.remove();append('assistant','Network error. Is the server running?','unclear',0);}
  sendBtn.disabled=false;input.focus();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    return HTMLResponse(content=_HTML)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Classify intent and return expert response. Logs every request."""
    intent_result = classify_intent(request.message)
    try:
        final_response = route_and_respond(request.message, intent_result)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    log_route(
        user_message=request.message,
        intent=intent_result["intent"],
        confidence=intent_result["confidence"],
        final_response=final_response,
    )
    return ChatResponse(
        intent=intent_result["intent"],
        confidence=intent_result["confidence"],
        response=final_response,
    )


@app.get("/api/logs")
async def get_logs(limit: int = 20):
    """Return the most recent log entries from route_log.jsonl."""
    limit = _sanitize_limit(limit)
    if not os.path.exists(config.LOG_FILE):
        return JSONResponse(content={"entries": [], "total": 0})
    entries = []
    try:
        with open(config.LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return JSONResponse(content={"entries": [], "total": 0})
    return JSONResponse(content={"entries": list(reversed(entries[-limit:])), "total": len(entries)})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


def run_cli():
    print("LLM Prompt Router — Interactive CLI")
    print("=" * 50)
    print("Type your message and press Enter. Type 'quit' to exit.")
    print("Tip: prefix with @code, @data, @writing, or @career to override routing.")
    print("=" * 50)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break
        if not user_input:
            continue
        intent_result = classify_intent(user_input)
        print(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        try:
            response = route_and_respond(user_input, intent_result)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            continue
        log_route(
            user_message=user_input,
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            final_response=response,
        )
        print(f"\nAssistant [{intent_result['intent']}]:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="LLM Prompt Router")
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode.")
    args = parser.parse_args()
    if args.cli:
        run_cli()
    else:
        uvicorn.run("src.main:app", host="0.0.0.0", port=config.APP_PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
