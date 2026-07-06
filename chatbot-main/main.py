import asyncio
import json as _json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from chatbot_engine import MedicalChatSession, detect_lang, get_error_msg

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Chatbot API")

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if _allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "null",
    ]
    logger.warning(
        "ALLOWED_ORIGINS is not set in .env — defaulting to localhost + "
        "file:// (null) for local development only. Set it explicitly "
        "before deploying to production: "
        "ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com "
        "(null will not be allowed automatically once set)."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

RATE_LIMIT_WINDOW_SEC    = 60
RATE_LIMIT_MAX_REQUESTS  = 20

_rate_limit_lock  = threading.Lock()
_rate_limit_hits: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        hits = _rate_limit_hits[client_ip]
        while hits and now - hits[0] > RATE_LIMIT_WINDOW_SEC:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        hits.append(now)
        return True


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

SESSION_TTL_SECONDS  = 30 * 60
CLEANUP_INTERVAL_SEC = 5 * 60

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _create_session_entry(user_id: str | None = None) -> tuple[str, MedicalChatSession]:
    session_id = str(uuid.uuid4())
    session    = MedicalChatSession(user_id=user_id)
    with _sessions_lock:
        _sessions[session_id] = {
            "session":     session,
            "last_active": time.time(),
            "lock":        asyncio.Lock(),
        }
    return session_id, session


def _require_session_entry(session_id: str) -> dict:
    with _sessions_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            raise HTTPException(
                404,
                "Session not found or expired. Please start a new session "
                "via /chat/start.",
            )
        entry["last_active"] = time.time()
        return entry


def _cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        now = time.time()
        with _sessions_lock:
            expired = [sid for sid, e in _sessions.items()
                       if now - e["last_active"] > SESSION_TTL_SECONDS]
            for sid in expired:
                del _sessions[sid]
        if expired:
            logger.info("Session cleanup: removed %d expired session(s)", len(expired))


@app.on_event("startup")
def _start_cleanup_thread() -> None:
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    logger.info("Session cleanup thread started (TTL=%ds)", SESSION_TTL_SECONDS)


@app.post("/chat/start")
def start_session(user_id: str | None = Form(default=None)):
    """
    Create a new session.

    `user_id` is optional. Pass it once the parent app has a stable patient
    identity to hand over (any string works — it's just a lookup key for
    patient memory, it doesn't have to be a phone number). If omitted, the
    session behaves exactly as before: no cross-session memory.
    """
    session_id, _ = _create_session_entry(user_id=user_id)
    return {"session_id": session_id}


@app.post("/chat/message")
async def send_message(request: Request, session_id: str = Form(...), message: str = Form(...)):
    """Send a message - full (non-streaming) response."""
    if not message.strip():
        raise HTTPException(400, "Empty message")
    if not _check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please slow down")

    entry = _require_session_entry(session_id)
    async with entry["lock"]:
        result = await entry["session"].send_message(message)
    return result


@app.post("/chat/stream")
async def chat_stream(request: Request, session_id: str = Form(...), message: str = Form(...)):
    """
    Send a message - streaming (SSE) response.

    The client receives events:
    - data: <text_chunk>
    - data: [SOURCES]<json_array>
    - data: [REPORT]<json_object>
    - data: [ERROR]<error_message>
    - data: [DONE]
    """
    if not message.strip():
        raise HTTPException(400, "Empty message")
    if not _check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please slow down")

    entry   = _require_session_entry(session_id)
    session = entry["session"]
    lock    = entry["lock"]
    lang    = detect_lang(session.history, message)

    async def generate():
        async with lock:
            try:
                async for event_type, value in session.stream_message(message):
                    if event_type == "text":
                        safe = (value or "").replace("\n", "\\n")
                        yield f"data: {safe}\n\n"

                    elif event_type == "sources":
                        json_str = _json.dumps(value, ensure_ascii=False)
                        yield f"data: [SOURCES]{json_str}\n\n"

                    elif event_type == "report":
                        json_str = _json.dumps(value, ensure_ascii=False)
                        yield f"data: [REPORT]{json_str}\n\n"

                    elif event_type == "error":
                        yield f"data: [ERROR]{value}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    elif event_type == "done":
                        yield "data: [DONE]\n\n"
                        return

                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.error("chat_stream error: %s", exc)
                yield f"data: [ERROR]{get_error_msg(exc, lang)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/chat/{session_id}/report")
def get_report(session_id: str):
    """Get the medical report for a session."""
    entry  = _require_session_entry(session_id)
    report = entry["session"].get_report()
    if not report:
        return {"report": None, "message": "No report generated yet"}
    return {"report": report}


@app.delete("/chat/{session_id}")
def end_session(session_id: str):
    """End a session."""
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
    return {"message": "Session ended"}


@app.get("/health")
def health():
    from key_manager import key_manager
    with _sessions_lock:
        active_sessions = len(_sessions)
    return {
        "status":          "ok",
        "keys_loaded":     len(key_manager.keys),
        "keys_ok":         len(key_manager.keys) > 0,
        "active_sessions": active_sessions,
    }
