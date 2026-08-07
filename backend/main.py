import asyncio
import contextvars
import json
import os
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from agent_graph import build_and_run, stream_answer
from llm import LLMQuotaError
from safety import SafetyError, check_input, check_output
from observability import (
    metrics_endpoint,
    CHAT_REQUESTS_TOTAL,
    ACTIVE_CHATS,
)

load_dotenv()

# Request-correlation id, exposed to every log call (in this module or any
# other - llm.py, tools.*, agent_graph.py, even third-party loggers like
# uvicorn/httpx) without needing to pass `extra=` at every call site.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
)

_base_record_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _base_record_factory(*args, **kwargs)
    record.request_id = _request_id_var.get()
    return record


logging.setLogRecordFactory(_record_factory)
logger = logging.getLogger(__name__)

RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "20/minute")
APP_API_KEY = os.getenv("APP_API_KEY")  # optional: set to require an API key on /chat
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = [key for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)]
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("(<any LLM provider key>)")
    if missing:
        logger.warning("Missing required environment variables: %s", ", ".join(missing))
    if not APP_API_KEY:
        logger.warning(
            "APP_API_KEY is not set - /chat is unauthenticated and reachable by anyone "
            "who can hit this URL. Set APP_API_KEY to require an X-API-Key header."
        )
    yield


app = FastAPI(title="Live AI Assistant", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Session-Id", "X-API-Key", "traceparent"],
    expose_headers=["X-Session-Id", "X-Request-Id"],
)

# Metrics / tracing middleware.
from observability import MetricsMiddleware

app.add_middleware(MetricsMiddleware)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Tags every request with a correlation id (for log tracing across a
    distributed deployment) and enforces a hard wall-clock timeout so a
    stuck upstream call can't hold a rate-limit slot / connection forever."""
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    token = _request_id_var.set(request_id)
    start = time.monotonic()

    try:
        try:
            response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error(
                "Request timed out after %.1fs: %s %s",
                REQUEST_TIMEOUT_SECONDS,
                request.method,
                request.url.path,
            )
            return JSONResponse(status_code=504, content={"detail": "The request took too long to complete."})

        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    finally:
        _request_id_var.reset(token)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _SESSION_ID_RE.match(value):
            raise ValueError("session_id must match ^[A-Za-z0-9_-]{1,128}$")
        return value


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


class SessionListResponse(BaseModel):
    sessions: list[dict]


def _require_api_key(x_api_key: str | None) -> None:
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
async def chat(
    request: Request,
    req: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _require_api_key(x_api_key)
    CHAT_REQUESTS_TOTAL.labels("false").inc()
    try:
        check_input(req.message)
    except SafetyError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = await build_and_run(req.message, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMQuotaError as exc:
        logger.warning("LLM quota/rate limit reached: %s", exc)
        raise HTTPException(
            status_code=503, detail="The AI service is temporarily unavailable. Please try again later."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all boundary
        logger.exception("Chat request failed (session_id=%s)", session_id)
        raise HTTPException(
            status_code=502, detail="The assistant could not complete this request. Please try again."
        ) from exc

    answer = check_output(result["answer"])
    return ChatResponse(
        answer=answer,
        sources=[Source(**s) for s in result.get("sources", [])],
        session_id=session_id,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
@limiter.limit(RATE_LIMIT)
async def chat_stream(
    request: Request,
    req: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _require_api_key(x_api_key)
    CHAT_REQUESTS_TOTAL.labels("true").inc()
    try:
        check_input(req.message)
    except SafetyError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    session_id = req.session_id or str(uuid.uuid4())

    async def event_gen():
        ACTIVE_CHATS.inc()
        try:
            try:
                async for event in stream_answer(req.message, session_id):
                    if event["type"] == "token":
                        yield _sse("token", {"token": event["token"]})
                    elif event["type"] == "sources":
                        yield _sse("sources", {"sources": event["sources"]})
                    elif event["type"] == "metadata":
                        yield _sse("session", {"session_id": event["session_id"]})
                    elif event["type"] == "done":
                        yield _sse("done", {})
            except LLMQuotaError as exc:
                logger.warning("LLM quota during streaming: %s", exc)
                yield _sse(
                    "error",
                    {"detail": "The AI service is temporarily unavailable. Please try again later."},
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming chat failed")
                yield _sse("error", {"detail": "The assistant could not complete this request."})
        finally:
            ACTIVE_CHATS.dec()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions")
async def list_sessions():
    """For the frontend history sidebar. Requires API key if configured."""
    from memory.sqlite_memory import Memory

    mem = Memory()
    sessions = await asyncio.to_thread(mem.list_sessions)
    return SessionListResponse(sessions=sessions)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.get("/health/live", include_in_schema=False)
async def health_live():
    """Liveness: the process is up and serving requests."""
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
async def health_ready():
    """Readiness: the service can actually fulfill a chat request (has an
    LLM provider configured)."""
    has_llm = any(
        os.getenv(k) for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_llm:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "No LLM provider configured"})
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    return await metrics_endpoint(request)


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
