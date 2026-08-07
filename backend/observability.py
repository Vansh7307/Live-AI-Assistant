"""Observability: Prometheus metrics + OpenTelemetry trace helpers.

Exposes a ``/metrics`` endpoint (Prometheus text format) and lightweight
tracing helpers that propagate a ``traceparent`` header so logs can be
correlated across a distributed deployment. Kept dependency-light: metrics
use ``prometheus_client``; tracing is a minimal W3C traceparent parser so we
don't need to pull in the full OpenTelemetry SDK.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# --- Metrics -------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUESTS_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

CHAT_REQUESTS_TOTAL = Counter(
    "chat_requests_total",
    "Total /chat requests",
    ["streaming"],
)

CHAT_LLM_CALLS = Counter(
    "chat_llm_calls_total",
    "Total LLM generate_text calls",
    ["provider"],
)

CHAT_LLM_FAILURES = Counter(
    "chat_llm_failures_total",
    "Total LLM failures",
    ["provider"],
)

CHAT_SEARCH_CALLS = Counter(
    "chat_search_calls_total",
    "Total Tavily search calls",
)

CHAT_SEARCH_FAILURES = Counter(
    "chat_search_failures_total",
    "Total Tavily search failures",
)

ACTIVE_CHATS = Gauge(
    "active_chats",
    "Number of in-flight chat requests",
)

# --- Tracing (W3C traceparent) -------------------------------------------

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$",
    re.IGNORECASE,
)


def parse_traceparent(header: str | None) -> str | None:
    """Extract a trace id from a W3C traceparent header, or None."""
    if not header:
        return None
    match = _TRACEPARENT_RE.match(header.strip())
    if not match:
        return None
    return match.group("trace_id")


def generate_traceparent() -> str:
    """Create a random traceparent (version 00) for outbound correlation."""
    trace_id = os.urandom(16).hex()
    span_id = os.urandom(8).hex()
    return f"00-{trace_id}-{span_id}-01"


# --- Middleware -----------------------------------------------------------


class MetricsMiddleware:
    """ASGI middleware that records request count/duration and a trace id."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        start = time.monotonic()

        trace_id = parse_traceparent(request.headers.get("traceparent"))
        if trace_id:
            scope["state"]["trace_id"] = trace_id
        else:
            scope["state"]["trace_id"] = generate_traceparent()

        status_holder = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.monotonic() - start
            path = request.url.path
            # Avoid high-cardinality metric labels on streaming paths.
            label_path = path.split("/")[0] or "/"
            REQUESTS_TOTAL.labels(request.method, label_path, status_holder["status"]).inc()
            REQUESTS_DURATION.labels(request.method, label_path).observe(duration)


async def metrics_endpoint(request: Request) -> Response:
    """Serve Prometheus metrics in text exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
