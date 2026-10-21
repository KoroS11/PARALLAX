"""ASGI middleware components for Request ID, Access Logging, Bulkhead, and Payload limits."""

import json
import logging
import threading
import time
import uuid
from typing import Any

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PayloadTooLargeError(StarletteHTTPException):
    """Raised when an incoming payload exceeds the configured maximum size."""

    def __init__(self, max_size: int) -> None:
        super().__init__(
            status_code=413,
            detail=f"Payload size exceeds maximum allowed limit of {max_size} bytes",
        )


class RequestIdMiddleware:
    """Middleware ensuring every request has an X-Request-ID attached."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id: str | None = None
        for key, value in scope.get("headers", []):
            if key.lower() == b"x-request-id":
                req_id = value.decode("latin1").strip()
                break

        if not req_id:
            req_id = str(uuid.uuid4())

        state = scope.setdefault("state", {})
        state["request_id"] = req_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Add X-Request-ID header to response
                headers.append((b"x-request-id", req_id.encode("latin1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class BulkheadMiddleware:
    """Bulkhead concurrency limiter returning 503 with Retry-After: 1 when overloaded."""

    def __init__(
        self,
        app: ASGIApp,
        max_concurrency: int = 50,
        exempt_paths: set[str] | None = None,
    ) -> None:
        self.app = app
        self.max_concurrency = max_concurrency
        self.exempt_paths = exempt_paths or {"/health", "/ready", "/metrics"}
        self._active_requests = 0
        self._lock = threading.Lock()

    @property
    def active_requests(self) -> int:
        """Return the current number of active requests."""
        with self._lock:
            return self._active_requests

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        with self._lock:
            if self._active_requests >= self.max_concurrency:
                overloaded = True
            else:
                self._active_requests += 1
                overloaded = False

        if overloaded:
            req_id = scope.get("state", {}).get("request_id", "")
            payload = json.dumps(
                {
                    "detail": "Service Unavailable: server concurrency limit reached",
                    "error": "Overloaded",
                    "request_id": req_id,
                }
            ).encode("utf-8")

            headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", b"1"),
                (b"content-length", str(len(payload)).encode("latin1")),
            ]
            if req_id:
                headers.append((b"x-request-id", req_id.encode("latin1")))

            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": payload,
                    "more_body": False,
                }
            )
            return

        try:
            await self.app(scope, receive, send)
        finally:
            with self._lock:
                self._active_requests -= 1


class PayloadSizeLimitMiddleware:
    """Validates request body size and rejects payloads larger than max_input_size with 413."""

    def __init__(self, app: ASGIApp, max_input_size: int = 32 * 1024) -> None:
        self.app = app
        self.max_input_size = max_input_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = scope.get("state", {}).get("request_id", "")

        # Fast check via Content-Length header
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    content_length = int(value.decode("latin1"))
                    if content_length > self.max_input_size:
                        await self._send_rejection(
                            send,
                            status_code=413,
                            detail=f"Payload size ({content_length} bytes) exceeds limit ({self.max_input_size} bytes)",
                            error="PayloadTooLarge",
                            req_id=req_id,
                        )
                        return
                except ValueError:
                    await self._send_rejection(
                        send,
                        status_code=400,
                        detail="Invalid Content-Length header",
                        error="BadRequest",
                        req_id=req_id,
                    )
                    return
                break

        # Streaming body check
        received_bytes = 0

        async def limit_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                received_bytes += len(chunk)
                if received_bytes > self.max_input_size:
                    raise PayloadTooLargeError(self.max_input_size)
            return message

        try:
            await self.app(scope, limit_receive, send)
        except PayloadTooLargeError:
            await self._send_rejection(
                send,
                status_code=413,
                detail=f"Payload size exceeds limit of {self.max_input_size} bytes",
                error="PayloadTooLarge",
                req_id=req_id,
            )

    async def _send_rejection(
        self,
        send: Send,
        status_code: int,
        detail: str,
        error: str,
        req_id: str,
    ) -> None:
        payload = json.dumps(
            {
                "detail": detail,
                "error": error,
                "request_id": req_id,
            }
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode("latin1")),
        ]
        if req_id:
            headers.append((b"x-request-id", req_id.encode("latin1")))

        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload,
                "more_body": False,
            }
        )


class AccessLogMiddleware:
    """Structured JSON access logger measuring request duration and status."""

    def __init__(self, app: ASGIApp, logger_name: str = "parallax.api.access") -> None:
        self.app = app
        self.logger = logging.getLogger(logger_name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        status_code = 500

        async def capturing_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, capturing_send)
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            req_id = scope.get("state", {}).get("request_id", "")
            method = scope.get("method", "")
            path = scope.get("path", "")
            query = scope.get("query_string", b"").decode("latin1")
            full_path = f"{path}?{query}" if query else path

            log_entry: dict[str, Any] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": req_id,
                "method": method,
                "path": full_path,
                "status_code": status_code,
                "duration_ms": elapsed_ms,
            }
            self.logger.info(json.dumps(log_entry))


class GlobalExceptionMiddleware:
    """Catches any unhandled exceptions, sanitizes to JSON, and prevents stack trace leakage."""

    def __init__(self, app: ASGIApp, metrics: Any | None = None) -> None:
        self.app = app
        self.metrics = metrics
        self.logger = logging.getLogger("parallax.api")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except Exception:
            self.logger.exception("Unhandled server error processing request")
            if self.metrics is not None:
                self.metrics.record_error()
            req_id = scope.get("state", {}).get("request_id", "")
            payload = json.dumps(
                {
                    "detail": "An internal server error occurred.",
                    "error": "InternalServerError",
                    "request_id": req_id,
                }
            ).encode("utf-8")
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("latin1")),
            ]
            if req_id:
                headers.append((b"x-request-id", req_id.encode("latin1")))

            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": payload,
                    "more_body": False,
                }
            )
