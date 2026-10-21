# PARALLAX Bug Tracker

Documenting bugs, edge-case failures, and anomalies identified during adversarial fuzzing and stress testing.

---

### BUG-001: Unhandled `TypeError: Object of type bytes is not JSON serializable` in `validation_exception_handler`

- **Severity:** Medium
- **Impacted Component:** `parallax.api.app` (`validation_exception_handler`)
- **Discovered In:** Wave A (Adversarial Fuzzing)
- **Status:** Open

#### Description
When an incoming HTTP request contains raw non-JSON bytes (such as XML, raw binary, or malformed non-UTF8 bodies) and triggers a `fastapi.exceptions.RequestValidationError`, FastAPI populates `exc.errors()` where the field `'input'` contains raw `bytes` (e.g. `b'<xml><text>attack</text></xml>'`).

In `parallax/api/app.py`:
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    req_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": exc.errors(),
            "error": "ValidationError",
            "request_id": req_id,
        },
    )
```
Starlette's `JSONResponse` serializes `content` using standard Python `json.dumps()`. Because Python's `json` encoder does not support `bytes`, this raises:
```text
TypeError: Object of type bytes is not JSON serializable
```
This unhandled exception escapes the validation handler and is caught by `GlobalExceptionMiddleware`, returning an unexpected `HTTP 500 Internal Server Error` instead of `HTTP 422 Unprocessable Content`. (Note: `GlobalExceptionMiddleware` prevents stack trace leakage to the client, but the server log records an unhandled 500 error).

#### Reproduction Steps
1. Run the PARALLAX API server.
2. Send a POST request to `/v1/scan` with raw XML bytes:
   ```bash
   curl -X POST http://localhost:8000/v1/scan \
     -H "Content-Type: application/xml" \
     -d "<xml><text>attack</text></xml>"
   ```
3. Observe `HTTP 500 Internal Server Error` returned instead of `HTTP 422 Unprocessable Content`.

#### Proposed Fix
In `parallax/api/app.py`, use FastAPI's `jsonable_encoder`:
```python
from fastapi.encoders import jsonable_encoder


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    req_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=jsonable_encoder(
            {
                "detail": exc.errors(),
                "error": "ValidationError",
                "request_id": req_id,
            }
        ),
    )
```

---

### BUG-002: Negative `Content-Length` header not rejected by `PayloadSizeLimitMiddleware`

- **Severity:** Low
- **Impacted Component:** `parallax.api.middlewares` (`PayloadSizeLimitMiddleware`)
- **Discovered In:** Wave A (Adversarial Fuzzing)
- **Status:** Open

#### Description
In `PayloadSizeLimitMiddleware.__call__`, incoming requests are checked for the `Content-Length` header:
```python
content_length = int(value.decode("latin1"))
if content_length > self.max_input_size:
    await self._send_rejection(...)
```
If a client sends a negative `Content-Length` header (e.g. `Content-Length: -1`), `int("-1")` succeeds and does not exceed `max_input_size`. Consequently, the middleware does not reject the malformed header and allows the request through to downstream streaming processing instead of returning `HTTP 400 Bad Request`.

#### Reproduction Steps
1. Run the PARALLAX API server.
2. Send a POST request to `/v1/scan` with header `Content-Length: -1`:
   ```bash
   curl -X POST http://localhost:8000/v1/scan \
     -H "Content-Type: application/json" \
     -H "Content-Length: -1" \
     -d '{"text": "hello"}'
   ```
3. Observe that the request succeeds (`HTTP 200 OK`) instead of being rejected as malformed with `HTTP 400 Bad Request`.

#### Proposed Fix
In `parallax/api/middlewares.py`:
```python
content_length = int(value.decode("latin1"))
if content_length < 0:
    await self._send_rejection(
        send,
        status_code=400,
        detail="Invalid Content-Length header: value cannot be negative",
        error="BadRequest",
        req_id=req_id,
    )
    return
if content_length > self.max_input_size:
    ...
```
