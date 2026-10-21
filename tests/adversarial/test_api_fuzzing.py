"""Adversarial API fuzzing and stress testing suite for PARALLAX serving plane.

Tests:
1. Huge payloads exceeding max_input_size (32 KB, 64 KB, 1 MB) via Content-Length and streaming.
2. Null bytes (\\x00), control characters, and malformed encoding.
3. Deeply nested JSON payloads and unexpected types.
4. Unicode edge cases: RTL overrides (\\u202E), unmatched surrogates, non-characters (\\uFFFF), zero-width floods.
5. High concurrency bursts (50-100 parallel requests) verifying bulkhead 503 with Retry-After: 1 without leaking.
6. Information leakage verification (no tracebacks, file paths, or internal stack traces).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from parallax.api import Settings, create_app
from parallax.contracts.api import (
    Decision,
    ScanResponse,
)
from parallax.contracts.detectors import Detector, DetectorResult


class MockDetector:
    """Deterministic mock detector for adversarial testing."""

    def __init__(
        self,
        name: str = "mock-fuzz-detector",
        attack_on: str = "attack",
        score: float = 0.95,
        latency_ms: float = 1.0,
    ) -> None:
        self.name = name
        self.attack_on = attack_on.lower()
        self.score = score
        self.latency_ms = latency_ms

    async def predict(self, text: str) -> DetectorResult:
        is_attack = self.attack_on in text.lower()
        return DetectorResult(
            name=self.name,
            is_attack=is_attack,
            score=self.score if is_attack else 0.05,
            attack_class="instruction_override" if is_attack else None,
            latency_ms=self.latency_ms,
            error=None,
        )


class DelayedDetector:
    """Mock detector that adds a controlled delay for concurrency testing."""

    def __init__(self, name: str = "delayed-detector", delay: float = 0.15) -> None:
        self.name = name
        self.delay = delay

    async def predict(self, text: str) -> DetectorResult:
        await asyncio.sleep(self.delay)
        return DetectorResult(
            name=self.name,
            is_attack=False,
            score=0.1,
            latency_ms=self.delay * 1000.0,
        )


def assert_no_information_leak(response: httpx.Response) -> None:
    """Verify that an HTTP response never leaks stack traces, file paths, or internals."""
    text = response.text

    # 1. Traceback signatures
    assert "traceback (most recent call last)" not in text.lower(), (
        f"Traceback leaked in response body: {text[:300]}"
    )

    # 2. Python line/file reference signatures
    assert '.py", line' not in text, (
        f"Python source line leaked in response body: {text[:300]}"
    )
    assert 'File "' not in text, (
        f"Source file path leaked in response body: {text[:300]}"
    )

    # 3. Directory paths
    assert "C:\\Users\\" not in text, f"Local user directory leaked: {text[:300]}"
    assert "/home/" not in text, f"Unix home directory leaked: {text[:300]}"
    assert "\\parallax\\" not in text and "/parallax/" not in text, (
        f"Internal package path leaked: {text[:300]}"
    )

    # 4. Python exception class patterns
    for exc in ("RuntimeError:", "ZeroDivisionError:", "ValueError:", "KeyError:"):
        assert exc not in text, f"Raw exception string '{exc}' leaked: {text[:300]}"


@pytest.fixture
def fuzz_detectors() -> list[Detector]:
    return [
        MockDetector(name="d1-fuzz", attack_on="attack"),
        MockDetector(name="d2-fuzz", attack_on="malicious"),
    ]


@pytest.fixture
def default_fuzz_settings() -> Settings:
    return Settings(
        environment="test",
        max_input_size=32 * 1024,  # Standard 32 KB limit
        max_concurrency=20,
        default_timeout=0.5,
    )


@pytest_asyncio.fixture
async def fuzz_client(
    fuzz_detectors: list[Detector], default_fuzz_settings: Settings
) -> AsyncIterator[AsyncClient]:
    app = create_app(
        settings=default_fuzz_settings,
        detectors=fuzz_detectors,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ===========================================================================
# 1. Huge Payloads Exceeding max_input_size (32 KB, 64 KB, 1 MB)
# ===========================================================================


class TestHugePayloads:
    """Tests for payload size limits and streaming rejection."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload_size", "label"),
        [
            (32 * 1024 + 1, "32KB_plus_1B"),
            (64 * 1024, "64KB"),
            (1024 * 1024, "1MB"),
        ],
    )
    async def test_oversized_payload_rejected_with_413(
        self,
        fuzz_detectors: list[Detector],
        default_fuzz_settings: Settings,
        payload_size: int,
        label: str,
    ) -> None:
        """Verify payloads exceeding max_input_size (32KB+1B, 64KB, 1MB) return 413."""
        app = create_app(settings=default_fuzz_settings, detectors=fuzz_detectors)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            oversized_text = "A" * payload_size
            raw_body = json.dumps({"text": oversized_text}).encode("utf-8")
            assert len(raw_body) >= payload_size

            resp = await ac.post(
                "/v1/scan",
                content=raw_body,
                headers={"content-type": "application/json"},
            )

            assert resp.status_code == 413, (
                f"Expected 413 for {label}, got {resp.status_code}"
            )
            data = resp.json()
            assert data.get("error") in ("PayloadTooLarge", "PayloadTooLargeError")
            assert "exceeds limit" in data.get("detail", "").lower()
            assert "x-request-id" in resp.headers
            assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_streaming_chunked_oversized_payload(
        self,
        fuzz_detectors: list[Detector],
        default_fuzz_settings: Settings,
    ) -> None:
        """Verify streaming body without Content-Length header is terminated when exceeding limit."""
        app = create_app(settings=default_fuzz_settings, detectors=fuzz_detectors)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            chunk = b"x" * 8192

            async def stream_chunks() -> AsyncIterator[bytes]:
                # Yield 5 chunks of 8KB = 40KB (exceeds 32KB)
                for _ in range(5):
                    yield chunk

            resp = await ac.post(
                "/v1/scan",
                content=stream_chunks(),
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 413
            data = resp.json()
            assert data.get("error") in ("PayloadTooLarge", "PayloadTooLargeError")
            assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_boundary_payload_size(
        self,
        fuzz_detectors: list[Detector],
    ) -> None:
        """Verify exact boundary behavior at limit vs limit + 1 byte."""
        limit = 1000
        boundary_settings = Settings(max_input_size=limit)
        app = create_app(settings=boundary_settings, detectors=fuzz_detectors)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Construct body of exactly limit bytes
            prefix = '{"text": "'
            suffix = '"}'
            padding_len = limit - len(prefix) - len(suffix)
            exact_body = (prefix + ("b" * padding_len) + suffix).encode("utf-8")
            assert len(exact_body) == limit

            # Exact limit -> should pass payload check (200 status)
            resp_exact = await ac.post(
                "/v1/scan",
                content=exact_body,
                headers={"content-type": "application/json"},
            )
            assert resp_exact.status_code == 200

            # Exactly limit + 1 byte -> should return 413
            over_body = (prefix + ("b" * (padding_len + 1)) + suffix).encode("utf-8")
            assert len(over_body) == limit + 1

            resp_over = await ac.post(
                "/v1/scan",
                content=over_body,
                headers={"content-type": "application/json"},
            )
            assert resp_over.status_code == 413
            assert resp_over.json().get("error") == "PayloadTooLarge"


# ===========================================================================
# 2. Null Bytes, Control Characters, and Malformed Encoding
# ===========================================================================


class TestNullBytesAndMalformedEncoding:
    """Tests for null bytes, control characters, and malformed encoding."""

    @pytest.mark.asyncio
    async def test_null_bytes_in_prompt_handled_safely(
        self, fuzz_client: AsyncClient
    ) -> None:
        """Verify that null bytes in prompt do not crash the service and are sanitized."""
        payload = {"text": "prefix\x00middle\x00suffix"}
        resp = await fuzz_client.post("/v1/scan", json=payload)
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        assert scan.decision in (Decision.ALLOW, Decision.BLOCK)
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_control_characters_in_prompt(self, fuzz_client: AsyncClient) -> None:
        """Verify ASCII control characters (0x01-0x1F) do not cause unhandled errors."""
        control_chars = "".join(chr(i) for i in range(1, 32) if i not in (9, 10, 13))
        payload = {"text": f"safe text {control_chars} prompt end"}
        resp = await fuzz_client.post("/v1/scan", json=payload)
        assert resp.status_code == 200
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_malformed_utf8_in_request_body(
        self, fuzz_client: AsyncClient
    ) -> None:
        """Verify invalid raw byte sequences (non-UTF-8) are rejected without leaking."""
        invalid_bytes = b'{"text": "invalid \x80\xff\xfe bytes"}'
        resp = await fuzz_client.post(
            "/v1/scan",
            content=invalid_bytes,
            headers={"content-type": "application/json"},
        )
        # Should be rejected with 400 or 422 or 500 cleanly
        assert resp.status_code in (400, 422, 500)
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_cl",
        ["abc", "12.34", "9999999999999999999999999999999999999999"],
    )
    async def test_malformed_content_length_header(
        self, fuzz_client: AsyncClient, bad_cl: str
    ) -> None:
        """Verify non-integer or overflowing Content-Length returns 400 or 413 without crashing."""
        resp = await fuzz_client.post(
            "/v1/scan",
            content=b'{"text": "hello"}',
            headers={
                "content-type": "application/json",
                "content-length": bad_cl,
            },
        )
        assert resp.status_code in (400, 413)
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_negative_content_length_header(
        self, fuzz_client: AsyncClient
    ) -> None:
        """Test negative Content-Length header behavior (see docs/BUGS.md BUG-002)."""
        resp = await fuzz_client.post(
            "/v1/scan",
            content=b'{"text": "hello"}',
            headers={
                "content-type": "application/json",
                "content-length": "-1",
            },
        )
        # PayloadSizeLimitMiddleware permits negative int through without 400
        assert resp.status_code in (200, 400)
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_unsupported_content_type(self, fuzz_client: AsyncClient) -> None:
        """Verify non-JSON Content-Type requests do not leak information (see docs/BUGS.md BUG-001)."""
        resp = await fuzz_client.post(
            "/v1/scan",
            content=b"<xml><text>attack</text></xml>",
            headers={"content-type": "application/xml"},
        )
        # BUG-001: Returns 500 instead of 422 due to bytes serialization in validation handler,
        # but GlobalExceptionMiddleware ensures zero information leakage.
        assert resp.status_code in (422, 500)
        assert_no_information_leak(resp)


# ===========================================================================
# 3. Deeply Nested JSON Payloads and Unexpected Types
# ===========================================================================


class TestDeeplyNestedAndUnexpectedTypes:
    """Tests for schema fuzzing, deeply nested structures, and type confusion."""

    @pytest.mark.asyncio
    async def test_deeply_nested_json_dict(self, fuzz_client: AsyncClient) -> None:
        """Verify deeply nested dict structures are rejected cleanly without RecursionError."""
        nested = '{"nested": ' * 80 + '{"text": "deep"}' + "}" * 80
        resp = await fuzz_client.post(
            "/v1/scan",
            content=nested.encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_deeply_nested_json_array(self, fuzz_client: AsyncClient) -> None:
        """Verify deeply nested array structures return 422 cleanly."""
        nested = "[" * 80 + "]" * 80
        resp = await fuzz_client.post(
            "/v1/scan",
            content=nested.encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_payload",
        [
            {"text": 12345},  # Int instead of string
            {"text": 3.14159},  # Float
            {"text": True},  # Bool
            {"text": None},  # Null
            {"text": ["a", "b", "c"]},  # List
            {"text": {"sub": "dict"}},  # Dict
            {"text": ""},  # Empty string (violates min_length=1)
            {},  # Missing text
            {"text": "hello", "mode": 999},  # Non-string mode
            {"text": "hello", "mode": "hyperdrive_mode"},  # Unknown enum
            {"text": "hello", "trace_id": 12345},  # Non-string trace_id
            {"text": "hello", "extra_injected_field": "val"},  # Extra field forbidden
            {
                "text": "hello",
                "__proto__": {"admin": True},
            },  # Prototype pollution probe
        ],
    )
    async def test_unexpected_types_and_fields(
        self, fuzz_client: AsyncClient, bad_payload: dict[str, Any]
    ) -> None:
        """Verify type fuzzing yields structured 422 ValidationError with no leak."""
        resp = await fuzz_client.post("/v1/scan", json=bad_payload)
        assert resp.status_code == 422
        data = resp.json()
        assert data.get("error") == "ValidationError"
        assert "detail" in data
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("primitive", ['"plain string"', "12345", "true", "null"])
    async def test_primitive_json_roots(
        self, fuzz_client: AsyncClient, primitive: str
    ) -> None:
        """Verify JSON primitives at root yield 422 ValidationError."""
        resp = await fuzz_client.post(
            "/v1/scan",
            content=primitive.encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        assert_no_information_leak(resp)


# ===========================================================================
# 4. Unicode Edge Cases (RTL, Surrogates, Non-characters, Zero-Width)
# ===========================================================================


class TestUnicodeEdgeCases:
    """Tests for Unicode directional overrides, surrogates, non-characters, and zalgo."""

    @pytest.mark.asyncio
    async def test_rtl_directional_overrides(self, fuzz_client: AsyncClient) -> None:
        """Verify RTL overrides (\\u202E, \\u202B) do not cause exceptions."""
        rtl_text = (
            "Hello \u202e reverse text instruction \u202c normal text \u2067 isolate"
        )
        resp = await fuzz_client.post("/v1/scan", json={"text": rtl_text})
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        assert scan.decision in (Decision.ALLOW, Decision.BLOCK)
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_unmatched_surrogates_handling(
        self, fuzz_client: AsyncClient
    ) -> None:
        """Verify unmatched surrogates in escaped JSON are sanitized cleanly."""
        surrogate_body = b'{"text": "safe prefix \\ud800\\udfff suffix"}'
        resp = await fuzz_client.post(
            "/v1/scan",
            content=surrogate_body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_unicode_non_characters(self, fuzz_client: AsyncClient) -> None:
        """Verify Unicode non-characters (\\uFFFF, \\uFFFE, \\uFDD0) are handled safely."""
        non_chars = "Special chars: \uffff \ufffe \ufdd0 \ufdef valid text"
        resp = await fuzz_client.post("/v1/scan", json={"text": non_chars})
        assert resp.status_code == 200
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_zero_width_character_flood(self, fuzz_client: AsyncClient) -> None:
        """Verify thousands of zero-width characters are stripped without hanging."""
        # 4000 zero-width characters interspersed
        zw_flood = "attack" + ("\u200b\u200c\u200d\ufeff" * 1000)
        resp = await fuzz_client.post("/v1/scan", json={"text": zw_flood})
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        # "attack" should still be detected after zero-width stripping
        assert scan.decision == Decision.BLOCK
        assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_combining_characters_zalgo_flood(
        self, fuzz_client: AsyncClient
    ) -> None:
        """Verify zalgo combining character text is normalized safely."""
        zalgo_text = "H" + ("\u0300\u0301\u0302\u0303\u0304" * 100) + "ello attack"
        resp = await fuzz_client.post("/v1/scan", json={"text": zalgo_text})
        assert resp.status_code == 200
        assert_no_information_leak(resp)


# ===========================================================================
# 5. High Concurrency Bursts (Bulkhead 503 & No Resource Leaks)
# ===========================================================================


class TestConcurrencyAndBulkhead:
    """Tests for high concurrency bursts, bulkhead saturation, and memory/counter safety."""

    @pytest.mark.asyncio
    async def test_high_concurrency_burst_bulkhead_limit(self) -> None:
        """Verify burst of 60 requests against max_concurrency=10 returns 503 with Retry-After: 1."""
        bulkhead_settings = Settings(
            max_concurrency=10,
            default_timeout=2.0,
        )
        delayed_detector = DelayedDetector(delay=0.1)
        app = create_app(settings=bulkhead_settings, detectors=[delayed_detector])
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            burst_size = 60
            tasks = [
                ac.post("/v1/scan", json={"text": f"burst-request-{i}"})
                for i in range(burst_size)
            ]
            responses = await asyncio.gather(*tasks)

            status_codes = [r.status_code for r in responses]
            successes = [r for r in responses if r.status_code == 200]
            overloaded = [r for r in responses if r.status_code == 503]

            # At least some must be 200 and some must be 503
            assert len(successes) > 0, "Expected some requests to succeed"
            assert len(overloaded) > 0, (
                f"Expected bulkhead 503 rejections, got codes: {set(status_codes)}"
            )
            assert len(successes) + len(overloaded) == burst_size

            # Check 503 structure and Retry-After header
            for r in overloaded:
                assert r.headers.get("retry-after") == "1"
                data = r.json()
                assert data.get("error") == "Overloaded"
                assert "server concurrency limit reached" in data.get("detail", "")
                assert_no_information_leak(r)

            # Check exempt endpoints remained operational during or right after burst
            health_resp = await ac.get("/health")
            assert health_resp.status_code == 200
            metrics_resp = await ac.get("/metrics")
            assert metrics_resp.status_code == 200

            # Verify bulkhead counter reset: a subsequent single request must succeed
            post_burst_resp = await ac.post(
                "/v1/scan", json={"text": "post-burst-check"}
            )
            assert post_burst_resp.status_code == 200, (
                "Bulkhead concurrency leaked active request count!"
            )


# ===========================================================================
# 6. Global Information Leakage Audit
# ===========================================================================


class TestInformationLeakageAudit:
    """Systematic check verifying error responses never reveal stack traces or paths."""

    @pytest.mark.asyncio
    async def test_internal_server_error_sanitized(self) -> None:
        """Verify unhandled exceptions return sanitized 500 without internal trace."""
        app = create_app(
            settings=Settings(environment="test"),
            detectors=[MockDetector()],
        )
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Trigger intentional exception endpoint
            resp = await ac.get("/_test/raise-unhandled")
            assert resp.status_code == 500
            data = resp.json()
            assert data.get("error") == "InternalServerError"
            assert data.get("detail") == "An internal server error occurred."
            # Confirm no secret or code path was returned
            assert "supersecret" not in resp.text
            assert_no_information_leak(resp)

    @pytest.mark.asyncio
    async def test_not_found_sanitized(self, fuzz_client: AsyncClient) -> None:
        """Verify 404 responses are clean and sanitized."""
        resp = await fuzz_client.get("/non_existent_adversarial_probe")
        assert resp.status_code == 404
        assert_no_information_leak(resp)
