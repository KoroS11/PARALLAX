"""D5: Local LLM judge detector (Hermes 3 / Qwen via Ollama) producing verdict and reason."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from parallax.contracts.detectors import DetectorResult
from parallax.detectors.base import BaseDetector
from parallax.detectors.prompts.judge_prompt import (
    DEFAULT_ATTACK_CLASS,
    JUDGE_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    create_repair_user_prompt,
    format_user_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "hermes3:latest"
DEFAULT_BASE_URL: str = "http://localhost:11434"
DEFAULT_TIMEOUT: float = 10.0
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_SEED: int = 42


class JudgeOutput(BaseModel):
    """Validated structured output schema expected from the LLM judge."""

    model_config = ConfigDict(extra="ignore")

    is_attack: bool
    confidence: float
    attack_class: str = Field(default="none")
    reasoning: str = Field(default="")

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Validate and clamp confidence score into [0.0, 1.0]."""
        num = float(v)
        if num < 0.0 or num > 1.0:
            if -0.05 <= num <= 1.05:
                return max(0.0, min(1.0, num))
            raise ValueError(f"Confidence {num} is outside allowable [0.0, 1.0] range")
        return num


def extract_json_candidate(text: str) -> str:
    """Extract candidate JSON substring from raw model response text.

    Strips optional markdown code fences and isolates the outermost JSON object braces.
    """
    trimmed = text.strip()

    # Strip markdown code blocks (e.g. ```json ... ```)
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        trimmed = "\n".join(lines).strip()

    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed

    # Search for outermost JSON braces if commentary surrounds the JSON
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start != -1 and end != -1 and end > start:
        return trimmed[start : end + 1]

    return trimmed


def parse_judge_output(raw_text: str) -> JudgeOutput:
    """Parse and strictly validate raw judge output against JudgeOutput schema.

    Raises
    ------
    ValueError, json.JSONDecodeError, pydantic.ValidationError
        If text cannot be parsed or does not conform to the schema.
    """
    candidate = extract_json_candidate(raw_text)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise TypeError(f"Parsed JSON is not an object/dict: {type(data)}")
    return JudgeOutput.model_validate(data)


def _extract_content_from_ollama_response(data: Any) -> str:
    """Extract assistant content string from Ollama chat or generate response."""
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", ""))
        if "response" in data:
            return str(data.get("response", ""))
        if "content" in data:
            return str(data.get("content", ""))
    return str(data)


class LLMJudgeDetector(BaseDetector):
    """D5: Local LLM judge detector (Hermes 3 / Qwen via Ollama).

    Conforms to the PARALLAX Detector protocol:
      name: str = "d5_judge"
      async def predict(self, text: str) -> DetectorResult

    Provides structured JSON evaluation with strict data isolation, prompt-injection
    resistance against meta-jailbreaks, and automated JSON repair retry.
    """

    def __init__(
        self,
        name: str = "d5_judge",
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT,
        endpoint: str = "/api/chat",
        max_retries: int = 1,
        system_prompt: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.timeout = float(timeout)
        self.endpoint = endpoint
        self.max_retries = int(max_retries)
        self.system_prompt = system_prompt or JUDGE_SYSTEM_PROMPT
        self._client = client
        self._owns_client = client is None

    async def _call_ollama(
        self,
        client: httpx.AsyncClient,
        messages: list[dict[str, str]],
    ) -> str:
        """Call Ollama API endpoint and return raw assistant response text."""
        url = f"{self.base_url}{self.endpoint}"

        if "chat" in self.endpoint:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                },
                "format": "json",
            }
        else:
            # Fallback for /api/generate
            prompt_content = "\n\n".join(
                f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                for m in messages
            )
            payload = {
                "model": self.model_name,
                "prompt": prompt_content,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                },
                "format": "json",
            }

        response = await client.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return _extract_content_from_ollama_response(data)

    async def _call_repair(
        self,
        client: httpx.AsyncClient,
        malformed_output: str,
    ) -> str:
        """Execute a clean JSON repair request to Ollama."""
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": create_repair_user_prompt(malformed_output)},
        ]
        return await self._call_ollama(client, repair_messages)

    async def predict(self, text: str) -> DetectorResult:
        """Asynchronously run LLM judge prediction on the given prompt text."""
        start_time = time.perf_counter()

        # Input validation: abstain on empty or non-string input
        if not isinstance(text, str) or not text.strip():
            latency_ms = self.measure_latency_ms(start_time)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=round(latency_ms, 3),
                error="Empty or invalid input prompt",
            )

        user_content = format_user_prompt(text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        client = self._client
        owns_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            owns_client = True

        try:
            try:
                raw_output = await self._call_ollama(client, messages)
            except httpx.TimeoutException:
                latency_ms = self.measure_latency_ms(start_time)
                return DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=round(latency_ms, 3),
                    error="Ollama request timed out",
                )
            except Exception as e:  # noqa: BLE001
                latency_ms = self.measure_latency_ms(start_time)
                return DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=round(latency_ms, 3),
                    error=f"Ollama request error: {e}",
                )

            # Output parsing & recovery
            parsed: JudgeOutput | None = None
            try:
                parsed = parse_judge_output(raw_output)
            except Exception:  # noqa: BLE001
                # If output is malformed, attempt one clean JSON repair retry
                if self.max_retries > 0:
                    try:
                        repair_output = await self._call_repair(client, raw_output)
                        parsed = parse_judge_output(repair_output)
                    except Exception:  # noqa: BLE001
                        parsed = None

            if parsed is None:
                latency_ms = self.measure_latency_ms(start_time)
                return DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=round(latency_ms, 3),
                    error="Malformed judge output",
                )

            is_attack = bool(parsed.is_attack)
            score = round(float(parsed.confidence), 4)

            if is_attack:
                attack_class = (
                    parsed.attack_class.strip()
                    if parsed.attack_class
                    else DEFAULT_ATTACK_CLASS
                )
                if attack_class.lower() == "none":
                    attack_class = DEFAULT_ATTACK_CLASS
            else:
                attack_class = "none"

            latency_ms = self.measure_latency_ms(start_time)
            return DetectorResult(
                name=self.name,
                is_attack=is_attack,
                score=score,
                attack_class=attack_class,
                latency_ms=round(latency_ms, 3),
                error=None,
            )
        finally:
            if owns_client:
                await client.aclose()

    def predict_sync(self, text: str) -> DetectorResult:
        """Synchronously execute predict for environments without an active event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(lambda: asyncio.run(self.predict(text))).result()
        return asyncio.run(self.predict(text))

    async def close(self) -> None:
        """Close managed HTTP client if owned."""
        if self._client is not None and not self._owns_client:
            # Client injected externally; do not close
            return
        if self._client is not None and self._owns_client:
            await self._client.aclose()


# Verify protocol compliance
assert issubclass(LLMJudgeDetector, BaseDetector)
