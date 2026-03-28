"""
HTTP client for the Ollama OpenAI-compatible API.

Ollama exposes a `/v1/chat/completions` endpoint that is compatible with the
OpenAI SDK wire format, so we use plain httpx rather than adding a heavy SDK
dependency.  The interface is intentionally minimal; Borya's telemetry module
can wrap `chat()` with a decorator to count tokens and measure latency.
"""

import httpx

from rentals_agents.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from rentals_agents.benchmark import Benchmark


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error or is unreachable."""


def chat(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
) -> str:
    """
    Send a chat completion request to Ollama and return the assistant's text.

    Args:
        model:         Ollama model name, e.g. "qwen2.5-coder:7b".
        system_prompt: System message text.
        user_message:  User turn content.
        temperature:   Sampling temperature (low = more deterministic).

    Returns:
        The raw assistant reply string (may contain JSON, prose, etc.).

    Raises:
        OllamaError: on HTTP errors, timeouts, or unexpected response shape.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    try:
        response = httpx.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT}s "
            f"(model={model})"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {exc}"
        ) from exc

    try:
        data = response.json()
        # Извлекаем токены (OpenAI-совместимый формат)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        Benchmark().add_tokens(prompt_tokens, completion_tokens)

        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise OllamaError(
            f"Unexpected Ollama response shape: {response.text[:300]}"
        ) from exc