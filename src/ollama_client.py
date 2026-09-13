import time
from typing import Any

from ollama import Client, ResponseError

from .config import OLLAMA_MODEL
from .observability import finish_generation, start_generation


class OllamaClientError(RuntimeError):
    """Raised when a request to the local Ollama server cannot be completed."""


def chat_completion(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
):
    """Send a chat completion request to Ollama, optionally with tools."""
    if not OLLAMA_MODEL:
        raise OllamaClientError(
            "OLLAMA_MODEL is not configured. Add it to the project's "
            ".env file."
        )
    observation = start_generation(messages, OLLAMA_MODEL)
    started_at = time.perf_counter()

    try:
        client = Client(host="http://localhost:11434")
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )
        response_message = _response_message(response)
        usage = _usage_from_response(response)
        finish_generation(
            observation,
            response_message,
            None,
            (time.perf_counter() - started_at) * 1000,
            usage,
        )
        return response
    except ResponseError as error:
        error_text = str(error)
        finish_generation(
            observation,
            None,
            error_text,
            (time.perf_counter() - started_at) * 1000,
        )
        if error.status_code == 404 or "not found" in error_text.lower():
            raise OllamaClientError(
                f"The Ollama model '{OLLAMA_MODEL}' was not found."
            ) from None
        raise OllamaClientError(
            f"Ollama returned an error: {error_text}"
        ) from None
    except Exception as error:
        error_text = str(error).lower()
        finish_generation(
            observation,
            None,
            "Unexpected Ollama error",
            (time.perf_counter() - started_at) * 1000,
        )
        if any(
            term in error_text
            for term in ("connect", "connection", "refused", "localhost:11434")
        ):
            raise OllamaClientError(
                "Could not connect to Ollama. Make sure the Ollama server "
                "is running."
            ) from None
        raise OllamaClientError(
            "An unexpected error occurred while contacting Ollama."
        ) from None


def _usage_from_response(response: Any) -> dict[str, int] | None:
    """Map Ollama token counts to Langfuse usage fields when available."""
    input_tokens = getattr(response, "prompt_eval_count", None)
    output_tokens = getattr(response, "eval_count", None)
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }


def _response_message(response: Any) -> dict[str, object]:
    """Serialize a real Ollama message or a simple test double."""
    message = response.message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    result = {"content": getattr(message, "content", "")}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def generate_response(
    prompt: str,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """Send a prompt to the configured Ollama model and return its response."""
    if not prompt.strip():
        raise OllamaClientError("The prompt cannot be empty.")

    if not OLLAMA_MODEL:
        raise OllamaClientError(
            "OLLAMA_MODEL is not configured. Add it to the project's "
            ".env file."
        )

    messages = conversation or [{"role": "user", "content": prompt}]
    try:
        response = chat_completion(messages)
        generated_text = response.message.content
        return generated_text
    except OllamaClientError:
        raise
