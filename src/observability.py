from typing import Any

from langfuse import Langfuse

from .config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)


def _create_client() -> tuple[Langfuse | None, str | None]:
    """Create one optional Langfuse client without exposing credentials."""
    if not all(
        (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)
    ):
        return None, (
            "Langfuse is not configured. Chat will continue without "
            "observability."
        )

    try:
        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            base_url=LANGFUSE_BASE_URL,
        )
        if not client.auth_check():
            return None, (
                "Langfuse authentication failed. Chat will continue without "
                "observability."
            )
        return client, None
    except Exception:
        return None, (
            "Langfuse could not be initialized. Chat will continue without "
            "observability."
        )


LANGFUSE_CLIENT, LANGFUSE_WARNING = _create_client()


def start_generation(
    messages: list[dict[str, object]],
    model: str,
) -> Any | None:
    """Start an observation for one actual Ollama generation request."""
    if LANGFUSE_CLIENT is None:
        return None

    try:
        return LANGFUSE_CLIENT.start_observation(
            name="ollama-chat",
            as_type="generation",
            input={"messages": messages},
            model=model,
        )
    except Exception:
        return None


def finish_generation(
    observation: Any | None,
    response: Any | None,
    error: str | None,
    latency_ms: float,
    usage: dict[str, int] | None = None,
) -> None:
    """Record the result and flush it without affecting the chatbot."""
    if observation is None or LANGFUSE_CLIENT is None:
        return

    try:
        metadata = {"latency_ms": round(latency_ms, 2)}
        if error:
            observation.update(
                output=None,
                metadata=metadata,
                level="ERROR",
                status_message=error,
            )
        else:
            observation.update(
                output=response,
                metadata=metadata,
                usage_details=usage,
            )
        observation.end()
    except Exception:
        return
    finally:
        try:
            LANGFUSE_CLIENT.flush()
        except Exception:
            pass
