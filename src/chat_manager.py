from .agent import run_agent


def get_chat_response(
    prompt: str,
    conversation: list[dict[str, str]],
) -> str:
    """Return an assistant response using the current conversation context."""
    return run_agent(prompt, conversation)
