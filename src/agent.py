import ast
import math
import operator
import re
from datetime import datetime
from typing import Any

from .ollama_client import chat_completion


class AgentError(RuntimeError):
    """Raised when an agent tool request cannot be completed."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_number(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_number(node.left)
        right = _evaluate_number(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_number(node.operand))
    raise AgentError("Only basic arithmetic is supported by the calculator.")


def calculate(expression: str) -> str:
    """Safely calculate a basic arithmetic expression without eval()."""
    if len(expression) > 100:
        raise AgentError("The calculator expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_number(tree.body)
    except (
        SyntaxError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
    ):
        raise AgentError(
            "The calculator could not evaluate that expression."
        ) from None
    if isinstance(result, float) and not math.isfinite(result):
        raise AgentError("The calculator result is not finite.")
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def current_datetime() -> str:
    """Return the computer's current local date and time."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a basic arithmetic expression safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the computer's current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _run_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "calculator":
        return calculate(arguments.get("expression", ""))
    if name == "current_datetime":
        return current_datetime()
    raise AgentError(f"Unknown agent tool: {name}")


_ARITHMETIC_OPERATOR_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*[+\-*/%]\s*\d+(?:\.\d+)?"
)
_CALCULATION_REQUEST_PATTERN = re.compile(
    r"\b(?:calculate|compute|evaluate|solve)\b[^.!?]*\d",
    re.IGNORECASE,
)
_DATE_TIME_REQUEST_PATTERN = re.compile(
    r"(?:today(?:'s)?\s+date|current\s+(?:date|time)|"
    r"what(?:\s+is|'s)\s+(?:the\s+)?(?:date|time)|"
    r"what\s+time\s+is\s+it|time\s+now|date\s+today)",
    re.IGNORECASE,
)


def _tool_call_is_relevant(prompt: str, tool_name: str) -> bool:
    """Prevent an LLM tool call that does not match the user's request."""
    if tool_name == "calculator":
        return bool(
            _ARITHMETIC_OPERATOR_PATTERN.search(prompt)
            or _CALCULATION_REQUEST_PATTERN.search(prompt)
        )
    if tool_name == "current_datetime":
        return bool(_DATE_TIME_REQUEST_PATTERN.search(prompt))
    return False


def run_agent(
    prompt: str,
    conversation: list[dict[str, str]],
) -> str:
    """Let Ollama decide whether to use one of the safe local tools."""
    messages: list[dict[str, Any]] = list(conversation)
    response = chat_completion(messages, TOOLS)
    tool_calls = response.message.tool_calls or []
    if not tool_calls:
        return response.message.content

    if not all(
        _tool_call_is_relevant(prompt, tool_call.function.name)
        for tool_call in tool_calls
    ):
        direct_response = chat_completion(messages)
        return direct_response.message.content

    messages.append(response.message.model_dump(exclude_none=True))
    for tool_call in tool_calls:
        try:
            result = _run_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
        except AgentError as error:
            result = f"Tool error: {error}"
        messages.append(
            {
                "role": "tool",
                "content": result,
            }
        )

    final_response = chat_completion(messages, TOOLS)
    return final_response.message.content
