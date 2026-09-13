import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import ANY, patch

from ollama import ResponseError

from src import config
from src.agent import calculate, current_datetime, run_agent
from src.conversation_store import ConversationStore
from src.ollama_client import (
    OllamaClientError,
    chat_completion,
    generate_response,
)


class OllamaClientTests(unittest.TestCase):
    def test_configuration_loads_model(self):
        self.assertEqual(config.OLLAMA_MODEL, "llama3.2:3b")

    @patch("src.ollama_client.finish_generation")
    @patch("src.ollama_client.start_generation", return_value=None)
    @patch("src.ollama_client.Client")
    def test_success_sends_conversation_history(
        self,
        client_class,
        _start_generation,
        _finish_generation,
    ):
        client = client_class.return_value
        client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="Your name is Kalyan.")
        )
        conversation = [
            {"role": "user", "content": "My name is Kalyan."},
            {"role": "assistant", "content": "Nice to meet you."},
            {"role": "user", "content": "What is my name?"},
        ]

        response = generate_response("What is my name?", conversation)

        self.assertEqual(response, "Your name is Kalyan.")
        client.chat.assert_called_once_with(
            model="llama3.2:3b",
            messages=conversation,
            tools=None,
        )

    @patch("src.ollama_client.Client")
    def test_connection_failure_has_clear_message(self, client_class):
        client_class.return_value.chat.side_effect = ConnectionError(
            "Connection refused: localhost:11434"
        )

        with self.assertRaisesRegex(OllamaClientError, "Could not connect"):
            generate_response("Hello")

    @patch("src.ollama_client.Client")
    def test_missing_model_has_clear_message(self, client_class):
        client_class.return_value.chat.side_effect = ResponseError(
            "model 'missing-model' not found", status_code=404
        )

        with self.assertRaisesRegex(
            OllamaClientError,
            "model .* was not found",
        ):
            generate_response("Hello")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(OllamaClientError, "cannot be empty"):
            generate_response("   ")

    @patch("src.ollama_client.finish_generation")
    @patch("src.ollama_client.start_generation", return_value="observation")
    @patch("src.ollama_client.Client")
    def test_chat_completion_records_ollama_usage(
        self,
        client_class,
        start_generation,
        finish_generation,
    ):
        client_class.return_value.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="ready"),
            prompt_eval_count=12,
            eval_count=7,
        )
        messages = [{"role": "user", "content": "Say ready."}]

        response = chat_completion(messages)

        self.assertEqual(response.message.content, "ready")
        start_generation.assert_called_once_with(messages, "llama3.2:3b")
        finish_generation.assert_called_once_with(
            "observation",
            {"content": "ready"},
            None,
            ANY,
            {"input": 12, "output": 7, "total": 19},
        )

    def test_safe_tools(self):
        self.assertEqual(calculate("25 * 18"), "450")
        self.assertEqual(calculate("1250 / 5 + 30"), "280")
        self.assertTrue(current_datetime())

    @patch("src.agent.chat_completion")
    def test_agent_executes_tool_and_returns_final_response(self, chat):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="calculator",
                arguments={"expression": "25 * 18"},
            )
        )
        first_response = SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=[tool_call],
                content="",
                model_dump=lambda exclude_none: {
                    "role": "assistant",
                    "tool_calls": [],
                },
            )
        )
        final_response = SimpleNamespace(
            message=SimpleNamespace(tool_calls=[], content="450")
        )
        chat.side_effect = [first_response, final_response]

        result = run_agent(
            "What is 25 * 18?",
            [{"role": "user", "content": "What is 25 * 18?"}],
        )

        self.assertEqual(result, "450")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(
            chat.call_args_list[1].args[0][-1],
            {"role": "tool", "content": "450"},
        )

    @patch("src.agent.chat_completion")
    def test_greeting_rejects_irrelevant_calculator_call(self, chat):
        calculator_call = SimpleNamespace(
            function=SimpleNamespace(
                name="calculator",
                arguments={"expression": "Hello, how are you?"},
            )
        )
        chat.side_effect = [
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[calculator_call],
                    content="",
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[],
                    content="I am doing well, thank you.",
                )
            ),
        ]
        conversation = [{"role": "user", "content": "Hello, how are you?"}]

        result = run_agent("Hello, how are you?", conversation)

        self.assertEqual(result, "I am doing well, thank you.")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args_list[1].args, (conversation,))

    @patch("src.agent.chat_completion")
    def test_normal_question_rejects_irrelevant_date_call(self, chat):
        date_call = SimpleNamespace(
            function=SimpleNamespace(name="current_datetime", arguments={})
        )
        chat.side_effect = [
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=[date_call], content="")
            ),
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[],
                    content="Python is a programming language.",
                )
            ),
        ]
        conversation = [{"role": "user", "content": "What is Python?"}]

        result = run_agent("What is Python?", conversation)

        self.assertEqual(result, "Python is a programming language.")
        self.assertEqual(chat.call_args_list[1].args, (conversation,))

    @patch("src.agent.current_datetime", return_value="2026-09-13 12:00 UTC")
    @patch("src.agent.chat_completion")
    def test_date_question_allows_date_time_tool(
        self,
        chat,
        get_datetime,
    ):
        date_call = SimpleNamespace(
            function=SimpleNamespace(name="current_datetime", arguments={})
        )
        first_response = SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=[date_call],
                content="",
                model_dump=lambda exclude_none: {
                    "role": "assistant",
                    "tool_calls": [],
                },
            )
        )
        final_response = SimpleNamespace(
            message=SimpleNamespace(tool_calls=[], content="2026-09-13 12:00 UTC")
        )
        chat.side_effect = [first_response, final_response]

        result = run_agent(
            "What time is it?",
            [{"role": "user", "content": "What time is it?"}],
        )

        self.assertEqual(result, "2026-09-13 12:00 UTC")
        get_datetime.assert_called_once_with()


class ConversationStoreTests(unittest.TestCase):
    def test_messages_persist_when_store_is_reopened(self):
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "conversations.db"
            store = ConversationStore(database_path)
            store.save_message("default", "user", "My name is Kalyan.")
            store.save_message("default", "assistant", "Nice to meet you.")

            reopened_store = ConversationStore(database_path)

            self.assertEqual(
                reopened_store.load_messages("default"),
                [
                    {"role": "user", "content": "My name is Kalyan."},
                    {"role": "assistant", "content": "Nice to meet you."},
                ],
            )

    def test_clear_removes_only_the_selected_conversation(self):
        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.db")
            store.save_message("default", "user", "Keep this separate.")
            store.save_message("other", "user", "Do not delete this.")

            store.clear_conversation("default")

            self.assertEqual(store.load_messages("default"), [])
            self.assertEqual(
                store.load_messages("other"),
                [{"role": "user", "content": "Do not delete this."}],
            )

    def test_saved_message_has_a_timestamp(self):
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "conversations.db"
            store = ConversationStore(database_path)
            store.save_message("default", "user", "Hello")

            with store._connect() as connection:
                timestamp = connection.execute(
                    "SELECT timestamp FROM messages"
                ).fetchone()[0]

            self.assertTrue(timestamp)


def test_ollama_connection() -> str:
    """Verify that Python can send a prompt to Ollama and receive text."""
    response = generate_response("Explain what Python is in one sentence.")
    assert response.strip(), "Ollama returned an empty response."
    return response


if __name__ == "__main__":
    unittest.main()
