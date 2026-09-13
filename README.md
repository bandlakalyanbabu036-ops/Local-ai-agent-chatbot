# Local AI Agent Chatbot

A local AI chatbot and agent built with Python, Streamlit, Ollama, and a local language model. The application keeps conversation memory in the current Streamlit session and can decide when to use safe local tools.

Implemented capabilities:

- Local chat through Ollama and `llama3.2:3b`
- Streamlit chat interface with conversation history
- Agent tool calling
- Safe arithmetic calculator
- Local date and time tool
- Optional Langfuse observability
- Friendly handling for missing servers, invalid models, empty prompts, and unavailable observability

## Architecture

```text
User
	|
	v
Streamlit UI (app.py)
	|
	v
Chat Manager
	|
	v
AI Agent (src/agent.py)
	|
	v
Ollama Client (src/ollama_client.py)
	|
	v
Ollama Server
	|
	v
Local LLM (llama3.2:3b)
	|
	+--> Tool selection
					|
					+--> Calculator
					+--> Date/time tool
					|
					v
			 Tool result
					|
					v
			 Final response
```

Langfuse is optional and its observability hooks are attached to the shared Ollama `chat_completion()` method. This means the agent's tool-decision and final-response calls are recorded when valid credentials are configured. No telemetry is required for the chatbot to work.

## Technologies

### Python

Python provides the application logic, tool implementations, configuration loading, and tests. Its standard library is sufficient for the safe calculator, date/time tool, and unittest-based test suite.

### Streamlit

Streamlit provides the browser chat interface without requiring a separate frontend. It renders messages, accepts prompts, shows a loading spinner, and stores the current conversation in `st.session_state`.

### Ollama

Ollama runs the language model locally and provides a local API for chat and tool calls. This avoids sending prompts to a cloud LLM provider and avoids cloud API keys for the core chatbot.

### Local LLM

The local LLM is the language model that generates responses and decides whether one of the declared tools is needed. This project uses the locally installed `llama3.2:3b` model.

### AI Agent

The application is agentic because the model can request a tool instead of always answering directly. Python executes the requested safe tool, sends the result back to the model, and displays the model's final response.

### Tools

- **Calculator:** Parses basic arithmetic with Python's AST module and approved operators. It does not use unrestricted `eval()`.
- **Date/time:** Returns the computer's current local date and time using Python's standard library. It does not call an external API.

### Conversation Memory

Messages are persisted in a local SQLite database at `data/conversations.db` by default. The path can be changed with `CONVERSATION_DATABASE_PATH`. On startup, the default conversation's user and assistant messages are loaded into `st.session_state.messages`, then passed through the chat manager and agent to Ollama as context. New user messages and successful assistant responses are saved with their role, content, conversation ID, and UTC timestamp.

The Clear Chat button deletes only the messages for the current conversation and resets the in-memory collection. It does not delete the SQLite database or other conversations.

### Langfuse

Langfuse provides optional observability for LLM applications. The generation hook includes the messages, model, response or tool-call output, latency, errors, and Ollama token usage when available. Langfuse is optional: missing, invalid, or unavailable configuration produces a warning and does not prevent Ollama chat from working.

## Installation

These commands assume Windows PowerShell or Command Prompt and Python is available on `PATH`.

### 1. Clone or download

Clone the repository:

```powershell
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>\ai_chatbot
```

Alternatively, download the project as a ZIP file, extract it, and open the `ai_chatbot` folder in a terminal.

### 2. Create the virtual environment

From the `ai_chatbot` directory:

```powershell
python -m venv .venv
```

### 3. Activate it

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Python dependencies

```powershell
pip install -r requirements.txt
```

The requirements are `streamlit`, `ollama`, `python-dotenv`, and `langfuse`.

### 5. Install and start Ollama

If Ollama is not installed, install it manually from the official Ollama website: <https://ollama.com/>.

Verify the installation:

```powershell
ollama --version
```

Start the Ollama server if necessary:

```powershell
ollama serve
```

### 6. Use the required model

The project is configured for `llama3.2:3b`:

```powershell
ollama pull llama3.2:3b
ollama list
```

The model supports the tool-calling capability used by the agent.

### 7. Configure environment variables

Create a local `.env` file from the template:

```powershell
Copy-Item .env.example .env
```

Set the Ollama model in `.env`:

```text
OLLAMA_MODEL=llama3.2:3b
```

Langfuse is optional. Add valid local values only if observability is wanted:

```text
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
```

Never commit `.env` or expose the secret key. `.env.example` contains placeholders only.

### 8. Run the chatbot

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, normally <http://localhost:8501>.

## Example Usage

```text
User: What is Python?
```

The agent can answer this directly through the local LLM.

```text
User: What is 25 * 18?
Assistant: The answer to 25 * 18 is 450.
```

The agent can use the calculator tool.

```text
User: What is today's date?
```

The agent can use the local date/time tool.

```text
User: My name is Kalyan.
User: What is my name?
```

The second request uses the conversation history loaded from SQLite.

## Testing

The project uses Python's built-in `unittest` runner. Pytest is not configured or required.

Run the fast unit tests from the `ai_chatbot` directory:

```powershell
python -m unittest tests.test_chat -v
```

The tests cover configuration loading, conversation-history forwarding, safe tools, agent tool execution, empty prompts, connection failures, and invalid models. They use mocks and do not require a Langfuse account or model downloads.

For a real Ollama smoke test:

```powershell
python -c "from src.ollama_client import generate_response; print(generate_response('Explain Python in one sentence.'))"
```

During development, the real application was also verified with normal questions, calculator requests, date/time requests, conversation memory, and Clear Chat.

## Error Handling

- **Ollama unavailable:** The client raises a simple connection error, and Streamlit displays it without a long traceback.
- **Invalid model:** The client reports that the configured model was not found. The model name is not silently replaced.
- **Langfuse unavailable:** The application shows a short warning and continues using Ollama. Agent tool calls also continue because telemetry is optional.
- **Missing Langfuse configuration:** Langfuse is disabled when its variables are empty; chatbot responses continue normally.
- **Empty input:** Empty prompts are not sent to Ollama. The client rejects blank text if called directly.

## Project Structure

```text
AI_Chatbot/
└── ai_chatbot/
		├── app.py
		├── requirements.txt
		├── .env
		├── .env.example
		├── .gitignore
		├── README.md
		├── src/
		│   ├── __init__.py
		│   ├── agent.py
		│   ├── chat_manager.py
		│   ├── config.py
		│   ├── observability.py
		│   └── ollama_client.py
		└── tests/
				├── __init__.py
				└── test_chat.py
```

The local `.venv` directory and Python cache directories are development artifacts and are not part of the source structure described for GitHub.

## Interview Discussion

### 1. Why Ollama?

Ollama makes it possible to run the selected LLM locally, avoiding cloud LLM API keys for the chatbot's core requests.

### 2. Why Streamlit?

Streamlit provides a simple Python-first chat UI with input controls, message rendering, session state, and loading indicators.

### 3. What is an AI agent?

An AI agent is an application where the model can decide to use an available capability before producing its final answer.

### 4. What is tool calling?

Tool calling is when the model returns a structured request for a named function and arguments. Python executes the approved function and sends the result back to the model.

### 5. How does the calculator work?

The calculator parses an expression with `ast.parse()` and recursively evaluates only approved numeric and arithmetic AST nodes. It never executes unrestricted Python input.

### 6. How does the date/time tool work?

It uses `datetime.now().astimezone()` to read the computer's local date and time without an external service.

### 7. How does conversation memory work?

Streamlit stores user and assistant message dictionaries in `st.session_state.messages`. That list is passed through the chat manager and agent on later requests.

### 8. What is Langfuse?

Langfuse is an observability platform. This project contains an optional generation-observation hook for messages, model, response or tool-call output, latency, errors, and Ollama token usage when available.

### 9. Why is Langfuse optional?

The chatbot should remain useful when telemetry is not configured or the telemetry service is unavailable. Langfuse failures are isolated from Ollama requests, and the current agent does not depend on telemetry.

### 10. What happens when Ollama is unavailable?

The client converts the connection failure into a clear `OllamaClientError`, and Streamlit displays a friendly message instead of a traceback.

### 11. What is the complete request flow?

Streamlit receives the prompt, the chat manager invokes the agent, Ollama decides whether a tool is needed, Python executes an approved tool when requested, the result returns to Ollama, and Streamlit displays the final response. The optional Langfuse generation hook records each actual Ollama call in this flow when configured.

## Resume Description

- Built a local AI agent chatbot with Python, Streamlit, Ollama, and the `llama3.2:3b` local LLM.
- Implemented Ollama tool calling with safe AST-based arithmetic and local date/time tools.
- Added Streamlit session-state conversation memory and a Clear Chat workflow.
- Added optional Langfuse generation observability hooks for prompts, model responses, latency, and errors.
