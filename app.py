import streamlit as st

from src.chat_manager import get_chat_response
from src.config import CONVERSATION_DATABASE_PATH, CONVERSATION_ID
from src.conversation_store import ConversationStore, ConversationStoreError
from src.ollama_client import OllamaClientError
from src.observability import LANGFUSE_WARNING


st.set_page_config(page_title="Local AI Chatbot", page_icon="💬")

st.title("Local AI Chatbot")
st.write("Chat with a language model running on your computer.")
if LANGFUSE_WARNING:
    st.warning(LANGFUSE_WARNING)

try:
    conversation_store = ConversationStore(CONVERSATION_DATABASE_PATH)
except ConversationStoreError as error:
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    try:
        st.session_state.messages = conversation_store.load_messages(
            CONVERSATION_ID
        )
    except ConversationStoreError as error:
        st.error(str(error))
        st.session_state.messages = []

if st.button("Clear chat"):
    try:
        conversation_store.clear_conversation(CONVERSATION_ID)
    except ConversationStoreError as error:
        st.error(str(error))
    else:
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your local AI assistant something"):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        conversation_store.save_message(CONVERSATION_ID, "user", prompt)
    except ConversationStoreError as error:
        st.error(str(error))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_chat_response(
                    prompt,
                    st.session_state.messages,
                )
            except OllamaClientError as error:
                st.error(str(error))
            else:
                st.markdown(response)
                assistant_message = {"role": "assistant", "content": response}
                st.session_state.messages.append(assistant_message)
                try:
                    conversation_store.save_message(
                        CONVERSATION_ID,
                        "assistant",
                        response,
                    )
                except ConversationStoreError as error:
                    st.error(str(error))
