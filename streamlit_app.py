# streamlit_app.py
import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Handle the API Key ---
# Set the OpenAI API key from Streamlit's secrets.
# This MUST be the first Streamlit command.
st.set_page_config(page_title="RAG Agent 🤖", page_icon="🤖")

try:
    # Set the API key as an environment variable *before* importing the agent
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OPENAI_API_KEY secret is not set. Please add it in your Streamlit settings.", icon="🚨")
    st.stop()

# --- 2. Import Your Agent ---
# Now that the API key is set, we can safely import the agent
from rag_agent import app

# --- 3. Set Up the Streamlit UI ---
st.title("Conversational RAG Agent 🤖")
st.caption("Ask me questions about your documents. I can check for relevance, rewrite queries, and remember our conversation.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle New Chat Input ---
if prompt := st.chat_input("What is LangGraph?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert Streamlit history to the LangGraph BaseMessages format
    chat_history_messages = []
    for msg in st.session_state.messages[:-1]: # Get all messages *except* the new one
        if msg["role"] == "user":
            chat_history_messages.append(HumanMessage(content=msg["content"]))
        else:
            chat_history_messages.append(AIMessage(content=msg["content"]))

    # Prepare the input for the LangGraph agent
    initial_state_input = {
        "question": prompt,
        "original_question": prompt,
        "chat_history": chat_history_messages,
        "documents": [],
        "generation": "",
        "relevance_grade": "",
        "hallucination_grade": "",
        "rewrite_attempts": 0
    }

    # Invoke the agent and get the final response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (this may take a moment)"):
            try:
                # Use .invoke() to get the final state
                final_state = app.invoke(initial_state_input)
                response_text = final_state.get("generation", "Sorry, I couldn't generate a response.")
                
                st.markdown(response_text)
                # Add the response to session state
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🔥")