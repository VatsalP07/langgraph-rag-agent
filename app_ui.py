# In app_ui.py

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage


from rag_agent import app

print("Successfully imported LangGraph agent application.")

# THIS IS THE FUNCTION YOU NEED TO REPLACE
def agent_chat_interface(message: str, history: list):
    """
    Wrapper function to interface the Gradio Chatbot with the LangGraph agent.
    """
    print(f"User message: '{message}'")
    print(f"Current history from Gradio: {history}")

    # Convert Gradio's history format to the list of BaseMessages our agent expects.
    chat_history_messages = []
    for user_msg, ai_msg in history:
        chat_history_messages.append(HumanMessage(content=user_msg))
        chat_history_messages.append(AIMessage(content=ai_msg))

    # Prepare the input for the LangGraph agent
    # The keys must match the fields in our AgentState
    initial_state_input = {
        "question": message,
        "original_question": message,
        "chat_history": chat_history_messages,
        "documents": [],
        "generation": "",
        "relevance_grade": "",
        "hallucination_grade": "",
        "rewrite_attempts": 0
    }

    # Stream the response from the agent
    # We accumulate the state as we stream to get the final complete state.
    accumulated_state = {}
    for chunk in app.stream(initial_state_input):
        # The output of `stream` is a dictionary where keys are node names
        # and values are the state *after* that node ran.
        print(f"--- Stream Chunk ---\n{chunk}\n--------------------")
        
        # Merge the updates from the current chunk into our accumulated state.
        # This ensures we always have the most complete picture of the state.
        for key, value in chunk.items():
            accumulated_state.update(value)

    # Extract the final generation from the fully accumulated state
    if "generation" in accumulated_state and accumulated_state["generation"]:
        response_text = accumulated_state["generation"]
    else:
        response_text = "Sorry, I could not generate a response. Please try rephrasing."
        print("Error: 'generation' key not found in the final accumulated state.")

    return response_text


# THIS IS THE PART THAT CALLS THE FUNCTION
if __name__ == "__main__":
    print("Launching Gradio Chat Interface...")

    # Create the Gradio ChatInterface
    chat_ui = gr.ChatInterface(
        fn=agent_chat_interface,
        title="RAG Agent",
        description="Ask me questions about the provided documents. I can check for relevance, rewrite queries, check my own answers, and remember our conversation.",
        examples=[
            ["What is LangGraph?"],
            ["How does ChromaDB work?"],
            ["What is RAG?"]
        ],
        cache_examples=False
    )

    # Launch the UI
    chat_ui.launch()