import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence 
from langchain_core.messages import BaseMessage # For chat history 
import operator # For chat history 

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # Updated line
from langchain_chroma import Chroma                     # Updated line
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.documents import Document 

# --- Configuration (Copied and adapted from simple_rag.py) ---
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "gpt-4o-mini"
RELEVANCE_THRESHOLD = 0.5 

# --- LangGraph Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our RAG agent.
    This dictionary will be passed between nodes in the graph.
    """
    question: str               # The input question from the user
    documents: List[Document]   # List of retrieved documents
    generation: str             # The LLM-generated answer

load_dotenv()

# Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

RAG_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_node(state: AgentState) -> dict:
    """
    Retrieves documents relevant to the question from the vector store.
    Updates the 'documents' field in the AgentState.
    """
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    print(f"Retrieving documents for question: '{question}'")

    # Retrieve documents
    # The retriever was initialized globally
    retrieved_docs = retriever.invoke(question)

    # For now, we're not formatting them here, just storing the Document objects
    print(f"Retrieved {len(retrieved_docs)} documents.")
    return {"documents": retrieved_docs} # Return a dictionary to update the state


def generate_node(state: AgentState) -> dict:
    """
    Generates an answer using the LLM based on the question and retrieved documents.
    Updates the 'generation' field in the AgentState.
    """
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"] # Documents are now List[Document]

    if not documents:
        print("No documents found to generate an answer.")
        # Handle cases where no documents are retrieved (e.g., set a default message)
        # For now, we'll let it proceed, but the LLM might struggle or say it doesn't know.
        # A more robust solution might be a conditional edge after retrieval.
        formatted_context = "No context provided."
    else:
        # Format the documents for the prompt
        formatted_context = format_docs(documents)

    print(f"Generating answer for question: '{question}' with provided context.")

    # Create the RAG chain for this specific invocation
    # (prompt | llm | parser)
    # We could also define this chain globally if it doesn't change
    generation_chain = rag_prompt | llm | StrOutputParser()

    # Invoke the chain
    answer = generation_chain.invoke({
        "context": formatted_context,
        "question": question
    })

    print(f"Generated answer: '{answer}'")
    return {"generation": answer} # Update the 'generation' field in the state




from langgraph.graph import StateGraph, END

# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow...")
workflow = StateGraph(AgentState)


workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_edge("generate", END)

# Compile the graph into a runnable application
print("Compiling LangGraph...")
app = workflow.compile()
print("LangGraph compiled successfully.")

if __name__ == "__main__":
    print("\n--- Running RAG Agent (LangGraph) ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_query = input("\nAsk a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        initial_state_input = {"question": user_query, "documents": [], "generation": ""} # Ensure all keys are present

        print("Invoking RAG agent graph...")
        try:
            final_state = app.invoke(initial_state_input)

            print("\n--- Agent Response ---")
            print(final_state.get("generation", "No generation found."))

        except Exception as e:
            print(f"An error occurred while running the agent: {e}")
            import traceback
            traceback.print_exc() 