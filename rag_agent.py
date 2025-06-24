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
    relevance_grade: str         

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




def grade_documents_node(state: AgentState) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    Updates the 'relevance_grade' field in the AgentState.
    """
    print("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("No documents to grade.")
        return {"relevance_grade": "no"} # Or some other indicator like 'error' or 'no_docs'

    # Format documents for the grader LLM
    # We can reuse format_docs or create a slightly different version if needed for the grader prompt
    formatted_docs = format_docs(documents)

    # Grader prompt
    # Inside grade_documents_node
    grader_prompt_template = """You are a grader assessing the relevance of retrieved documents to a user question.
    Consider if the documents contain information that is helpful, related, or could contribute to answering the question.
    It's okay if the documents don't provide a complete, direct answer, as long as they offer some relevant information.
    Respond with 'yes' if the documents are relevant or related, and 'no' if they are clearly off-topic or unhelpful.
    Provide only 'yes' or 'no'.

    Retrieved Documents:
    {documents_context}

    User Question:
    {question}

    Are these documents relevant or related to the question? (yes/no):
    """
    grader_prompt = ChatPromptTemplate.from_template(grader_prompt_template)

    # Grader LLM call (can use the same LLM instance or a different one if needed)
    # For simplicity, we use the same LLM instance 'llm' initialized globally.
    # If you wanted different settings (e.g., lower temperature for grading), you could define a new LLM instance.
    grader_chain = grader_prompt | llm | StrOutputParser()

    print(f"Grading relevance for question: '{question}'")
    try:
        grade_output = grader_chain.invoke({
            "documents_context": formatted_docs,
            "question": question
        })
        grade = grade_output.strip().lower()
        print(f"Relevance grade from LLM: '{grade}'")

        if "yes" in grade:
            print("Relevance Grade: YES")
            return {"relevance_grade": "yes"}
        elif "no" in grade:
            print("Relevance Grade: NO")
            return {"relevance_grade": "no"}
        else:
            print(f"Warning: Unexpected grade format: '{grade}'. Defaulting to 'no'.")
            return {"relevance_grade": "no"} # Fallback for unexpected LLM output

    except Exception as e:
        print(f"Error during relevance grading: {e}")
        return {"relevance_grade": "no"} # Default to 'no' on error
    
def handle_irrelevance_node(state: AgentState) -> dict:
    """
    Handles the case where documents are deemed irrelevant.
    Sets a specific message in the 'generation' field.
    """
    print("---NODE: HANDLE IRRELEVANCE---")
    # We can set the generation directly here.
    # In more complex scenarios, this node might trigger other actions.
    return {"generation": "Sorry, I could not find relevant documents to answer your question."}

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


# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow...")
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node) # New node
workflow.add_node("generate", generate_node)
workflow.add_node("handle_irrelevance", handle_irrelevance_node) # New node

# Set the entry point
workflow.set_entry_point("retrieve")

# Define edges and conditional logic
workflow.add_edge("retrieve", "grade_documents") # retrieve -> grade_documents

# The 'decide_to_generate_or_handle_irrelevance' function will determine the next node.
def decide_to_generate_or_handle_irrelevance(state: AgentState) -> str:
    """
    Based on 'relevance_grade', decides the next step.
    Returns the name of the next node to execute.
    """
    print("---CONDITIONAL EDGE: DECIDE GENERATE/IRRELEVANT---")
    relevance_grade = state.get("relevance_grade", "no") # Default to "no" if not set

    if relevance_grade == "yes":
        print("Decision: Relevant documents found. Proceed to generate.")
        return "generate"  # Name of the node to go to
    else:
        print("Decision: Documents are not relevant. Handle irrelevance.")
        return "handle_irrelevance" # Name of the node to go to

workflow.add_conditional_edges(
    "grade_documents",  # The node from which the conditional logic starts
    decide_to_generate_or_handle_irrelevance, # The function that returns the next node's name
    { # A dictionary mapping the condition's output to node names
        "generate": "generate",
        "handle_irrelevance": "handle_irrelevance"
    }
)

# Edges to END
workflow.add_edge("generate", END) # If generation happens, then end.
workflow.add_edge("handle_irrelevance", END) # If irrelevance is handled, then end.

# Compile the graph
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

        initial_state_input = {"question": user_query, "documents": [], "generation": "", "relevance_grade": "" } # Ensure all keys are present

        print("Invoking RAG agent graph...")
        try:
            final_state = app.invoke(initial_state_input)

            print("\n--- Agent Response ---")
            print(final_state.get("generation", "No generation found."))

        except Exception as e:
            print(f"An error occurred while running the agent: {e}")
            import traceback
            traceback.print_exc() 