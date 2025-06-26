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
MAX_REWRITE_ATTEMPTS = 2


# --- LangGraph Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our RAG agent.
    This dictionary will be passed between nodes in the graph.
    """
    question: str               # The input question from the user
    original_question:str       # For referencing the original question
    documents: List[Document]   # List of retrieved documents
    generation: str             # The LLM-generated answer
    relevance_grade: str       
    rewrite_attempts:int    #how many times the question has been rewritten
    hallucination_grade: str 

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
    
def transform_query_node(state: AgentState) -> dict:
    """
    Transforms the original query to improve retrieval results if initial documents were irrelevant.
    Updates the 'question' field with the new query and increments 'rewrite_attempts'.
    """
    print("---NODE: TRANSFORM QUERY---")
    original_question = state["original_question"] # Use the original question as base for rewriting
    current_question = state["question"]           # The question that led to irrelevant docs
    rewrite_attempts = state.get("rewrite_attempts", 0)

    print(f"Original question: '{original_question}'")
    print(f"Current (failed) question: '{current_question}'")
    print(f"Rewrite attempt number: {rewrite_attempts + 1}")


    transform_prompt_template = """You are a query rewriting expert.
    The user's original question was: '{original_question}'
    A previous search attempt using the question '{current_question}' did not yield relevant documents.
    Your task is to rewrite the *original question* to be clearer, more specific, or to use different keywords
    that might improve search results against a general knowledge base.
    Do not just add "explain" or "tell me about". Try to rephrase substantively.
    If you think the original question is already very good and cannot be improved, you can return the original question.

    Original Question: {original_question}
    Potentially Failed Question (for context, do not just slightly alter this): {current_question}

    Rewritten Question:
    """
    transform_prompt = ChatPromptTemplate.from_template(transform_prompt_template)

    # LLM call for transformation
    # We can use the same global 'llm' instance.
    transformer_chain = transform_prompt | llm | StrOutputParser()

    try:
        rewritten_question = transformer_chain.invoke({
            "original_question": original_question,
            "current_question": current_question
        })
        print(f"Rewritten question from LLM: '{rewritten_question.strip()}'")

        # Update state with the new question and incremented attempt count
        return {
            "question": rewritten_question.strip(), # This now becomes the 'current' question for the next retrieval
            "rewrite_attempts": rewrite_attempts + 1
        }
    except Exception as e:
        print(f"Error during query transformation: {e}")
        # If transformation fails, maybe we should just give up or revert to original?
        # For now, let's just keep the current question and increment attempts to avoid infinite loop on error.
        return {
            "question": current_question, # Fallback to current question on error
            "rewrite_attempts": rewrite_attempts + 1
        }
    
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
def decide_next_step_after_grading(state: AgentState) -> str:
    """
    Based on 'relevance_grade' and 'rewrite_attempts', decides the next step.
    """
    print("---CONDITIONAL EDGE: DECIDE AFTER GRADING---")
    relevance_grade = state.get("relevance_grade", "no")
    rewrite_attempts = state.get("rewrite_attempts", 0)

    if relevance_grade == "yes":
        print("Decision: Relevant documents found. Proceed to generate.")
        return "generate"
    else: # Documents are not relevant
        print("Decision: Documents are NOT relevant.")
        if rewrite_attempts < MAX_REWRITE_ATTEMPTS:
            print(f"Rewrite attempts ({rewrite_attempts}) < max ({MAX_REWRITE_ATTEMPTS}). Proceed to transform query.")
            return "transform_query"
        else:
            print(f"Max rewrite attempts ({MAX_REWRITE_ATTEMPTS}) reached. Handle irrelevance (cannot answer).")
            return "handle_irrelevance"
        
def grade_generation_node(state: AgentState) -> dict:
    """
    Determines whether the generated answer is grounded in the provided documents
    and addresses the original question.
    Updates the 'hallucination_grade' field in the AgentState.
    """
    print("---NODE: GRADE GENERATION (HALLUCINATION/GROUNDEDNESS)---")
    original_question = state["original_question"] # Check against the original intent
    documents = state["documents"]
    generation = state["generation"]

    if not documents: # Should ideally not happen if relevance check passed, but as a safeguard
        print("No documents provided for generation grading. Marking as not grounded.")
        return {"hallucination_grade": "no"}
    if not generation: # If generation failed or is empty
         print("No generation to grade. Marking as not graded or error.")
         return {"hallucination_grade": "no"} # Or a different status like 'not_graded'

    formatted_docs = format_docs(documents)

    # Grader prompt for hallucination/groundedness
    hallucination_grader_prompt_template = """You are a grader assessing whether an answer is grounded in / supported by a set of retrieved documents.
    Your goal is to determine if all claims in the answer can be directly verified from the provided documents.
    The answer should also be relevant to the original user question.
    Respond with 'yes' if the answer is well-grounded and relevant, and 'no' otherwise.
    Do not provide any explanation or other text, just 'yes' or 'no'.

    Retrieved Documents (Context):
    {documents_context}

    Original User Question:
    {question}

    Generated Answer:
    {generation}

    Is the Generated Answer grounded in the Context and relevant to the Question? (yes/no):
    """
    hallucination_grader_prompt = ChatPromptTemplate.from_template(hallucination_grader_prompt_template)

    # LLM call for grading
    grader_chain = hallucination_grader_prompt | llm | StrOutputParser()

    print(f"Grading generation for question: '{original_question}'")
    try:
        grade_output = grader_chain.invoke({
            "documents_context": formatted_docs,
            "question": original_question, # Use original_question for relevance part of check
            "generation": generation
        })
        grade = grade_output.strip().lower()
        print(f"Groundedness grade from LLM: '{grade}'")

        if "yes" in grade:
            print("Groundedness Grade: YES (Answer is grounded and relevant)")
            return {"hallucination_grade": "yes"}
        elif "no" in grade:
            print("Groundedness Grade: NO (Answer may not be grounded or relevant)")
            return {"hallucination_grade": "no"}
        else:
            print(f"Warning: Unexpected grade format from hallucination grader: '{grade}'. Defaulting to 'no'.")
            return {"hallucination_grade": "no"}

    except Exception as e:
        print(f"Error during generation grading: {e}")
        return {"hallucination_grade": "no"} # Default to 'no' on error
    

def handle_hallucination_node(state: AgentState) -> dict:
    """
    Handles the case where the generation is deemed not grounded or hallucinatory.
    Modifies the 'generation' to include a warning.
    """
    print("---NODE: HANDLE HALLUCINATION/UNGROUNDED ANSWER---")
    original_generation = state["generation"]
    warning_message = "[Warning: The following answer may not be fully supported by the provided documents or could be speculative.]\n"
    updated_generation = warning_message + original_generation
    return {"generation": updated_generation} # Overwrite the generation with the warning



from langgraph.graph import StateGraph, END

# --- Define the LangGraph Workflow ---


print("Defining LangGraph workflow...")
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_generation", grade_generation_node) # New node
workflow.add_node("handle_irrelevance", handle_irrelevance_node)
workflow.add_node("handle_hallucination", handle_hallucination_node) # New node


# Set the entry point
workflow.set_entry_point("retrieve")

# Define edges and conditional logic

workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("transform_query", "retrieve") # Query rewrite loop

# Conditional edge after 'grade_documents'
# MAX_REWRITE_ATTEMPTS defined earlier (e.g., = 2)
workflow.add_conditional_edges(
    start_node_name="grade_documents",
    condition=decide_next_step_after_grading, # This function was defined on Day 4
    conditional_edge_mapping={
        "generate": "generate",
        "transform_query": "transform_query",
        "handle_irrelevance": "handle_irrelevance"
    }
)

# After generation, always grade the generation
workflow.add_edge("generate", "grade_generation")

# Conditional edge after 'grade_generation'
def decide_after_generation_grading(state: AgentState) -> str:
    """
    Based on 'hallucination_grade', decides if the answer is final or needs handling.
    """
    print("---CONDITIONAL EDGE: DECIDE AFTER GENERATION GRADING---")
    hallucination_grade = state.get("hallucination_grade", "no") # Default to "no"

    if hallucination_grade == "yes":
        print("Decision: Generation is grounded. Proceed to END.")
        return "END_SUCCESS" # Using a more descriptive name for the branch
    else:
        print("Decision: Generation may not be grounded. Handle hallucination.")
        return "handle_hallucination"

workflow.add_conditional_edges(
    start_node_name="grade_generation",
    condition=decide_after_generation_grading,
    conditional_edge_mapping={
        "END_SUCCESS": END, # If grounded, go straight to END
        "handle_hallucination": "handle_hallucination"
    }
)

# Final Edges to END
workflow.add_edge("handle_hallucination", END) # After handling hallucination, end.
workflow.add_edge("handle_irrelevance", END)   # If cannot answer, end.


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

        initial_state_input = {"question": user_query, "original_question":user_query,"documents": [], "generation": "", "relevance_grade": "" ,"rewrite_attempts":0,"hallucination_grade":""} # Ensure all keys are present

        print("Invoking RAG agent graph...")
        try:
            final_state = app.invoke(initial_state_input)

            print("\n--- Agent Response ---")
            print(final_state.get("generation", "No generation found."))

        except Exception as e:
            print(f"An error occurred while running the agent: {e}")
            import traceback
            traceback.print_exc() 