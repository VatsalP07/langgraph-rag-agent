"""
This module defines a conversational RAG (Retrieval-Augmented Generation) agent
using LangGraph. The agent is designed to answer questions based on a provided
set of documents. It features several advanced capabilities:
- Conversational Memory: Remembers past interactions to answer follow-up questions.
- Document Relevance Grading: Checks if retrieved documents are relevant to the question.
- Query Rewriting: Attempts to rephrase the question if initial documents are irrelevant.
- Hallucination Grading: Checks if the generated answer is grounded in the retrieved documents.
"""

import os
import logging
import operator
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain and Community Imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from openai import APIError

# --- Setup Logging ---
# Configure logging to provide detailed output with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Directory for ChromaDB to persist its data
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
# HuggingFace embedding model for vectorizing text
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# OpenAI model used for all LLM tasks (grading, generation, etc.)
OPENAI_MODEL_NAME = "gpt-4o-mini"
# Maximum number of times to attempt query rewriting
MAX_REWRITE_ATTEMPTS = 2

# --- LangGraph Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our RAG agent.
    This dictionary will be passed between nodes in the graph.
    """
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    relevance_grade: str
    rewrite_attempts: int
    hallucination_grade: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]

# --- Load Environment Variables and Initialize Components ---
load_dotenv()

logger.info("Initializing components...")
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
logger.info("Components initialized successfully.")

# --- Prompt Templates ---
RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# --- Helper Functions ---
def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Graph Nodes ---

def condense_question_node(state: AgentState) -> dict:
    """Condenses chat history and a new question into a standalone question."""
    logger.info("---NODE: CONDENSE QUESTION---")
    chat_history = state["chat_history"]
    question = state["question"]

    if not chat_history:
        logger.info("No chat history, using question as is.")
        return {"question": question}

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    condenser_chain = condense_prompt | llm | StrOutputParser()

    logger.info("Condensing question with chat history...")
    try:
        condensed_question = condenser_chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
        logger.info(f"Condensed question: '{condensed_question.strip()}'")
        return {"question": condensed_question.strip()}
    except APIError as e:
        logger.error(f"OpenAI API Error during question condensing: {e}")
        return {"question": question}
    except Exception as e:
        logger.error(f"Unexpected error during question condensing: {e}. Using original question as fallback.")
        return {"question": question}

def retrieve_node(state: AgentState) -> dict:
    """Retrieves documents relevant to the question from the vector store."""
    logger.info("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    logger.info(f"Retrieving documents for question: '{question}'")
    retrieved_docs = retriever.invoke(question)
    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
    return {"documents": retrieved_docs}

def grade_documents_node(state: AgentState) -> dict:
    """Determines whether the retrieved documents are relevant to the question."""
    logger.info("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("No documents to grade.")
        return {"relevance_grade": "no"}

    formatted_docs = format_docs(documents)
    grader_prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing document relevance. Respond 'yes' if the documents are relevant to the question, 'no' otherwise.
        Retrieved Documents: {documents_context}
        User Question: {question}
        Are these documents relevant? (yes/no):"""
    )
    grader_chain = grader_prompt | llm | StrOutputParser()
    logger.info(f"Grading relevance for question: '{question}'")
    try:
        grade_output = grader_chain.invoke({"documents_context": formatted_docs, "question": question})
        grade = grade_output.strip().lower()
        logger.info(f"Relevance grade from LLM: '{grade}'")
        if "yes" in grade:
            return {"relevance_grade": "yes"}
        else:
            return {"relevance_grade": "no"}
    except APIError as e:
        logger.error(f"OpenAI API Error during relevance grading: {e}")
        return {"relevance_grade": "no"}
    except Exception as e:
        logger.error(f"Unexpected error during relevance grading: {e}")
        return {"relevance_grade": "no"}

def transform_query_node(state: AgentState) -> dict:
    """Transforms the query to improve retrieval results."""
    logger.info("---NODE: TRANSFORM QUERY---")
    original_question = state["original_question"]
    current_question = state["question"]
    rewrite_attempts = state.get("rewrite_attempts", 0)
    logger.info(f"Rewrite attempt number: {rewrite_attempts + 1}")

    transform_prompt = ChatPromptTemplate.from_template(
        """You are a query rewriting expert. Rewrite the following question to improve search results.
        Original Question: {original_question}
        Potentially Failed Question: {current_question}
        Rewritten Question:"""
    )
    transformer_chain = transform_prompt | llm | StrOutputParser()
    try:
        rewritten_question = transformer_chain.invoke({"original_question": original_question, "current_question": current_question})
        logger.info(f"Rewritten question: '{rewritten_question.strip()}'")
        return {"question": rewritten_question.strip(), "rewrite_attempts": rewrite_attempts + 1}
    except APIError as e:
        logger.error(f"OpenAI API Error during query transformation: {e}")
        return {"rewrite_attempts": rewrite_attempts + 1}
    except Exception as e:
        logger.error(f"Unexpected error during query transformation: {e}")
        return {"rewrite_attempts": rewrite_attempts + 1}

def generate_node(state: AgentState) -> dict:
    """Generates an answer using the LLM."""
    logger.info("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    formatted_context = format_docs(documents)
    generation_chain = RAG_PROMPT_WITH_HISTORY | llm | StrOutputParser()
    logger.info(f"Generating answer for question: '{question}'")
    answer = generation_chain.invoke({"context": formatted_context, "question": question, "chat_history": chat_history})
    logger.info("Answer generated.")
    return {"generation": answer}

def grade_generation_node(state: AgentState) -> dict:
    """Determines if the generated answer is grounded in the documents."""
    logger.info("---NODE: GRADE GENERATION---")
    original_question = state["original_question"]
    documents = state["documents"]
    generation = state["generation"]

    if not documents or not generation:
        logger.warning("No documents or generation to grade.")
        return {"hallucination_grade": "no"}

    formatted_docs = format_docs(documents)
    grader_prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing if an answer is grounded in context. Respond 'yes' if the answer is supported by the documents, 'no' otherwise.
        Documents (Context): {documents_context}
        User Question: {question}
        Generated Answer: {generation}
        Is the Answer supported by the Context? (yes/no):"""
    )
    grader_chain = grader_prompt | llm | StrOutputParser()
    logger.info(f"Grading generation for question: '{original_question}'")
    try:
        grade_output = grader_chain.invoke({"documents_context": formatted_docs, "question": original_question, "generation": generation})
        grade = grade_output.strip().lower()
        logger.info(f"Groundedness grade from LLM: '{grade}'")
        if "yes" in grade:
            return {"hallucination_grade": "yes"}
        else:
            return {"hallucination_grade": "no"}
    except APIError as e:
        logger.error(f"OpenAI API Error during generation grading: {e}")
        return {"hallucination_grade": "no"}
    except Exception as e:
        logger.error(f"Unexpected error during generation grading: {e}")
        return {"hallucination_grade": "no"}

def handle_irrelevance_node(state: AgentState) -> dict:
    """Handles the case where documents are deemed irrelevant after all retries."""
    logger.info("---NODE: HANDLE IRRELEVANCE---")
    return {"generation": "Sorry, I could not find relevant documents to answer your question after several attempts."}

def handle_hallucination_node(state: AgentState) -> dict:
    """Handles the case where the generation is not grounded."""
    logger.info("---NODE: HANDLE HALLUCINATION---")
    original_generation = state["generation"]
    warning_message = "[Warning: The following answer may not be fully supported by the provided documents.]\n\n"
    return {"generation": warning_message + original_generation}

# --- Conditional Edges ---
def decide_next_step_after_grading(state: AgentState) -> str:
    """Decides the next step based on document relevance and rewrite attempts."""
    logger.info("---CONDITIONAL EDGE: DECIDE AFTER GRADING---")
    if state["relevance_grade"] == "yes":
        logger.info("Decision: Relevant documents found. Proceed to generate.")
        return "generate"
    else:
        logger.info("Decision: Documents are NOT relevant.")
        if state["rewrite_attempts"] < MAX_REWRITE_ATTEMPTS:
            logger.info("Proceed to transform query.")
            return "transform_query"
        else:
            logger.info("Max rewrite attempts reached. Handle irrelevance.")
            return "handle_irrelevance"

def decide_after_generation_grading(state: AgentState) -> str:
    """Decides if the answer is final or needs handling."""
    logger.info("---CONDITIONAL EDGE: DECIDE AFTER GENERATION GRADING---")
    if state["hallucination_grade"] == "yes":
        logger.info("Decision: Generation is grounded. Final answer is ready.")
        return "END"
    else:
        logger.info("Decision: Generation is NOT grounded. Handle hallucination.")
        return "handle_hallucination"

# --- Build the Graph ---
logger.info("Defining LangGraph workflow...")
workflow = StateGraph(AgentState)
workflow.add_node("condense_question", condense_question_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_generation", grade_generation_node)
workflow.add_node("handle_irrelevance", handle_irrelevance_node)
workflow.add_node("handle_hallucination", handle_hallucination_node)

workflow.set_entry_point("condense_question")
workflow.add_edge("condense_question", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step_after_grading,
    {"generate": "generate", "transform_query": "transform_query", "handle_irrelevance": "handle_irrelevance"}
)
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges(
    "grade_generation",
    decide_after_generation_grading,
    {"END": END, "handle_hallucination": "handle_hallucination"}
)
workflow.add_edge("handle_hallucination", END)
workflow.add_edge("handle_irrelevance", END)

logger.info("Compiling LangGraph...")
app = workflow.compile()
logger.info("LangGraph compiled successfully.")

# --- Main Execution Block for Standalone Testing ---
if __name__ == "__main__":
    logger.info("--- Running RAG Agent in Standalone Mode ---")
    logger.info("Type 'exit' or 'quit' to stop.")

    session_history = []
    while True:
        user_query = input("\nAsk a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        initial_state_input = {
            "question": user_query,
            "original_question": user_query,
            "chat_history": session_history,
            "documents": [], "generation": "", "relevance_grade": "",
            "rewrite_attempts": 0, "hallucination_grade": ""
        }

        logger.info("Invoking RAG agent graph...")
        try:
            final_state = app.invoke(initial_state_input)
            final_generation = final_state.get("generation", "No generation found.")
            print("\n--- Agent Response ---")
            print(final_generation)
            session_history.append(HumanMessage(content=user_query))
            session_history.append(AIMessage(content=final_generation))
        except Exception as e:
            logger.error(f"An error occurred while running the agent: {e}", exc_info=True)