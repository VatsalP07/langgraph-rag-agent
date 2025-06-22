import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Path to the directory where ChromaDB stored its data
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
# Name of the HuggingFace embedding model (must match the one used in ingest.py)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# OpenAI model to use for generation
OPENAI_MODEL_NAME = "gpt-4o-mini" # or "gpt-3.5-turbo" or "gpt-4-turbo"

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """Main function to run the RAG pipeline."""
    load_dotenv() # Load OPENAI_API_KEY from .env file

    # 1. Initialize Embeddings and Load Vector Store
    print(f"Initializing embeddings with model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Loading vector store from: {CHROMA_PERSIST_DIRECTORY}")
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    # Make the vector store usable as a retriever
    # k=3 means it will retrieve the top 3 most similar chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Vector store loaded and retriever created.")

    # 2. Initialize LLM
    print(f"Initializing LLM: {OPENAI_MODEL_NAME}")
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=0.1, # Low temperature for more factual answers
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3. Define Prompt Template
    # This template instructs the LLM on how to use the context.
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # 4. Create RAG Chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("RAG chain created. Ready to answer questions.")
    print("Type 'exit' or 'quit' to stop.")

    # 5. Interactive Q&A Loop
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        print("Thinking...")
        try:
            response = rag_chain.invoke(query)
            print("\nAnswer:")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()