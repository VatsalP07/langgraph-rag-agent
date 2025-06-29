# Conversational RAG Agent with Self-Correction using LangGraph

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a sophisticated, conversational RAG (Retrieval-Augmented Generation) agent built with LangGraph. It goes beyond simple RAG by implementing a cyclical, self-correcting flow to provide more accurate and reliable answers based on a given set of documents.

---

## Key Features

-   **Dynamic Agentic Flows:** Built with **LangGraph** to create a non-linear, stateful agent that can loop, branch, and make decisions.
-   **Conversational Memory:** Remembers previous turns in the conversation to understand context and answer follow-up questions accurately.
-   **Self-Correction Loop - Document Relevance:** The agent first retrieves documents and then uses an LLM to **grade their relevance**. If documents are not relevant, it doesn't proceed with a poor answer.
-   **Self-Correction Loop - Query Rewriting:** If documents are deemed irrelevant, the agent uses an LLM to **rewrite the user's query** and retries the retrieval process, attempting to find better information.
-   **Self-Correction Loop - Hallucination Check:** After generating an answer, the agent performs a final check to ensure the response is **factually grounded** in the retrieved documents, reducing the chance of hallucinations.

## Architecture & Flow

The agent's logic is structured as a state machine using LangGraph. The diagram below illustrates the conditional paths and loops that make the agent robust. It's not a simple linear chain; it's a graph that can reason about its own progress.

```mermaid
graph TD
    A[Start: User Input] --> B(condense_question_node);
    B --> C(retrieve_node);
    C --> D{grade_documents_node};
    D -- Relevant --> F(generate_node);
    D -- Not Relevant --> E{Rewrite Attempts < 2?};
    E -- Yes --> G(transform_query_node);
    G --> C;
    E -- No --> H(handle_irrelevance_node);
    F --> I{grade_generation_node};
    I -- Grounded --> J[END: Final Answer];
    I -- Not Grounded --> K(handle_hallucination_node);
    K --> J;
    H --> J;
