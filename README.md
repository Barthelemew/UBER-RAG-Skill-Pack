# UBER-RAG Skill Pack

An advanced agentic RAG implementation toolkit designed for building highly precise, production-ready Q&A copilots, inspired by Uber Engineering's internal copilot architecture (**EAg-RAG**).

This repository contains modular implementation skills designed for developer assistants and engineers. These skills combine the agentic orchestration power of **LangChain** and **LangGraph** with the indexing and retrieval capabilities of **LlamaIndex**.

---

## 🚀 Architecture Overview

Unlike traditional RAG systems that suffer from noisy retrievals and format-breaking document splits, the **EAg-RAG** (Enhanced Agentic RAG) architecture introduces offline document enrichment and online preprocessing/post-processing agent layers.

```
       Offline Document Processing (Ingestion)
[Google Docs/HTML] ──> [Enrichment LLM] ──> [Table-Aware Chunking] ──> [Vector/Feature Store]

       Online Answer Generation (Query Time)
[User Query] ──> [Query Optimizer] ──> [Source Identifier] ──> [Hybrid Retrieve (Vector + BM25)] ──> [Post-Processor] ──> [LLM Generator] ──> [User Answer]
```

---

## 🛠️ Repository Structure & Skills

This repository is organized into 4 logical, sequentially executed pipeline skills under the `skills/` directory:

1.  **[01-enriched-document-ingestion](skills/01-enriched-document-ingestion/SKILL.md)**:
    *   Extracts Google Docs/HTML tables recursively.
    *   Uses LLM calls to format raw HTML tables into clean Markdown tables.
    *   Performs **Table-Aware Chunking** (keeping tables insubdivided in a single chunk) and attaches summaries/keywords metadata to each chunk.
2.  **[02-agentic-query-routing](skills/02-agentic-query-routing/SKILL.md)**:
    *   **Query Optimizer**: Clarifies ambiguous questions and splits multi-hop queries.
    *   **Source Identifier**: Compares queries against a cached index of document summaries to filter search space before retrieval.
3.  **[03-hybrid-retrieval-postprocessing](skills/03-hybrid-retrieval-postprocessing/SKILL.md)**:
    *   Combines **Vector Search** (semantic) and **BM25 Search** (lexical/keywords).
    *   Applies a **Positional Post-Processor** to de-duplicate results and sort chunks chronologically in respect to the source documents.
4.  **[04-llm-as-a-judge-evaluation](skills/04-llm-as-a-judge-evaluation/SKILL.md)**:
    *   Implements an automated batch test framework.
    *   Runs the LLM-as-a-Judge protocol using the evaluation function:
        $$\mathcal{E} \leftarrow \mathcal{P}_{LLM}(x \oplus C)$$
    *   Generates quality metrics (0-5 score scale, correctness label, detailed reasoning) compared to SME (Subject Matter Expert) answers.

---

## 📋 Prerequisites & Installation

To run the implementation templates provided in these skills, you need:

1.  **Python 3.10+**
2.  Install required packages:
    ```bash
    pip install llama-index langchain langchain-google-genai langgraph pydantic beautifulsoup4
    ```
3.  Configure your environment variables:
    ```bash
    export GOOGLE_API_KEY="your-gemini-api-key"
    # Or for OpenAI
    # export OPENAI_API_KEY="your-openai-api-key"
    ```

---

## 📖 License

This repository is for educational and implementation guidance purposes. Inspired by the public publication from Uber Engineering.
