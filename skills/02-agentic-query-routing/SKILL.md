---
name: eag-rag-query-routing
description: Agentic query preprocessing, query optimization, and source identification using LangChain and LangGraph
---

# Agentic Query Preprocessing & Routing (EAg-RAG Pre-Process Router)

This skill implements the online agentic pre-retrieval layer. It leverages lightweight LLMs to rewrite user questions (Query Optimizer) and target high-probability source documents (Source Identifier), filtering the search space to eliminate vector database noise.

---

## 1. Graph Pipeline Architecture

Orchestrated using **LangGraph** as a sequential processing graph:

```
[User Query] 
     │
     ▼
┌──────────────────┐
│ Query Optimizer  │  ◄── (Cleans, reformulates, and splits query)
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ Source Identifier│  ◄── (Matches query against document summaries index)
└──────────────────┘
     │
     ▼
[Optimized Queries + Metadata Document Filters]
     │
     ▼
[Downstream Hybrid Retrieval Engine]
```

---

## 2. Implementation Guide

### Step A: Query Optimizer Agent
Cleans chat dialogue noise and greetings (e.g., *"Hi team, how do I configure data encryption?"* to *"Data encryption configuration policy"*). If a question is double-barreled (e.g., *"What is the password policy and how do I report a security incident?"*), it splits the user request into separate, distinct search queries.

### Step B: Source Identifier Agent
Instead of running a vector search over the entire corpus, we feed this agent a list of document titles paired with their global summaries. The agent returns a list of target document titles.

---

## 3. Reference Implementation (Python)

This template uses **LangChain** for structured output generation and **LangGraph** to build the processing graph.

```python
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- 1. Structured Output Data Models ---

class OptimizedQueries(BaseModel):
    queries: List[str] = Field(description="List of optimized and cleaned search queries.")

class IdentifiedSources(BaseModel):
    document_titles: List[str] = Field(description="List of target document titles relevant to the query.")

# --- 2. LangGraph State Definition ---

class RouterState(TypedDict):
    raw_query: str
    optimized_queries: List[str]
    identified_sources: List[str]

# Initialize the LLM
llm_small = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# --- 3. Agent Node Logic ---

# Node A: Query Optimizer
OPTIMIZER_PROMPT = """
You are an expert search query optimization assistant.
Analyze the user's raw query below and:
1. Remove conversational noise, greetings, and filler text.
2. Make the query explicit if it lacks context or details.
3. If the query contains multiple questions, split them into a list of simpler queries.

Raw User Query:
{raw_query}
"""

optimizer_prompt_tmpl = ChatPromptTemplate.from_template(OPTIMIZER_PROMPT)
optimizer_chain = optimizer_prompt_tmpl | llm_small.with_structured_output(OptimizedQueries)

def query_optimizer_node(state: RouterState) -> Dict[str, Any]:
    output = optimizer_chain.invoke({"raw_query": state["raw_query"]})
    return {"optimized_queries": output.queries}


# Node B: Source Identifier
SOURCE_PROMPT = """
You are a documentation targeting assistant.
Your job is to match the optimized query against a catalog of internal corporate documents (containing title and summary).
Select only the document titles that are highly likely to contain the answer. Be precise to prevent retrieving irrelevant documents.

Optimized Queries:
{optimized_queries}

Internal Document Catalog:
{document_catalog}
"""

source_prompt_tmpl = ChatPromptTemplate.from_template(SOURCE_PROMPT)
source_chain = source_prompt_tmpl | llm_small.with_structured_output(IdentifiedSources)

# Document Catalog (retrieved from a feature store or cache)
DOCUMENT_CATALOG = """
- Title: "Password Complexity Policy" | Summary: Defines password length, special character rules, and rotation frequencies.
- Title: "Data Retention Charter" | Summary: Details retention periods for user and driver storage.
- Title: "Data Sharing Directive" | Summary: Outlines permission protocols for sharing data with external partners.
- Title: "Physical Security Guide" | Summary: Covers badging rules and workspace CCTV monitoring.
"""

def source_identifier_node(state: RouterState) -> Dict[str, Any]:
    output = source_chain.invoke({
        "optimized_queries": "\n".join(state["optimized_queries"]),
        "document_catalog": DOCUMENT_CATALOG
    })
    return {"identified_sources": output.document_titles}

# --- 4. Assemble the LangGraph ---

workflow = StateGraph(RouterState)

# Add Nodes
workflow.add_node("optimize_query", query_optimizer_node)
workflow.add_node("identify_sources", source_identifier_node)

# Set Edges
workflow.set_entry_point("optimize_query")
workflow.add_edge("optimize_query", "identify_sources")
workflow.add_edge("identify_sources", END)

# Compile Graph
compiled_graph = workflow.compile()

# --- 5. Execution Example ---
if __name__ == "__main__":
    inputs = {"raw_query": "Hello! I would like to know how long we keep logs and what is the policy for sharing data with external partners."}
    result = compiled_graph.invoke(inputs)
    print("Raw query:", result["raw_query"])
    print("Optimized queries:", result["optimized_queries"])
    print("Identified sources:", result["identified_sources"])
```

---

## 4. Downstream Impact

Restricting vector queries to specific documents (via metadata filters such as `doc_title` equals one of the identified sources) guarantees:
1.  **Lower noise**: Eliminates chunks sharing vocabulary but belonging to unrelated policies.
2.  **Context Optimization**: Keeps the LLM's prompt context clean and focused on verified sources.
