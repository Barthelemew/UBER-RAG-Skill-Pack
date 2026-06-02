---
name: eag-rag-hybrid-retrieval
description: Hybrid retrieval (Vector + BM25) and contextual post-processing using LlamaIndex and LangChain
---

# Hybrid Retrieval & Context Post-Processing (EAg-RAG Hybrid Retrieve)

This skill implements the online extraction layer. It queries semantic vectors and lexical indexes simultaneously, applying document filters from the router node, and compiles the context chronologically.

---

## 1. Retrieval Pipeline Architecture

The query-time retrieval flows as follows:

```
             ┌─────────────────────────────┐
             │ Optimized Query + Filters   │
             └──────────────┬──────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│     Vector Search     │       │     Lexical Search    │
│      (LlamaIndex)     │       │     BM25 (LangChain)  │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │     Fusion & De-duplication │
             └──────────────┬─────────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Positional Post-Processor   │ ◄── (Sorts by index and source pages)
             └──────────────┬──────────────┘
                            │
                            ▼
             [Clean Context for LLM Prompts]
```

---

## 2. Implementation Guide

### Step A: Metadata-Filtered Vector Search
Utilize **LlamaIndex** to execute semantic cosine-similarity search over document nodes. Pass a dynamic `MetadataFilters` object (containing `doc_title` equality constraints) derived from the prior *Source Identifier* step.

### Step B: Keyword Search (BM25)
Concurrently, query a BM25 keyword matching engine (`BM25Retriever` in LangChain) indexing standard text supplemented with the pre-calculated FAQ and keyword metadata. This ensures exact matching of system codes, terms, and functions that vector search might skip.

### Step C: Fusion and De-duplication
Perform a union of the retrieved vectors and lexical results. If a node appears in both, de-duplicate it by retaining the highest relevance score.

### Step D: Positional Context Sorting (*Positional Post-Processor*)
To prevent context truncation issues (*lost-in-the-middle*) and preserve reading flow, sort final chunks sequentially by mapping the metadata values of `doc_title` and `chunk_index` before building the generation prompt.

---

## 3. Reference Implementation (Python)

This template combines **LlamaIndex** for Vector Index querying and **LangChain** for BM25 retrieval.

```python
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.schema import NodeWithScore, TextNode
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

class EAgRAGHybridRetriever:
    def __init__(self, documents: List[Document]):
        # 1. Initialize LlamaIndex Vector Index
        self.vector_index = VectorStoreIndex.from_documents(documents)
        
        # 2. Setup LangChain BM25 Retriever
        self.lc_docs = []
        for doc in documents:
            # Supplement matching content with metadata keywords and FAQs
            faq_text = "\n".join(doc.metadata.get("faq", []))
            keywords_text = " ".join(doc.metadata.get("keywords", []))
            enriched_content = f"{doc.text}\n{faq_text}\n{keywords_text}"
            
            self.lc_docs.append(
                LCDocument(
                    page_content=enriched_content,
                    metadata={
                        "doc_title": doc.metadata.get("doc_title", ""),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "original_text": doc.text
                    }
                )
            )
        self.bm25_retriever = BM25Retriever.from_documents(self.lc_docs)
        self.bm25_retriever.k = 5

    def retrieve(self, query: str, target_docs: List[str] = None) -> List[NodeWithScore]:
        # --- Step A: Filtered Vector Search ---
        filters = None
        if target_docs:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="doc_title", value=title, operator=FilterOperator.EQ)
                    for title in target_docs
                ],
                condition="or"
            )
            
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=5,
            filters=filters
        )
        vector_results = vector_retriever.retrieve(query)
        
        # --- Step B: BM25 Lexical Retrieval ---
        bm25_results = self.bm25_retriever.invoke(query)
        
        # --- Step C: Fusion & De-duplication ---
        merged_nodes: Dict[str, NodeWithScore] = {}
        
        for res in vector_results:
            text_id = res.node.text
            merged_nodes[text_id] = res
            
        for doc in bm25_results:
            # Filter manually since BM25 retriever does not natively support metadata filters
            if target_docs and doc.metadata["doc_title"] not in target_docs:
                continue
                
            text_content = doc.metadata["original_text"]
            if text_content not in merged_nodes:
                node = TextNode(
                    text=text_content,
                    metadata={
                        "doc_title": doc.metadata["doc_title"],
                        "chunk_index": doc.metadata["chunk_index"]
                    }
                )
                merged_nodes[text_content] = NodeWithScore(node=node, score=0.7)
                
        final_nodes = list(merged_nodes.values())
        
        # --- Step D: Chronological/Positional Post-processing ---
        final_nodes.sort(key=lambda x: (x.node.metadata.get("doc_title", ""), x.node.metadata.get("chunk_index", 0)))
        
        return final_nodes

# --- Execution Example ---
if __name__ == "__main__":
    # Mock Document Segment Nodes
    fictional_chunks = [
        Document(
            text="Security logs retention is exactly 90 days.",
            metadata={"doc_title": "Security Directive", "chunk_index": 2, "keywords": ["logs", "retention"]}
        ),
        Document(
            text="User passwords must be at least 12 characters long.",
            metadata={"doc_title": "Security Directive", "chunk_index": 1, "keywords": ["passwords", "security"]}
        )
    ]
    
    retriever = EAgRAGHybridRetriever(fictional_chunks)
    results = retriever.retrieve(
        query="retention logs",
        target_docs=["Security Directive"]
    )
    
    print(f"Total retrieved nodes: {len(results)}")
    for node_w_score in results:
        print(f"\n[Index {node_w_score.node.metadata['chunk_index']}] : {node_w_score.node.text}")
```

---

## 4. Context Prompt Formulation Guidance

Compile the final context template fed into the generator LLM as follows:

```markdown
Answer the user query relying only on the document sources provided below.
If the context doesn't contain the information, state clearly that you do not know.

--- CONTEXT START ---
Source: "[doc_title]", Node index: [chunk_index]
[Text content of Node 1]

Source: "[doc_title]", Node index: [chunk_index + 1]
[Text content of Node 2]
--- CONTEXT END ---
```
