---
name: eag-rag-enriched-ingestion
description: Enriched document processing and table-aware chunking (HTML/Markdown parsing, LLM table formatting, metadata attachment)
---

# Enriched Document Ingestion & Enrichment (EAg-RAG Ingestion)

This skill enables extraction, formatting, and semantic enrichment of Google Docs/HTML contents prior to indexing in the vector database. It preserves structural cohabitation of tables and lists using LLMs to format raw elements and generate metadata.

---

## 1. Ingestion Pipeline Architecture

The offline document ingestion flows as follows:

```
[Google Doc HTML] 
       │
       ▼
[GDoc API Recursive Parser] ──(Extract paragraphs & tables)──► [Raw Text + HTML Table]
                                                                        │
                                                                        ▼
[LLM Table Enrichment] ◄──(Invokes LLM to output clean Markdown)────────┤
                                                                        │
                                                                        ▼
[Table-Aware Chunking] ◄──(Ensures table remains inside one chunk)──────┤
                                                                        │
                                                                        ▼
[LLM Metadata (FAQ, Keywords)] ◄────────────────────────────────────────┤
                                                                        │
                                                                        ▼
[Embeddings Generation & Indexing (LlamaIndex)] ◄───────────────────────┘
```

---

## 2. Implementation Guide

### Step A: Google Docs/HTML Structural Parsing
Extract documents (via HTML elements or Google Docs API) identifying nested `<table>` tags and paragraphs recursively.

### Step B: LLM-Powered Table Enrichment
When a table is extracted, its raw HTML content is passed to a fast LLM (e.g., Gemini Flash or GPT-4o-mini). The LLM is instructed to return a clean Markdown table, a 2-line description summary of the table, and 5 related keywords.

### Step C: Table-Aware Chunking
To prevent standard text splitters from slicing tables across chunks, wrap tables in custom boundary markers (e.g., `<!-- START_TABLE -->` ... `<!-- END_TABLE -->`). The parser yields a single node for the table along with its generated summary and keywords, maximizing semantic vector similarity search accuracy.

---

## 3. Reference Implementation (Python)

This template uses **LlamaIndex** for document indexing and chunking, and **LangChain** for LLM orchestration.

```python
import os
import re
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 1. Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# 2. Define the JSON extraction prompt for tables
TABLE_PROMPT = """
You are an expert document processing assistant.
You are given a raw HTML table. Your task is to:
1. Convert it into a clean, valid Markdown table.
2. Generate a 2-line summary describing the table's purpose.
3. Extract 5 relevant keywords/phrases from this table.

Output format must be strict JSON:
{{
    "markdown_table": "... (Markdown formatted table) ...",
    "summary": "... (2-line summary description) ...",
    "keywords": ["key1", "key2", "key3", "key4", "key5"]
}}

Raw HTML Table:
{raw_html}
"""

table_enrichment_prompt = ChatPromptTemplate.from_template(TABLE_PROMPT)
table_chain = table_enrichment_prompt | llm | JsonOutputParser()

def enrich_table_content(raw_html_table: str) -> Dict[str, Any]:
    """Invokes LLM to clean and summarize table HTML."""
    try:
        return table_chain.invoke({"raw_html": raw_html_table})
    except Exception as e:
        print(f"Error enriching table: {e}")
        return {
            "markdown_table": raw_html_table,
            "summary": "Table data.",
            "keywords": []
        }

# 3. Table-Aware Node Parser
class TableAwareNodeParser:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    def get_nodes_from_documents(self, documents: List[Document]) -> List[Any]:
        nodes = []
        for doc in documents:
            text = doc.text
            # Identify delimited tables
            parts = re.split(r'(<!-- START_TABLE -->.*?<!-- END_TABLE -->)', text, flags=re.DOTALL)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if part.startswith("<!-- START_TABLE -->"):
                    # Process table as a single, indivisible chunk
                    html_content = part.replace("<!-- START_TABLE -->", "").replace("<!-- END_TABLE -->", "")
                    enrichment = enrich_table_content(html_content)
                    
                    enriched_text = (
                        f"Table Summary: {enrichment['summary']}\n"
                        f"Keywords: {', '.join(enrichment['keywords'])}\n"
                        f"\n{enrichment['markdown_table']}"
                    )
                    
                    # Create LlamaIndex Document Node
                    node = Document(
                        text=enriched_text,
                        metadata={
                            **doc.metadata,
                            "contains_table": True,
                            "table_summary": enrichment['summary'],
                            "keywords": enrichment['keywords']
                        }
                    )
                    nodes.append(node)
                else:
                    # Regular text chunks
                    sub_docs = [Document(text=part, metadata=doc.metadata)]
                    sub_nodes = self.splitter.get_nodes_from_documents(sub_docs)
                    nodes.extend(sub_nodes)
        return nodes
```

---

## 4. Recommended Metadata Fields

Incorporate these attributes to boost search relevancy and enforce security controls:

| Metadata Attribute | Description | Rôle in RAG |
| :--- | :--- | :--- |
| `doc_summary` | Global summary of the source document | Pre-retrieval routing context |
| `chunk_faq` | Automated QA pairs generated from the chunk | High-relevance lexical target (BM25) |
| `keywords` | Semantic tokens representing the chunk | Retrieval matching enhancement |
| `user_roles` | Permitted access groups/roles for this chunk | Post-retrieval access control filtering |
