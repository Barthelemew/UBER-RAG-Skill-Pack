---
name: eag-rag-evaluation-judge
description: Automated RAG evaluation (LLM-as-a-Judge) using LangChain structured outputs and scoring metrics
---

# LLM-as-a-Judge Evaluation (EAg-RAG Judge Eval)

This skill implements the automated batch evaluation layer. It measures the quality and accuracy of chatbot responses against a reference Golden Test Set containing verified answers from experts (SMEs).

---

## 1. Evaluation Pipeline Architecture

The automated batch evaluation process runs as follows:

```
┌─────────────────────┐
│   Golden Test Set   ├──(Test Queries)──► [Chatbot RAG under test]
│                     │                                   │
│                     │                                   ▼
│                     │                         [Chatbot Answers]
│                     │                         [Retrieved Contexts]
│                     │                                   │
│  (Expert Answers)   │                                   │
└──────────┬──────────┘                                   │
           │                                              │
           └───────────────┬──────────────────────────────┘
                           │ Evaluation context (C) + Answer (x)
                           ▼
               ┌───────────────────────┐
               │     Judge Prompt      │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │    LLM Judge (Pro)    │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │ Scores & Reasonings   │
               └───────────────────────┘
```

---

## 2. Implementation Guide

### Step A: Golden Test Set Structure
Establish a test list of dictionary instances containing:
1.  The initial user query (`query`).
2.  The ideal ground-truth response drafted by a Subject Matter Expert (`sme_response`).

### Step B: Mathematical Evaluation Model
The judge LLM computes the evaluation result ($\mathcal{E}$) using the chatbot response ($x$) combined with context ($C$) consisting of user query, expert response, and RAG document context:

$$\mathcal{E} \leftarrow \mathcal{P}_{LLM}(x \oplus C)$$

We use structured JSON output parsers to ensure the judge returns schema-conforming metrics.

### Step C: Rating Rubric (0-5 Scale)
Provide the judge LLM with the following standardized scale:
*   **Score 5 :** Perfect response: accurate, comprehensive, contains no fluff, and matches the expert answer.
*   **Score 4 :** Accurate and complete, but contains mild redundancy or verbose phrasing.
*   **Score 3 :** Partially correct, but incomplete (misses a key security policy or exception).
*   **Score 2 :** Major inaccuracies or clear lack of contextual understanding.
*   **Score 1 :** Incorrect or misleading advice (critical failure).
*   **Score 0 :** Total hallucination or irrelevant response.

---

## 3. Reference Implementation (Python)

This template uses **LangChain** and Pydantic to structure the LLM judge outputs, and handles evaluation requests in parallel.

```python
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define LLM Judge Output Schema
class JudgeEvaluation(BaseModel):
    score: int = Field(description="Quality rating score from 0 to 5 based on the rubric.")
    reasoning: str = Field(description="Detailed explanation of the rating, detailing delta with expert answer.")
    correctness_label: str = Field(description="Categorical label: 'CORRECT', 'INCOMPLETE', 'INCORRECT', or 'HALLUCINATION'.")

# 2. Judge Prompt Template
JUDGE_PROMPT_TEMPLATE = """
You are an objective AI Judge evaluating a corporate Q&A chatbot response.
Compare the chatbot response against the ground-truth answer written by a Subject Matter Expert (SME).

Evaluation Context:
- User Query: {query}
- Expert Answer (SME): {sme_response}
- Retrieved Context (RAG documents): {retrieved_context}

Chatbot Response to Evaluate:
{bot_response}

---
Rating Rubric:
- 5: Perfect, complete, accurate, matches expert, supported by context.
- 4: Accurate and complete, but contains mild redundancy.
- 3: Partially correct but incomplete (omits key details from expert answer).
- 2: Major inaccuracies or shows clear lack of precision.
- 1: Incorrect or dangerous guidance (critical failure).
- 0: Total hallucination or irrelevant output.

Generate your evaluation containing score, detailed reasoning, and final correctness label.
"""

judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
llm_judge = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0) # Larger model recommended for judging
judge_chain = judge_prompt | llm_judge.with_structured_output(JudgeEvaluation)

# 3. Asynchronous Batch Evaluator
async def evaluate_single_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a single test case from the Golden Set."""
    # Integrate your actual RAG chatbot query execution here
    bot_response = test_case.get("simulated_bot_response", "I don't know.")
    retrieved_context = test_case.get("simulated_context", "No documents found.")
    
    try:
        evaluation: JudgeEvaluation = await judge_chain.ainvoke({
            "query": test_case["query"],
            "sme_response": test_case["sme_response"],
            "retrieved_context": retrieved_context,
            "bot_response": bot_response
        })
        return {
            "query": test_case["query"],
            "score": evaluation.score,
            "reasoning": evaluation.reasoning,
            "label": evaluation.correctness_label
        }
    except Exception as e:
        return {
            "query": test_case["query"],
            "score": 0,
            "reasoning": f"Judge invocation failed: {str(e)}",
            "label": "ERROR"
        }

async def run_batch_evaluation(golden_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs evaluations in parallel to speed up test execution."""
    tasks = [evaluate_single_case(case) for case in golden_set]
    return await asyncio.gather(*tasks)

# --- 4. Execution Example ---
if __name__ == "__main__":
    # Sample Golden Set
    sample_golden_set = [
        {
            "query": "What is the security logs retention period?",
            "sme_response": "Security logs must be retained for exactly 90 days.",
            "simulated_bot_response": "Uber security logs are stored for 90 days under our standard policy.",
            "simulated_context": "Retention Policy: Security logs must be kept for 90 days."
        },
        {
            "query": "Can I use password123 as admin?",
            "sme_response": "No, using simple password combinations is strictly prohibited for admin accounts.",
            "simulated_bot_response": "Yes, admins can use password123 as long as they promise not to tell anyone.",
            "simulated_context": "Exceptions: Admins can use password123 (Obsolete mock policy)."
        }
    ]
    
    # Run evaluation
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_batch_evaluation(sample_golden_set))
    
    print("Batch Evaluation Results:")
    print("=========================")
    for idx, res in enumerate(results):
        print(f"\nTest Case #{idx+1}: {res['query']}")
        print(f"  Score: {res['score']}/5 | Label: {res['label']}")
        print(f"  Reasoning: {res['reasoning']}")
```

---

## 4. Key Performance Aggregated Metrics

Track these indicators to measure RAG improvements across code iterations:

*   **Acceptable Answer Rate :** Percentage of test cases rated **4** or **5**.
*   **Incorrect Advice Rate :** Percentage of test cases rated **1**. This must be kept at **0%** for production.
*   **Incomplete Rate :** Percentage of test cases rated **3** (correct but missing details).
