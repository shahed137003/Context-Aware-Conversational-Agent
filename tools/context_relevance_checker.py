from typing import List, Dict
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
import json

# Load embedding model once (faster)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def context_relevance_checker(user_question: str, candidate_context: List[str], threshold: float = 0.15) -> Dict:
    """
    Check if each candidate context is relevant to the question using embeddings.
    Returns a dict with relevant contexts, scores, and debug info.
    """
    # Embed question
    question_vec = np.array(embeddings.embed_query(user_question))

    # Embed all candidate contexts at once
    context_vecs = np.array(embeddings.embed_documents(candidate_context))

    # Cosine similarity
    def cosine_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom != 0 else 0.0

    scores = [cosine_sim(q_vec, question_vec) for q_vec in context_vecs]

    # Filter based on threshold
    relevant = [
        {"context": ctx, "score": score}
        for ctx, score in zip(candidate_context, scores)
        if score >= threshold
    ]

    return {
        "question": user_question,
        "threshold": threshold,
        "relevant_contexts": relevant,
        "all_scores": list(zip(candidate_context, scores))
    }

# Wrapper for LangChain Tool
def _context_relevance_checker_wrapper(input_str: str) -> str:
    """
    LangChain Tool wrapper that expects a JSON string with:
    {
        "user_question": "...",
        "candidate_context": ["...", "..."],
        "threshold": 0.15
    }
    """
    try:
        params = json.loads(input_str)
        return json.dumps(
            context_relevance_checker(
                params["user_question"],
                params["candidate_context"],
                params.get("threshold", 0.15)
            ),
            indent=2
        )
    except Exception as e:
        return json.dumps({"error": str(e)})

def context_relevance_checker_tool():
    """Create the LangChain tool for context relevance checking."""
    return Tool(
        name="context_relevance_checker",
        func=_context_relevance_checker_wrapper,
        description="Given a question and candidate contexts (JSON format), returns only the contexts relevant to the question with similarity scores."
    )
