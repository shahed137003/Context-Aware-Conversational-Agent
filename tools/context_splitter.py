import re
from typing import Dict, List
from transformers import pipeline
from langchain.tools import Tool

def context_splitter(user_input: str) -> Dict[str, List[str]]:
    # Preprocessing
    user_input = re.sub(r"\s+", " ", user_input.strip())

    # Sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', user_input)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Zero-shot classification model
    clf_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    background = []
    questions = []

    question_keywords = {
        "who", "what", "where", "when", "why", "how", "can", "could", "would", "will",
        "is", "are", "do", "does", "should", "shall", "tell", "give", "recommend"
    }

    for sentence in sentences:
        try:
            first_word = sentence.lower().split()[0] if sentence else ""

            # Heuristic check
            if sentence.endswith("?") or first_word in question_keywords:
                questions.append(sentence)
                continue

            # Model classification
            result = clf_pipeline(sentence, candidate_labels=["background", "question"])
            label = result["labels"][0].lower()
            score = result["scores"][0]

            # Apply confidence threshold
            if label == "question" and score >= 0.65:
                questions.append(sentence)
            else:
                background.append(sentence)

        except Exception:
            background.append(sentence)

    return {
        "background": background,
        "question": questions
    }


def context_splitter_tool() -> Tool:
    """Return LangChain Tool for context splitting."""
    return Tool(
        name="context_splitter",
        func=context_splitter,
        description="Separates background information from the actual question in user input."
    )
