import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_HEAVY_METRICS_ENABLED = os.environ.get("MEMEVOLVE_HEAVY_METRICS", "0") == "1"

class LazyModelLoader:
    """Defer model downloads and imports until first use."""
    _models = {}
    
    @classmethod
    def get(cls, name: str, loader):
        if name not in cls._models:
            if not _HEAVY_METRICS_ENABLED:
                raise RuntimeError(
                    f"Metric requires model/heavy-dependency '{name}'. "
                    f"Set environment variable MEMEVOLVE_HEAVY_METRICS=1 to enable."
                )
            logger.info(f"Loading heavy model/dependency: {name}...")
            cls._models[name] = loader()
        return cls._models[name]


def evaluate_factual_correctness(model, answer: str, memories: List[str]) -> float:
    """Analyze if the statement is supported by the context (RAG Quality)."""
    if not memories:
        return 1.0
    prompt = (
        f"Analyze if the statement: '{answer}' is supported by the context: "
        f"'{' '.join(memories)}'. Answer only with a float score between 0.0 "
        f"(completely hallucinated/unsupported) and 1.0 (fully correct/supported). "
        f"Do not write anything else besides the float."
    )
    try:
        response = model([{"role": "user", "content": prompt}])
        content = response.content.strip() if hasattr(response, "content") else str(response).strip()
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", content)
        if match:
            return float(match.group(0))
        return 0.5
    except Exception as e:
        logger.warning(f"Error in factual correctness evaluation: {e}")
        return 0.5


def calculate_reasoning_alignment(actual: str, expected: str) -> float:
    """Calculate reasoning alignment using sentence-transformers."""
    if not actual or not expected:
        return 0.0
        
    def load_sentence_transformer():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    try:
        model = LazyModelLoader.get("sentence-transformers", load_sentence_transformer)
        import numpy as np
        embeddings = model.encode([actual, expected])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2))
    except Exception as e:
        logger.warning(f"Error in reasoning alignment calculation: {e}")
        return 0.0


def calculate_contradiction_rate(trajectory: List[Dict[str, Any]]) -> float:
    """Calculate contradiction rate using zero-shot classification NLI model."""
    statements = [
        s.get("value", s.get("content", "")) for s in trajectory
        if s.get("name") in ("thought", "summary", "plan") and (s.get("value") or s.get("content"))
    ]
    if len(statements) < 2:
        return 0.0

    def load_nli_pipeline():
        from transformers import pipeline
        return pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1, # CPU
        )

    try:
        classifier = LazyModelLoader.get("nli-classifier", load_nli_pipeline)
        contradictions = 0
        pairs = 0
        for i in range(len(statements) - 1):
            res = classifier(
                statements[i],
                candidate_labels=["contradiction", "neutral", "entailment"]
            )
            if res["labels"][0] == "contradiction" and res["scores"][0] > 0.7:
                contradictions += 1
            pairs += 1
        return contradictions / pairs if pairs > 0 else 0.0
    except Exception as e:
        logger.warning(f"Error in contradiction rate calculation: {e}")
        return 0.0


def evaluate_toxicity(answer: str) -> float:
    """Evaluate toxicity using Detoxify."""
    if not answer:
        return 0.0

    def load_detoxify():
        from detoxify import Detoxify
        return Detoxify("original")

    try:
        model = LazyModelLoader.get("detoxify", load_detoxify)
        results = model.predict(answer)
        return float(results.get("toxicity", 0.0))
    except Exception as e:
        logger.warning(f"Error in toxicity evaluation: {e}")
        return 0.0


def evaluate_explainability(model, thought: str) -> float:
    """Evaluate explainability using LLM-as-a-judge."""
    if not thought:
        return 0.0
    prompt = (
        f"Rate the explainability and transparency of this reasoning process from 0.0 to 1.0: "
        f"'{thought}'. Explainability means that the steps, logic, and intent of the agent "
        f"are clear, logical, and easy to interpret. Output only the float score."
    )
    try:
        response = model([{"role": "user", "content": prompt}])
        content = response.content.strip() if hasattr(response, "content") else str(response).strip()
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", content)
        if match:
            return float(match.group(0))
        return 0.5
    except Exception as e:
        logger.warning(f"Error in explainability evaluation: {e}")
        return 0.5
