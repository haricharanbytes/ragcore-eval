import logging
from functools import lru_cache

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from app.core.embeddings import get_embedding_model
from app.core.rag_chain import _get_llm

logger = logging.getLogger(__name__)


@lru_cache
def _get_evaluator_llm() -> LangchainLLMWrapper:
    """Wraps our existing ChatGroq instance so RAGAS can use it as the
    judge model. Cached like every other model/client in this app."""
    return LangchainLLMWrapper(_get_llm())


@lru_cache
def _get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """Wraps our existing HuggingFace embeddings for the metrics that
    need them (Answer Relevancy compares embedding similarity between
    the question and a few generated 'ideal' questions)."""
    return LangchainEmbeddingsWrapper(get_embedding_model())


@lru_cache
def _get_faithfulness_metric() -> Faithfulness:
    return Faithfulness(llm=_get_evaluator_llm())


@lru_cache
def _get_answer_relevancy_metric() -> AnswerRelevancy:
    return AnswerRelevancy(llm=_get_evaluator_llm(), embeddings=_get_evaluator_embeddings())


async def evaluate_answer(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Scores one answer against its source contexts.

    Returns:
        {"faithfulness": float, "answer_relevancy": float}   (each 0-1, higher is better)

    Runs the two metrics sequentially rather than concurrently — both
    ultimately hit the same Groq API, and running them one after another
    is simpler to reason about and avoids any rate-limit surprises from
    firing two judge calls at once for a single user click.
    """
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    faithfulness_score = await _get_faithfulness_metric().single_turn_ascore(sample)
    relevancy_score = await _get_answer_relevancy_metric().single_turn_ascore(sample)

    logger.info(
        "Evaluated answer: faithfulness=%.3f, answer_relevancy=%.3f",
        faithfulness_score, relevancy_score,
    )

    return {
        "faithfulness": round(float(faithfulness_score), 4),
        "answer_relevancy": round(float(relevancy_score), 4),
    }