import logging
from configs import get_settings
from services.ai.engine import AIEngine

from .analyser_prompt import build_analyser_prompt
from .analyser_schema import DatabaseOperation, QueryIntent, QueryPlan

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes natural-language user queries to determine BIS domain relevance,
    user intent, normalized English phrasing, retrieval sources (PostgreSQL vs Vector RAG),
    high-level database operations, extracted parameters, and missing information.
    """

    def __init__(self, ai_engine: AIEngine | None = None):
        """
        Initialize the QueryAnalyzer with an injected AIEngine instance or default provider.
        """
        self.ai_engine = ai_engine or AIEngine(get_settings().AI_MODEL)

    def analyze(self, query: str) -> QueryPlan:
        """
        Analyze the given user query and return a validated QueryPlan.

        :param query: Natural language input string from the user.
        :return: Validated QueryPlan instance.
        """
        clean_query = query.strip() if query else ""

        if not clean_query:
            return QueryPlan(
                normalized_query="",
                relevant=False,
                intent=QueryIntent.OUT_OF_SCOPE,
                response_language="en",
                needs_db=False,
                needs_rag=False,
                db_operations=[],
                parameters={},
                missing_information=["Empty query content"],
            )

        prompt = build_analyser_prompt(clean_query)

        try:
            raw_response = self.ai_engine.complete(
                prompt=prompt,
                response_schema=QueryPlan,
            )

            if isinstance(raw_response, QueryPlan):
                return raw_response

            return QueryPlan.model_validate(raw_response)

        except Exception as e:
            logger.exception(f"QueryAnalyzer failed to analyze query: {clean_query}")
            raise e
