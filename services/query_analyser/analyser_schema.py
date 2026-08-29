from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class QueryIntent(str, Enum):
    """Supported intent classifications for user queries."""
    STANDARD_LOOKUP = "STANDARD_LOOKUP"
    PRODUCT_STANDARD_RECOMMENDATION = "PRODUCT_STANDARD_RECOMMENDATION"
    CERTIFICATION_REQUIREMENT = "CERTIFICATION_REQUIREMENT"
    CERTIFICATION_PROCESS = "CERTIFICATION_PROCESS"
    TESTING_REQUIREMENT = "TESTING_REQUIREMENT"
    LABORATORY_LOOKUP = "LABORATORY_LOOKUP"
    BIS_SERVICE_LOOKUP = "BIS_SERVICE_LOOKUP"
    TECHNICAL_QUESTION = "TECHNICAL_QUESTION"
    GENERAL_BIS_QUERY = "GENERAL_BIS_QUERY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class DatabaseOperation(str, Enum):
    """High-level database operations mapped to safe backend SQL queries."""
    FIND_STANDARD = "FIND_STANDARD"
    FIND_PRODUCT = "FIND_PRODUCT"
    FIND_APPLICABLE_STANDARDS = "FIND_APPLICABLE_STANDARDS"
    GET_CERTIFICATION_REQUIREMENT = "GET_CERTIFICATION_REQUIREMENT"
    GET_CERTIFICATION_SCHEME = "GET_CERTIFICATION_SCHEME"
    GET_BIS_SERVICE = "GET_BIS_SERVICE"
    FIND_LABORATORIES = "FIND_LABORATORIES"


class QueryPlan(BaseModel):
    """
    Structured execution plan produced by the Query Analyzer.
    
    Informs downstream retrieval and generation stages about user intent,
    relevance, data requirements, and parameter extraction without executing queries.
    Supports multiple concurrent database operations.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    normalized_query: str = Field(
        ...,
        description="Normalized English translation and rephrasing of the user's query, preserving the original intent."
    )

    relevant: bool = Field(
        ...,
        description="True if the query is within the domain of BIS, Indian Standards, or conformity assessment; False otherwise."
    )

    intent: QueryIntent = Field(
        ...,
        description="Classified user intent for domain routing."
    )

    response_language: str = Field(
        default="en",
        description="ISO 639-1 language code or language name for the final response to match the user's language (e.g., 'en', 'hi')."
    )

    needs_db: bool = Field(
        ...,
        description="Whether structured metadata retrieval from PostgreSQL (standards, labs, schemes, services) is required."
    )

    needs_rag: bool = Field(
        ...,
        description="Whether semantic vector search / RAG over standard documents, clauses, test methods, or gazettes is required."
    )

    db_operations: list[DatabaseOperation] = Field(
        default_factory=list,
        description="List of high-level database operations to execute when needs_db is true. Must be empty when needs_db is false."
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted key-value parameters for DB lookup/filtering (e.g., standard_number, product, voltage, state, district)."
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Essential details missing from the user query needed to provide a complete answer (e.g., ['product type', 'voltage rating'])."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_and_normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle legacy single db_operation if supplied
            if "db_operation" in data and "db_operations" not in data:
                single_op = data.pop("db_operation")
                data["db_operations"] = [single_op] if single_op else []

            # If not relevant or not needs_db, ensure db_operations is empty
            if not data.get("relevant", True) or not data.get("needs_db", False):
                data["db_operations"] = []

        return data

    @property
    def db_operation(self) -> DatabaseOperation | None:
        """Backward compatibility property returning the first operation if present."""
        return self.db_operations[0] if self.db_operations else None
