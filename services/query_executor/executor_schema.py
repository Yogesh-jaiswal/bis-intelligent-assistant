from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from services.query_analyser.analyser_schema import DatabaseOperation


class ExecutionStatus(str, Enum):
    """Status indicating the outcome of database query execution."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_RECORDS_FOUND = "NO_RECORDS_FOUND"
    SKIPPED_NOT_RELEVANT = "SKIPPED_NOT_RELEVANT"
    SKIPPED_DB_NOT_REQUIRED = "SKIPPED_DB_NOT_REQUIRED"
    SKIPPED_MISSING_REQUIRED_PARAMS = "SKIPPED_MISSING_REQUIRED_PARAMS"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    ERROR = "ERROR"


class QueryExecutionResult(BaseModel):
    """
    Structured outcome of the Database Execution layer.
    
    Preserves records and errors grouped by individual DatabaseOperation,
    allowing the downstream ContextBuilder and LLM to identify the exact provenance
    of each data record without flattening them ambiguously.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    executed: bool = Field(
        ...,
        description="True if at least one database operation was executed; False if execution was skipped or blocked."
    )

    status: ExecutionStatus = Field(
        ...,
        description="Detailed execution status outcome (SUCCESS, PARTIAL_SUCCESS, NO_RECORDS_FOUND, ERROR, etc.)."
    )

    results: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Grouped records mapped by operation name (e.g. {'FIND_STANDARD': [...], 'FIND_LABORATORIES': [...]})."
    )

    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Error messages mapped by operation name for any operations that failed."
    )

    operations_executed: list[DatabaseOperation] = Field(
        default_factory=list,
        description="List of database operations executed during this run."
    )

    record_count: int = Field(
        default=0,
        description="Total number of records returned across all executed operations."
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Required parameters that were missing and prevented query execution."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata including applied filters and derived parameters."
    )

    @property
    def data(self) -> list[dict[str, Any]]:
        """Convenience property flattening all records across operations."""
        flattened = []
        for records in self.results.values():
            flattened.extend(records)
        return flattened

    @property
    def operation(self) -> DatabaseOperation | None:
        """Backward-compatibility property returning the first operation executed or None."""
        return self.operations_executed[0] if self.operations_executed else None

    @property
    def error_message(self) -> str | None:
        """Backward-compatibility property returning aggregated error messages."""
        if not self.errors:
            return None
        return "; ".join(f"{op}: {err}" for op, err in self.errors.items())
