from unittest.mock import MagicMock
import pytest

from exceptions import DatabaseError
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.query_executor import ExecutionStatus, QueryExecutionResult, QueryExecutor


@pytest.fixture
def mock_repos():
    """Fixture providing mocked repositories for dependency injection."""
    return {
        "standard_repo": MagicMock(),
        "product_repo": MagicMock(),
        "cert_repo": MagicMock(),
        "lab_repo": MagicMock(),
        "service_repo": MagicMock(),
    }


@pytest.fixture
def executor(mock_repos):
    """Fixture providing a QueryExecutor instance configured with mocked repositories."""
    return QueryExecutor(
        standard_repo=mock_repos["standard_repo"],
        product_repo=mock_repos["product_repo"],
        cert_repo=mock_repos["cert_repo"],
        lab_repo=mock_repos["lab_repo"],
        service_repo=mock_repos["service_repo"],
    )


def test_find_standard_mapping(executor, mock_repos):
    """Test FIND_STANDARD operation maps parameters and returns results."""
    mock_repos["standard_repo"].find_standard.return_value = [
        {"id": "1", "is_number": "IS 694:2010", "title": "PVC Cables"}
    ]

    plan = QueryPlan(
        normalized_query="What is IS 694:2010?",
        relevant=True,
        intent=QueryIntent.STANDARD_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_STANDARD,
        parameters={"standard_number": "IS 694:2010", "limit": 5},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.operation == DatabaseOperation.FIND_STANDARD
    assert result.record_count == 1
    assert result.data[0]["is_number"] == "IS 694:2010"

    mock_repos["standard_repo"].find_standard.assert_called_once_with(
        standard_number="IS 694:2010",
        title=None,
        status=None,
        technical_department=None,
        limit=5,
    )


def test_find_product_mapping(executor, mock_repos):
    """Test FIND_PRODUCT operation maps parameters correctly."""
    mock_repos["product_repo"].find_product.return_value = [
        {"id": "1", "name": "PVC Cable", "category": "Electrical"}
    ]

    plan = QueryPlan(
        normalized_query="Find PVC cable product",
        relevant=True,
        intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_PRODUCT,
        parameters={"product": "PVC Cable", "category": "Electrical"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["product_repo"].find_product.assert_called_once_with(
        name="PVC Cable",
        category="Electrical",
        keyword=None,
        limit=10,
    )


def test_find_applicable_standards_mapping(executor, mock_repos):
    """Test FIND_APPLICABLE_STANDARDS maps product name and category."""
    mock_repos["product_repo"].find_applicable_standards.return_value = [
        {"product_name": "PVC Cable", "is_number": "IS 694:2010", "relevance": "Primary"}
    ]

    plan = QueryPlan(
        normalized_query="Which standard applies to PVC cables?",
        relevant=True,
        intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_APPLICABLE_STANDARDS,
        parameters={"product": "PVC Cable"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["product_repo"].find_applicable_standards.assert_called_once_with(
        product_name="PVC Cable",
        category=None,
        standard_number=None,
        relevance=None,
        limit=10,
    )


def test_find_laboratories_filtering(executor, mock_repos):
    """Test FIND_LABORATORIES passes state, district, and scope filters."""
    mock_repos["lab_repo"].find_laboratories.return_value = [
        {"id": "1", "name": "CETL", "state": "Tamil Nadu", "district": "Thiruvallur"}
    ]

    plan = QueryPlan(
        normalized_query="Find labs testing PVC cables in Tamil Nadu",
        relevant=True,
        intent=QueryIntent.LABORATORY_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_LABORATORIES,
        parameters={"state": "Tamil Nadu", "district": "Thiruvallur", "product": "PVC cables"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["lab_repo"].find_laboratories.assert_called_once_with(
        state="Tamil Nadu",
        district="Thiruvallur",
        scope_keyword=None,
        standard_number=None,
        product="PVC cables",
        lab_code=None,
        name=None,
        limit=20,
    )


def test_get_certification_requirement(executor, mock_repos):
    """Test GET_CERTIFICATION_REQUIREMENT operation."""
    mock_repos["cert_repo"].find_certification_requirements.return_value = [
        {"is_number": "IS 694:2010", "requirement_type": "Compulsory under QCO", "mandatory": "Yes"}
    ]

    plan = QueryPlan(
        normalized_query="Is BIS certification compulsory for IS 694?",
        relevant=True,
        intent=QueryIntent.CERTIFICATION_REQUIREMENT,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.GET_CERTIFICATION_REQUIREMENT,
        parameters={"standard_number": "IS 694"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["cert_repo"].find_certification_requirements.assert_called_once_with(
        standard_number="IS 694",
        scheme_code=None,
        mandatory=None,
        requirement_type=None,
        limit=10,
    )


def test_get_certification_scheme(executor, mock_repos):
    """Test GET_CERTIFICATION_SCHEME operation."""
    mock_repos["cert_repo"].find_certification_scheme.return_value = [
        {"scheme_code": "Scheme-I", "name": "Scheme I (ISI Mark Scheme)"}
    ]

    plan = QueryPlan(
        normalized_query="What is Scheme-I?",
        relevant=True,
        intent=QueryIntent.CERTIFICATION_PROCESS,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.GET_CERTIFICATION_SCHEME,
        parameters={"scheme_code": "Scheme-I"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["cert_repo"].find_certification_scheme.assert_called_once_with(
        scheme_code="Scheme-I",
        name=None,
        certification_type=None,
        limit=10,
    )


def test_get_bis_service(executor, mock_repos):
    """Test GET_BIS_SERVICE operation."""
    mock_repos["service_repo"].get_bis_service.return_value = [
        {"name": "Grant of Licence", "service_type": "Product Certification"}
    ]

    plan = QueryPlan(
        normalized_query="How to grant licence for ISI mark?",
        relevant=True,
        intent=QueryIntent.BIS_SERVICE_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.GET_BIS_SERVICE,
        parameters={"service_type": "Product Certification"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.record_count == 1
    mock_repos["service_repo"].get_bis_service.assert_called_once_with(
        name=None,
        service_type="Product Certification",
        keyword=None,
        limit=5,
    )


def test_not_relevant_skips_execution(executor, mock_repos):
    """Test that out-of-scope queries skip database execution."""
    plan = QueryPlan(
        normalized_query="What is the weather today?",
        relevant=False,
        intent=QueryIntent.OUT_OF_SCOPE,
        response_language="en",
        needs_db=False,
        needs_rag=False,
        db_operation=None,
        parameters={},
    )

    result = executor.execute(plan)

    assert result.executed is False
    assert result.status == ExecutionStatus.SKIPPED_NOT_RELEVANT
    assert len(result.data) == 0
    mock_repos["standard_repo"].find_standard.assert_not_called()


def test_needs_db_false_skips_execution(executor):
    """Test that needs_db=False skips DB execution even if relevant."""
    plan = QueryPlan(
        normalized_query="Explain testing methodology text",
        relevant=True,
        intent=QueryIntent.TECHNICAL_QUESTION,
        response_language="en",
        needs_db=False,
        needs_rag=True,
        db_operation=None,
        parameters={},
    )

    result = executor.execute(plan)

    assert result.executed is False
    assert result.status == ExecutionStatus.SKIPPED_DB_NOT_REQUIRED
    assert len(result.data) == 0


def test_missing_required_params_blocks_execution(executor, mock_repos):
    """Test that missing mandatory parameters prevents invalid execution."""
    plan = QueryPlan(
        normalized_query="Which standard applies to my product?",
        relevant=True,
        intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_APPLICABLE_STANDARDS,
        parameters={},
        missing_information=["product type"],
    )

    result = executor.execute(plan)

    assert result.executed is False
    assert result.status == ExecutionStatus.SKIPPED_MISSING_REQUIRED_PARAMS
    assert "product type" in result.missing_information
    mock_repos["product_repo"].find_applicable_standards.assert_not_called()


def test_empty_result_returns_no_records_found(executor, mock_repos):
    """Test that empty DB query returns NO_RECORDS_FOUND status."""
    mock_repos["standard_repo"].find_standard.return_value = []

    plan = QueryPlan(
        normalized_query="What is IS 999999?",
        relevant=True,
        intent=QueryIntent.STANDARD_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_STANDARD,
        parameters={"standard_number": "IS 999999"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.NO_RECORDS_FOUND
    assert result.record_count == 0
    assert len(result.data) == 0


def test_database_exception_handling(executor, mock_repos):
    """Test that repository DatabaseError is caught and packaged into error status."""
    mock_repos["standard_repo"].find_standard.side_effect = DatabaseError("Connection timed out")

    plan = QueryPlan(
        normalized_query="What is IS 694:2010?",
        relevant=True,
        intent=QueryIntent.STANDARD_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
        db_operation=DatabaseOperation.FIND_STANDARD,
        parameters={"standard_number": "IS 694:2010"},
    )

    result = executor.execute(plan)

    assert result.executed is True
    assert result.status == ExecutionStatus.ERROR
    assert "Connection timed out" in result.error_message
