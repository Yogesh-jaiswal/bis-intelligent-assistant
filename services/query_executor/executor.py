import logging
from typing import Any

from services.query_analyser.analyser_schema import DatabaseOperation, QueryPlan
from .executor_schema import ExecutionStatus, QueryExecutionResult

from repositories.standard_repository import StandardRepository
from repositories.product_repository import ProductRepository
from repositories.certification_repository import CertificationRepository
from repositories.laboratory_repository import LaboratoryRepository
from repositories.service_repository import ServiceRepository

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Executes deterministic database queries mapped from a validated QueryPlan.
    
    Supports sequential multi-operation execution, parameter sharing/derivation
    between operations, and isolated per-operation error handling.
    """

    def __init__(
        self,
        standard_repo: type[StandardRepository] | Any = StandardRepository,
        product_repo: type[ProductRepository] | Any = ProductRepository,
        cert_repo: type[CertificationRepository] | Any = CertificationRepository,
        lab_repo: type[LaboratoryRepository] | Any = LaboratoryRepository,
        service_repo: type[ServiceRepository] | Any = ServiceRepository,
    ):
        """
        Initialize QueryExecutor with injected or default repositories.
        """
        self.standard_repo = standard_repo
        self.product_repo = product_repo
        self.cert_repo = cert_repo
        self.lab_repo = lab_repo
        self.service_repo = service_repo

        # Explicit deterministic dispatch mapping
        self._dispatch_map = {
            DatabaseOperation.FIND_STANDARD: self._execute_find_standard,
            DatabaseOperation.FIND_PRODUCT: self._execute_find_product,
            DatabaseOperation.FIND_APPLICABLE_STANDARDS: self._execute_find_applicable_standards,
            DatabaseOperation.GET_CERTIFICATION_REQUIREMENT: self._execute_get_certification_requirement,
            DatabaseOperation.GET_CERTIFICATION_SCHEME: self._execute_get_certification_scheme,
            DatabaseOperation.GET_BIS_SERVICE: self._execute_get_bis_service,
            DatabaseOperation.FIND_LABORATORIES: self._execute_find_laboratories,
        }

    def execute(self, plan: QueryPlan) -> QueryExecutionResult:
        """
        Execute all database operations in the QueryPlan sequentially.

        :param plan: Validated QueryPlan from QueryAnalyzer.
        :return: QueryExecutionResult containing grouped results or skip/error reasons.
        """
        # Step 1: Check relevance
        if not plan.relevant:
            return QueryExecutionResult(
                executed=False,
                status=ExecutionStatus.SKIPPED_NOT_RELEVANT,
                results={},
                errors={},
                operations_executed=[],
                metadata={"reason": "Query was classified as out-of-scope for BIS."},
            )

        # Step 2: Check if database retrieval is required
        if not plan.needs_db or not plan.db_operations:
            return QueryExecutionResult(
                executed=False,
                status=ExecutionStatus.SKIPPED_DB_NOT_REQUIRED,
                results={},
                errors={},
                operations_executed=[],
                metadata={"reason": "Query plan indicates structured database retrieval is not required."},
            )

        results: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}
        operations_executed: list[DatabaseOperation] = []
        derived_params: dict[str, Any] = dict(plan.parameters or {})

        # Step 3: Execute each operation in plan.db_operations
        for op in plan.db_operations:
            handler = self._dispatch_map.get(op)
            op_key = op.value if isinstance(op, DatabaseOperation) else str(op)

            if not handler:
                errors[op_key] = f"Database operation '{op}' is not supported."
                continue

            try:
                op_records, op_skipped = handler(derived_params, plan)

                if op_skipped:
                    continue

                results[op_key] = op_records
                operations_executed.append(op)

                # Parameter Derivation: enrich derived_params for downstream operations
                if op == DatabaseOperation.FIND_APPLICABLE_STANDARDS or op == DatabaseOperation.FIND_STANDARD:
                    if op_records and "standard_number" not in derived_params:
                        first_std = op_records[0].get("is_number")
                        if first_std:
                            derived_params["standard_number"] = first_std
                            derived_params["is_number"] = first_std

                elif op == DatabaseOperation.FIND_PRODUCT:
                    if op_records and "product" not in derived_params:
                        first_prod = op_records[0].get("name")
                        if first_prod:
                            derived_params["product"] = first_prod
                            derived_params["product_name"] = first_prod

            except Exception as e:
                logger.exception(f"Error executing database operation '{op}'")
                errors[op_key] = str(e)
                # Note: We continue loop so other operations can still succeed!

        # Step 4: Determine overall status
        total_records = sum(len(recs) for recs in results.values())
        executed_count = len(operations_executed)

        if executed_count == 0:
            if errors:
                status = ExecutionStatus.ERROR
            elif plan.missing_information:
                status = ExecutionStatus.SKIPPED_MISSING_REQUIRED_PARAMS
            else:
                status = ExecutionStatus.SKIPPED_DB_NOT_REQUIRED
            executed = False
        else:
            executed = True
            if errors:
                status = ExecutionStatus.PARTIAL_SUCCESS if total_records > 0 else ExecutionStatus.ERROR
            elif total_records > 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.NO_RECORDS_FOUND

        return QueryExecutionResult(
            executed=executed,
            status=status,
            results=results,
            errors=errors,
            operations_executed=operations_executed,
            record_count=total_records,
            missing_information=plan.missing_information if total_records == 0 and plan.missing_information else [],
            metadata={
                "applied_parameters": derived_params,
                "operations_planned": [op.value if isinstance(op, DatabaseOperation) else str(op) for op in plan.db_operations],
            },
        )

    # ------------------------------------------------------------------
    # Operation Handlers (return tuple: (records, is_skipped))
    # ------------------------------------------------------------------

    def _execute_find_standard(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        std_num = params.get("standard_number") or params.get("is_number")
        title = params.get("title") or params.get("keyword")

        # Skip if mandatory identification parameter is missing
        if not std_num and not title and plan.missing_information:
            return [], True

        data = self.standard_repo.find_standard(
            standard_number=str(std_num) if std_num else None,
            title=str(title) if title else None,
            status=params.get("status"),
            technical_department=params.get("technical_department") or params.get("department"),
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_find_product(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        name = params.get("product") or params.get("name") or params.get("product_name")
        category = params.get("category") or params.get("product_category")
        keyword = params.get("keyword")

        data = self.product_repo.find_product(
            name=str(name) if name else None,
            category=str(category) if category else None,
            keyword=str(keyword) if keyword else None,
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_find_applicable_standards(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        product_name = params.get("product") or params.get("product_name") or params.get("product_type")
        category = params.get("category")
        std_num = params.get("standard_number") or params.get("is_number")

        # If no search terms exist and missing information was flagged
        if not product_name and not category and not std_num and plan.missing_information:
            return [], True

        data = self.product_repo.find_applicable_standards(
            product_name=str(product_name) if product_name else None,
            category=str(category) if category else None,
            standard_number=str(std_num) if std_num else None,
            relevance=params.get("relevance"),
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_get_certification_requirement(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        std_num = params.get("standard_number") or params.get("is_number")
        scheme_code = params.get("scheme_code") or params.get("certification_scheme")
        req_type = params.get("requirement_type")
        mandatory = params.get("mandatory")

        data = self.cert_repo.find_certification_requirements(
            standard_number=str(std_num) if std_num else None,
            scheme_code=str(scheme_code) if scheme_code else None,
            mandatory=str(mandatory) if mandatory else None,
            requirement_type=str(req_type) if req_type else None,
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_get_certification_scheme(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        scheme_code = params.get("scheme_code") or params.get("certification_scheme")
        name = params.get("name") or params.get("scheme_name")
        cert_type = params.get("certification_type")

        data = self.cert_repo.find_certification_scheme(
            scheme_code=str(scheme_code) if scheme_code else None,
            name=str(name) if name else None,
            certification_type=str(cert_type) if cert_type else None,
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_get_bis_service(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        name = params.get("service") or params.get("name") or params.get("service_name")
        service_type = params.get("service_type")
        keyword = params.get("keyword")

        data = self.service_repo.get_bis_service(
            name=str(name) if name else None,
            service_type=str(service_type) if service_type else None,
            keyword=str(keyword) if keyword else None,
            limit=params.get("limit", 10),
        )
        return data, False

    def _execute_find_laboratories(self, params: dict[str, Any], plan: QueryPlan) -> tuple[list[dict[str, Any]], bool]:
        state = params.get("state")
        district = params.get("district")
        std_num = params.get("standard_number") or params.get("is_number")
        product = params.get("product") or params.get("product_name")
        scope_kw = params.get("scope") or params.get("scope_keyword")
        lab_code = params.get("lab_code")
        name = params.get("name") or params.get("laboratory_name")

        data = self.lab_repo.find_laboratories(
            state=str(state) if state else None,
            district=str(district) if district else None,
            scope_keyword=str(scope_kw) if scope_kw else None,
            standard_number=str(std_num) if std_num else None,
            product=str(product) if product else None,
            lab_code=str(lab_code) if lab_code else None,
            name=str(name) if name else None,
            limit=params.get("limit", 20),
        )
        return data, False
