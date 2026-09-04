# BIS Intelligent Assistant — Testing Guide

This document describes the testing architecture, fixture organization, test doubles, and execution instructions for the BIS Intelligent Assistant.

---

## 1. Test Suite Architecture

The repository divides tests into two isolated tiers:

```text
tests/
├── conftest.py              # Root pytest fixtures (session app, test client, mock repos)
├── features/                # Fast, deterministic unit and integration tests (CI-Ready)
│   ├── test_ai_engine.py
│   ├── test_conversation_context.py
│   ├── test_ingestion.py
│   ├── test_query_analyser.py
│   ├── test_query_executor.py
│   ├── test_query_pipeline.py
│   ├── test_response_serialization.py
│   ├── test_semantic_preservation.py
│   ├── test_service_routing_and_rag_prevention.py
│   └── test_synthesis_prompt.py
├── fixtures/                # Service-specific test fixtures and payload factories
│   ├── __init__.py          # Re-exports all service fixtures
│   ├── chat_fixtures.py     # Chat schemas, message builders, sample user queries
│   ├── standard_fixtures.py # Indian Standards mock records and catalog entries
│   ├── service_fixtures.py  # BIS certification and conformity schemes
│   └── sample_payloads.py   # Backward-compatible aggregate module
├── fakes/                   # Deterministic mocks & test doubles
│   └── fake_services.py     # In-memory repositories & fake AI provider
├── profiles/                # Seed profiles for testing
│   └── test_profiles.py
└── e2e/                     # Standalone real-world system harness (NOT for CI)
    ├── run_e2e.py           # CLI runner
    ├── scenarios.py         # Multi-turn conversation scenarios
    ├── helpers.py           # Infrastructure checks & API client
    ├── results/             # Timestamped JSON run reports
    └── README.md            # Detailed E2E documentation
```

---

## 2. Infrastructure Requirements for Testing

> [!IMPORTANT]
> **Tests do NOT require the containerized application to be running.**
> Only the PostgreSQL database container is needed when running tests.

* **Database Service**:
  Start PostgreSQL with pgvector:
  ```bash
  docker compose up -d
  ```
* **No App Startup Needed**:
  Pytest runs using Flask's in-process `test_client()` and mocked or local providers. You do **not** need to run `docker-compose.app.yml` during test execution.

---

## 3. Service Fixtures (`tests/fixtures/`)

Fixtures are organized by domain rather than lumped into a single monolithic file:

| Module | Contents | Primary Usage |
| :--- | :--- | :--- |
| [`./tests/fixtures/chat_fixtures.py`](./tests/fixtures/chat_fixtures.py) | `make_chat_request`, `SAMPLE_CABLE_QUERY`, `SAMPLE_SERVICE_QUERY`, `sample_cable_chat_request`, etc. | Testing query analysis, intent classification, and chat endpoints |
| [`./tests/fixtures/standard_fixtures.py`](./tests/fixtures/standard_fixtures.py) | `SAMPLE_STANDARD_RECORDS`, `sample_standard_records` | Testing standards repository queries and vector search integration |
| [`./tests/fixtures/service_fixtures.py`](./tests/fixtures/service_fixtures.py) | `SAMPLE_SERVICE_RECORDS`, `sample_service_records` | Testing BIS scheme lookups, ISI mark queries, CRS schemes |
| [`./tests/fixtures/sample_payloads.py`](./tests/fixtures/sample_payloads.py) | Legacy aggregate module re-exporting all service fixtures | Backward compatibility for tests importing `sample_payloads` |

---

## 4. Running the Feature Tests (CI Suite)

The feature test suite under `tests/features/` is completely deterministic, fast, and does not depend on an external Ollama endpoint.

### Step 1: Start Database Container
Before running tests, start the PostgreSQL database container (`docker-compose.yml` only provisions the `db` service):
```bash
docker compose up -d
```

### Step 2: Ensure Migrations Are Applied
```bash
alembic upgrade head
```

### Step 3: Run Feature Tests
```bash
pytest tests/features/ -v
```

### Run Specific Test Modules
* **Synthesis Prompt Structure & Response Schema**:
  ```bash
  pytest tests/features/test_synthesis_prompt.py -v
  ```
* **Service Query Routing & RAG Prevention**:
  ```bash
  pytest tests/features/test_service_routing_and_rag_prevention.py -v
  ```
* **Semantic Preservation & Multi-turn Continuity**:
  ```bash
  pytest tests/features/test_semantic_preservation.py -v
  ```
* **Query Pipeline & End-to-End Multi-Hop**:
  ```bash
  pytest tests/features/test_query_pipeline.py -v
  ```

---

## 5. End-to-End System Tests (`tests/e2e/`)

The E2E harness tests the entire live stack against real-world conversations using the actual Ollama LLM (`qwen3:8b`), live PostgreSQL database, Alembic migrations, and the HTTP API server.

```bash
python tests/e2e/run_e2e.py
```

For complete documentation on scenarios, pre-flight checks, and output reports, refer to [`./tests/e2e/README.md`](./tests/e2e/README.md).

---

## 6. Continuous Integration (GitHub Actions)

CI is configured under [`.github/workflows/ci.yml`](./.github/workflows/ci.yml):
* Provisions a PostgreSQL 16 + pgvector container service.
* Installs dependencies from `requirements.txt`.
* Applies Alembic migrations (`alembic upgrade head`).
* Runs `pytest tests/features/ -v` with `AI_PROVIDER=FAKE`.
