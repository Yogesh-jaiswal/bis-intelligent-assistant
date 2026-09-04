# BIS Intelligent Assistant - End-to-End (E2E) System Testing Harness

The `tests/e2e/` package provides an automated, standalone end-to-end verification harness for the BIS Intelligent Assistant. It validates the full real-world stack—from infrastructure services (PostgreSQL, Alembic migrations, Docker) through Flask API request handling, Ollama AI model completion, deterministic retrieval, and response synthesis—against the live HTTP API.

---

## 1. Why These E2E Tests Are Separate From CI Tests

> [!IMPORTANT]
> **This is NOT a CI test suite.**
>
> Standard CI pipelines run fast unit and integration tests under `tests/features/` with deterministic fixtures or mock providers (`AI_PROVIDER=FAKE`).
>
> These E2E tests require:
> 1. A live **Ollama** server hosting the real target model (`MODEL_NAME=qwen3:8b` or configured equivalent).
> 2. A live **PostgreSQL** database with `pgvector` extension and seeded Indian Standards/services dataset.
> 3. Applied database migrations at `head` revision.
>
> Do **NOT** include `tests/e2e/run_e2e.py` in regular CI workflows unless the CI environment explicitly provisions Ollama with GPU acceleration and PostgreSQL with vector extensions.

---

## 2. Directory Structure

```text
tests/
└── e2e/
    ├── __init__.py
    ├── run_e2e.py            # Main executable runner CLI
    ├── scenarios.py          # Defined multi-turn conversation scenarios
    ├── helpers.py            # Pre-flight infrastructure checks & API client
    ├── results/              # Timestamped output JSON reports
    │   └── .gitkeep
    └── README.md             # Documentation & operational guide
```

---

## 3. Required Services & Pre-Flight Checks

Before executing conversations, the runner performs automated pre-flight checks:

| Step | Component | Detection & Verification Mechanism | Failure Handling |
| :--- | :--- | :--- | :--- |
| **1** | **Configuration** | Reads settings using existing `configs.get_settings()`. Checks `MODEL_URL`, `POSTGRES_*`, `HOST`, `PORT`. | Aborts with status code `1` if required configuration is missing. |
| **2** | **Ollama Model** | Sends a lightweight HTTP GET request to `{MODEL_URL}` and `{MODEL_URL}/api/tags` with a 4.0s timeout. | Fails immediately: `E2E TEST ABORTED\nReason: Ollama/model endpoint unavailable`. |
| **3** | **PostgreSQL** | Attempts connection via `psycopg.connect()` using project DB credentials and queries `SELECT 1;`. | If unreachable, attempts non-destructive startup via `docker compose up -d db`. If still down, aborts with status code `1`. |
| **4** | **Migrations** | Uses Alembic `MigrationContext` and `ScriptDirectory` on `migrations/` to verify current DB revision matches the `head` revision. | Aborts if revisions mismatch or unapplied migrations exist. |
| **5** | **API Server** | Checks `http://127.0.0.1:5000/v1/health`. If alive, reuses existing server. If not alive, spawns `sys.executable run.py` as a subprocess and polls `/v1/health` until ready. | Terminates child process cleanly on normal exit, SIGINT/SIGTERM, or error. |

---

## 4. Test Limits & Execution Model

To avoid stressing local LLM endpoints and to observe pipeline behavior reliably:

* **Maximum Conversations**: 3
* **Maximum Total Questions**: 10
* **Execution**: Strictly synchronous (one request at a time; no concurrent requests)
* **Cooldown**: ~1.0 second pause between consecutive questions

---

## 5. How to Run

Run the suite from the repository root:

```bash
# Using the project virtual environment
python tests/e2e/run_e2e.py
```

Or on Windows:

```powershell
.venv\Scripts\python tests/e2e/run_e2e.py
```

### Expected Output Example:

```text
================================================================
BIS CHATBOT E2E SYSTEM TEST
================================================================

[1/6] Checking configuration...
[PASS] Configuration loaded (Model: qwen3:8b, Provider: OLLAMA)

[2/6] Checking Ollama...
[PASS] Ollama endpoint reachable (200)

[3/6] Checking PostgreSQL...
[PASS] PostgreSQL reachable and responding to queries

[4/6] Checking migrations...
[PASS] Database schema is up to date (Revision: 433254697eef)

[5/6] Checking API server...
[PASS] Server started and healthy after 1.82s

[6/6] Running E2E conversations...

----------------------------------------------------------------
Conversation 1/3: general_services_and_laboratories
(Validates structured DB retrieval for BIS services and follow-up lab inquiries.)
----------------------------------------------------------------

Question 1/3
> What services does BIS provide?
[REQUEST] Sending request...
[PASS] HTTP 200 in 7.42s
[CONVERSATION] conv_01K8V6...
[MESSAGE TYPE] answer
[CARDS] 5 | [CITATIONS] 0
Waiting 1 second cooldown before next request...
...
```

---

## 6. Response Validation Logic

For every request to `POST /v1/query`, the harness validates:
1. **HTTP Status**: Must be `200`.
2. **Envelope**: Must contain `success: true`, `error: null`, and `data: {...}`.
3. **Payload Shape**:
   - `conversation_id`: Non-empty string.
   - `message_type`: Either `"answer"` or `"clarification"`.
   - `message`: Non-empty synthesized text response.
   - `citations`: List of structured citation objects.
   - `data`: List of structured data cards.
4. **Classification**:
   - **`PASS`**: Structurally valid, all expected evidence present.
   - **`WARNING`**: Structurally valid, but optional soft expectation not met (e.g. expected cards on structured intent).
   - **`FAIL`**: HTTP error, JSON schema violation, empty message, or server crash.

---

## 7. Results & JSON Reports

Every test run writes a timestamped report to:

```text
tests/e2e/results/e2e_YYYY-MM-DD_HH-MM-SS.json
```

Previous test runs are never overwritten.

### JSON Structure:

```json
{
  "run": {
    "started_at": "2026-09-04T15:45:00.000000+00:00",
    "finished_at": "2026-09-04T15:46:15.000000+00:00",
    "duration_ms": 75120.4,
    "status": "passed",
    "configuration": {
      "ai_provider": "OLLAMA",
      "model_name": "qwen3:8b",
      "model_url": "http://localhost:11434"
    },
    "ollama_available": true,
    "postgres_available": true,
    "migrations_ok": true,
    "api_server_available": true
  },
  "conversations": [
    {
      "name": "general_services_and_laboratories",
      "conversation_id": "conv_xxx",
      "status": "passed",
      "questions": [
        {
          "question": "What services does BIS provide?",
          "conversation_id_before": null,
          "conversation_id_after": "conv_xxx",
          "started_at": "...",
          "completed_at": "...",
          "latency_ms": 7420.5,
          "http_status": 200,
          "status": "PASS",
          "success": true,
          "message_type": "answer",
          "message": "BIS offers several key conformity assessment and certification services...",
          "citation_count": 0,
          "data_card_count": 5,
          "warnings": [],
          "error": null
        }
      ]
    }
  ],
  "summary": {
    "total_conversations": 3,
    "passed_conversations": 3,
    "failed_conversations": 0,
    "total_questions": 9,
    "passed_questions": 9,
    "failed_questions": 0,
    "average_latency_ms": 6820.1,
    "min_latency_ms": 4210.3,
    "max_latency_ms": 8940.6
  }
}
```

---

## 8. How to Add or Modify Scenarios

Edit [`tests/e2e/scenarios.py`](./scenarios.py).

Add or update a `ConversationScenario`:

```python
ConversationScenario(
    name="my_new_scenario",
    description="Validates specific standard technical requirements.",
    questions=[
        QuestionSpec(
            text="What are the test requirements for cement under IS 1489?",
            expectation=QuestionExpectation(
                description="Technical requirement RAG query",
                expect_citations=True,
                language="en",
            ),
        ),
    ],
)
```

> [!NOTE]
> Hard safety limits enforce `len(SCENARIOS) <= 3` and `total_questions <= 10`. Modifying scenarios beyond these limits will raise an assertion error.

---

## 9. Process Exit Codes

* **`0`**: Success. All pre-flight checks succeeded, and all conversation questions passed.
* **`1`**: Failure. Infrastructure unavailable, server failed to start, or one or more conversation questions failed validation.
