"""
tests/e2e/helpers.py
====================
Infrastructure verification, process lifecycle management, API client,
and response validation for the BIS Assistant E2E system testing harness.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from configs import get_settings
from tests.e2e.scenarios import QuestionSpec


# ==============================================================================
# 1. Pre-Flight Infrastructure Checks
# ==============================================================================

def check_configuration() -> tuple[bool, str, dict[str, Any]]:
    """
    Verify application configuration presence using existing settings.
    Does not expose sensitive credentials.
    """
    try:
        settings = get_settings()
        info = {
            "ai_provider": settings.AI_PROVIDER,
            "model_url": settings.MODEL_URL,
            "model_name": settings.MODEL_NAME,
            "postgres_host": settings.POSTGRES_HOST,
            "postgres_port": settings.POSTGRES_PORT,
            "postgres_db": settings.POSTGRES_DB,
            "api_host": settings.HOST,
            "api_port": settings.PORT,
        }
        if not settings.MODEL_URL:
            return False, "Missing MODEL_URL in configuration", info
        if not settings.POSTGRES_HOST:
            return False, "Missing POSTGRES_HOST in configuration", info
        return True, "Configuration loaded successfully", info
    except Exception as e:
        return False, f"Failed to load application settings: {e}", {}


def check_ollama(model_url: str, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    """
    Perform a lightweight HTTP check against the configured model endpoint.
    Fails early if unreachable.
    """
    cleaned_url = model_url.rstrip("/")
    probe_url = f"{cleaned_url}/api/tags"

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            resp = client.get(probe_url)
            if resp.status_code == 200:
                return True, f"Ollama endpoint reachable ({resp.status_code})"
            if resp.status_code == 530:
                return False, "Ollama/model endpoint unavailable (Cloudflare tunnel HTTP 530: Tunnel inactive or origin unreachable)"
            if resp.status_code == 404:
                # Some proxies only expose root
                r2 = client.get(cleaned_url)
                if r2.status_code == 200 or "ollama" in r2.text.lower():
                    return True, f"Ollama endpoint reachable at root ({r2.status_code})"
                return False, f"Ollama endpoint responded with HTTP {resp.status_code}"
            return False, f"Ollama endpoint responded with unexpected HTTP {resp.status_code}"
    except httpx.ConnectTimeout:
        return False, f"Ollama/model endpoint unavailable (Connection timed out after {timeout_seconds}s)"
    except httpx.ConnectError as e:
        return False, f"Ollama/model endpoint unavailable (Connection refused: {e})"
    except Exception as e:
        return False, f"Ollama/model endpoint unavailable ({e})"


def check_postgres(timeout_seconds: float = 3.0) -> tuple[bool, str]:
    """
    Verify PostgreSQL availability using existing psycopg / database settings.
    If down, attempts a safe docker-compose up for the db service if available.
    """
    import psycopg

    settings = get_settings()

    def _try_connect() -> bool:
        try:
            conn = psycopg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                dbname=settings.POSTGRES_DB,
                connect_timeout=int(timeout_seconds),
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.close()
            return True
        except Exception:
            return False

    if _try_connect():
        return True, "PostgreSQL reachable and responding to queries"

    # Attempt safe recovery via docker-compose if docker is present
    try:
        compose_file = Path("docker-compose.yml")
        if compose_file.exists():
            subprocess.run(
                ["docker", "compose", "up", "-d", "db"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            time.sleep(2.0)
            if _try_connect():
                return True, "PostgreSQL started via docker-compose and responding"
    except Exception:
        pass

    return False, "PostgreSQL unavailable"


def check_migrations() -> tuple[bool, str]:
    """
    Verify database schema migrations using Alembic ScriptDirectory and MigrationContext.
    Checks that the current database revision matches the latest head revision.
    """
    try:
        from alembic.config import Config
        from alembic.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine

        settings = get_settings()
        migrations_dir = Path("migrations")
        ini_file = migrations_dir / "alembic.ini"

        if not migrations_dir.exists():
            return False, "Migrations directory not found"

        cfg = Config(str(ini_file) if ini_file.exists() else None)
        cfg.set_main_option("script_location", str(migrations_dir))
        script = ScriptDirectory.from_config(cfg)
        head_rev = script.get_current_head()

        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_rev = context.get_current_revision()

        if current_rev is None:
            return False, f"Database has unapplied migrations (Head: {head_rev}, Current: None)"

        if current_rev != head_rev:
            return False, f"Database migration mismatch (Head: {head_rev}, Current: {current_rev})"

        return True, f"Database schema is up to date (Revision: {current_rev})"

    except Exception as e:
        return False, f"Migration check failed: {e}"


# ==============================================================================
# 2. Server Process Lifecycle Manager
# ==============================================================================

class ServerManager:
    """
    Manages the API server lifecycle.
    Detects if an API server is already running. If not, spawns the local dev
    server as a child process and ensures clean termination on exit.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.health_url = f"{self.base_url}/v1/health"
        self.query_url = f"{self.base_url}/v1/query"
        self.process: subprocess.Popen[str] | None = None
        self.managed_mode: str = "none"  # "none", "docker", "subprocess"
        self.log_buffer: list[str] = []

        # Register cleanup handlers
        atexit.register(self.stop)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def is_server_healthy(self, timeout_seconds: float = 1.0) -> bool:
        """Check if the API health endpoint responds with HTTP 200."""
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                r = client.get(self.health_url)
                return r.status_code == 200
        except Exception:
            return False

    def _drain_output(self) -> None:
        """Background thread reader to prevent pipe buffer deadlocks."""
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ""):
                self.log_buffer.append(line)
                if len(self.log_buffer) > 500:
                    self.log_buffer.pop(0)

    def ensure_server_running(self, startup_timeout: float = 45.0) -> tuple[bool, str]:
        """
        Check if the server is alive. If not, start it using Docker Compose,
        falling back to local development subprocess if Docker is unavailable.
        """
        if self.is_server_healthy(timeout_seconds=1.5):
            self.managed_mode = "none"
            return True, "Existing API server detected and healthy"

        # 1. Attempt Docker Compose startup if compose files exist
        compose_file = Path("docker-compose.yml")
        app_compose_file = Path("docker-compose.app.yml")
        docker_started = False
        if compose_file.exists() and app_compose_file.exists():
            try:
                # Check if docker daemon is running
                d_check = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if d_check.returncode == 0:
                    cmd = subprocess.run(
                        [
                            "docker",
                            "compose",
                            "-f",
                            "docker-compose.yml",
                            "-f",
                            "docker-compose.app.yml",
                            "up",
                            "-d",
                            "app",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if cmd.returncode == 0:
                        docker_started = True
                        self.managed_mode = "docker"
            except Exception:
                docker_started = False

        if docker_started:
            start_time = time.monotonic()
            while time.monotonic() - start_time < startup_timeout:
                if self.is_server_healthy(timeout_seconds=1.0):
                    elapsed = time.monotonic() - start_time
                    return True, f"API server started via Docker Compose and healthy after {elapsed:.2f}s"
                time.sleep(1.0)
            # If docker startup timed out, stop container and fall back
            self.stop()

        # 2. Fallback: Spawn local development server as child process
        try:
            import threading

            self.process = subprocess.Popen(
                [sys.executable, "run.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.managed_mode = "subprocess"
            reader_thread = threading.Thread(target=self._drain_output, daemon=True)
            reader_thread.start()
        except Exception as e:
            return False, f"Failed to spawn application server: {e}"

        # Poll health endpoint until healthy or timeout
        start_time = time.monotonic()
        while time.monotonic() - start_time < startup_timeout:
            if self.process.poll() is not None:
                output = "".join(self.log_buffer[-20:])
                return False, f"Server subprocess exited unexpectedly with code {self.process.returncode}:\n{output}"

            if self.is_server_healthy(timeout_seconds=1.0):
                elapsed = time.monotonic() - start_time
                return True, f"Server started and healthy after {elapsed:.2f}s"

            time.sleep(0.5)

        output = "".join(self.log_buffer[-20:])
        self.stop()
        return False, f"Server failed to become healthy within {startup_timeout}s. Last logs:\n{output}"

    def stop(self) -> None:
        """Terminate the server process (Docker container or subprocess)."""
        if self.managed_mode == "docker":
            try:
                subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        "docker-compose.yml",
                        "-f",
                        "docker-compose.app.yml",
                        "stop",
                        "app",
                    ],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
            self.managed_mode = "none"

        elif self.managed_mode == "subprocess" and self.process is not None:
            pid = self.process.pid
            if self.process.poll() is None:
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            timeout=5.0,
                        )
                    else:
                        self.process.terminate()
                        self.process.wait(timeout=5.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            self.process = None
            self.managed_mode = "none"

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.stop()
        sys.exit(128 + signum)


# ==============================================================================
# 3. HTTP Client & Response Validator
# ==============================================================================

@dataclass
class ValidationResult:
    status: str  # "PASS", "WARNING", "FAIL"
    http_status: int
    success: bool
    conversation_id: str | None
    message_type: str | None
    message: str | None
    citation_count: int
    data_card_count: int
    latency_ms: float
    error: str | None = None
    warnings: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class E2EApiClient:
    """Synchronous HTTP client for executing E2E query requests."""

    def __init__(self, query_url: str, timeout_seconds: float = 60.0):
        self.query_url = query_url
        self.timeout_seconds = timeout_seconds

    def send_query(
        self,
        question: QuestionSpec,
        conversation_id: str | None = None,
    ) -> ValidationResult:
        """
        Send a query request to POST /v1/query with monotonic latency measurement.
        """
        payload: dict[str, Any] = {
            "message": {
                "content": question.text,
            }
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if question.expectation.language:
            payload["message"]["language"] = question.expectation.language

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(self.query_url, json=payload)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return self._validate_response(response, question, latency_ms)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ValidationResult(
                status="FAIL",
                http_status=504,
                success=False,
                conversation_id=conversation_id,
                message_type=None,
                message=None,
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error=f"Request timed out after {self.timeout_seconds}s",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ValidationResult(
                status="FAIL",
                http_status=0,
                success=False,
                conversation_id=conversation_id,
                message_type=None,
                message=None,
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error=f"HTTP request error: {e}",
            )

    def _validate_response(
        self,
        response: httpx.Response,
        question: QuestionSpec,
        latency_ms: float,
    ) -> ValidationResult:
        """Validate API response envelope and schema conformance."""
        http_status = response.status_code
        warnings: list[str] = []

        if http_status != 200:
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=False,
                conversation_id=None,
                message_type=None,
                message=response.text[:300],
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error=f"API returned HTTP {http_status}: {response.text[:200]}",
            )

        try:
            body = response.json()
        except Exception as e:
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=False,
                conversation_id=None,
                message_type=None,
                message=response.text[:200],
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error=f"Failed to parse JSON response: {e}",
            )

        # Check response envelope
        if not isinstance(body, dict):
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=False,
                conversation_id=None,
                message_type=None,
                message=str(body)[:200],
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error="Response is not a JSON object",
            )

        if not body.get("success", False):
            err = body.get("error") or "Unknown API error"
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=False,
                conversation_id=None,
                message_type=None,
                message=None,
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error=f"API returned success=false: {err}",
            )

        data = body.get("data")
        if not isinstance(data, dict):
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=False,
                conversation_id=None,
                message_type=None,
                message=None,
                citation_count=0,
                data_card_count=0,
                latency_ms=latency_ms,
                error="Envelope 'data' field is not an object",
            )

        conv_id = data.get("conversation_id")
        msg_type = data.get("message_type")
        msg = data.get("message")
        citations = data.get("citations") or []
        cards = data.get("data") or []

        if not conv_id or not isinstance(conv_id, str):
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=True,
                conversation_id=None,
                message_type=msg_type,
                message=str(msg)[:200],
                citation_count=len(citations),
                data_card_count=len(cards),
                latency_ms=latency_ms,
                error="Missing or invalid 'conversation_id' in response",
            )

        if msg_type not in ("answer", "clarification"):
            warnings.append(f"Unexpected message_type: '{msg_type}'")

        if not msg or not isinstance(msg, str) or not msg.strip():
            return ValidationResult(
                status="FAIL",
                http_status=http_status,
                success=True,
                conversation_id=conv_id,
                message_type=msg_type,
                message="",
                citation_count=len(citations),
                data_card_count=len(cards),
                latency_ms=latency_ms,
                error="Empty or missing response message",
            )

        # Soft checks based on scenario expectation
        if question.expectation.expect_data_cards and len(cards) == 0:
            warnings.append("Expected data cards for structured query, but 0 returned")

        if question.expectation.expect_citations and len(citations) == 0:
            warnings.append("Expected citations for technical RAG query, but 0 returned")

        status = "WARNING" if warnings else "PASS"

        return ValidationResult(
            status=status,
            http_status=http_status,
            success=True,
            conversation_id=conv_id,
            message_type=msg_type,
            message=msg,
            citation_count=len(citations),
            data_card_count=len(cards),
            latency_ms=latency_ms,
            warnings=warnings,
        )
