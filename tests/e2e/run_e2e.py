#!/usr/bin/env python3
"""
tests/e2e/run_e2e.py
====================
Standalone End-to-End System Testing Harness for the BIS Intelligent Assistant.

Executes a small, realistic battery of multi-turn conversations through the actual
HTTP API against live PostgreSQL, applied migrations, and the configured AI model.

Usage:
    python tests/e2e/run_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enable UTF-8 encoding on stdout/stderr for multilingual (Hindi) console output
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from tests.e2e.helpers import (
    E2EApiClient,
    ServerManager,
    check_configuration,
    check_migrations,
    check_ollama,
    check_postgres,
)
from tests.e2e.scenarios import get_e2e_scenarios


def _format_header(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n{title}\n{line}\n", flush=True)


def _format_divider(title: str) -> None:
    line = "-" * 64
    print(f"\n{line}\n{title}\n{line}\n", flush=True)


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        try:
            print(safe_msg, flush=True)
        except Exception:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()


def run_e2e() -> int:
    """Main entrypoint for the standalone E2E system testing harness."""
    run_started_utc = datetime.now(timezone.utc)
    run_start_monotonic = time.perf_counter()

    _format_header("BIS CHATBOT E2E SYSTEM TEST")

    # --------------------------------------------------------------------------
    # [1/6] Check Environment Configuration
    # --------------------------------------------------------------------------
    log("[1/6] Checking configuration...")
    cfg_ok, cfg_msg, cfg_info = check_configuration()
    if not cfg_ok:
        log(f"[FAIL] {cfg_msg}")
        log("\nE2E TEST ABORTED")
        log(f"Reason: Configuration error ({cfg_msg})")
        return 1

    model_url = cfg_info["model_url"]
    api_host = cfg_info.get("api_host", "127.0.0.1")
    api_port = cfg_info.get("api_port", 5000)
    log(f"[PASS] Configuration loaded (Model: {cfg_info.get('model_name')}, Provider: {cfg_info.get('ai_provider')})")

    # --------------------------------------------------------------------------
    # [2/6] Check Ollama / Model Endpoint Reachability
    # --------------------------------------------------------------------------
    log("\n[2/6] Checking Ollama...")
    ollama_ok, ollama_msg = check_ollama(model_url, timeout_seconds=3.0)
    if not ollama_ok:
        log(f"[FAIL] {ollama_msg}")
        log("\nE2E TEST ABORTED")
        log(f"Reason: Ollama/model endpoint unavailable ({model_url})")
        return 1
    log(f"[PASS] {ollama_msg}")

    # --------------------------------------------------------------------------
    # [3/6] Check PostgreSQL Availability
    # --------------------------------------------------------------------------
    log("\n[3/6] Checking PostgreSQL...")
    pg_ok, pg_msg = check_postgres(timeout_seconds=3.0)
    if not pg_ok:
        log(f"[FAIL] {pg_msg}")
        log("\nE2E TEST ABORTED")
        log("Reason: PostgreSQL unavailable")
        return 1
    log(f"[PASS] {pg_msg}")

    # --------------------------------------------------------------------------
    # [4/6] Check Database Migrations State
    # --------------------------------------------------------------------------
    log("\n[4/6] Checking migrations...")
    mig_ok, mig_msg = check_migrations()
    if not mig_ok:
        log(f"[FAIL] {mig_msg}")
        log("\nE2E TEST ABORTED")
        log(f"Reason: Database migrations issue ({mig_msg})")
        return 1
    log(f"[PASS] {mig_msg}")

    # --------------------------------------------------------------------------
    # [5/6] Check / Start Application Server
    # --------------------------------------------------------------------------
    log("\n[5/6] Checking API server...")
    server = ServerManager(host=api_host, port=api_port)
    server_ok, server_msg = server.ensure_server_running(startup_timeout=45.0)
    if not server_ok:
        log(f"[FAIL] {server_msg}")
        server.stop()
        log("\nE2E TEST ABORTED")
        log(f"Reason: Application server unavailable ({server_msg})")
        return 1
    log(f"[PASS] {server_msg}")

    # --------------------------------------------------------------------------
    # [6/6] Run E2E Conversations
    # --------------------------------------------------------------------------
    log("\n[6/6] Running E2E conversations...")
    scenarios = get_e2e_scenarios()
    client = E2EApiClient(query_url=server.query_url, timeout_seconds=90.0)

    conversation_results: list[dict[str, Any]] = []
    total_questions = 0
    passed_questions = 0
    failed_questions = 0
    latencies: list[float] = []

    try:
        for s_idx, scenario in enumerate(scenarios, start=1):
            _format_divider(f"Conversation {s_idx}/{len(scenarios)}: {scenario.name}\n({scenario.description})")

            current_conv_id: str | None = None
            scenario_questions_data: list[dict[str, Any]] = []
            scenario_passed = True

            for q_idx, question in enumerate(scenario.questions, start=1):
                total_questions += 1
                q_start_utc = datetime.now(timezone.utc).isoformat()

                log(f"Question {q_idx}/{len(scenario.questions)}")
                log(f"> {question.text}")
                log("[REQUEST] Sending request...")

                # Execute query
                res = client.send_query(question, conversation_id=current_conv_id)
                q_end_utc = datetime.now(timezone.utc).isoformat()
                latencies.append(res.latency_ms)

                sec = res.latency_ms / 1000.0

                if res.status in ("PASS", "WARNING"):
                    passed_questions += 1
                    log(f"[{res.status}] HTTP {res.http_status} in {sec:.2f}s")
                    log(f"[CONVERSATION] {res.conversation_id}")
                    log(f"[MESSAGE TYPE] {res.message_type}")
                    log(f"[CARDS] {res.data_card_count} | [CITATIONS] {res.citation_count}")
                    if res.warnings:
                        for w in res.warnings:
                            log(f"[WARNING] {w}")
                    # Update conversation id for multi-turn continuity
                    if res.conversation_id:
                        current_conv_id = res.conversation_id
                else:
                    failed_questions += 1
                    scenario_passed = False
                    log(f"[FAIL] HTTP {res.http_status} in {sec:.2f}s")
                    log(f"[ERROR] {res.error}")

                question_record = {
                    "question": question.text,
                    "specification": question.to_dict(),
                    "conversation_id_before": current_conv_id if q_idx > 1 else None,
                    "conversation_id_after": res.conversation_id,
                    "started_at": q_start_utc,
                    "completed_at": q_end_utc,
                    "latency_ms": round(res.latency_ms, 2),
                    "http_status": res.http_status,
                    "status": res.status,
                    "success": res.success,
                    "message_type": res.message_type,
                    "message": res.message,
                    "citation_count": res.citation_count,
                    "data_card_count": res.data_card_count,
                    "warnings": res.warnings,
                    "error": res.error,
                }
                scenario_questions_data.append(question_record)

                # Cooldown between questions (approx 1 second)
                if q_idx < len(scenario.questions) or s_idx < len(scenarios):
                    log("Waiting 1 second cooldown before next request...")
                    time.sleep(1.0)
                log()

            conversation_results.append({
                "name": scenario.name,
                "description": scenario.description,
                "conversation_id": current_conv_id,
                "status": "passed" if scenario_passed else "failed",
                "questions": scenario_questions_data,
            })

    finally:
        # Gracefully shut down server if spawned by this process
        server.stop()

    # --------------------------------------------------------------------------
    # Aggregation & Metrics
    # --------------------------------------------------------------------------
    run_finished_utc = datetime.now(timezone.utc)
    total_run_duration_ms = (time.perf_counter() - run_start_monotonic) * 1000.0

    passed_conversations = sum(1 for c in conversation_results if c["status"] == "passed")
    failed_conversations = len(conversation_results) - passed_conversations

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    min_latency = round(min(latencies), 2) if latencies else 0.0
    max_latency = round(max(latencies), 2) if latencies else 0.0

    overall_status = "passed" if failed_questions == 0 else "failed"

    report_data = {
        "run": {
            "started_at": run_started_utc.isoformat(),
            "finished_at": run_finished_utc.isoformat(),
            "duration_ms": round(total_run_duration_ms, 2),
            "status": overall_status,
            "configuration": cfg_info,
            "ollama_available": ollama_ok,
            "postgres_available": pg_ok,
            "migrations_ok": mig_ok,
            "api_server_available": server_ok,
        },
        "conversations": conversation_results,
        "summary": {
            "total_conversations": len(conversation_results),
            "passed_conversations": passed_conversations,
            "failed_conversations": failed_conversations,
            "total_questions": total_questions,
            "passed_questions": passed_questions,
            "failed_questions": failed_questions,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
        },
    }

    # --------------------------------------------------------------------------
    # Save JSON Report
    # --------------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = run_started_utc.strftime("%Y-%m-%d_%H-%M-%S")
    report_file = results_dir / f"e2e_{timestamp_str}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # Terminal Summary
    # --------------------------------------------------------------------------
    _format_header("E2E TEST RUN SUMMARY")
    log(f"Status:               {overall_status.upper()}")
    log(f"Report File:          {report_file}")
    log(f"Total Run Time:       {total_run_duration_ms / 1000.0:.2f}s")
    log(f"Conversations:        {passed_conversations}/{len(conversation_results)} passed")
    log(f"Questions:            {passed_questions}/{total_questions} passed")
    if latencies:
        log(f"Latency (Avg/Min/Max): {avg_latency / 1000.0:.2f}s / {min_latency / 1000.0:.2f}s / {max_latency / 1000.0:.2f}s")
    log("=" * 64 + "\n")

    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    sys.exit(run_e2e())
