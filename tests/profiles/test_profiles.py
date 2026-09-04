"""
tests/profiles/test_profiles.py
===============================

Environment configurations and execution profiles for test execution.
"""

from dataclasses import dataclass


@dataclass
class TestProfile:
    name: str
    ai_provider: str
    enable_ratelimit: bool
    description: str


CI_PROFILE = TestProfile(
    name="ci",
    ai_provider="FAKE",
    enable_ratelimit=False,
    description="Deterministic testing profile for automated CI environments.",
)

LOCAL_E2E_PROFILE = TestProfile(
    name="local_e2e",
    ai_provider="OLLAMA",
    enable_ratelimit=True,
    description="Live stack testing profile requiring running Ollama and PostgreSQL.",
)
