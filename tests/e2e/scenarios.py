"""
tests/e2e/scenarios.py
======================
Scenario and question definitions for the BIS Assistant E2E system testing harness.

Constraints:
- Maximum 3 conversations.
- Maximum 10 questions total.
- Covers multiple domains: Services, Certification, Testing, Standards, Multilingual (Hindi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionExpectation:
    """Expected behavioral characteristics for a question."""
    description: str = ""
    expected_intent_hint: str | None = None
    expect_data_cards: bool = False
    expect_citations: bool = False
    language: str = "en"


@dataclass
class QuestionSpec:
    """Individual question specification within a conversation scenario."""
    text: str
    expectation: QuestionExpectation = field(default_factory=QuestionExpectation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "description": self.expectation.description,
            "expected_intent_hint": self.expectation.expected_intent_hint,
            "expect_data_cards": self.expectation.expect_data_cards,
            "expect_citations": self.expectation.expect_citations,
            "language": self.expectation.language,
        }


@dataclass
class ConversationScenario:
    """Multi-turn conversation scenario executed sequentially."""
    name: str
    description: str
    questions: list[QuestionSpec]


# ==============================================================================
# E2E TEST SCENARIO DEFINITIONS (3 Conversations, 9 Questions Total)
# ==============================================================================

SCENARIOS: list[ConversationScenario] = [
    # --------------------------------------------------------------------------
    # Scenario 1: BIS Services & Laboratory Operations (Structured DB Retrieval)
    # --------------------------------------------------------------------------
    ConversationScenario(
        name="general_services_and_laboratories",
        description="Validates structured DB retrieval for BIS services and follow-up lab inquiries.",
        questions=[
            QuestionSpec(
                text="What services does BIS provide?",
                expectation=QuestionExpectation(
                    description="Structured lookup for BIS services; expects ServiceCards, no standard citations.",
                    expected_intent_hint="BIS_SERVICE_LOOKUP",
                    expect_data_cards=True,
                    expect_citations=False,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="Does BIS operate testing laboratories for these services?",
                expectation=QuestionExpectation(
                    description="Follow-up on laboratory operations testing conversation continuity.",
                    expected_intent_hint="LABORATORY_LOOKUP",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="What are the fee requirements or eligibility for product certification?",
                expectation=QuestionExpectation(
                    description="Follow-up inquiry about certification service details.",
                    expected_intent_hint="BIS_SERVICE_LOOKUP",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="en",
                ),
            ),
        ],
    ),

    # --------------------------------------------------------------------------
    # Scenario 2: Certification Schemes & Multilingual Hindi (Context Continuity)
    # --------------------------------------------------------------------------
    ConversationScenario(
        name="certification_process_and_multilingual",
        description="Validates Scheme I certification process, follow-up continuity, and Hindi response synthesis.",
        questions=[
            QuestionSpec(
                text="What is the certification process for obtaining an ISI mark under Scheme I?",
                expectation=QuestionExpectation(
                    description="Queries certification procedure under Scheme I.",
                    expected_intent_hint="CERTIFICATION_PROCESS",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="Is this certification mandatory or voluntary for manufacturers?",
                expectation=QuestionExpectation(
                    description="Contextual follow-up regarding mandatory order applicability.",
                    expected_intent_hint="CERTIFICATION_REQUIREMENT",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="आईएसआई मार्क प्राप्त करने की क्या प्रक्रिया है?",
                expectation=QuestionExpectation(
                    description="Multilingual Hindi query validating language detection and Hindi synthesis.",
                    expected_intent_hint="CERTIFICATION_PROCESS",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="hi",
                ),
            ),
        ],
    ),

    # --------------------------------------------------------------------------
    # Scenario 3: Standards, Testing Requirements & General Overview (RAG Retrieval)
    # --------------------------------------------------------------------------
    ConversationScenario(
        name="technical_standards_and_rag",
        description="Validates standard identification, technical testing RAG retrieval, and general BIS query.",
        questions=[
            QuestionSpec(
                text="What is the Indian Standard for packaged drinking water?",
                expectation=QuestionExpectation(
                    description="Product-to-standard mapping query (IS 14543).",
                    expected_intent_hint="PRODUCT_STANDARD_RECOMMENDATION",
                    expect_data_cards=True,
                    expect_citations=False,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="What are the key testing requirements specified in the standard for packaged drinking water?",
                expectation=QuestionExpectation(
                    description="Documentary RAG query over standard PDF, expects citations.",
                    expected_intent_hint="TESTING_REQUIREMENT",
                    expect_data_cards=False,
                    expect_citations=True,
                    language="en",
                ),
            ),
            QuestionSpec(
                text="What is BIS and what is its primary role in India?",
                expectation=QuestionExpectation(
                    description="General institutional overview query.",
                    expected_intent_hint="GENERAL_BIS_QUERY",
                    expect_data_cards=False,
                    expect_citations=False,
                    language="en",
                ),
            ),
        ],
    ),
]


def get_e2e_scenarios() -> list[ConversationScenario]:
    """Retrieve all defined scenarios, asserting hard constraints."""
    assert len(SCENARIOS) <= 3, f"Scenario count ({len(SCENARIOS)}) exceeds maximum allowed (3)"
    total_questions = sum(len(s.questions) for s in SCENARIOS)
    assert total_questions <= 10, f"Total questions ({total_questions}) exceeds maximum allowed (10)"
    return SCENARIOS
