"""
tests/features/test_conversation_context.py
===========================================

Comprehensive test suite for Prompt Optimization and MVP Conversation Context:
1. New conversation: new conversation_id -> answer works, summary created and persisted.
2. Follow-up query: same conversation_id -> prior summary injected, context preserved.
3. Hindi continuity: Hindi query followed by Hindi follow-up -> referent resolved and context preserved.
4. Mixed language: English query followed by Hindi follow-up -> context preserved.
5. Evidence conflict: Prior summary says X, current verified evidence says Y -> current evidence wins.
6. Missing conversation: Unknown conversation_id -> query succeeds as new conversation.
7. Persistence failure: DB failure during load/save -> query succeeds gracefully.
8. Synthesis failure: LLM synthesis failure -> deterministic fallback produces answer + summary.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.extensions import db
from app.factory import create_app
from models.conversation import Conversation
from models.enums import DocumentBlockType, DocumentTypes
from services.file_processors.document.doc_representation import DocumentBlock
from services.query.query_service import QueryService, SynthesisResponse
from services.query_analyser.analyser import QueryAnalyzer
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import ChatRequest, UserMessage


@pytest.fixture
def app():
    from flask import Flask
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


def _make_chunk(text: str, filename: str, url: str) -> RetrievedChunk:
    return RetrievedChunk(
        score=0.92,
        chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text=text, metadata={"page": 1}),
        filename=filename,
        author=None,
        source_type=DocumentTypes.PDF,
        source_url=url,
    )


def _build_mock_qs(mock_ai: MagicMock) -> QueryService:
    qs = QueryService(ai_engine=mock_ai)
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        _make_chunk("IS 694:2010 details and specifications", "694.pdf", "https://standards.bis.gov.in/694")
    ]
    qs.retriever = mock_retriever

    mock_product_repo = MagicMock()
    mock_product_repo.find_applicable_standards.return_value = [
        {"is_number": "IS 694:2010", "title": "PVC Insulated Cables", "product_name": "PVC Cables", "source_url": "https://standards.bis.gov.in/694"}
    ]
    mock_standard_repo = MagicMock()
    mock_standard_repo.find_by_number.return_value = {
        "is_number": "IS 694:2010", "title": "PVC Insulated Cables", "status": "Active", "source_url": "https://standards.bis.gov.in/694"
    }
    qs.executor.product_repo = mock_product_repo
    qs.executor.standard_repo = mock_standard_repo
    return qs


class TestConversationContextFlow:
    def test_new_conversation_creates_and_persists_summary(self, app):
        """A new conversation_id processes successfully and persists an initial conversation summary."""
        with app.app_context():
            conv_id = "test_conv_new_123"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            db.session.commit()

            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Which BIS standard applies to PVC cables",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="en",
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "PVC cable"},
                ),
                SynthesisResponse(
                    answer="IS 694:2010 applies to PVC insulated cables up to 1100 V.",
                    conversation_summary="Topic: PVC cables. Relevant Standard: IS 694:2010. Language: en.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="Which BIS standard applies to my PVC cable?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "IS 694" in resp["message"]

            # Verify persisted in database
            saved_conv = Conversation.query.filter_by(conversation_id=conv_id).first()
            assert saved_conv is not None
            assert "IS 694:2010" in saved_conv.summary
            assert "PVC cables" in saved_conv.summary

    def test_followup_query_injects_previous_summary(self, app):
        """Follow-up query with same conversation_id loads prior summary and injects into synthesis and analyzer."""
        with app.app_context():
            conv_id = "test_conv_followup_456"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            conv = Conversation(
                conversation_id=conv_id,
                summary="Topic: PVC cables. Relevant Standard: IS 694:2010. Language: en.",
            )
            db.session.add(conv)
            db.session.commit()

            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="What is the voltage limit for PVC cables under IS 694?",
                    relevant=True,
                    intent=QueryIntent.TECHNICAL_QUESTION,
                    response_language="en",
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters={"standard_number": "IS 694:2010"},
                ),
                SynthesisResponse(
                    answer="Under IS 694:2010, the rated voltage limit is up to and including 1100 V AC.",
                    conversation_summary="Topic: PVC cables. Standard: IS 694:2010. Voltage limit: up to 1100 V. Language: en.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="What is its voltage limit?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "1100 V" in resp["message"]

            # Verify synthesis prompt was passed the prior conversation summary
            synthesis_call_prompt = mock_ai.complete.call_args_list[1][0][0]
            assert "PREVIOUS CONVERSATION CONTEXT" in synthesis_call_prompt
            assert "Relevant Standard: IS 694:2010" in synthesis_call_prompt
            assert "PRECEDENCE HIERARCHY" in synthesis_call_prompt

            # Verify updated summary was saved
            saved_conv = Conversation.query.filter_by(conversation_id=conv_id).first()
            assert "Voltage limit: up to 1100 V" in saved_conv.summary

    def test_hindi_continuity_preserves_language_and_context(self, app):
        """Hindi follow-up resolves referent 'इसकी' using prior summary and responds in Hindi."""
        with app.app_context():
            conv_id = "test_conv_hindi_789"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            conv = Conversation(
                conversation_id=conv_id,
                summary="विषय: पीवीसी केबल (PVC Cables). मानक: IS 694:2010. भाषा: hi.",
            )
            db.session.add(conv)
            db.session.commit()

            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="What is the voltage limit for PVC cables under IS 694?",
                    relevant=True,
                    intent=QueryIntent.TECHNICAL_QUESTION,
                    response_language="hi",
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters={"standard_number": "IS 694:2010"},
                ),
                SynthesisResponse(
                    answer="IS 694:2010 के तहत पीवीसी केबलों की वोल्टेज सीमा 1100 V तक है।",
                    conversation_summary="विषय: PVC केबल. मानक: IS 694:2010. वोल्टेज: 1100 V. भाषा: hi.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="और इसकी voltage limit क्या है?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "IS 694:2010" in resp["message"]
            assert "1100 V" in resp["message"]

            synthesis_prompt = mock_ai.complete.call_args_list[1][0][0]
            assert "Target Response Language: \"hi\"" in synthesis_prompt
            assert "DO NOT translate technical identifiers" in synthesis_prompt

    def test_mixed_language_continuity(self, app):
        """Hinglish/English initial turn followed by Hindi follow-up retains context."""
        with app.app_context():
            conv_id = "test_conv_mixed_101"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            conv = Conversation(
                conversation_id=conv_id,
                summary="Topic: PVC cables. Standard: IS 694:2010 (PVC insulated cables). Language: en/hi.",
            )
            db.session.add(conv)
            db.session.commit()

            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="When was IS 694 revised?",
                    relevant=True,
                    intent=QueryIntent.STANDARD_LOOKUP,
                    response_language="hi",
                    needs_db=True,
                    needs_rag=False,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters={"standard_number": "IS 694:2010"},
                ),
                SynthesisResponse(
                    answer="IS 694 का चौथा संशोधन 2010 में प्रकाशित हुआ था।",
                    conversation_summary="Topic: PVC cables. Standard: IS 694:2010. Revision: 4th revision in 2010. Language: hi.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="इसका revision कब हुआ था?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "IS 694" in resp["message"]

    def test_evidence_conflict_current_evidence_wins(self, app):
        """When prior summary contains incorrect/outdated fact, current verified DB evidence strictly overrides it."""
        with app.app_context():
            conv_id = "test_conv_conflict_202"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            # Old summary contains erroneous/outdated standard IS 9999
            conv = Conversation(
                conversation_id=conv_id,
                summary="Old context: The user previously thought IS 9999 applied to PVC cables.",
            )
            db.session.add(conv)
            db.session.commit()

            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Which BIS standard applies to PVC cables",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="en",
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "PVC cable"},
                ),
                SynthesisResponse(
                    answer="Authoritative BIS records confirm that IS 694:2010 applies to PVC cables, NOT IS 9999.",
                    conversation_summary="Active Topic: PVC cables. Verified Standard: IS 694:2010 (superseding earlier reference to IS 9999).",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="Which BIS standard applies to my PVC cable?"),
            )
            resp = qs.process(req)

            assert "IS 694:2010" in resp["message"]

            synthesis_prompt = mock_ai.complete.call_args_list[1][0][0]
            assert "PRECEDENCE HIERARCHY" in synthesis_prompt
            assert "CURRENT EVIDENCE STRICTLY WINS" in synthesis_prompt

    def test_missing_conversation_id_succeeds_as_fresh(self, app):
        """When an unknown conversation_id is passed, query executes cleanly without error."""
        with app.app_context():
            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="What is BIS?",
                    relevant=True,
                    intent=QueryIntent.GENERAL_BIS_QUERY,
                    response_language="en",
                    needs_db=False,
                    needs_rag=True,
                    db_operations=[],
                    parameters={},
                ),
                SynthesisResponse(
                    answer="BIS is the National Standard Body of India.",
                    conversation_summary="Topic: General BIS mandate.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id="non_existent_conv_9999",
                message=UserMessage(content="What is BIS?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "BIS is the National Standard Body" in resp["message"]

    def test_persistence_failure_does_not_break_query(self, app):
        """If database load or save fails (e.g. connection error), the query still completes successfully."""
        with app.app_context():
            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Which standard applies to PVC cables?",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="en",
                    needs_db=True,
                    needs_rag=False,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "PVC cable"},
                ),
                SynthesisResponse(
                    answer="IS 694:2010 applies to PVC cables.",
                    conversation_summary="Topic: PVC cables.",
                ),
            ]

            qs = _build_mock_qs(mock_ai)

            # Simulate database errors during both load and save
            with patch.object(qs, "_load_conversation_summary", side_effect=Exception("DB connection dropped")):
                with patch.object(qs, "_save_conversation_summary", side_effect=Exception("DB write error")):
                    req = ChatRequest(
                        conversation_id="failing_db_conv",
                        message=UserMessage(content="Which standard applies to PVC cables?"),
                    )
                    resp = qs.process(req)

                    assert resp["message_type"] == "answer"
                    assert "IS 694:2010" in resp["message"]

    def test_synthesis_failure_uses_deterministic_fallback_and_summary(self, app):
        """When synthesis LLM fails, deterministic fallback produces both valid answer and summary."""
        with app.app_context():
            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Which BIS standard applies to PVC cable",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="en",
                    needs_db=True,
                    needs_rag=False,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "PVC cable"},
                ),
                Exception("Synthesis LLM Timeout/Empty response"),
            ]

            conv_id = "test_synthesis_fallback_conv"
            Conversation.query.filter_by(conversation_id=conv_id).delete()
            db.session.commit()

            qs = _build_mock_qs(mock_ai)
            req = ChatRequest(
                conversation_id=conv_id,
                message=UserMessage(content="Which BIS standard applies to my PVC cable?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "IS 694" in resp["message"]

            saved_conv = Conversation.query.filter_by(conversation_id=conv_id).first()
            assert saved_conv is not None
            assert "PVC cable" in saved_conv.summary
            assert "IS 694" in saved_conv.summary
