"""
services/query/synthesis_prompt.py
==================================

Dedicated module for constructing the response synthesis prompt and structured response schema.
Separates LLM prompt design from pipeline orchestration in QueryService.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from services.query_analyser.analyser_schema import QueryPlan
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import Citation


SYNTHESIS_FALLBACK_TEMPLATES = {
    "en": {
        "applicable_standards": "Applicable Indian Standards: {standards}.",
        "certification_schemes": "Certification Scheme: {schemes}.",
        "mandatory": "Mandatory",
        "voluntary": "Voluntary",
        "laboratories": "Recognized Testing Laboratories: {labs}.",
        "services": "BIS Services: {services}.",
        "no_records_found": "Based on BIS records for '{query}', no matching standard, certification, or service could be found. Please provide additional product details.",
        "summary_topic": "Topic: {topic}. Facts: {facts}.",
        "product_label": "Product",
        "standard_label": "Standard",
        "language_label": "Language",
        "clarification_default": "Could you please specify more details to help identify the exact standard or requirement?",
    },
    "hi": {
        "applicable_standards": "लागू भारतीय मानक: {standards}।",
        "certification_schemes": "प्रमाणन योजना: {schemes}।",
        "mandatory": "अनिवार्य",
        "voluntary": "ऐच्छिक",
        "laboratories": "मान्यता प्राप्त परीक्षण प्रयोगशालाएँ: {labs}।",
        "services": "BIS सेवाएँ: {services}।",
        "no_records_found": "BIS अभिलेखों के अनुसार '{query}' के लिए कोई संबंधित मानक, प्रमाणन या सेवा नहीं मिली। कृपया उत्पाद के संबंध में अतिरिक्त विवरण प्रदान करें।",
        "summary_topic": "विषय: {topic}। मुख्य तथ्य: {facts}।",
        "product_label": "उत्पाद",
        "standard_label": "मानक",
        "language_label": "भाषा",
        "clarification_default": "सटीक मानक या आवश्यकता की पहचान करने के लिए कृपया अतिरिक्त विवरण प्रदान करें।",
    },
}


class SynthesisResponse(BaseModel):
    """Structured response schema for the final synthesis LLM."""
    answer: str = Field(
        ...,
        description="Authoritative, evidence-based, concise answer synthesized from database and document evidence.",
    )
    conversation_summary: str | None = Field(
        default=None,
        description="Compact conversation summary (2-4 bullet points or concise sentences) capturing the topic, key facts/standards discussed, and language context.",
    )


def build_synthesis_prompt(
    user_query: str,
    plan: QueryPlan,
    accumulated_db_results: dict[str, list[dict[str, Any]]],
    rag_chunks: list[RetrievedChunk],
    citations: list[Citation],
    conversation_summary: str | None = None,
) -> str:
    """
    Construct the authoritative synthesis prompt for the final response LLM.

    Treats retrieved evidence as source material, avoiding raw RAG dumping and verbatim copying.
    Preserves technical terminology, Indian Standard numbers, units, and valid citation tags.
    """
    valid_cit_ids = [c.id for c in citations]
    has_db_records = any(len(recs) > 0 for recs in accumulated_db_results.values())

    context_block = ""
    if conversation_summary and conversation_summary.strip():
        context_block = f"""
==================================================
PREVIOUS CONVERSATION CONTEXT (BACKGROUND REFERENCE)
==================================================
{conversation_summary.strip()}

* PRECEDENCE HIERARCHY:
  Current Verified DB/RAG Evidence > Current User Query > Previous Conversation Context.
  The previous summary is contextual memory only, not authoritative facts.
  If current evidence contradicts previous context, CURRENT EVIDENCE STRICTLY WINS.
"""

    passages_block = (
        "\n".join([
            f"[Passage {i+1}] (Citation ID: <cit_{i+1}>, Source: {c.filename}): {c.chunk.text}"
            for i, c in enumerate(rag_chunks)
        ])
        if rag_chunks
        else "No document passages retrieved."
    )

    intent_str = getattr(plan.intent, "value", str(plan.intent))

    prompt = f"""You are the Bureau of Indian Standards (BIS) Conversational Assistant.
Synthesize a clear, authoritative, direct, and concise response to the user's question using ONLY the provided evidence.
{context_block}
==================================================
EVIDENCE SUMMARY
==================================================
User Query: "{user_query}"
Normalized English Query: "{plan.normalized_query}"
Classified User Intent: "{intent_str}"
Target Response Language: "{plan.response_language}"

--- STRUCTURED DATABASE RECORDS (AUTHORITATIVE METADATA) ---
{accumulated_db_results if has_db_records else "No database records retrieved."}

--- DOCUMENTARY RAG PASSAGES (SUPPORTING SPECIFICATIONS) ---
{passages_block}

--- VALID CITATION TAGS YOU MAY REFERENCE ---
{valid_cit_ids if valid_cit_ids else "None"}

==================================================
INSTRUCTIONS FOR SYNTHESIS & LOCALIZATION
==================================================
1. SYNTHESIS OVER VERBATIM EXTRACTION:
   - Retrieved evidence is SOURCE MATERIAL to synthesize into an answer, NOT text to copy verbatim.
   - Synthesize a natural, coherent explanation directly answering the user's actual question.
   - DO NOT dump raw passages, clauses, or large chunks of standard text verbatim into your response.
   - For example: if asked which standard applies to PVC cable, state the applicable standard and briefly explain why it applies, rather than reproducing unrelated clauses.
2. DISTINGUISH EVIDENCE ROLES:
   - Structured Database Records: Authoritative metadata (standard numbers, titles, certification requirements, laboratories, services).
   - Documentary RAG Passages: Supporting context, technical definitions, test methods, and voltage/rating provisions.
3. CITATIONS:
   - When citing information from documentary passages, use inline tags like <cit_1>, <cit_2>.
   - STRICT RULE: ONLY reference citation IDs that exist in the valid citation list above: {valid_cit_ids}.
   - Do NOT invent or make up non-existent citation tags.
4. TECHNICAL TERMINOLOGY & LOCALIZATION:
   - Respond in the requested language: '{plan.response_language}' (e.g. Hindi if 'hi', English if 'en').
   - DO NOT translate technical identifiers, Indian Standard numbers (e.g., IS 694, IS 7098, IS 1786), scheme codes (e.g., Scheme-I, FMCS, CRS), or technical units (e.g., 1100 V, 1.1 kV, sq mm, °C).
   - Maintain natural sentence structure in the user's language without translating technical symbols.
5. COMPLETENESS & UNCERTAINTY:
   - If the user query asked multi-part questions (e.g., applicable standard + certification requirement + laboratories), address all parts from the retrieved evidence.
   - If the retrieved evidence does not fully answer the question, state clearly what is established and what is missing or requires additional details. Do not guess.
6. CONVERSATION SUMMARY:
   - Generate an updated, compact `conversation_summary` (2-4 bullet points or concise sentences).
   - Capture the active topic, key standards or facts established, language context, and any pending follow-up context.
   - Preserve key Hindi/English terms if the conversation is multilingual.
7. INTENT-COMPATIBLE EVIDENCE GROUNDING:
   - Base your answer strictly on the evidence that matches the user's classified intent ({intent_str}).
   - For service queries (BIS_SERVICE_LOOKUP), answer directly from the structured BIS Services records. Do NOT discuss or cite technical standard documents or material specifications (such as steel, pressure cookers, or cables) unless specifically requested by the user.
"""
    return prompt
