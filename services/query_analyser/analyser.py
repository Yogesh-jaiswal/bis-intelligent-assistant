import logging
import re
import time
from configs import get_settings
from services.ai.engine import AIEngine

from .analyser_prompt import build_analyser_prompt
from .analyser_schema import DatabaseOperation, QueryIntent, QueryPlan

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes natural-language user queries to determine BIS domain relevance,
    user intent, normalized English phrasing, retrieval sources (PostgreSQL vs Vector RAG),
    high-level database operations, extracted parameters, and missing information.
    Supports injecting prior conversation summary for multi-turn contextual continuity.
    """

    def __init__(self, ai_engine: AIEngine | None = None):
        """
        Initialize the QueryAnalyzer with an injected AIEngine instance or default provider.
        """
        self.ai_engine = ai_engine or AIEngine(get_settings().AI_MODEL)

    def analyze(self, query: str, conversation_summary: str | None = None) -> QueryPlan:
        """
        Analyze the given user query and return a validated QueryPlan.

        :param query: Natural language input string from the user.
        :param conversation_summary: Optional compact summary of previous conversation context.
        :return: Validated QueryPlan instance.
        """
        clean_query = query.strip() if query else ""

        if not clean_query:
            logger.warning("[ANALYZER] Empty user query provided -> marking OUT_OF_SCOPE")
            return QueryPlan(
                normalized_query="",
                relevant=False,
                intent=QueryIntent.OUT_OF_SCOPE,
                response_language="en",
                needs_db=False,
                needs_rag=False,
                db_operations=[],
                parameters={},
                missing_information=["Empty query content"],
            )

        if self.ai_engine.provider == "FAKE":
            logger.info("[ANALYZER] FAKE provider active -> using deterministic heuristic analysis")
            return self._heuristic_fallback(clean_query, conversation_summary=conversation_summary)

        prompt = build_analyser_prompt(clean_query, conversation_summary=conversation_summary)
        start_t = time.perf_counter()

        try:
            logger.info(
                "[ANALYZER: LLM] Sending prompt to AI provider '%s' (think=False, max_tokens=300)...",
                self.ai_engine.provider,
            )
            raw_response = self.ai_engine.complete(
                prompt=prompt,
                response_schema=QueryPlan,
                think=False,
                options={"temperature": 0.0, "num_predict": 300},
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info("[ANALYZER: LLM] Received structured response in %.2f ms", elapsed_ms)

            if isinstance(raw_response, QueryPlan):
                return raw_response

            plan = QueryPlan.model_validate(raw_response)
            if plan.relevant and not plan.db_operations and not plan.needs_rag:
                logger.warning("[ANALYZER] LLM produced empty retrieval plan for relevant query -> supplementing with heuristic fallback")
                return self._heuristic_fallback(clean_query, conversation_summary=conversation_summary)
            return plan

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(
                "[ANALYZER: FALLBACK] AI model completion failed after %.2f ms (%s: %s) -> activating rule-based heuristic fallback for query: '%s'",
                elapsed_ms,
                type(e).__name__,
                e,
                clean_query,
            )
            return self._heuristic_fallback(clean_query, conversation_summary=conversation_summary)

    def _heuristic_fallback(self, query: str, conversation_summary: str | None = None) -> QueryPlan:
        """
        Deterministic rule-based fallback analyzer recovering basic intent and
        explicitly requested operations without over-inferring unrelated operations.
        Uses prior conversation summary when resolving anaphoric follow-up queries.
        """
        q_lower = query.lower()

        # Out-of-scope check
        out_of_scope_keywords = ["weather", "cricket", "movie", "recipe", "stock market", "president"]
        if any(w in q_lower for w in out_of_scope_keywords):
            logger.info("[ANALYZER: HEURISTIC] Matched out-of-scope keywords -> OUT_OF_SCOPE")
            return QueryPlan(
                normalized_query=query,
                relevant=False,
                intent=QueryIntent.OUT_OF_SCOPE,
                response_language="en",
                needs_db=False,
                needs_rag=False,
                db_operations=[],
                parameters={},
            )

        # 1. Standard Lookup or Question by number directly in query (e.g. "What is IS 694", "Is IS 1234 required?", "IS 1554")
        is_match = re.search(r"\bIS\s*[:\s\-]?\s*(\d{2,5}(?::\d{4})?(?:\s*\(Part\s*\d+\))?)", query, re.IGNORECASE)
        std_in_query = f"IS {is_match.group(1).strip()}" if is_match else None

        # Extract explicit request dimensions (English, Hinglish, and Hindi Devanagari)
        asks_standards = bool(re.search(r"\b(which\s+(?:bis\s+)?standard|what\s+(?:bis\s+)?standard|applicable standard|recommend standard|standard applies|kaun\s*sa standard|konsa standard)\b|मानक|स्टैंडर्ड", q_lower))
        asks_cert = bool(re.search(r"\b(certif\w*|licen[cs]e|isi\s*mark|scheme|mandatory|compulsory|necessary|need to take|required|obligation|do i need|zaroori|anivarya|lena zaroori)\b|प्रमाणन|अनिवार्य|आवश्यक|ज़रूरी|जरूरी|लाइसेंस", q_lower))
        asks_lab = bool(re.search(r"\b(lab\w*|where can i (?:get\s+.*?\s+)?test\w*|testing facilit\w*|tested|get\s+tested)\b|प्रयोगशाला", q_lower))
        asks_test = bool(re.search(r"\b(test\w*|sampling|clause\w*)\b|परीक्षण", q_lower)) and not asks_lab
        asks_technical = bool(re.search(r"\b(voltage|limit|rating|specification|what if|less than|greater than|below|above|scope|formula)\b|सीमा|रेटिंग|वोल्टेज", q_lower))

        is_hindi = any("\u0900" <= c <= "\u097f" for c in query)
        resp_lang = "hi" if is_hindi or any(w in q_lower for w in ("kya", "hai", "zaroori", "mera", "meri", "ke liye", "iska", "iski", "anivarya")) else "en"

        # 2. BIS Services listing
        if re.search(r"\b(what\s+(?:are\s+)?(?:the\s+)?services|which\s+services|list\s+services|services\s+(?:provided|offered)\s+by\s+bis|services\s+of\s+bis|services\s+does\s+bis\s+provide|services\s+provided\s+bis)\b", q_lower) or (
            "service" in q_lower and "bis" in q_lower
        ) or any(w in query for w in ("सेवाएं", "सेवाएँ")):
            logger.info("[ANALYZER: HEURISTIC] Recognized BIS Services inquiry")
            norm_q = "What services does BIS provide?" if ("what" in q_lower or "which" in q_lower or "कौन" in query) else "What are the services provided by BIS?"
            return QueryPlan(
                normalized_query=norm_q,
                relevant=True,
                intent=QueryIntent.BIS_SERVICE_LOOKUP,
                response_language=resp_lang,
                needs_db=True,
                needs_rag=False,
                db_operations=[DatabaseOperation.GET_BIS_SERVICE],
                parameters={},
            )

        # 3. General BIS information ("What is BIS", "What does BIS do")
        if re.search(r"\bwhat is bis\b|\bwhat does bis do\b|\babout bis\b|\bwho is bis\b", q_lower):
            logger.info("[ANALYZER: HEURISTIC] Recognized General BIS mandate inquiry")
            return QueryPlan(
                normalized_query="What is the Bureau of Indian Standards (BIS) and what are its functions?",
                relevant=True,
                intent=QueryIntent.GENERAL_BIS_QUERY,
                response_language=resp_lang,
                needs_db=False,
                needs_rag=True,
                db_operations=[],
                parameters={},
            )

        # 4. Contextual entity extraction from conversation_summary (domain-general)
        context_std_num = None
        context_prod = None
        if conversation_summary:
            std_in_summary = re.search(r"\bIS\s*[:\s\-]?\s*(\d{2,5}(?::\d{4})?(?:\s*\(Part\s*\d+\))?)", conversation_summary, re.IGNORECASE)
            if std_in_summary:
                context_std_num = f"IS {std_in_summary.group(1).strip()}"
            m_top = re.search(r"(?:Topic|Product|विषय)\s*[:=]\s*([^.,;\n]+)", conversation_summary, re.IGNORECASE)
            if m_top:
                context_prod = m_top.group(1).strip()

        # Extract product candidate from query
        product_name = None
        m_prod = re.search(
            r"(?:for|applies to|about|regarding)\s+(?:my\s+|this\s+)?([a-zA-Z0-9\s\-]+?)(?:\s+business|\s+product|\s+manufacturing|\?|\.|$)",
            query,
            re.IGNORECASE,
        )
        if m_prod:
            candidate = m_prod.group(1).strip()
            candidate = re.sub(r"\b(do i need|and do i need|and|is it|is this)\b.*$", "", candidate, flags=re.IGNORECASE).strip()
            if candidate and len(candidate) > 2:
                product_name = candidate

        if not product_name and context_prod:
            product_name = context_prod

        # Extract voltage constraint if present
        voltage_val = None
        m_volt = re.search(r"\b(?:(less than|below|up to|above|greater than)\s+)?(\d+(?:\.\d+)?\s*(?:v|kv|volts?))\b", q_lower)
        if m_volt:
            prefix = (m_volt.group(1) or "").strip()
            voltage_val = f"{prefix} {m_volt.group(2)}".strip()

        # 5. Direct Standard in Query (e.g. "Is IS 1234 required?", "What is IS 694?")
        if std_in_query:
            if asks_cert:
                logger.info("[ANALYZER: HEURISTIC] Recognized Standard Necessity / Certification query for '%s'", std_in_query)
                norm_q = f"Is {std_in_query} mandatory or required for {product_name or 'my product'}?"
                params = {"standard_number": std_in_query}
                if product_name:
                    params["product"] = product_name
                return QueryPlan(
                    normalized_query=norm_q,
                    relevant=True,
                    intent=QueryIntent.CERTIFICATION_REQUIREMENT,
                    response_language=resp_lang,
                    needs_db=True,
                    needs_rag=asks_technical,
                    db_operations=[DatabaseOperation.GET_CERTIFICATION_REQUIREMENT],
                    parameters=params,
                )
            elif asks_test:
                logger.info("[ANALYZER: HEURISTIC] Recognized Testing Requirements for '%s'", std_in_query)
                return QueryPlan(
                    normalized_query=f"What testing requirements and methods apply under {std_in_query}?",
                    relevant=True,
                    intent=QueryIntent.TESTING_REQUIREMENT,
                    response_language=resp_lang,
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters={"standard_number": std_in_query},
                )
            else:
                logger.info("[ANALYZER: HEURISTIC] Recognized Standard Lookup pattern for '%s'", std_in_query)
                return QueryPlan(
                    normalized_query=f"Details and specifications for standard {std_in_query}",
                    relevant=True,
                    intent=QueryIntent.STANDARD_LOOKUP,
                    response_language=resp_lang,
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters={"standard_number": std_in_query},
                )

        # 6. Follow-up queries referencing context standard (e.g. "Is it necessary to take this standard?", "Do I need it?")
        is_followup = bool(re.search(r"\b(it|its|this|voltage|limit|rating|clause|revision|amendment|scope|test|do i need|is it necessary|is it mandatory|इसकी|इसका|इस|ये)\b", q_lower))
        if is_followup and context_std_num and not asks_standards:
            logger.info("[ANALYZER: HEURISTIC] Resolved follow-up to context standard '%s'", context_std_num)
            params = {"standard_number": context_std_num}
            if product_name:
                params["product"] = product_name
            if voltage_val:
                params["voltage"] = voltage_val

            if asks_cert:
                # E.g. "Then what if my wires are less than 1110V? Is it necessary to take this standard?"
                # or "Is it necessary to take this standard?"
                norm_q = query.strip()
                if "this standard" in norm_q.lower():
                    norm_q = re.sub(r"(?i)\bthis standard\b", context_std_num, norm_q)
                elif re.search(r"(?i)\b(it|इसकी|इसका)\b", norm_q):
                    norm_q = re.sub(r"(?i)\b(it|इसकी|इसका)\b", context_std_num, norm_q)
                else:
                    norm_q = f"Is {context_std_num} mandatory/necessary for {product_name or 'my product'}?"

                sec_intents = [QueryIntent.TECHNICAL_QUESTION] if asks_technical else []
                return QueryPlan(
                    normalized_query=norm_q,
                    relevant=True,
                    intent=QueryIntent.CERTIFICATION_REQUIREMENT,
                    secondary_intents=sec_intents,
                    response_language=resp_lang,
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.GET_CERTIFICATION_REQUIREMENT],
                    parameters=params,
                )
            else:
                norm_q = f"{query.strip()} (Standard: {context_std_num})"
                return QueryPlan(
                    normalized_query=norm_q,
                    relevant=True,
                    intent=QueryIntent.TECHNICAL_QUESTION if asks_technical else QueryIntent.STANDARD_LOOKUP,
                    response_language=resp_lang,
                    needs_db=True,
                    needs_rag=True,
                    db_operations=[DatabaseOperation.FIND_STANDARD],
                    parameters=params,
                )

        # 7. Multi-question and Single-question queries for new entities
        product_val = product_name or query.strip("? .")
        params = {"product": product_val}
        if voltage_val:
            params["voltage"] = voltage_val

        # Check Hindi / Hinglish translation normalization
        norm_q = query.strip()
        if is_hindi or any(w in q_lower for w in ("kya", "hai", "zaroori", "anivarya")):
            if asks_standards and asks_cert:
                norm_q = f"Which BIS standard applies to {product_val}, and is BIS certification mandatory?"
            elif asks_standards:
                norm_q = f"Which BIS standard applies to {product_val}?"
            elif asks_cert:
                norm_q = f"Is BIS certification mandatory for {product_val}?"

        ops: list[DatabaseOperation] = []
        sec_intents: list[QueryIntent] = []
        intent: QueryIntent | None = None

        if asks_standards:
            ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
            intent = QueryIntent.PRODUCT_STANDARD_RECOMMENDATION

        if asks_cert:
            if DatabaseOperation.FIND_APPLICABLE_STANDARDS not in ops:
                ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
            ops.append(DatabaseOperation.GET_CERTIFICATION_REQUIREMENT)
            if intent is None:
                intent = QueryIntent.CERTIFICATION_REQUIREMENT
            elif QueryIntent.CERTIFICATION_REQUIREMENT not in sec_intents:
                sec_intents.append(QueryIntent.CERTIFICATION_REQUIREMENT)

        if asks_lab:
            ops.append(DatabaseOperation.FIND_LABORATORIES)
            if intent is None:
                intent = QueryIntent.LABORATORY_LOOKUP
                params = {"scope": product_val}
            elif QueryIntent.LABORATORY_LOOKUP not in sec_intents:
                sec_intents.append(QueryIntent.LABORATORY_LOOKUP)

        if not ops:
            intent = QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
            ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
            if not is_hindi and not asks_standards:
                norm_q = f"Which BIS standard applies to {product_val}"
        elif intent is None:
            intent = QueryIntent.PRODUCT_STANDARD_RECOMMENDATION

        logger.info(
            "[ANALYZER: HEURISTIC] Fallback determined intent=%s, secondary=%s, ops=%s, params=%s",
            intent.value,
            [s.value for s in sec_intents],
            [o.value for o in ops],
            params,
        )

        return QueryPlan(
            normalized_query=norm_q,
            relevant=True,
            intent=intent,
            secondary_intents=sec_intents,
            response_language=resp_lang,
            needs_db=True,
            needs_rag=True,
            db_operations=ops,
            parameters=params,
        )
