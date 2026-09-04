from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, ConfigDict


# ============================================================
# REQUEST
# ============================================================


class UserMessage(BaseModel):
    """User message submitted to the BIS conversational assistant."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        ...,
        min_length=1,
        description="Natural-language user message.",
        examples=["Which BIS standard applies to my PVC cable?"],
    )

    language: str | None = Field(
        default=None,
        description=(
            "ISO 639-1 language code when known. "
            "The backend may detect the language when omitted."
        ),
        examples=["en", "hi"],
    )


class ChatRequest(BaseModel):
    """Request body for the BIS conversational assistant."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the ongoing conversation. "
            "Null for a new conversation."
        ),
        examples=["conv_01HXYZ123"],
    )

    message: UserMessage


# ============================================================
# CITATIONS
# ============================================================


class Citation(BaseModel):
    """Authoritative source referenced by a citation ID in the message."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description=(
            "Citation identifier embedded inside the response message "
            "using the <cit_id> syntax."
        ),
        examples=["cit_1"],
    )

    source_type: Literal[
        "standard",
        "certification",
        "laboratory",
        "service",
        "document",
        "other",
    ]

    title: str = Field(
        ...,
        description="Human-readable source title.",
        examples=["IS 694:2010"],
    )

    reference: str | None = Field(
        default=None,
        description="Formal source reference when available.",
        examples=["IS 694:2010"],
    )

    source_url: HttpUrl = Field(
        ...,
        description="Direct URL to the authoritative source.",
    )


# ============================================================
# DATA CARDS
# ============================================================


class StandardCard(BaseModel):
    """Structured data returned for a BIS standard."""

    model_config = ConfigDict(extra="forbid")

    data_type: Literal["standard"] = "standard"

    id: str = Field(
        ...,
        examples=["std_001"],
    )

    is_number: str = Field(
        ...,
        examples=["IS 694:2010"],
    )

    title: str

    revision_number: int | None = None

    publication_year: int | None = None

    status: str

    technical_department: str | None = None

    relevance: Literal[
        "Primary",
        "Supporting",
        "Related",
    ] | None = Field(
        default=None,
        description=(
            "Query-specific relevance assigned to the standard. "
            "This may be generated during response construction "
            "and does not necessarily originate from the database."
        ),
    )

    source_url: HttpUrl

    document_url: HttpUrl | None = None


class CertificationCard(BaseModel):
    """Structured data returned for a BIS certification scheme."""

    model_config = ConfigDict(extra="forbid")

    data_type: Literal["certification"] = "certification"

    id: str = Field(
        ...,
        examples=["cert_001"],
    )

    name: str = Field(
        ...,
        examples=["Scheme I (ISI Mark Scheme)"],
    )

    scheme_code: str | None = Field(
        default=None,
        examples=["Scheme-I"],
    )

    certification_type: str | None = None

    mandatory: str | None = Field(
        default=None,
        description=(
            "Whether certification is mandatory in the applicable context."
        ),
    )

    authority: str | None = Field(
        default=None,
        examples=["Bureau of Indian Standards"],
    )

    requirements: list[str] | None = Field(
        default=None,
        description=(
            "Requirements relevant to the user's query. "
            "These may be assembled from structured data and "
            "authoritative retrieved documents."
        ),
    )

    source_url: HttpUrl


class LaboratoryCard(BaseModel):
    """Structured data returned for a testing laboratory."""

    model_config = ConfigDict(extra="forbid")

    data_type: Literal["laboratory"] = "laboratory"

    id: str = Field(
        ...,
        examples=["lab_001"],
    )

    lab_code: str | None = Field(
        default=None,
        examples=["6104524"],
    )

    name: str

    address: str | None = None

    state: str | None = None

    district: str | None = None

    phone: str | None = None

    email: str | None = None

    validity_date: str | None = None

    scope: str | None = Field(
        default=None,
        description="Laboratory testing scope published by the source.",
    )

    source_url: HttpUrl


class ServiceCard(BaseModel):
    """Structured data returned for a BIS service."""

    model_config = ConfigDict(extra="forbid")

    data_type: Literal["service"] = "service"

    id: str = Field(
        ...,
        examples=["service_001"],
    )

    name: str = Field(
        ...,
        examples=["Grant of Licence for Product Certification"],
    )

    service_type: str | None = None

    description: str | None = None

    source_url: HttpUrl


# ============================================================
# DATA BLOCK
# ============================================================


DataBlock = (
    StandardCard
    | CertificationCard
    | LaboratoryCard
    | ServiceCard
)


# ============================================================
# CLARIFICATION
# ============================================================


class ClarificationQuestion(BaseModel):
    """Question that the frontend should present to the user."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description=(
            "Unique identifier for the clarification question. "
            "The identifier is used when submitting the answer."
        ),
        examples=["q_voltage"],
    )

    question: str = Field(
        ...,
        description="Question displayed to the user.",
        examples=["What is the rated voltage of the cable?"],
    )

    input_type: Literal[
        "text",
        "number",
        "select",
        "multi_select",
        "boolean",
    ]

    required: bool = True

    options: list[str] | None = Field(
        default=None,
        description=(
            "Available options for select and multi-select inputs."
        ),
    )


# ============================================================
# CHAT RESPONSE
# ============================================================


class ChatResponse(BaseModel):
    """Successful response returned by the query endpoint."""

    model_config = ConfigDict(extra="forbid")

    message_type: Literal[
        "answer",
        "clarification",
    ]

    conversation_id: str

    message: str = Field(
        ...,
        description=(
            "Natural-language assistant response. "
            "Citation identifiers may be embedded directly "
            "inside the message using <cit_id> syntax."
        ),
    )

    citations: list[Citation] = Field(
        default_factory=list,
    )

    data: list[DataBlock] = Field(
        default_factory=list,
        description=(
            "Structured data cards. Each card contains its own "
            "data_type so mixed card types can be returned together."
        ),
    )

    questions: list[ClarificationQuestion] | None = Field(
        default=None,
        description=(
            "Clarification questions returned when message_type "
            "is clarification."
        ),
    )