from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.canonical import (
    NodeType,
    NodeStatus,
    ApprovalStatus,
    DocumentType,
    TranslationMode,
    QALevel,
    DocumentNode,
    Chapter,
    Asset,
    CanonicalDocument,
    DocumentMetadata,
)


# Project Schemas
class ProjectCreate(BaseModel):
    title: str = "New Project"
    description: Optional[str] = None
    source_language: str = "en"
    target_language: str = "vi"
    document_type: DocumentType = DocumentType.GENERAL
    translation_mode: TranslationMode = TranslationMode.NATURAL
    selected_model: str = "qwen2.5:7b"


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[DocumentType] = None
    translation_mode: Optional[TranslationMode] = None
    custom_instructions: Optional[str] = None
    current_stage: Optional[str] = None
    selected_model: Optional[str] = None
    qa_level: Optional[QALevel] = None
    style_guide: Optional[Dict[str, Any]] = None
    structure_confirmed: Optional[bool] = None


class ProjectResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    source_language: str
    target_language: str
    document_type: str
    translation_mode: str
    custom_instructions: Optional[str]
    current_stage: str
    structure_version: int
    structure_confirmed: bool
    selected_model: str
    qa_level: str
    style_guide: Optional[Dict[str, Any]]
    total_pages: int = 0
    total_words: int = 0
    total_nodes: int = 0
    translatable_nodes: int = 0
    skipped_nodes: int = 0
    translated_nodes: int = 0
    progress_percent: float = 0.0
    created_at: datetime
    updated_at: datetime


# Document Schemas
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_format: str
    file_size_bytes: int
    page_count: int
    word_count_est: int


# Structure Schemas
class NodeUpdate(BaseModel):
    node_type: Optional[NodeType] = None
    content: Optional[str] = None
    translated_content: Optional[str] = None
    approval_status: Optional[ApprovalStatus] = None
    heading_level: Optional[int] = None
    order_index: Optional[int] = None


class ChapterUpdate(BaseModel):
    title: Optional[str] = None
    translated_title: Optional[str] = None
    summary: Optional[str] = None
    order_index: Optional[int] = None


class StructureConfirmRequest(BaseModel):
    lock_version: bool = True


# Translation Schemas
class TranslationStartRequest(BaseModel):
    model_name: Optional[str] = None
    translation_mode: Optional[TranslationMode] = None
    custom_instructions: Optional[str] = None
    worker_count: int = 1


class RetranslateNodeRequest(BaseModel):
    instruction: Optional[str] = None
    custom_model: Optional[str] = None


class TranslationPreviewRequest(BaseModel):
    model_name: Optional[str] = None
    translation_mode: TranslationMode = TranslationMode.NATURAL
    document_type: DocumentType = DocumentType.GENERAL
    custom_instructions: Optional[str] = None
    style_register: str = "ACCESSIBLE"
    sentence_style: str = "MODERATE"


class TranslationPreviewSample(BaseModel):
    node_id: str
    source: str
    translation: str
    quality: Dict[str, Any]


class TranslationPreviewResponse(BaseModel):
    samples: List[TranslationPreviewSample]
    profile: Dict[str, Any]
    prompt_version: str


class TranslationStatusResponse(BaseModel):
    project_id: str
    status: str  # PENDING, RUNNING, PAUSED, COMPLETED, FAILED
    total_nodes: int
    translatable_nodes: int = 0
    skipped_nodes: int = 0
    translated_nodes: int
    failed_nodes: int
    needs_review_nodes: int
    progress_percent: float
    current_chapter_title: Optional[str]
    current_node_id: Optional[str]
    estimated_time_remaining_sec: Optional[int]
    current_chunk_id: Optional[str] = None
    context_mode: Optional[str] = None
    retry_count: int = 0
    quality_state: Optional[str] = None
    naturalness_score: Optional[float] = None
    naturalness_status: Optional[str] = None
    naturalness_critic_calls: int = 0
    naturalness_latency_ms: float = 0.0
    editorial_rewrite_count: int = 0
    editorial_rewrite_success: bool = False
    execution_status: str = "IDLE"
    document_status: str = "TRANSLATION_CONFIGURED"


# Glossary Schemas
class GlossaryItemCreate(BaseModel):
    source_term: str
    target_term: str
    category: str = "GENERAL"
    notes: Optional[str] = None
    locked: bool = True


class GlossaryItemUpdate(BaseModel):
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None
    locked: Optional[bool] = None


class GlossaryItemResponse(BaseModel):
    id: str
    source_term: str
    target_term: str
    category: str
    notes: Optional[str]
    locked: bool
    created_at: datetime


# Entity ledger Schemas
class EntityDecisionCreate(BaseModel):
    source_key: str = Field(..., min_length=1, max_length=255)
    preferred_translation: str = Field(..., min_length=1, max_length=255)
    entity_type: str = Field(default="OTHER", min_length=1, max_length=32)
    aliases: List[str] = Field(default_factory=list)
    locked: bool = False


# QA Schemas
class QAIssueResponse(BaseModel):
    id: str
    node_id: Optional[str]
    issue_type: str
    severity: str
    message: str
    source_snippet: Optional[str]
    translation_snippet: Optional[str]
    suggested_fix: Optional[str]
    status: str
    created_at: datetime


class QAIssueUpdate(BaseModel):
    status: str  # OPEN, RESOLVED, IGNORED


class ConsistencyScanResponse(BaseModel):
    inconsistencies: List[Dict[str, Any]]
    total_issues: int


# Layout Schemas
class LayoutProfileCreate(BaseModel):
    name: str = "Custom Profile"
    page_size: str = "A5"
    page_width_mm: float = 148.0
    page_height_mm: float = 210.0
    margin_top_mm: float = 20.0
    margin_bottom_mm: float = 20.0
    margin_left_mm: float = 20.0
    margin_right_mm: float = 20.0
    body_font: str = "Noto Serif"
    heading_font: str = "Noto Serif"
    body_font_size_pt: float = 11.0
    line_height: float = 1.5
    paragraph_spacing_pt: float = 4.0
    first_line_indent_mm: float = 5.0
    text_alignment: str = "justify"
    chapter_break_mode: str = "next_page"
    show_header: bool = True
    show_footer: bool = True
    show_page_number: bool = True


class LayoutProfileResponse(LayoutProfileCreate):
    id: str
    project_id: str
    is_default: bool
    created_at: datetime


class PreviewRequest(BaseModel):
    chapter_ids: Optional[List[str]] = None
    page_limit: int = 10
    layout_profile_id: Optional[str] = None
    sample_type: str = "representative"  # representative, first_pages, chapter


# Export Schemas
class ExportRequest(BaseModel):
    format: str = "pdf"  # pdf, epub, mobi
    layout_profile_id: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    translator: Optional[str] = "Antigravity Local AI"
    include_cover: bool = True


class ExportResponse(BaseModel):
    export_id: str
    export_format: str
    download_url: str
    file_size_bytes: int
    created_at: datetime


# Hardware & System Schemas
class HardwareInfoResponse(BaseModel):
    cpu_name: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    vram_free_gb: Optional[float]
    cuda_available: bool
    disk_free_gb: float
    ollama_running: bool
    installed_models: List[str]
    tesseract_available: bool
    calibre_available: bool
    recommended_preset: str
