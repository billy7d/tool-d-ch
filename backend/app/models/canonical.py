from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    QUOTE = "quote"
    LIST = "list"
    LIST_ITEM = "list_item"
    IMAGE = "image"
    CAPTION = "caption"
    TABLE = "table"
    FOOTNOTE = "footnote"
    FOOTNOTE_REFERENCE = "footnote_reference"
    EQUATION = "equation"
    CODE_BLOCK = "code_block"
    HORIZONTAL_RULE = "horizontal_rule"
    PAGE_BREAK_HINT = "page_break_hint"


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    TRANSLATING = "TRANSLATING"
    TRANSLATED = "TRANSLATED"
    QA_PENDING = "QA_PENDING"
    QA_PASSED = "QA_PASSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class ApprovalStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    APPROVED = "APPROVED"
    NEEDS_WORK = "NEEDS_WORK"


class DocumentType(str, Enum):
    GENERAL = "GENERAL"
    ACADEMIC = "ACADEMIC"
    TECHNICAL = "TECHNICAL"
    BUSINESS = "BUSINESS"
    FINANCE = "FINANCE"
    LEGAL = "LEGAL"
    LITERATURE = "LITERATURE"
    SELF_HELP = "SELF_HELP"
    MANUAL = "MANUAL"
    BIOGRAPHY = "BIOGRAPHY"
    OTHER = "OTHER"


class TranslationMode(str, Enum):
    NATURAL = "NATURAL"
    BALANCED = "BALANCED"
    FAITHFUL = "FAITHFUL"
    ACADEMIC = "ACADEMIC"
    TECHNICAL = "TECHNICAL"
    CUSTOM = "CUSTOM"


class QALevel(str, Enum):
    OFF = "OFF"
    DETERMINISTIC_ONLY = "DETERMINISTIC_ONLY"
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_QUALITY = "HIGH_QUALITY"
    PUBLISHING = "PUBLISHING"


class SourceMapping(BaseModel):
    source_document: str = ""
    source_page_start: int = 1
    source_page_end: int = 1
    source_block_ids: List[int] = Field(default_factory=list)
    bounding_box: Optional[List[float]] = None  # [x0, y0, x1, y1]


class NodeMetadata(BaseModel):
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None
    is_italic: bool = False
    is_bold: bool = False
    heading_level: Optional[int] = None  # 1-4 for headings
    list_type: Optional[str] = None  # bullet, numbered
    table_as_image: bool = False
    table_rows: Optional[List[List[str]]] = None
    image_asset_id: Optional[str] = None
    image_caption: Optional[str] = None
    equation_latex: Optional[str] = None
    footnote_id: Optional[str] = None
    footnote_number: Optional[str] = None
    code_language: Optional[str] = None
    confidence: float = 1.0  # Structure / OCR confidence
    structure_issues: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentNode(BaseModel):
    id: str  # Stable ID e.g., paragraph_0001_0005
    type: NodeType
    content: str  # Original English content
    translated_content: Optional[str] = None  # Vietnamese translation
    source_mapping: SourceMapping = Field(default_factory=SourceMapping)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)
    status: NodeStatus = NodeStatus.PENDING
    approval_status: ApprovalStatus = ApprovalStatus.UNREVIEWED
    version: int = 1
    order_index: int = 0


class Chapter(BaseModel):
    id: str  # e.g., chapter_0001
    number: Optional[str] = None
    title: str = "Untitled Chapter"
    translated_title: Optional[str] = None
    level: int = 1
    source_pages: List[int] = Field(default_factory=list)
    summary: Optional[str] = None  # Chapter memory / context summary
    order_index: int = 0
    nodes: List[DocumentNode] = Field(default_factory=list)


class Asset(BaseModel):
    id: str
    original_path: str
    derived_path: Optional[str] = None
    preview_path: Optional[str] = None
    source_page: int = 1
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str = "image/png"


class DocumentMetadata(BaseModel):
    title: str = "Untitled Document"
    author: Optional[str] = None
    translator: Optional[str] = "Antigravity Local AI"
    language: str = "en"
    target_language: str = "vi"
    document_type: DocumentType = DocumentType.GENERAL
    total_pages: int = 0
    total_words: int = 0
    total_chapters: int = 0
    total_nodes: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CanonicalDocument(BaseModel):
    id: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    front_matter: List[DocumentNode] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    back_matter: List[DocumentNode] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)
