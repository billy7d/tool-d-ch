from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False, default="New Project")
    description = Column(Text, nullable=True)
    source_language = Column(String(16), default="en")
    target_language = Column(String(16), default="vi")
    document_type = Column(String(32), default="GENERAL")
    translation_mode = Column(String(32), default="NATURAL")
    custom_instructions = Column(Text, nullable=True)
    current_stage = Column(String(32), default="IMPORTED")
    structure_version = Column(Integer, default=1)
    structure_confirmed = Column(Boolean, default=False)
    selected_model = Column(String(128), default="qwen2.5:7b")
    qa_level = Column(String(32), default="BALANCED")
    style_guide = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_documents = relationship("SourceDocumentModel", back_populates="project", cascade="all, delete-orphan")
    chapters = relationship("ChapterModel", back_populates="project", cascade="all, delete-orphan", order_by="ChapterModel.order_index")
    glossary_items = relationship("GlossaryModel", back_populates="project", cascade="all, delete-orphan")
    qa_issues = relationship("QAIssueModel", back_populates="project", cascade="all, delete-orphan")
    assets = relationship("AssetModel", back_populates="project", cascade="all, delete-orphan")
    layout_profiles = relationship("LayoutProfileModel", back_populates="project", cascade="all, delete-orphan")
    exports = relationship("ExportModel", back_populates="project", cascade="all, delete-orphan")


class SourceDocumentModel(Base):
    __tablename__ = "source_documents"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_format = Column(String(32), nullable=False)  # pdf, epub, docx, txt, md
    file_size_bytes = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    word_count_est = Column(Integer, default=0)
    order_index = Column(Integer, default=0)
    analysis_status = Column(String(32), default="PENDING")
    scanned_pages_count = Column(Integer, default=0)
    text_pages_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="source_documents")


class ChapterModel(Base):
    __tablename__ = "chapters"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    number = Column(String(64), nullable=True)
    title = Column(String(255), nullable=False, default="Untitled Chapter")
    translated_title = Column(String(255), nullable=True)
    level = Column(Integer, default=1)
    source_pages = Column(JSON, default=list)  # list of int
    summary = Column(Text, nullable=True)  # chapter memory
    order_index = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="chapters")
    nodes = relationship("NodeModel", back_populates="chapter", cascade="all, delete-orphan", order_by="NodeModel.order_index")


class NodeModel(Base):
    __tablename__ = "nodes"

    id = Column(String(64), primary_key=True)  # stable ID e.g., paragraph_0001_0005
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_id = Column(String(64), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=True, index=True)
    node_type = Column(String(32), nullable=False, default="paragraph")
    content = Column(Text, nullable=False)  # English text
    translated_content = Column(Text, nullable=True)  # Vietnamese translation
    order_index = Column(Integer, default=0, index=True)
    status = Column(String(32), default="PENDING", index=True)  # PENDING, TRANSLATING, TRANSLATED, etc.
    approval_status = Column(String(32), default="UNREVIEWED")  # UNREVIEWED, APPROVED, NEEDS_WORK
    confidence = Column(Float, default=1.0)
    version = Column(Integer, default=1)
    
    # Metadata and source mapping stored as JSON
    node_metadata = Column(JSON, default=dict)
    source_mapping = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chapter = relationship("ChapterModel", back_populates="nodes")
    translations = relationship("TranslationModel", back_populates="node", cascade="all, delete-orphan")
    versions = relationship("TranslationVersionModel", back_populates="node", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_nodes_project_order", "project_id", "order_index"),
        Index("ix_nodes_project_status", "project_id", "status"),
    )


class TranslationModel(Base):
    __tablename__ = "translations"

    id = Column(String(64), primary_key=True)
    node_id = Column(String(64), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    translated_text = Column(Text, nullable=False)
    model_name = Column(String(128), nullable=True)
    prompt_version = Column(String(32), default="v1")
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    node = relationship("NodeModel", back_populates="translations")


class TranslationVersionModel(Base):
    __tablename__ = "translation_versions"

    id = Column(String(64), primary_key=True)
    node_id = Column(String(64), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, default=1)
    source_content = Column(Text, nullable=False)
    translated_content = Column(Text, nullable=False)
    instruction = Column(Text, nullable=True)
    created_by = Column(String(32), default="ai")  # ai or user
    created_at = Column(DateTime, default=datetime.utcnow)

    node = relationship("NodeModel", back_populates="versions")


class GlossaryModel(Base):
    __tablename__ = "glossary"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    source_term = Column(String(255), nullable=False)
    target_term = Column(String(255), nullable=False)
    category = Column(String(64), default="GENERAL")
    notes = Column(Text, nullable=True)
    locked = Column(Boolean, default=True)  # locked glossary is strictly enforced
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="glossary_items")

    __table_args__ = (
        Index("ix_glossary_project_source", "project_id", "source_term"),
    )


class TranslationMemoryModel(Base):
    __tablename__ = "translation_memory"

    id = Column(String(64), primary_key=True)
    source_hash = Column(String(64), nullable=False, index=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    style_hash = Column(String(64), default="")
    glossary_hash = Column(String(64), default="")
    model_name = Column(String(128), default="")
    prompt_version = Column(String(32), default="v1")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_tm_lookup", "source_hash", "style_hash", "glossary_hash"),
    )


class QAIssueModel(Base):
    __tablename__ = "qa_issues"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    node_id = Column(String(64), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=True, index=True)
    issue_type = Column(String(64), nullable=False)  # NUMBER_MISMATCH, LENGTH_ANOMALY, TERM_INCONSISTENCY, etc.
    severity = Column(String(32), default="WARNING")  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    source_snippet = Column(Text, nullable=True)
    translation_snippet = Column(Text, nullable=True)
    suggested_fix = Column(Text, nullable=True)
    status = Column(String(32), default="OPEN")  # OPEN, RESOLVED, IGNORED
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="qa_issues")


class AssetModel(Base):
    __tablename__ = "assets"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    original_path = Column(String(512), nullable=False)
    derived_path = Column(String(512), nullable=True)
    preview_path = Column(String(512), nullable=True)
    source_page = Column(Integer, default=1)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    mime_type = Column(String(64), default="image/png")
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="assets")


class LayoutProfileModel(Base):
    __tablename__ = "layout_profiles"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(128), default="Classic Book")
    page_size = Column(String(32), default="A5")  # A4, A5, LETTER, 6X9, CUSTOM
    page_width_mm = Column(Float, default=148.0)
    page_height_mm = Column(Float, default=210.0)
    margin_top_mm = Column(Float, default=20.0)
    margin_bottom_mm = Column(Float, default=20.0)
    margin_left_mm = Column(Float, default=20.0)
    margin_right_mm = Column(Float, default=20.0)
    body_font = Column(String(128), default="Noto Serif")
    heading_font = Column(String(128), default="Noto Serif")
    body_font_size_pt = Column(Float, default=11.0)
    line_height = Column(Float, default=1.5)
    paragraph_spacing_pt = Column(Float, default=4.0)
    first_line_indent_mm = Column(Float, default=5.0)
    text_alignment = Column(String(32), default="justify")  # justify, left
    chapter_break_mode = Column(String(32), default="next_page")  # next_page, right_page, continuous
    show_header = Column(Boolean, default=True)
    show_footer = Column(Boolean, default=True)
    show_page_number = Column(Boolean, default=True)
    is_default = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="layout_profiles")


class ExportModel(Base):
    __tablename__ = "exports"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    export_format = Column(String(32), nullable=False)  # pdf, epub, mobi
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    status = Column(String(32), default="COMPLETED")  # GENERATING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ProjectModel", back_populates="exports")


class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), nullable=False, index=True)
    job_type = Column(String(64), nullable=False)  # ANALYSIS, OCR, TRANSLATION, QA, EXPORT
    status = Column(String(32), default="CREATED")  # CREATED, QUEUED, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED
    total_units = Column(Integer, default=0)
    completed_units = Column(Integer, default=0)
    failed_units = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SettingModel(Base):
    __tablename__ = "settings"

    key = Column(String(128), primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
