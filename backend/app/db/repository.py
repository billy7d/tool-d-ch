import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from app.db.models import (
    ProjectModel,
    SourceDocumentModel,
    ChapterModel,
    NodeModel,
    TranslationModel,
    TranslationVersionModel,
    GlossaryModel,
    TranslationMemoryModel,
    QAIssueModel,
    AssetModel,
    LayoutProfileModel,
    ExportModel,
    JobModel,
    SettingModel,
)
from app.models.canonical import (
    DocumentNode,
    Chapter,
    Asset,
    CanonicalDocument,
    NodeType,
    NodeStatus,
    ApprovalStatus,
    DocumentMetadata,
    SourceMapping,
    NodeMetadata,
)
from app.services.translation.node_policy import translatable_values


class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_project(self, title: str, description: Optional[str] = None, project_id: Optional[str] = None, **kwargs) -> ProjectModel:
        pid = project_id or str(uuid.uuid4())
        existing = self.db.query(ProjectModel).filter(ProjectModel.id == pid).first()
        if existing:
            return existing

        project = ProjectModel(
            id=pid,
            title=title,
            description=description,
            **kwargs
        )
        self.db.add(project)
        
        # Create default layout profile if not exists
        default_layout = LayoutProfileModel(
            id=str(uuid.uuid4()),
            project_id=pid,
            name="Classic Book",
            is_default=True
        )
        self.db.add(default_layout)
        
        self.db.commit()
        self.db.refresh(project)
        return project

    def get_project(self, project_id: str) -> Optional[ProjectModel]:
        return self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()

    def list_projects(self) -> List[ProjectModel]:
        return self.db.query(ProjectModel).order_by(desc(ProjectModel.updated_at)).all()

    def update_project(self, project_id: str, **kwargs) -> Optional[ProjectModel]:
        project = self.get_project(project_id)
        if not project:
            return None
        for key, value in kwargs.items():
            if hasattr(project, key) and value is not None:
                setattr(project, key, value)
        project.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(project)
        return project

    def delete_project(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        if not project:
            return False
        self.db.delete(project)
        self.db.commit()
        return True

    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        # Always check project database for accurate node & document counts
        from app.db.engine import get_project_db
        try:
            pdb = get_project_db(project_id)
            translatable_types = tuple(translatable_values())
            total_nodes = pdb.query(func.count(NodeModel.id)).filter(NodeModel.project_id == project_id).scalar() or 0
            translatable_nodes = pdb.query(func.count(NodeModel.id)).filter(
                NodeModel.project_id == project_id,
                NodeModel.node_type.in_(translatable_types),
            ).scalar() or 0
            translated_nodes = pdb.query(func.count(NodeModel.id)).filter(
                and_(NodeModel.project_id == project_id, NodeModel.node_type.in_(translatable_types), NodeModel.status.in_(["TRANSLATED", "QA_PASSED"]))
            ).scalar() or 0
            failed_nodes = pdb.query(func.count(NodeModel.id)).filter(
                and_(NodeModel.project_id == project_id, NodeModel.node_type.in_(translatable_types), NodeModel.status == "FAILED")
            ).scalar() or 0
            needs_review_nodes = pdb.query(func.count(NodeModel.id)).filter(
                and_(NodeModel.project_id == project_id, NodeModel.node_type.in_(translatable_types), NodeModel.status == "NEEDS_REVIEW")
            ).scalar() or 0
            
            doc_stats = pdb.query(
                func.sum(SourceDocumentModel.page_count),
                func.sum(SourceDocumentModel.word_count_est)
            ).filter(SourceDocumentModel.project_id == project_id).first()
            
            total_pages = (doc_stats[0] if doc_stats else 0) or 0
            total_words = (doc_stats[1] if doc_stats else 0) or 0
            skipped_nodes = max(0, total_nodes - translatable_nodes)
            terminal_nodes = translated_nodes + failed_nodes + needs_review_nodes
            progress_percent = (translated_nodes / translatable_nodes * 100.0) if translatable_nodes > 0 else 100.0
            pdb.close()
        except Exception:
            total_nodes = 0
            translatable_nodes = 0
            translated_nodes = 0
            failed_nodes = 0
            needs_review_nodes = 0
            total_pages = 0
            total_words = 0
            progress_percent = 0.0
            skipped_nodes = 0
            terminal_nodes = 0

        return {
            "total_nodes": total_nodes,
            "translatable_nodes": translatable_nodes,
            "skipped_nodes": skipped_nodes,
            "terminal_nodes": terminal_nodes,
            "translated_nodes": translated_nodes,
            "failed_nodes": failed_nodes,
            "needs_review_nodes": needs_review_nodes,
            "total_pages": total_pages,
            "total_words": total_words,
            "progress_percent": round(progress_percent, 1),
        }


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_source_document(self, project_id: str, filename: str, file_path: str, file_format: str, file_size: int, page_count: int = 0, word_count: int = 0) -> SourceDocumentModel:
        # Ensure project exists in this DB
        proj = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not proj:
            proj = ProjectModel(
                id=project_id,
                title=filename.replace(f".{file_format}", "").replace("_", " ").title()
            )
            self.db.add(proj)
            self.db.commit()

        doc = SourceDocumentModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            filename=filename,
            file_path=file_path,
            file_format=file_format,
            file_size_bytes=file_size,
            page_count=page_count,
            word_count_est=word_count,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_source_documents(self, project_id: str) -> List[SourceDocumentModel]:
        return self.db.query(SourceDocumentModel).filter(SourceDocumentModel.project_id == project_id).order_by(SourceDocumentModel.order_index).all()


class StructureRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_canonical_document(self, project_id: str, canonical_doc: CanonicalDocument):
        # Ensure project exists
        proj = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not proj:
            proj = ProjectModel(id=project_id, title=canonical_doc.metadata.title or "Document")
            self.db.add(proj)
            self.db.commit()

        # Clean existing chapters and nodes
        self.db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        self.db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        
        node_order = 0
        for ch_idx, chapter in enumerate(canonical_doc.chapters):
            ch_model = ChapterModel(
                id=chapter.id,
                project_id=project_id,
                number=chapter.number,
                title=chapter.title,
                translated_title=chapter.translated_title,
                level=chapter.level,
                source_pages=chapter.source_pages,
                summary=chapter.summary,
                order_index=ch_idx,
            )
            self.db.add(ch_model)
            
            for node in chapter.nodes:
                node_model = NodeModel(
                    id=node.id,
                    project_id=project_id,
                    chapter_id=chapter.id,
                    node_type=node.type.value,
                    content=node.content,
                    translated_content=node.translated_content,
                    order_index=node_order,
                    status=node.status.value,
                    approval_status=node.approval_status.value,
                    confidence=node.metadata.confidence,
                    version=node.version,
                    node_metadata=node.metadata.model_dump(),
                    source_mapping=node.source_mapping.model_dump(),
                )
                self.db.add(node_model)
                node_order += 1
                
        self.db.commit()

    def get_canonical_document(self, project_id: str) -> CanonicalDocument:
        project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        chapters_models = self.db.query(ChapterModel).filter(ChapterModel.project_id == project_id).order_by(ChapterModel.order_index).all()
        
        chapters = []
        for ch in chapters_models:
            node_models = self.db.query(NodeModel).filter(NodeModel.chapter_id == ch.id).order_by(NodeModel.order_index).all()
            nodes = []
            for nm in node_models:
                nodes.append(DocumentNode(
                    id=nm.id,
                    type=NodeType(nm.node_type),
                    content=nm.content,
                    translated_content=nm.translated_content,
                    status=NodeStatus(nm.status),
                    approval_status=ApprovalStatus(nm.approval_status),
                    version=nm.version,
                    order_index=nm.order_index,
                    metadata=NodeMetadata(**(nm.node_metadata or {})),
                    source_mapping=SourceMapping(**(nm.source_mapping or {})),
                ))
            chapters.append(Chapter(
                id=ch.id,
                number=ch.number,
                title=ch.title,
                translated_title=ch.translated_title,
                level=ch.level,
                source_pages=ch.source_pages or [],
                summary=ch.summary,
                order_index=ch.order_index,
                nodes=nodes,
            ))
            
        assets_models = self.db.query(AssetModel).filter(AssetModel.project_id == project_id).all()
        assets = [
            Asset(
                id=a.id,
                original_path=a.original_path,
                derived_path=a.derived_path,
                preview_path=a.preview_path,
                source_page=a.source_page,
                width=a.width,
                height=a.height,
                mime_type=a.mime_type
            ) for a in assets_models
        ]
        
        return CanonicalDocument(
            id=project_id,
            metadata=DocumentMetadata(
                title=project.title if project else "Document",
                created_at=project.created_at.isoformat() if project and project.created_at else None,
                updated_at=project.updated_at.isoformat() if project and project.updated_at else None,
            ),
            chapters=chapters,
            assets=assets,
        )

    def get_node(self, node_id: str) -> Optional[NodeModel]:
        return self.db.query(NodeModel).filter(NodeModel.id == node_id).first()

    def update_node(self, node_id: str, **kwargs) -> Optional[NodeModel]:
        node = self.get_node(node_id)
        if not node:
            return None
        for k, v in kwargs.items():
            if hasattr(node, k) and v is not None:
                setattr(node, k, v)
        node.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(node)
        return node


class TranslationRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_node_translation(self, node_id: str, project_id: str, translated_text: str, model_name: str = "", instruction: Optional[str] = None, created_by: str = "ai", prompt_version: str = "v1", latency_ms: float = 0.0) -> TranslationModel:
        node = self.db.query(NodeModel).filter(NodeModel.id == node_id).first()
        if not node:
            raise ValueError(f"Node {node_id} not found")
            
        node.translated_content = translated_text
        node.status = "TRANSLATED"
        node.version = (node.version or 0) + 1
        node.updated_at = datetime.utcnow()
        
        # Save translation record
        trans = TranslationModel(
            id=str(uuid.uuid4()),
            node_id=node_id,
            project_id=project_id,
            translated_text=translated_text,
            model_name=model_name,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
        )
        self.db.add(trans)
        
        # Save version history for undo/restore
        version = TranslationVersionModel(
            id=str(uuid.uuid4()),
            node_id=node_id,
            version_number=node.version,
            source_content=node.content,
            translated_content=translated_text,
            instruction=instruction,
            created_by=created_by,
        )
        self.db.add(version)
        
        self.db.commit()
        return trans

    def get_node_versions(self, node_id: str) -> List[TranslationVersionModel]:
        return self.db.query(TranslationVersionModel).filter(TranslationVersionModel.node_id == node_id).order_by(desc(TranslationVersionModel.version_number)).all()


class GlossaryRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_glossary(self, project_id: str) -> List[GlossaryModel]:
        return self.db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).order_by(GlossaryModel.source_term).all()

    def add_term(self, project_id: str, source_term: str, target_term: str, category: str = "GENERAL", notes: Optional[str] = None, locked: bool = True) -> GlossaryModel:
        existing = self.db.query(GlossaryModel).filter(
            and_(GlossaryModel.project_id == project_id, GlossaryModel.source_term == source_term)
        ).first()
        if existing:
            existing.target_term = target_term
            existing.category = category
            existing.notes = notes
            existing.locked = locked
            self.db.commit()
            self.db.refresh(existing)
            return existing
            
        term = GlossaryModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            source_term=source_term,
            target_term=target_term,
            category=category,
            notes=notes,
            locked=locked,
        )
        self.db.add(term)
        self.db.commit()
        self.db.refresh(term)
        return term

    def update_term(self, term_id: str, **kwargs) -> Optional[GlossaryModel]:
        term = self.db.query(GlossaryModel).filter(GlossaryModel.id == term_id).first()
        if not term:
            return None
        for k, v in kwargs.items():
            if hasattr(term, k) and v is not None:
                setattr(term, k, v)
        self.db.commit()
        self.db.refresh(term)
        return term

    def delete_term(self, term_id: str) -> bool:
        term = self.db.query(GlossaryModel).filter(GlossaryModel.id == term_id).first()
        if not term:
            return False
        self.db.delete(term)
        self.db.commit()
        return True


class QARepository:
    def __init__(self, db: Session):
        self.db = db

    def add_issue(self, project_id: str, issue_type: str, message: str, node_id: Optional[str] = None, severity: str = "WARNING", source_snippet: Optional[str] = None, translation_snippet: Optional[str] = None, suggested_fix: Optional[str] = None) -> QAIssueModel:
        issue = QAIssueModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            node_id=node_id,
            issue_type=issue_type,
            severity=severity,
            message=message,
            source_snippet=source_snippet,
            translation_snippet=translation_snippet,
            suggested_fix=suggested_fix,
            status="OPEN",
        )
        self.db.add(issue)
        self.db.commit()
        self.db.refresh(issue)
        return issue

    def upsert_open_issue(
        self,
        project_id: str,
        node_id: Optional[str],
        issue_type: str,
        message: str,
        severity: str = "WARNING",
        source_snippet: Optional[str] = None,
        translation_snippet: Optional[str] = None,
        suggested_fix: Optional[str] = None,
    ) -> QAIssueModel:
        """Cập nhật lỗi đang mở theo node để retry không tạo QA_ERROR vô hạn."""
        existing = self.db.query(QAIssueModel).filter(
            QAIssueModel.project_id == project_id,
            QAIssueModel.node_id == node_id,
            QAIssueModel.issue_type == issue_type,
            QAIssueModel.status == "OPEN",
        ).order_by(desc(QAIssueModel.created_at)).first()
        if not existing:
            return self.add_issue(
                project_id=project_id,
                node_id=node_id,
                issue_type=issue_type,
                severity=severity,
                message=message,
                source_snippet=source_snippet,
                translation_snippet=translation_snippet,
                suggested_fix=suggested_fix,
            )

        existing.severity = severity
        existing.message = message
        existing.source_snippet = source_snippet
        existing.translation_snippet = translation_snippet
        existing.suggested_fix = suggested_fix
        self.db.commit()
        self.db.refresh(existing)
        return existing

    def list_issues(self, project_id: str, status: Optional[str] = None) -> List[QAIssueModel]:
        query = self.db.query(QAIssueModel).filter(QAIssueModel.project_id == project_id)
        if status:
            query = query.filter(QAIssueModel.status == status)
        return query.order_by(desc(QAIssueModel.created_at)).all()

    def update_issue_status(self, issue_id: str, status: str) -> Optional[QAIssueModel]:
        issue = self.db.query(QAIssueModel).filter(QAIssueModel.id == issue_id).first()
        if not issue:
            return None
        issue.status = status
        self.db.commit()
        self.db.refresh(issue)
        return issue
