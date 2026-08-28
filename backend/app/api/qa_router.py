from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

from app.db.engine import get_project_db
from app.db.models import NodeModel, ChapterModel, QAIssueModel, GlossaryModel, SemanticReviewModel
from app.db.repository import QARepository, ProjectRepository
from app.models.schemas import QAIssueResponse, QAIssueUpdate, ConsistencyScanResponse, RetranslateNodeRequest
from app.services.qa.deterministic_qa import DeterministicQA
from app.services.qa.ai_qa import AIQAEngine
from app.services.qa.consistency_scanner import ConsistencyScanner
from app.services.qa.issue_policy import evaluate_issue_resolution
from app.services.qa.result_validator import qa_error, validate_qa_result
from app.services.translation.worker import translation_worker
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import build_translation_signature_from_config
from app.services.translation.semantic_assurance import SemanticAssuranceService
from app.services.qa.global_consistency import GlobalConsistencyScanner
from app.services.translation.entity_ledger import EntityLedgerService

router = APIRouter(prefix="/api/projects/{project_id}/qa", tags=["QA"])


def _semantic_engine(db, project_id: str):
    project = ProjectRepository(db).get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    config = TranslationConfig.from_project(project)
    provider = translation_worker.get_provider(config.model_name)
    glossary = GlossaryService.get_locked_glossary_map(db, project_id)
    return project, config, ContextualTranslationEngine(db, project, provider, config, glossary)


def _review_existing_node(db, project_id: str, engine, node: NodeModel):
    if not node.translated_content:
        raise HTTPException(status_code=422, detail="Node chưa có bản dịch.")
    chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=422, detail="Node không thuộc chương hợp lệ.")
    entities = EntityLedgerService.relevant_decisions(
        db, project_id, node.content, engine.locked_glossary,
    )
    result = SemanticAssuranceService.evaluate_candidate(
        engine, chapter, engine.canonical_node(node), node.translated_content, entities,
        max_repairs=0,
    )
    SemanticAssuranceService.persist_existing_review(
        db, project_id, node, result, engine.config.semantic_critic_model or engine.config.model_name,
    )
    return result


@router.get("/semantic-summary")
def semantic_summary(project_id: str):
    db = get_project_db(project_id)
    try:
        nodes_total = db.query(NodeModel).filter(
            NodeModel.project_id == project_id, NodeModel.translated_content.isnot(None),
        ).count()
        rows = db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id,
            SemanticReviewModel.is_stale.is_(False),
        ).order_by(SemanticReviewModel.created_at).all()
        latest = {row.node_id: row for row in rows}
        values = list(latest.values())
        return {
            "nodes_total": nodes_total,
            "risk_low": sum(row.risk_level == "LOW" for row in values),
            "risk_medium": sum(row.risk_level == "MEDIUM" for row in values),
            "risk_high": sum(row.risk_level == "HIGH" for row in values),
            "critic_reviewed": sum(row.critic_status != "NOT_REQUIRED" for row in values),
            "semantic_pass": sum(row.critic_status in {"PASS", "NOT_REQUIRED"} for row in values),
            "semantic_failed": sum(row.critic_status == "FAIL" for row in values),
            "semantic_error": sum(row.critic_status == "ERROR" for row in values),
            "needs_review": db.query(NodeModel).filter(NodeModel.project_id == project_id, NodeModel.status == "NEEDS_REVIEW").count(),
        }
    finally:
        db.close()


@router.get("/semantic-reviews")
def semantic_reviews(project_id: str):
    db = get_project_db(project_id)
    try:
        rows = db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id,
            SemanticReviewModel.is_stale.is_(False),
        ).order_by(SemanticReviewModel.created_at.desc()).all()
        nodes = {node.id: node for node in db.query(NodeModel).filter(NodeModel.project_id == project_id).all()}
        return [{
            "id": row.id, "node_id": row.node_id, "translation_version": row.translation_version,
            "risk_score": row.risk_score, "risk_level": row.risk_level,
            "critic_status": row.critic_status, "critic_score": row.critic_score,
            "issues": row.issues_json or [], "model_name": row.model_name,
            "prompt_version": row.prompt_version, "critic_latency_ms": row.critic_latency_ms,
            "critic_request_tokens": row.critic_request_tokens,
            "node_status": nodes.get(row.node_id).status if nodes.get(row.node_id) else "",
            "source_excerpt": (nodes.get(row.node_id).content if nodes.get(row.node_id) else "")[:240],
            "translation_excerpt": (nodes.get(row.node_id).translated_content if nodes.get(row.node_id) else "")[:240],
        } for row in rows]
    finally:
        db.close()


@router.post("/run-semantic-review")
def run_semantic_review(project_id: str):
    db = get_project_db(project_id)
    try:
        _, _, engine = _semantic_engine(db, project_id)
        nodes = db.query(NodeModel).filter(
            NodeModel.project_id == project_id, NodeModel.translated_content.isnot(None),
        ).all()
        results = [_review_existing_node(db, project_id, engine, node) for node in nodes]
        return {
            "reviewed": len(results),
            "critic_calls": sum(item.critic_calls for item in results),
            "failed": sum(not item.approved for item in results),
        }
    finally:
        db.close()


@router.post("/review-node/{node_id}")
def review_semantic_node(project_id: str, node_id: str):
    db = get_project_db(project_id)
    try:
        _, _, engine = _semantic_engine(db, project_id)
        node = db.query(NodeModel).filter(NodeModel.project_id == project_id, NodeModel.id == node_id).first()
        if not node:
            raise HTTPException(status_code=404, detail="Không tìm thấy node.")
        result = _review_existing_node(db, project_id, engine, node)
        return result.__dict__
    finally:
        db.close()


@router.post("/repair-semantic/{node_id}")
def repair_semantic_node(project_id: str, node_id: str):
    db = get_project_db(project_id)
    try:
        _, config, engine = _semantic_engine(db, project_id)
        node = db.query(NodeModel).filter(NodeModel.project_id == project_id, NodeModel.id == node_id).first()
        if not node:
            raise HTTPException(status_code=404, detail="Không tìm thấy node.")
        chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
        review = db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id, SemanticReviewModel.node_id == node_id,
        ).order_by(SemanticReviewModel.created_at.desc()).first()
        errors = list(review.issues_json or []) if review else []
        repaired = engine.repair_node(
            chapter, engine.canonical_node(node),
            [{"code": item.get("type", "MEANING_DRIFT"), "message": item.get("message", "Sửa lỗi semantic.")} for item in errors],
            max_attempts=1,
        )
        if not repaired.passed:
            raise HTTPException(status_code=422, detail={"code": "DETERMINISTIC_REPAIR_FAILED", "issues": repaired.quality.issues})
        signature = build_translation_signature_from_config(config, engine.locked_glossary)
        resolved_ids = [issue.id for issue in db.query(QAIssueModel).filter(
            QAIssueModel.project_id == project_id,
            QAIssueModel.node_id == node_id,
            QAIssueModel.status == "OPEN",
        ).all() if issue.issue_type in {item.get("type") for item in errors}]
        result = SemanticAssuranceService.assure_and_commit(
            engine, chapter, node, repaired.translated_text, signature, config.model_name,
            "semantic_repair", instruction="Sửa theo semantic critic từ nguyên bản gốc.",
            resolved_issue_ids=resolved_ids, previous_repairs=1,
        )
        if not result.approved:
            raise HTTPException(status_code=422, detail={"code": "SEMANTIC_REPAIR_FAILED", "issues": result.errors})
        return result.__dict__
    finally:
        db.close()


@router.post("/global-consistency")
def run_global_consistency(project_id: str):
    db = get_project_db(project_id)
    try:
        findings = GlobalConsistencyScanner.scan_project(db, project_id, persist=True)
        return {"issues": findings, "total_issues": len(findings)}
    finally:
        db.close()


@router.post("/run")
def run_qa_checks(project_id: str, enable_ai_qa: bool = False):
    db = get_project_db(project_id)
    qa_repo = QARepository(db)

    try:
        locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)
        # Clear existing open issues
        db.query(QAIssueModel).filter(
            QAIssueModel.project_id == project_id,
            QAIssueModel.status == "OPEN"
        ).delete()
        db.commit()

        nodes = db.query(NodeModel).filter(
            NodeModel.project_id == project_id,
            NodeModel.translated_content != None
        ).all()

        created_issues_count = 0

        for n in nodes:
            # 1. Deterministic QA
            issues = DeterministicQA.audit_node(n, locked_glossary)
            for iss in issues:
                qa_repo.add_issue(
                    project_id=project_id,
                    node_id=n.id,
                    issue_type=iss["issue_type"],
                    severity=iss["severity"],
                    message=iss["message"],
                    source_snippet=iss["source_snippet"],
                    translation_snippet=iss["translation_snippet"],
                    suggested_fix=iss["suggested_fix"]
                )
                created_issues_count += 1

            # 2. AI QA (Optional)
            if enable_ai_qa:
                ai_issue = AIQAEngine.audit_node_ai(db, project_id, n)
                if ai_issue:
                    qa_repo.add_issue(
                        project_id=project_id,
                        node_id=n.id,
                        issue_type=ai_issue["issue_type"],
                        severity=ai_issue["severity"],
                        message=ai_issue["message"],
                        source_snippet=ai_issue["source_snippet"],
                        translation_snippet=ai_issue["translation_snippet"],
                        suggested_fix=ai_issue["suggested_fix"]
                    )
                    created_issues_count += 1

        return {
            "message": f"Kiểm tra QA hoàn tất. Phát hiện {created_issues_count} cảnh báo.",
            "total_issues": created_issues_count
        }
    finally:
        db.close()


@router.get("/issues", response_model=List[QAIssueResponse])
def list_qa_issues(project_id: str, status: Optional[str] = None):
    db = get_project_db(project_id)
    repo = QARepository(db)
    try:
        issues = repo.list_issues(project_id, status=status)
        return [
            QAIssueResponse(
                id=i.id,
                node_id=i.node_id,
                issue_type=i.issue_type,
                severity=i.severity,
                message=i.message,
                source_snippet=i.source_snippet,
                translation_snippet=i.translation_snippet,
                suggested_fix=i.suggested_fix,
                status=i.status,
                created_at=i.created_at
            ) for i in issues
        ]
    finally:
        db.close()


@router.patch("/issues/{issue_id}", response_model=QAIssueResponse)
def update_issue_status(project_id: str, issue_id: str, payload: QAIssueUpdate):
    db = get_project_db(project_id)
    repo = QARepository(db)
    try:
        updated = repo.update_issue_status(issue_id, payload.status)
        if not updated:
            raise HTTPException(status_code=404, detail="Không tìm thấy vấn đề QA.")
        return QAIssueResponse(
            id=updated.id,
            node_id=updated.node_id,
            issue_type=updated.issue_type,
            severity=updated.severity,
            message=updated.message,
            source_snippet=updated.source_snippet,
            translation_snippet=updated.translation_snippet,
            suggested_fix=updated.suggested_fix,
            status=updated.status,
            created_at=updated.created_at
        )
    finally:
        db.close()


@router.get("/consistency", response_model=ConsistencyScanResponse)
def get_consistency_scan(project_id: str):
    db = get_project_db(project_id)
    try:
        inconsistencies = ConsistencyScanner.scan_project(db, project_id)
        return ConsistencyScanResponse(
            inconsistencies=inconsistencies,
            total_issues=len(inconsistencies)
        )
    finally:
        db.close()


@router.post("/find_replace")
def find_and_replace(project_id: str, find_text: str, replace_text: str, apply_changes: bool = False):
    db = get_project_db(project_id)
    try:
        nodes = db.query(NodeModel).filter(
            NodeModel.project_id == project_id,
            NodeModel.translated_content.like(f"%{find_text}%")
        ).all()

        matches_count = 0
        affected_nodes = []

        for n in nodes:
            if n.translated_content and find_text in n.translated_content:
                count = n.translated_content.count(find_text)
                matches_count += count
                affected_nodes.append(n.id)
                if apply_changes:
                    n.translated_content = n.translated_content.replace(find_text, replace_text)
                    SemanticAssuranceService.invalidate_for_nodes(db, project_id, [n.id])

        if apply_changes:
            db.commit()

        return {
            "find_text": find_text,
            "replace_text": replace_text,
            "total_matches": matches_count,
            "affected_nodes_count": len(affected_nodes),
            "applied": apply_changes
        }
    finally:
        db.close()


@router.post("/retranslate_all_issues")
def retranslate_all_qa_issues(
    project_id: str,
    payload: Optional[RetranslateNodeRequest] = None
):
    """Dịch lại các node có lỗi QA bằng đúng engine và quality path chuẩn."""
    db = get_project_db(project_id)
    qa_repo = QARepository(db)
    try:
        issues = db.query(QAIssueModel).filter(
            QAIssueModel.project_id == project_id,
            QAIssueModel.status == "OPEN",
            QAIssueModel.node_id != None
        ).all()
        
        node_ids = list(dict.fromkeys([i.node_id for i in issues if i.node_id]))
        if not node_ids:
            return {
                "message": "Không có đoạn văn nào có cảnh báo QA cần dịch lại.",
                "total_nodes": 0,
                "fixed_nodes": 0
            }

        proj_repo = ProjectRepository(db)
        proj = proj_repo.get_project(project_id)
        
        custom_model = payload.custom_model if payload else None
        custom_instruction = payload.instruction if payload else None
        
        instruction = custom_instruction or "Dịch lại trọn vẹn, tự nhiên hơn, chuẩn văn phong xuất bản và đúng ngữ cảnh."
        try:
            PromptBuilder.validate_custom_instructions(instruction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        config = TranslationConfig.from_project(
            proj,
            model_override=custom_model,
            custom_instruction_override=instruction,
        )
        provider = translation_worker.get_provider(config.model_name)
        locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)
        engine = ContextualTranslationEngine(db, proj, provider, config, locked_glossary)
        signature = build_translation_signature_from_config(config, locked_glossary)
        fixed_count = 0
        total_nodes = len(node_ids)

        for idx, nid in enumerate(node_ids):
            node = db.query(NodeModel).filter(NodeModel.id == nid).first()
            if not node:
                continue

            try:
                chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
                node_issues = [
                    {"code": issue.issue_type, "message": issue.message}
                    for issue in issues if issue.node_id == nid
                ]
                if not chapter:
                    node.status = "NEEDS_REVIEW"
                    db.commit()
                    continue
                result = engine.repair_node(chapter, engine.canonical_node(node), node_issues)
                probe = NodeModel(content=node.content, translated_content=result.translated_text)
                deterministic_issues = DeterministicQA.audit_node(probe, locked_glossary)
                if not result.passed:
                    node.status = "NEEDS_REVIEW"
                    db.commit()
                    continue
                node_open_issues = [issue for issue in issues if issue.node_id == nid]
                remaining_codes = [issue["code"] for issue in result.quality.issues]
                remaining_codes.extend(issue["issue_type"] for issue in deterministic_issues)
                preliminary = evaluate_issue_resolution(
                    [issue.issue_type for issue in node_open_issues],
                    remaining_codes,
                )
                semantic_review = None
                if preliminary.semantic_review_required and not deterministic_issues:
                    try:
                        semantic_review = validate_qa_result(provider.review_translation(
                            source_text=node.content,
                            translated_text=result.translated_text,
                            glossary_terms=locked_glossary,
                            model=config.model_name,
                        ))
                    except Exception as exc:
                        semantic_review = qa_error(f"Semantic re-review thất bại: {exc}")
                elif preliminary.semantic_review_required:
                    semantic_review = {
                        "status": "FAIL", "is_passed": False, "score": 0.0,
                        "issues": ["Candidate repair chưa pass deterministic QA."],
                    }
                resolution = evaluate_issue_resolution(
                    [issue.issue_type for issue in node_open_issues],
                    remaining_codes,
                    semantic_review,
                )
                if resolution.qa_error:
                    qa_repo.upsert_open_issue(
                        project_id=project_id,
                        node_id=nid,
                        issue_type="QA_ERROR",
                        severity="ERROR",
                        message=resolution.qa_error,
                        source_snippet=node.content[:150],
                        translation_snippet=result.translated_text[:150],
                        suggested_fix="Kiểm tra provider semantic QA rồi chạy lại repair.",
                    )
                if deterministic_issues or any(status == "OPEN" for status in resolution.statuses.values()):
                    node.status = "NEEDS_REVIEW"
                    db.commit()
                    continue
                resolved_ids = [
                    issue.id for issue in node_open_issues
                    if resolution.statuses.get(issue.issue_type.upper()) == "RESOLVED"
                ]
                semantic = SemanticAssuranceService.assure_and_commit(
                    engine, chapter, node, result.translated_text, signature,
                    f"{config.model_name} (Bulk QA Fix)", "qa_bulk_fix",
                    instruction=instruction, resolved_issue_ids=resolved_ids,
                    previous_repairs=max(1, result.attempts - 1),
                )
                if semantic.approved:
                    fixed_count += 1
            except Exception as e_node:
                # Mọi lỗi trong bulk repair phải được lưu lại, không chỉ in ra console.
                db.rollback()
                failed_node = db.query(NodeModel).filter(NodeModel.id == nid).first()
                if failed_node:
                    failed_node.status = "NEEDS_REVIEW"
                    qa_repo.upsert_open_issue(
                        project_id=project_id,
                        node_id=nid,
                        issue_type="QA_ERROR",
                        severity="ERROR",
                        message=f"QA bulk repair failed: {e_node}",
                        source_snippet=(failed_node.content or "")[:150],
                        translation_snippet=(failed_node.translated_content or "")[:150],
                        suggested_fix="Kiểm tra provider và chạy lại repair; candidate chưa được coi là đạt.",
                    )
                print(f"[QA Bulk Fix] Error fixing node {nid}: {e_node}")

            # Broadcast SSE progress
            percent = round((idx + 1) / total_nodes * 100.0, 1)
            translation_worker.broadcast_event("QA_FIX_PROGRESS", {
                "project_id": project_id,
                "completed": idx + 1,
                "total": total_nodes,
                "percent": percent,
                "current_node_id": nid
            })

        db.commit()

        return {
            "message": f"Đã hoàn thành dịch lại {fixed_count}/{total_nodes} đoạn văn bản có cảnh báo.",
            "total_nodes": total_nodes,
            "fixed_nodes": fixed_count
        }
    finally:
        db.close()
