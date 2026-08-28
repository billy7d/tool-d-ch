import time
import asyncio
import threading
import json
import os
from typing import Dict, Any, Optional, Callable, List
from app.db.engine import get_project_db
from app.db.models import ProjectModel, ChapterModel, NodeModel
from app.db.repository import ProjectRepository
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.mock_provider import MockProvider
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import build_translation_signature_from_config
from app.services.translation.node_policy import translatable_values
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.semantic_assurance import SemanticAssuranceService


class TranslationWorker:
    def __init__(self):
        self.running_tasks: Dict[str, threading.Event] = {}  # project_id -> stop_event
        self.pause_tasks: Dict[str, threading.Event] = {}    # project_id -> pause_event
        self.active_threads: Dict[str, threading.Thread] = {}
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.telemetry: Dict[str, Dict[str, Any]] = {}

    def register_event_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(callback)

    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        payload = {"type": event_type, "data": data, "timestamp": time.time()}
        project_id = data.get("project_id")
        if project_id:
            current = self.telemetry.setdefault(project_id, {})
            if event_type == "TRANSLATION_PROGRESS":
                current.update({
                    "current_chapter_title": data.get("chapter_title"),
                    "current_chunk_id": data.get("chunk_id"),
                    "context_mode": data.get("context_mode"),
                    "quality_state": "PASSING_GATE",
                })
            elif event_type == "TRANSLATION_RETRYING":
                current.update({"retry_count": data.get("attempt", 0), "quality_state": "RETRYING"})
            elif event_type == "TRANSLATION_NODE_NEEDS_REVIEW":
                current.update({"quality_state": "NEEDS_REVIEW"})
        for sub in self.subscribers:
            try:
                sub(payload)
            except Exception:
                pass

    def recover_crashed_jobs(self, project_id: str):
        """
        PRD Section 74: Crash Recovery.
        Startup resets any nodes stuck in 'TRANSLATING' back to 'PENDING'.
        """
        db = get_project_db(project_id)
        try:
            stuck_nodes = db.query(NodeModel).filter(
                NodeModel.project_id == project_id,
                NodeModel.status == "TRANSLATING"
            ).all()
            for n in stuck_nodes:
                n.status = "PENDING"
            if stuck_nodes:
                db.commit()
                print(f"[Crash Recovery] Reset {len(stuck_nodes)} stuck nodes to PENDING for project {project_id}")
        finally:
            db.close()

    def get_provider(self, model_name: str) -> TranslationProvider:
        if model_name.lower().startswith("mock") or model_name.lower().startswith("demo"):
            return MockProvider()
        return OllamaProvider(default_model=model_name)

    def translate_project_sync(
        self,
        project_id: str,
        model_name: str = "mock",
        custom_instructions: Optional[str] = None,
        provider: Optional[TranslationProvider] = None
    ):
        """
        Synchronous translation runner for tests and batch scripting.
        """
        stop_event = threading.Event()
        stop_event.set()
        pause_event = threading.Event()
        self._run_translation_loop(
            project_id=project_id,
            model_name=model_name,
            custom_instructions=custom_instructions,
            stop_event=stop_event,
            pause_event=pause_event,
            override_provider=provider
        )

    def start_translation(self, project_id: str, model_name: str = "qwen2.5:7b", custom_instructions: Optional[str] = None):
        # First recover any stuck states
        self.recover_crashed_jobs(project_id)

        if project_id in self.running_tasks and self.running_tasks[project_id].is_set():
            # If paused, unpause
            if project_id in self.pause_tasks and self.pause_tasks[project_id].is_set():
                self.pause_tasks[project_id].clear()
                self.broadcast_event("TRANSLATION_RESUMED", {"project_id": project_id})
                return
            return  # Already running

        stop_event = threading.Event()
        stop_event.set()
        pause_event = threading.Event()

        self.running_tasks[project_id] = stop_event
        self.pause_tasks[project_id] = pause_event

        thread = threading.Thread(
            target=self._run_translation_loop,
            args=(project_id, model_name, custom_instructions, stop_event, pause_event),
            daemon=True
        )
        self.active_threads[project_id] = thread
        thread.start()

    def pause_translation(self, project_id: str):
        if project_id in self.pause_tasks:
            self.pause_tasks[project_id].set()
            self.telemetry.setdefault(project_id, {})["execution_status"] = "PAUSED"
            self.broadcast_event("TRANSLATION_PAUSED", {"project_id": project_id})

    def resume_translation(self, project_id: str):
        if project_id in self.pause_tasks:
            self.pause_tasks[project_id].clear()
            self.telemetry.setdefault(project_id, {})["execution_status"] = "RUNNING"
            self.broadcast_event("TRANSLATION_RESUMED", {"project_id": project_id})

    def stop_translation(self, project_id: str):
        if project_id in self.running_tasks:
            self.running_tasks[project_id].clear()
        if project_id in self.pause_tasks:
            self.pause_tasks[project_id].clear()
        self.telemetry.setdefault(project_id, {})["execution_status"] = "STOPPED"
        db = get_project_db(project_id)
        try:
            project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
            if project and project.current_stage == "TRANSLATING":
                project.current_stage = "TRANSLATION_CONFIGURED"
                db.commit()
        finally:
            db.close()
        self.broadcast_event("TRANSLATION_STOPPED", {"project_id": project_id})

    def retry_failed(self, project_id: str):
        db = get_project_db(project_id)
        model_name = "qwen2.5:7b"
        try:
            project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
            if project and project.selected_model:
                model_name = project.selected_model
            failed_nodes = db.query(NodeModel).filter(
                NodeModel.project_id == project_id,
                NodeModel.status.in_(["FAILED", "NEEDS_REVIEW"])
            ).all()
            for n in failed_nodes:
                n.status = "PENDING"
            db.commit()
        finally:
            db.close()
        self.start_translation(project_id, model_name=model_name)

    def _wait_for_ollama_recovery(self, provider: TranslationProvider, stop_event: threading.Event, project_id: str, max_wait_sec: int = 180) -> bool:
        """
        Waits for Ollama / LLM endpoint to come back online when connection drops.
        Uses exponential backoff polling.
        """
        start_wait = time.time()
        wait_interval = 2.0
        self.broadcast_event("TRANSLATION_WAITING", {
            "project_id": project_id,
            "message": "Đang chờ dịch vụ mô hình AI kết nối lại..."
        })
        while stop_event.is_set() and (time.time() - start_wait < max_wait_sec):
            time.sleep(wait_interval)
            if provider.health_check():
                self.broadcast_event("TRANSLATION_RESUMED", {
                    "project_id": project_id,
                    "message": "Đã kết nối lại mô hình AI thành công. Đang tự động tiếp tục dịch."
                })
                return True
            wait_interval = min(10.0, wait_interval * 1.5)
        return False

    def _run_translation_loop(
        self,
        project_id: str,
        model_name: str,
        custom_instructions: Optional[str],
        stop_event: threading.Event,
        pause_event: threading.Event,
        override_provider: Optional[TranslationProvider] = None
    ):
        db = get_project_db(project_id)
        provider = override_provider or self.get_provider(model_name)
        proj_repo = ProjectRepository(db)
        project = None
        systemic_failure = False
        self.telemetry.setdefault(project_id, {})["execution_status"] = "RUNNING"
        try:
            project = proj_repo.get_project(project_id)
            if not project:
                return
            if not provider.health_check():
                recovered = self._wait_for_ollama_recovery(provider, stop_event, project_id, max_wait_sec=20)
                if not recovered:
                    systemic_failure = True
                    project.current_stage = "TRANSLATION_FAILED"
                    db.commit()
                    self.broadcast_event("TRANSLATION_ERROR", {
                        "project_id": project_id,
                        "error_type": "PROVIDER_OFFLINE",
                        "message": "Không thể kết nối provider dịch local.",
                    })
                    return

            config = TranslationConfig.from_project(
                project,
                model_override=model_name,
                custom_instruction_override=custom_instructions,
            )
            PromptBuilder.validate_custom_instructions(config.custom_instructions)
            locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)
            signature = build_translation_signature_from_config(config, locked_glossary)
            engine = ContextualTranslationEngine(
                db, project, provider, config, locked_glossary, self.broadcast_event,
            )
            project.current_stage = "TRANSLATING"
            db.commit()
            chapters = db.query(ChapterModel).filter(
                ChapterModel.project_id == project_id
            ).order_by(ChapterModel.order_index).all()
            translatable_types = tuple(translatable_values())

            for chapter in chapters:
                while stop_event.is_set():
                    while pause_event.is_set() and stop_event.is_set():
                        time.sleep(0.25)
                    if not stop_event.is_set():
                        break
                    pending_models = db.query(NodeModel).filter(
                        NodeModel.chapter_id == chapter.id,
                        NodeModel.node_type.in_(translatable_types),
                        NodeModel.status.in_(["PENDING", "FAILED"]),
                    ).order_by(NodeModel.order_index).all()
                    if not pending_models:
                        break

                    first_model = pending_models[0]
                    tm_hit = TranslationMemoryService.lookup(
                        db,
                        first_model.content,
                        style_hash=signature.style_hash,
                        glossary_hash=signature.glossary_hash,
                        prompt_version=signature.prompt_version,
                        locked_glossary=locked_glossary,
                    )
                    if tm_hit:
                        tm_quality = TranslationQualityGate().validate(
                            first_model.content, tm_hit, locked_glossary,
                        )
                        if tm_quality.passed:
                            semantic = SemanticAssuranceService.assure_and_commit(
                                engine, chapter, first_model, tm_hit, signature,
                                f"TM ({config.model_name})", "translation_worker_tm",
                            )
                            if not semantic.approved:
                                self.broadcast_event("TRANSLATION_NODE_NEEDS_REVIEW", {
                                    "project_id": project_id, "node_id": first_model.id,
                                    "issues": [item.get("type") for item in semantic.errors],
                                })
                            continue

                    canonical_nodes = [engine.canonical_node(node) for node in pending_models]
                    chunk = engine.pack_next_chunk(chapter, canonical_nodes)
                    if not chunk:
                        break
                    self.broadcast_event("TRANSLATION_CHUNK_STARTED", {
                        "project_id": project_id,
                        "chapter_title": chapter.title,
                        "chunk_id": chunk.chunk_id,
                        "node_ids": [node.id for node in chunk.nodes],
                    })
                    for node in chunk.nodes:
                        db_node = db.query(NodeModel).filter(NodeModel.id == node.id).first()
                        if db_node:
                            db_node.status = "TRANSLATING"
                    db.commit()

                    batch_result = engine.translate_batch(chapter, chunk.nodes)
                    for result in batch_result.results:
                        source_node = next(node for node in chunk.nodes if node.id == result.node_id)
                        if result.passed:
                            db_node = db.query(NodeModel).filter(NodeModel.id == result.node_id).first()
                            semantic = SemanticAssuranceService.assure_and_commit(
                                engine, chapter, db_node, result.translated_text, signature,
                                config.model_name, "translation_worker",
                                latency_ms=result.telemetry.latency_ms,
                                previous_repairs=max(0, result.attempts - 1),
                            )
                            if not semantic.approved:
                                self.broadcast_event("TRANSLATION_NODE_NEEDS_REVIEW", {
                                    "project_id": project_id, "node_id": result.node_id,
                                    "attempt": result.attempts + semantic.repair_attempts,
                                    "issues": [item.get("type") for item in semantic.errors],
                                })
                        else:
                            db_node = db.query(NodeModel).filter(NodeModel.id == result.node_id).first()
                            if db_node:
                                db_node.status = "NEEDS_REVIEW"
                                db.commit()
                            self.broadcast_event("TRANSLATION_NODE_NEEDS_REVIEW", {
                                "project_id": project_id,
                                "node_id": result.node_id,
                                "attempt": result.attempts,
                                "issues": [issue["code"] for issue in result.quality.issues],
                            })
                    telemetry = batch_result.telemetry
                    if os.getenv("TRANSLATION_DEBUG", "").lower() in {"1", "true", "yes"}:
                        print("[TranslationContext] " + json.dumps(telemetry.__dict__, ensure_ascii=False))
                    stats = proj_repo.get_project_stats(project_id)
                    self.broadcast_event("TRANSLATION_CHUNK_COMPLETED", {
                        "project_id": project_id,
                        "chapter_title": chapter.title,
                        "chunk_id": chunk.chunk_id,
                        "context_mode": "CONTEXTUAL_STABLE",
                        "prompt_version": signature.prompt_version,
                        "context_limit": telemetry.context_limit,
                        "source_tokens": telemetry.source_tokens,
                        "total_tokens": telemetry.total_tokens,
                        **stats,
                    })
                    self.broadcast_event("TRANSLATION_PROGRESS", {
                        "project_id": project_id,
                        "chapter_title": chapter.title,
                        "chunk_id": chunk.chunk_id,
                        "context_mode": "CONTEXTUAL_STABLE",
                        **stats,
                    })

            stats = proj_repo.get_project_stats(project_id)
            if not stop_event.is_set():
                project.current_stage = "TRANSLATION_CONFIGURED"
            elif stats["translatable_nodes"] == 0 or stats["translated_nodes"] == stats["translatable_nodes"]:
                project.current_stage = "TRANSLATED"
            elif stats["terminal_nodes"] >= stats["translatable_nodes"]:
                project.current_stage = "TRANSLATED_WITH_REVIEW"
            elif systemic_failure:
                project.current_stage = "TRANSLATION_FAILED"
            else:
                project.current_stage = "TRANSLATION_FAILED"
            db.commit()
            self.broadcast_event("TRANSLATION_JOB_FINALIZED", {
                "project_id": project_id,
                "document_status": project.current_stage,
                "execution_status": "IDLE",
                **stats,
            })
        except Exception as exc:
            systemic_failure = True
            if project:
                project.current_stage = "TRANSLATION_FAILED"
                db.commit()
            self.broadcast_event("TRANSLATION_ERROR", {
                "project_id": project_id,
                "error_type": "TRANSLATION_JOB_ERROR",
                "message": str(exc),
            })
        finally:
            if project and project.current_stage == "TRANSLATING":
                project.current_stage = "TRANSLATION_FAILED" if systemic_failure else "TRANSLATION_CONFIGURED"
                db.commit()
            self.telemetry.setdefault(project_id, {})["execution_status"] = (
                "IDLE" if stop_event.is_set() else "STOPPED"
            )
            db.close()
            self.running_tasks.pop(project_id, None)
            self.pause_tasks.pop(project_id, None)
            self.active_threads.pop(project_id, None)


translation_worker = TranslationWorker()
