import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.db.engine import get_project_db
from app.db.models import ProjectModel, ChapterModel, NodeModel
from app.db.repository import ProjectRepository, StructureRepository, TranslationRepository
from app.models.canonical import DocumentNode, NodeType, NodeStatus, TranslationMode
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.mock_provider import MockProvider
from app.services.translation.chunker import SemanticChunker
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor





class TranslationWorker:
    def __init__(self):
        self.running_tasks: Dict[str, threading.Event] = {}  # project_id -> stop_event
        self.pause_tasks: Dict[str, threading.Event] = {}    # project_id -> pause_event
        self.active_threads: Dict[str, threading.Thread] = {}
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def register_event_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(callback)

    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        payload = {"type": event_type, "data": data, "timestamp": time.time()}
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
            self.broadcast_event("TRANSLATION_PAUSED", {"project_id": project_id})

    def resume_translation(self, project_id: str):
        if project_id in self.pause_tasks:
            self.pause_tasks[project_id].clear()
            self.broadcast_event("TRANSLATION_RESUMED", {"project_id": project_id})

    def stop_translation(self, project_id: str):
        if project_id in self.running_tasks:
            self.running_tasks[project_id].clear()
        if project_id in self.pause_tasks:
            self.pause_tasks[project_id].clear()
        self.broadcast_event("TRANSLATION_STOPPED", {"project_id": project_id})

    def retry_failed(self, project_id: str):
        db = get_project_db(project_id)
        try:
            failed_nodes = db.query(NodeModel).filter(
                NodeModel.project_id == project_id,
                NodeModel.status == "FAILED"
            ).all()
            for n in failed_nodes:
                n.status = "PENDING"
            db.commit()
        finally:
            db.close()
        self.start_translation(project_id)

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
        trans_repo = TranslationRepository(db)
        proj_repo = ProjectRepository(db)

        try:
            project = proj_repo.get_project(project_id)
            if not project:
                return

            # Check provider health before proceeding, wait briefly if momentarily initializing
            if not provider.health_check():
                recovered = self._wait_for_ollama_recovery(provider, stop_event, project_id, max_wait_sec=20)
                if not recovered:
                    err_msg = "Không thể kết nối đến Ollama (http://localhost:11434). Vui lòng khởi động Ollama trên Windows hoặc kiểm tra kết nối."
                    print(f"[Translation Worker] Health check failed: {err_msg}")
                    self.broadcast_event("TRANSLATION_ERROR", {
                        "project_id": project_id,
                        "error_type": "OLLAMA_OFFLINE",
                        "message": err_msg
                    })
                    project.current_stage = "TRANSLATION_CONFIGURED"
                    db.commit()
                    return

            project.current_stage = "TRANSLATING"
            db.commit()

            mode = TranslationMode(project.translation_mode) if project.translation_mode else TranslationMode.NATURAL
            doc_type = project.document_type or "GENERAL"
            style_guide = project.style_guide or {}
            locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)

            sys_prompt = PromptBuilder.build_system_prompt(
                document_type=doc_type,
                translation_mode=mode,
                style_guide=style_guide,
                custom_instructions=custom_instructions or project.custom_instructions
            )

            chapters = db.query(ChapterModel).filter(ChapterModel.project_id == project_id).order_by(ChapterModel.order_index).all()
            total_nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).count()

            for chapter in chapters:
                if not stop_event.is_set():
                    break

                # Get pending or failed nodes for this chapter
                nodes_to_translate = db.query(NodeModel).filter(
                    NodeModel.chapter_id == chapter.id,
                    NodeModel.status.in_(["PENDING", "FAILED"])
                ).order_by(NodeModel.order_index).all()

                if not nodes_to_translate:
                    continue

                # Generate chapter memory summary if not present
                if not chapter.summary and len(nodes_to_translate) > 0:
                    try:
                        sample_text = "\n".join([n.content for n in nodes_to_translate[:5]])
                        chapter.summary = provider.summarize_context(sample_text, model=model_name)
                        db.commit()
                    except Exception:
                        pass

                # Chunk nodes
                canonical_nodes = [
                    DocumentNode(
                        id=n.id,
                        type=NodeType(n.node_type.lower()) if n.node_type else NodeType.PARAGRAPH,
                        content=n.content,
                        status=NodeStatus(n.status)
                    ) for n in nodes_to_translate
                ]

                chunks = SemanticChunker.chunk_nodes(canonical_nodes)

                prev_context = ""

                for chunk in chunks:
                    if not stop_event.is_set():
                        break

                    # Check pause
                    while pause_event.is_set() and stop_event.is_set():
                        time.sleep(0.5)

                    chunk_to_call = []
                    # Check translation memory first (PRD Section 69)
                    for node in chunk.nodes:
                        tm_hit = TranslationMemoryService.lookup(db, node.content)
                        if tm_hit:
                            trans_repo.save_node_translation(
                                node_id=node.id,
                                project_id=project_id,
                                translated_text=tm_hit,
                                model_name=f"TM ({model_name})"
                            )
                        else:
                            chunk_to_call.append(node)
                            # Mark node as TRANSLATING in SQLite immediately
                            db_node = db.query(NodeModel).filter(NodeModel.id == node.id).first()
                            if db_node:
                                db_node.status = "TRANSLATING"
                    db.commit()

                    if chunk_to_call:
                        user_prompt = PromptBuilder.build_user_prompt(
                            nodes=chunk_to_call,
                            chapter_title=chapter.title,
                            chapter_summary=chapter.summary,
                            previous_context=prev_context,
                            glossary_terms=locked_glossary
                        )

                        retries = 3
                        success = False
                        while retries > 0 and not success and stop_event.is_set():
                            try:
                                translations = provider.translate(
                                    blocks=[{"id": n.id, "text": n.content} for n in chunk_to_call],
                                    system_prompt=sys_prompt,
                                    user_prompt=user_prompt,
                                    model=model_name
                                )
                                
                                # Map and commit each translation immediately (PRD Section 73 & 203)
                                trans_map = {t.get("node_id") or t.get("id"): t.get("text", "") for t in translations if isinstance(t, dict)}
                                
                                for cn in chunk_to_call:
                                    tr_text = trans_map.get(cn.id, "")
                                    if tr_text:
                                        tr_text = VietnamesePostProcessor.clean_vietnamese_text(tr_text)
                                        tr_text = VietnamesePostProcessor.enforce_locked_glossary(tr_text, cn.content, locked_glossary)
                                        
                                        # Strict Chinese character check: Reject and auto-retry if Chinese detected
                                        if VietnamesePostProcessor.contains_chinese(tr_text):
                                            print(f"[Translation Worker] Chinese characters detected in node {cn.id}! Auto-retrying pure Vietnamese...")
                                            try:
                                                tr_clean = provider.translate_single(
                                                    text=cn.content,
                                                    system_prompt=sys_prompt,
                                                    glossary_terms=locked_glossary,
                                                    model=model_name,
                                                    temperature=0.1
                                                )
                                                if tr_clean and not VietnamesePostProcessor.contains_chinese(tr_clean):
                                                    tr_text = VietnamesePostProcessor.clean_vietnamese_text(tr_clean)
                                                    tr_text = VietnamesePostProcessor.enforce_locked_glossary(tr_text, cn.content, locked_glossary)
                                                else:
                                                    tr_text = ""
                                            except Exception:
                                                tr_text = ""

                                    if tr_text and not VietnamesePostProcessor.contains_chinese(tr_text):
                                        trans_repo.save_node_translation(
                                            node_id=cn.id,
                                            project_id=project_id,
                                            translated_text=tr_text,
                                            model_name=model_name
                                        )
                                        # Save to Translation Memory
                                        TranslationMemoryService.store(
                                            db=db,
                                            source_text=cn.content,
                                            translated_text=tr_text,
                                            model_name=model_name
                                        )
                                    else:
                                        # Single-node fallback with strict pure Vietnamese requirement
                                        try:
                                            tr_single = provider.translate_single(
                                                text=cn.content,
                                                system_prompt=sys_prompt,
                                                glossary_terms=locked_glossary,
                                                model=model_name,
                                                temperature=0.1
                                            )
                                            if tr_single:
                                                tr_single = VietnamesePostProcessor.clean_vietnamese_text(tr_single)
                                                tr_single = VietnamesePostProcessor.enforce_locked_glossary(tr_single, cn.content, locked_glossary)
                                                if not VietnamesePostProcessor.contains_chinese(tr_single):
                                                    trans_repo.save_node_translation(
                                                        node_id=cn.id,
                                                        project_id=project_id,
                                                        translated_text=tr_single,
                                                        model_name=f"{model_name} (Single Fallback)"
                                                    )
                                                    TranslationMemoryService.store(
                                                        db=db,
                                                        source_text=cn.content,
                                                        translated_text=tr_single,
                                                        model_name=model_name
                                                    )
                                                else:
                                                    db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                                    if db_n:
                                                        db_n.status = "FAILED"
                                                        db.commit()
                                            else:
                                                db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                                if db_n:
                                                    db_n.status = "FAILED"
                                                    db.commit()
                                        except Exception:
                                            db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                            if db_n:
                                                db_n.status = "FAILED"
                                                db.commit()

                                success = True
                            except Exception as e:
                                # Check if Ollama went offline during translation
                                if not provider.health_check():
                                    print(f"[Translation Worker] Connection lost to provider, waiting for recovery...")
                                    if not self._wait_for_ollama_recovery(provider, stop_event, project_id):
                                        break
                                
                                retries -= 1
                                if retries == 0:
                                    print(f"[Translation Worker] Batch failed for chunk {chunk.chunk_id}: {e}. Trying single-node fallback for all chunk nodes...")
                                    # Fallback: Translate each node in this chunk individually
                                    for cn in chunk_to_call:
                                        if not stop_event.is_set():
                                            break
                                        try:
                                            tr_single = provider.translate_single(
                                                text=cn.content,
                                                system_prompt=sys_prompt,
                                                glossary_terms=locked_glossary,
                                                model=model_name,
                                                temperature=0.1
                                            )
                                            if tr_single:
                                                tr_single = VietnamesePostProcessor.clean_vietnamese_text(tr_single)
                                                tr_single = VietnamesePostProcessor.enforce_locked_glossary(tr_single, cn.content, locked_glossary)
                                                if not VietnamesePostProcessor.contains_chinese(tr_single):
                                                    trans_repo.save_node_translation(
                                                        node_id=cn.id,
                                                        project_id=project_id,
                                                        translated_text=tr_single,
                                                        model_name=f"{model_name} (Single Fallback)"
                                                    )
                                                    TranslationMemoryService.store(
                                                        db=db,
                                                        source_text=cn.content,
                                                        translated_text=tr_single,
                                                        model_name=model_name
                                                    )
                                                else:
                                                    db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                                    if db_n:
                                                        db_n.status = "FAILED"
                                                        db.commit()
                                            else:
                                                db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                                if db_n:
                                                    db_n.status = "FAILED"
                                                    db.commit()
                                        except Exception as e_ind:
                                            print(f"[Translation Worker] Single fallback failed for node {cn.id}: {e_ind}")
                                            db_n = db.query(NodeModel).filter(NodeModel.id == cn.id).first()
                                            if db_n:
                                                db_n.status = "FAILED"
                                                db.commit()
                                else:
                                    time.sleep(2.0)

                    # Update context
                    prev_context = " ".join([n.content for n in chunk.nodes])[-300:]

                    # Broadcast progress
                    stats = proj_repo.get_project_stats(project_id)
                    self.broadcast_event("TRANSLATION_PROGRESS", {
                        "project_id": project_id,
                        "chapter_title": chapter.title,
                        **stats
                    })

            # Auto-Healing Pass: Automatically repair any FAILED, PENDING, or Chinese-contaminated nodes
            if stop_event.is_set():
                max_healing_passes = 2
                for healing_pass in range(max_healing_passes):
                    if not stop_event.is_set():
                        break

                    all_project_nodes = db.query(NodeModel).filter(
                        NodeModel.project_id == project_id
                    ).order_by(NodeModel.order_index).all()

                    remaining_failed = []
                    for n in all_project_nodes:
                        if n.status in ["FAILED", "PENDING"]:
                            remaining_failed.append(n)
                        elif n.status == "TRANSLATED" and n.translated_content:
                            if VietnamesePostProcessor.contains_chinese(n.translated_content):
                                print(f"[Auto-Healing] Detected Chinese in translated node {n.id}, queuing for pure Vietnamese repair...")
                                remaining_failed.append(n)

                    if not remaining_failed:
                        break

                    print(f"[Translation Worker] Starting Auto-Healing Pass {healing_pass + 1} for {len(remaining_failed)} nodes...")
                    self.broadcast_event("TRANSLATION_AUTO_HEALING", {
                        "project_id": project_id,
                        "healing_pass": healing_pass + 1,
                        "remaining_nodes": len(remaining_failed),
                        "message": f"Đang tự động dịch lại {len(remaining_failed)} đoạn văn bản (vá lỗi / sửa chữ Hán)..."
                    })

                    for fn in remaining_failed:
                        if not stop_event.is_set():
                            break
                        while pause_event.is_set() and stop_event.is_set():
                            time.sleep(0.5)

                        # Check connection health
                        if not provider.health_check():
                            if not self._wait_for_ollama_recovery(provider, stop_event, project_id):
                                break

                        try:
                            tr_single = provider.translate_single(
                                text=fn.content,
                                system_prompt=sys_prompt,
                                glossary_terms=locked_glossary,
                                model=model_name,
                                temperature=0.1
                            )
                            if tr_single:
                                tr_single = VietnamesePostProcessor.clean_vietnamese_text(tr_single)
                                tr_single = VietnamesePostProcessor.enforce_locked_glossary(tr_single, fn.content, locked_glossary)
                                if not VietnamesePostProcessor.contains_chinese(tr_single):
                                    trans_repo.save_node_translation(
                                        node_id=fn.id,
                                        project_id=project_id,
                                        translated_text=tr_single,
                                        model_name=f"{model_name} (Auto-Healed)"
                                    )
                                    TranslationMemoryService.store(
                                        db=db,
                                        source_text=fn.content,
                                        translated_text=tr_single,
                                        model_name=model_name
                                    )
                                    # Broadcast progress after each healed node
                                    stats = proj_repo.get_project_stats(project_id)
                                    self.broadcast_event("TRANSLATION_PROGRESS", {
                                        "project_id": project_id,
                                        "chapter_title": "Tự động vá lỗi (Auto-Healing)",
                                        **stats
                                    })
                        except Exception as e_heal:
                            print(f"[Auto-Healing] Failed to heal node {fn.id}: {e_heal}")
                            time.sleep(1.0)



            # Check if all completed
            stats = proj_repo.get_project_stats(project_id)
            if stats["translated_nodes"] >= total_nodes and total_nodes > 0:
                project.current_stage = "TRANSLATED"
                db.commit()
                self.broadcast_event("TRANSLATION_COMPLETED", {
                    "project_id": project_id,
                    "total_nodes": total_nodes,
                    "translated_nodes": stats["translated_nodes"]
                })

        finally:
            db.close()
            if project_id in self.running_tasks:
                del self.running_tasks[project_id]
            if project_id in self.pause_tasks:
                del self.pause_tasks[project_id]


translation_worker = TranslationWorker()
