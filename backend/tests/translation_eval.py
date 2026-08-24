"""Chạy benchmark full engine: python -m tests.translation_eval [dataset.jsonl]."""

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.context_assembler import estimate_tokens
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_config import TranslationConfig


def build_phase1_baseline_prompt(node: DocumentNode, glossary: Dict[str, str]) -> str:
    return (
        "You are an English-to-Vietnamese publishing translator. Translate all meaning naturally. "
        "Preserve names, URLs, numbers, units and locked terms. Return only the translation.\n"
        f"Locked glossary: {json.dumps(glossary, ensure_ascii=False)}\nSource: {node.content}"
    )


def _provider(model_name: str):
    return MockProvider() if model_name.lower().startswith("mock") else OllamaProvider(default_model=model_name)


def _metric_bucket() -> Dict[str, Any]:
    return {
        "latency_ms": 0.0, "passed": 0, "request_tokens": 0,
        "provider_calls": 0, "repairs": 0, "context_overflow_count": 0,
        "source_tokens": 0, "context_tokens": 0,
        "wrong_target_language": 0, "missing_number": 0, "wrong_number": 0,
        "missing_reference": 0, "glossary_failure": 0, "mapping_failure": 0,
    }


def _record_issues(bucket: Dict[str, Any], issues: List[Dict[str, str]]) -> List[str]:
    codes = [issue.get("code", "") for issue in issues]
    bucket["wrong_target_language"] += int("WRONG_TARGET_LANGUAGE" in codes or "FOREIGN_SCRIPT_CONTAMINATION" in codes)
    bucket["missing_number"] += int("NUMBER_MISMATCH" in codes)
    bucket["wrong_number"] += int("NUMBER_MISMATCH" in codes)
    bucket["missing_reference"] += int("REFERENCE_MISMATCH" in codes)
    bucket["glossary_failure"] += int("GLOSSARY_MISMATCH" in codes)
    bucket["mapping_failure"] += int("NODE_MAPPING_ERROR" in codes or "MISSING_NODE_IDS" in codes)
    return codes


def run(dataset_path: Path, model_name: str = "mock-qwen2.5:7b", human_csv: Path | None = None) -> dict:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = _provider(model_name)
    if not provider.health_check():
        raise RuntimeError(f"Mô hình {model_name} chưa sẵn sàng qua provider local.")
    metrics: Dict[str, Any] = {
        "model": model_name, "samples": len(rows), "baseline": _metric_bucket(),
        "phase2_1": _metric_bucket(), "human_naturalness": None, "budget_example": None,
    }
    human_rows: List[Dict[str, Any]] = []
    original_projects_dir = settings.PROJECTS_DIR
    with tempfile.TemporaryDirectory(prefix="translation-eval-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        db_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db_engine)
        db = sessionmaker(bind=db_engine)()
        try:
            project = ProjectModel(
                id="phase2_1_eval", title="Phase 2.1 Eval", selected_model=model_name,
                document_type=(rows[0].get("domain", "GENERAL") if rows else "GENERAL"),
                translation_mode="NATURAL",
            )
            chapter = ChapterModel(id="eval_chapter", project_id=project.id, title="Evaluation", order_index=0)
            db.add(project)
            db.add(chapter)
            db.commit()
            db_nodes: List[NodeModel] = []
            for index, row in enumerate(rows):
                db_node = NodeModel(
                    id=f"eval_{index}", project_id=project.id, chapter_id=chapter.id,
                    node_type=row.get("node_type", "paragraph"), content=row["source"],
                    status="PENDING", order_index=index,
                )
                db.add(db_node)
                db_nodes.append(db_node)
            db.commit()
            config = TranslationConfig.from_project(project, model_override=model_name)
            all_glossary: Dict[str, str] = {}
            for row in rows:
                all_glossary.update(row.get("glossary", {}))
            contextual = ContextualTranslationEngine(db, project, provider, config, all_glossary)

            for index, (row, db_node) in enumerate(zip(rows, db_nodes)):
                node = DocumentNode(
                    id=db_node.id, type=NodeType(row.get("node_type", "paragraph")),
                    content=row["source"], status=NodeStatus.PENDING, order_index=index,
                )
                glossary = row.get("glossary", {})
                baseline_prompt = build_phase1_baseline_prompt(node, glossary)
                started = time.perf_counter()
                baseline = provider.translate_single(node.content, baseline_prompt, glossary, model=model_name)
                baseline_gate = TranslationQualityGate().validate(node.content, baseline, glossary)
                metrics["baseline"]["latency_ms"] += (time.perf_counter() - started) * 1000
                metrics["baseline"]["request_tokens"] += estimate_tokens(baseline_prompt)
                metrics["baseline"]["provider_calls"] += 1
                metrics["baseline"]["passed"] += int(baseline_gate.passed)
                baseline_flags = _record_issues(metrics["baseline"], baseline_gate.issues)

                started = time.perf_counter()
                result = contextual.translate_node(chapter, contextual.canonical_node(db_node))
                telemetry = result.telemetry
                if metrics["budget_example"] is None:
                    metrics["budget_example"] = {
                        "context_limit": telemetry.context_limit,
                        "system_tokens": telemetry.system_tokens,
                        "document_tokens": telemetry.document_tokens,
                        "chapter_tokens": telemetry.chapter_tokens,
                        "rolling_tokens": telemetry.rolling_tokens,
                        "glossary_tokens": telemetry.glossary_tokens,
                        "few_shot_tokens": telemetry.few_shot_tokens,
                        "source_tokens": telemetry.source_tokens,
                        "reserved_output": telemetry.reserved_output,
                        "safety_margin": telemetry.safety_margin,
                        "actual_total_tokens": telemetry.total_tokens,
                    }
                metrics["phase2_1"]["latency_ms"] += (time.perf_counter() - started) * 1000
                metrics["phase2_1"]["request_tokens"] += telemetry.total_tokens
                metrics["phase2_1"]["provider_calls"] += telemetry.provider_calls
                metrics["phase2_1"]["repairs"] += int(telemetry.retries > 0)
                metrics["phase2_1"]["context_overflow_count"] += int(telemetry.total_tokens > telemetry.context_limit)
                metrics["phase2_1"]["passed"] += int(result.passed)
                metrics["phase2_1"]["source_tokens"] += telemetry.source_tokens
                metrics["phase2_1"]["context_tokens"] += (
                    telemetry.document_tokens + telemetry.chapter_tokens + telemetry.rolling_tokens
                    + telemetry.glossary_tokens + telemetry.few_shot_tokens
                )
                phase_flags = _record_issues(metrics["phase2_1"], result.quality.issues)
                if result.passed:
                    db_node.translated_content = result.translated_text
                    db_node.status = "TRANSLATED"
                    db.commit()
                human_rows.append({
                    "sample_id": node.id, "domain": row.get("domain", "GENERAL"), "source": node.content,
                    "baseline_translation": baseline, "phase2_translation": result.translated_text,
                    "correctness_flags": json.dumps({"baseline": baseline_flags, "phase2_1": phase_flags}, ensure_ascii=False),
                    "human_naturalness_baseline": "", "human_naturalness_phase2": "",
                    "preferred_output": "", "notes": "",
                })
        finally:
            db.close()
            settings.PROJECTS_DIR = original_projects_dir

    for group_name in ("baseline", "phase2_1"):
        group = metrics[group_name]
        count = max(1, len(rows))
        group["validation_pass_rate"] = group["passed"] / count
        group["avg_latency_ms"] = group["latency_ms"] / count
        group["avg_request_tokens"] = group["request_tokens"] / count
        group["provider_calls_per_node"] = group["provider_calls"] / count
        group["repair_percentage"] = group["repairs"] / count * 100.0
        group["needs_review_rate"] = 1.0 - group["validation_pass_rate"]
        group["retry_rate"] = group["repairs"] / count
        group["avg_source_tokens_per_chunk"] = group["source_tokens"] / count
        group["avg_context_tokens_per_chunk"] = group["context_tokens"] / count
    rated = [row for row in human_rows if row["human_naturalness_baseline"] != "" and row["human_naturalness_phase2"] != ""]
    if rated:
        baseline_avg = sum(float(row["human_naturalness_baseline"]) for row in rated) / len(rated)
        phase_avg = sum(float(row["human_naturalness_phase2"]) for row in rated) / len(rated)
        metrics["human_naturalness"] = {
            "baseline": baseline_avg, "phase2_1": phase_avg,
            "improvement": phase_avg - baseline_avg, "rated_samples": len(rated),
        }
    if human_csv:
        human_csv.parent.mkdir(parents=True, exist_ok=True)
        with human_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = list(human_rows[0].keys()) if human_rows else ["sample_id"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(human_rows)
    return metrics


def run_chapter_benchmark(dataset_path: Path, model_name: str = "mock-qwen2.5:7b") -> Dict[str, Any]:
    """Chạy ba mini-book để kiểm tra rolling context, resume và tính nhất quán."""
    specs = json.loads(dataset_path.read_text(encoding="utf-8"))
    provider = _provider(model_name)
    original_projects_dir = settings.PROJECTS_DIR
    result: Dict[str, Any] = {
        "mini_books": len(specs), "nodes": 0, "rolling_context_requests": 0,
        "resume_reconstruction_passed": False, "context_overflow_count": 0,
        "entity_consistency_failures": 0, "term_consistency_failures": 0,
    }
    with tempfile.TemporaryDirectory(prefix="translation-chapter-eval-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        db_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db_engine)
        db = sessionmaker(bind=db_engine)()
        try:
            project = ProjectModel(
                id="chapter_eval", title="Chapter Eval", selected_model=model_name,
                document_type="GENERAL", translation_mode="NATURAL",
            )
            db.add(project)
            db.commit()
            chapters: List[ChapterModel] = []
            glossary: Dict[str, str] = {}
            for chapter_index, spec in enumerate(specs):
                chapter = ChapterModel(
                    id=f"chapter_{chapter_index}", project_id=project.id,
                    title=spec["chapter_title"], order_index=chapter_index,
                )
                db.add(chapter)
                chapters.append(chapter)
                glossary.update(spec.get("locked_glossary", {}))
                templates = spec["source_templates"]
                for node_index in range(int(spec["node_count"])):
                    db.add(NodeModel(
                        id=f"c{chapter_index}_n{node_index}", project_id=project.id,
                        chapter_id=chapter.id, node_type="paragraph",
                        content=templates[node_index % len(templates)].format(n=node_index + 1),
                        status="PENDING", order_index=node_index,
                    ))
                    result["nodes"] += 1
            db.commit()
            config = TranslationConfig.from_project(project, model_override=model_name)
            contextual = ContextualTranslationEngine(db, project, provider, config, glossary)
            processed = 0
            for chapter_index, (spec, chapter) in enumerate(zip(specs, chapters)):
                nodes = db.query(NodeModel).filter(NodeModel.chapter_id == chapter.id).order_by(NodeModel.order_index).all()
                for db_node in nodes:
                    if processed == 10:
                        # Mô phỏng restart: engine mới phải dựng rolling context từ SQLite.
                        contextual = ContextualTranslationEngine(db, project, provider, config, glossary)
                    context = contextual.assemble_context(chapter, [contextual.canonical_node(db_node)], "single", neighbors=True)
                    if context.previous_context:
                        result["rolling_context_requests"] += 1
                    if processed == 10:
                        result["resume_reconstruction_passed"] = bool(context.previous_context)
                    translated = contextual.translate_node(chapter, contextual.canonical_node(db_node))
                    result["context_overflow_count"] += int(
                        translated.telemetry.total_tokens > translated.telemetry.context_limit
                    )
                    if translated.passed:
                        db_node.translated_content = translated.translated_text
                        db_node.status = "TRANSLATED"
                        db.commit()
                    source_lower = db_node.content.lower()
                    target_lower = translated.translated_text.lower()
                    for entity in spec.get("entities", []):
                        if entity.lower() in source_lower and entity.lower() not in target_lower:
                            result["entity_consistency_failures"] += 1
                    if any(issue["code"] == "GLOSSARY_MISMATCH" for issue in translated.quality.issues):
                        result["term_consistency_failures"] += 1
                    processed += 1
        finally:
            db.close()
            settings.PROJECTS_DIR = original_projects_dir
    return result


if __name__ == "__main__":
    default = Path(__file__).parent / "fixtures" / "translation_eval" / "phase2_eval.jsonl"
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=default)
    parser.add_argument("--model", default="mock-qwen2.5:7b")
    parser.add_argument("--human-csv", type=Path)
    parser.add_argument(
        "--chapter-dataset", type=Path,
        default=Path(__file__).parent / "fixtures" / "translation_eval" / "phase2_1_chapters.json",
    )
    args = parser.parse_args()
    report = run(args.dataset, args.model, args.human_csv)
    report["chapter_benchmark"] = run_chapter_benchmark(args.chapter_dataset, args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
