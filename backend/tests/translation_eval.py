"""Chạy benchmark full engine: python -m tests.translation_eval [dataset.jsonl]."""

import argparse
import csv
import json
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
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
from app.services.translation.prompt_profiles import STYLE_PACK_VERSION
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import PROMPT_VERSION


DATASET_VERSION = "phase2-1-2-eval-v1"


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
        "wrong_target_language": 0, "number_mismatch": 0,
        "number_addition": 0, "reference_mismatch": 0, "reference_addition": 0,
        "url_mismatch": 0, "url_addition": 0, "glossary_failure": 0, "negation_loss": 0,
        "qa_error": 0,
        "mapping_failure": 0,
    }


def _record_issues(bucket: Dict[str, Any], issues: List[Dict[str, str]]) -> List[str]:
    codes = [issue.get("code", "") for issue in issues]
    bucket["wrong_target_language"] += int("WRONG_TARGET_LANGUAGE" in codes or "FOREIGN_SCRIPT_CONTAMINATION" in codes)
    bucket["number_mismatch"] += int("NUMBER_MISMATCH" in codes)
    bucket["number_addition"] += int("NUMBER_ADDITION" in codes)
    bucket["reference_mismatch"] += int("REFERENCE_MISMATCH" in codes)
    bucket["reference_addition"] += int("REFERENCE_ADDITION" in codes)
    bucket["url_mismatch"] += int("URL_MISMATCH" in codes)
    bucket["url_addition"] += int("URL_ADDITION" in codes)
    bucket["glossary_failure"] += int("GLOSSARY_MISMATCH" in codes)
    bucket["negation_loss"] += int("NEGATION_LOSS" in codes)
    bucket["qa_error"] += int("QA_ERROR" in codes)
    bucket["mapping_failure"] += int("NODE_MAPPING_ERROR" in codes or "MISSING_NODE_IDS" in codes)
    return codes


def _merge_bucket(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] += value


def _finalize_bucket(bucket: Dict[str, Any], count: int) -> Dict[str, Any]:
    count = max(1, count)
    bucket["validation_pass_rate"] = bucket["passed"] / count
    bucket["avg_latency_ms"] = bucket["latency_ms"] / count
    bucket["avg_request_tokens"] = bucket["request_tokens"] / count
    bucket["provider_calls_per_node"] = bucket["provider_calls"] / count
    bucket["needs_review_rate"] = 1.0 - bucket["validation_pass_rate"]
    bucket["retry_rate"] = bucket["repairs"] / count
    bucket["avg_source_tokens"] = bucket["source_tokens"] / count
    bucket["avg_context_tokens"] = bucket["context_tokens"] / count
    return bucket


def _metadata(model_name: str, domain: str, effective_context_window: int) -> Dict[str, Any]:
    return {
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "style_pack_version": STYLE_PACK_VERSION,
        "dataset_version": DATASET_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "effective_context_window": effective_context_window,
        "quality_profile": "BALANCED",
        "domain": domain,
    }


def run(dataset_path: Path, model_name: str = "mock-qwen2.5:7b", human_csv: Path | None = None) -> dict:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = _provider(model_name)
    if not provider.health_check():
        raise RuntimeError(f"Mô hình {model_name} chưa sẵn sàng qua provider local.")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("domain") or "GENERAL").upper()].append(row)
    metrics: Dict[str, Any] = {"domains": {}, "aggregate": {}}
    human_rows: List[Dict[str, Any]] = []
    original_projects_dir = settings.PROJECTS_DIR
    with tempfile.TemporaryDirectory(prefix="translation-eval-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        db_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db_engine)
        db = sessionmaker(bind=db_engine)()
        try:
            aggregate_baseline = _metric_bucket()
            aggregate_phase = _metric_bucket()
            capabilities = provider.get_model_capabilities(model_name)
            effective_context = min(capabilities.context_window, capabilities.recommended_context_window)
            for domain, domain_rows in sorted(grouped.items()):
                project = ProjectModel(
                    id=f"phase2_1_1_eval_{domain.lower()}", title=f"{domain} Eval",
                    selected_model=model_name, document_type=domain, translation_mode="NATURAL",
                )
                db.add(project)
                db.commit()
                domain_nodes: List[tuple[Dict[str, Any], ChapterModel, NodeModel]] = []
                domain_glossary: Dict[str, str] = {}
                for index, row in enumerate(domain_rows):
                    # Mỗi sample độc lập có chapter riêng để rolling context luôn rỗng.
                    chapter = ChapterModel(
                        id=f"eval_{domain.lower()}_chapter_{index}", project_id=project.id,
                        title=f"Independent sample {index + 1}", order_index=index,
                    )
                    db_node = NodeModel(
                        id=f"eval_{domain.lower()}_{index}", project_id=project.id, chapter_id=chapter.id,
                        node_type=row.get("node_type", "paragraph"), content=row["source"],
                        status="PENDING", order_index=0,
                    )
                    db.add_all([chapter, db_node])
                    domain_nodes.append((row, chapter, db_node))
                    domain_glossary.update(row.get("glossary", {}))
                db.commit()
                config = TranslationConfig.from_project(project, model_override=model_name)
                contextual = ContextualTranslationEngine(db, project, provider, config, domain_glossary)
                domain_baseline = _metric_bucket()
                domain_phase = _metric_bucket()
                for row, chapter, db_node in domain_nodes:
                    node = contextual.canonical_node(db_node)
                    glossary = row.get("glossary", {})
                    baseline_prompt = build_phase1_baseline_prompt(node, glossary)
                    started = time.perf_counter()
                    baseline = provider.translate_single(node.content, baseline_prompt, glossary, model=model_name)
                    baseline_gate = TranslationQualityGate().validate(node.content, baseline, glossary)
                    domain_baseline["latency_ms"] += (time.perf_counter() - started) * 1000
                    domain_baseline["request_tokens"] += estimate_tokens(baseline_prompt)
                    domain_baseline["provider_calls"] += 1
                    domain_baseline["passed"] += int(baseline_gate.passed)
                    _record_issues(domain_baseline, baseline_gate.issues)

                    started = time.perf_counter()
                    context = contextual.assemble_context(chapter, [node], "single", neighbors=True)
                    if context.previous_context:
                        raise AssertionError(f"Independent sample {node.id} nhận rolling context không liên quan")
                    translated = contextual.translate_node(chapter, node)
                    telemetry = translated.telemetry
                    domain_phase["latency_ms"] += (time.perf_counter() - started) * 1000
                    domain_phase["request_tokens"] += telemetry.total_tokens
                    domain_phase["provider_calls"] += telemetry.provider_calls
                    domain_phase["repairs"] += int(telemetry.retries > 0)
                    domain_phase["context_overflow_count"] += int(
                        telemetry.total_tokens > telemetry.context_limit or telemetry.source_tokens > capabilities.recommended_source_tokens
                    )
                    domain_phase["passed"] += int(translated.passed)
                    domain_phase["source_tokens"] += telemetry.source_tokens
                    domain_phase["context_tokens"] += (
                        telemetry.document_tokens + telemetry.chapter_tokens + telemetry.rolling_tokens
                        + telemetry.glossary_tokens + telemetry.few_shot_tokens
                    )
                    _record_issues(domain_phase, translated.quality.issues)
                    human_rows.append({
                        "sample_id": node.id, "domain": domain, "source": node.content,
                        "baseline_translation": baseline,
                        "phase2_1_1_translation": translated.translated_text,
                        "baseline_score": "", "new_score": "",
                        "preferred_output": "", "notes": "",
                    })
                _merge_bucket(aggregate_baseline, domain_baseline)
                _merge_bucket(aggregate_phase, domain_phase)
                metrics["domains"][domain] = {
                    "metadata": _metadata(model_name, domain, effective_context),
                    "project_id": project.id,
                    "samples": len(domain_rows),
                    "baseline": _finalize_bucket(domain_baseline, len(domain_rows)),
                    "phase2_1_1": _finalize_bucket(domain_phase, len(domain_rows)),
                }
            metrics["aggregate"] = {
                "metadata": _metadata(model_name, "ALL", effective_context),
                "samples": len(rows),
                "baseline": _finalize_bucket(aggregate_baseline, len(rows)),
                "phase2_1_1": _finalize_bucket(aggregate_phase, len(rows)),
            }
            critical_keys = (
                "wrong_target_language", "number_mismatch", "number_addition",
                "reference_mismatch", "reference_addition", "url_mismatch", "url_addition",
                "glossary_failure", "negation_loss", "qa_error", "mapping_failure",
            )
            baseline_critical = sum(metrics["aggregate"]["baseline"][key] for key in critical_keys)
            phase_critical = sum(metrics["aggregate"]["phase2_1_1"][key] for key in critical_keys)
            metrics["aggregate"]["critical_correctness_regression"] = max(0, phase_critical - baseline_critical)
        finally:
            db.close()
            settings.PROJECTS_DIR = original_projects_dir

    if human_csv:
        human_csv.parent.mkdir(parents=True, exist_ok=True)
        with human_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = list(human_rows[0].keys()) if human_rows else ["sample_id"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(human_rows)
    return metrics


def run_chapter_benchmark(dataset_path: Path, model_name: str = "mock-qwen2.5:7b") -> Dict[str, Any]:
    """Chạy mỗi mini-book trong project/domain/context riêng."""
    specs = json.loads(dataset_path.read_text(encoding="utf-8"))
    provider = _provider(model_name)
    original_projects_dir = settings.PROJECTS_DIR
    result: Dict[str, Any] = {
        "mini_books": len(specs), "nodes": 0, "rolling_context_requests": 0,
        "resume_reconstruction_passed": True, "context_overflow_count": 0,
        "entity_consistency_failures": 0, "term_consistency_failures": 0,
        "books": {},
    }
    with tempfile.TemporaryDirectory(prefix="translation-chapter-eval-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        db_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db_engine)
        db = sessionmaker(bind=db_engine)()
        try:
            for chapter_index, spec in enumerate(specs):
                book_id = str(spec["book_id"])
                domain = str(spec["domain"]).upper()
                project = ProjectModel(
                    id=f"chapter_eval_{book_id}", title=f"{book_id} Eval", selected_model=model_name,
                    document_type=domain, translation_mode="NATURAL",
                )
                chapter = ChapterModel(
                    id=f"chapter_{book_id}", project_id=project.id,
                    title=spec["chapter_title"], order_index=0,
                )
                db.add_all([project, chapter])
                db.commit()
                glossary = dict(spec.get("locked_glossary", {}))
                templates = spec["source_templates"]
                for node_index in range(int(spec["node_count"])):
                    db.add(NodeModel(
                        id=f"{book_id}_n{node_index}", project_id=project.id,
                        chapter_id=chapter.id, node_type="paragraph",
                        content=templates[node_index % len(templates)].format(n=node_index + 1),
                        status="PENDING", order_index=node_index,
                    ))
                    result["nodes"] += 1
                db.commit()
                config = TranslationConfig.from_project(project, model_override=model_name)
                contextual = ContextualTranslationEngine(db, project, provider, config, glossary)
                processed = 0
                book_result = {
                    "project_id": project.id,
                    "domain": domain,
                    "nodes": int(spec["node_count"]),
                    "rolling_context_requests": 0,
                    "resume_reconstruction_passed": False,
                    "context_overflow_count": 0,
                    "entity_consistency_failures": 0,
                    "term_consistency_failures": 0,
                }
                nodes = db.query(NodeModel).filter(NodeModel.chapter_id == chapter.id).order_by(NodeModel.order_index).all()
                for db_node in nodes:
                    if processed == 10:
                        # Mô phỏng restart trong đúng project; context phải dựng lại từ SQLite.
                        contextual = ContextualTranslationEngine(db, project, provider, config, glossary)
                    context = contextual.assemble_context(chapter, [contextual.canonical_node(db_node)], "single", neighbors=True)
                    if context.previous_context:
                        result["rolling_context_requests"] += 1
                        book_result["rolling_context_requests"] += 1
                    if processed == 10:
                        book_result["resume_reconstruction_passed"] = bool(context.previous_context)
                    translated = contextual.translate_node(chapter, contextual.canonical_node(db_node))
                    overflow = int(
                        translated.telemetry.total_tokens > translated.telemetry.context_limit
                    )
                    result["context_overflow_count"] += overflow
                    book_result["context_overflow_count"] += overflow
                    if translated.passed:
                        db_node.translated_content = translated.translated_text
                        db_node.status = "TRANSLATED"
                        db.commit()
                    source_lower = db_node.content.lower()
                    target_lower = translated.translated_text.lower()
                    for entity in spec.get("entities", []):
                        if entity.lower() in source_lower and entity.lower() not in target_lower:
                            result["entity_consistency_failures"] += 1
                            book_result["entity_consistency_failures"] += 1
                    if any(issue["code"] == "GLOSSARY_MISMATCH" for issue in translated.quality.issues):
                        result["term_consistency_failures"] += 1
                        book_result["term_consistency_failures"] += 1
                    processed += 1
                result["resume_reconstruction_passed"] = (
                    result["resume_reconstruction_passed"] and book_result["resume_reconstruction_passed"]
                )
                result["books"][book_id] = book_result
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
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--chapter-dataset", type=Path,
        default=Path(__file__).parent / "fixtures" / "translation_eval" / "phase2_1_chapters.json",
    )
    args = parser.parse_args()
    report = run(args.dataset, args.model, args.human_csv)
    report["chapter_benchmark"] = run_chapter_benchmark(args.chapter_dataset, args.model)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
