"""Chạy benchmark P1/P2 offline: python -m tests.p1_p2_eval --model mock."""

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.services.qa.vietnamese_naturalness_critic import VietnameseNaturalnessCritic
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_eval_dataset import (
    EVAL_DATASET_VERSION,
    HUMAN_EVAL_COLUMNS,
    load_official_dataset,
    validate_official_dataset,
)
from app.services.translation.translation_eval_metrics import compare_regression, summarize_evaluations
from app.services.translation.translation_config import TranslationConfig


CRITICAL_CODES = {
    "WRONG_TARGET_LANGUAGE", "FOREIGN_SCRIPT_CONTAMINATION", "NUMBER_MISMATCH", "NUMBER_ADDITION",
    "REFERENCE_MISMATCH", "REFERENCE_ADDITION", "URL_MISMATCH", "URL_ADDITION",
    "GLOSSARY_MISMATCH", "NEGATION_LOSS", "EMPTY_TRANSLATION",
}


def _provider(model_name: str):
    return MockProvider() if model_name.lower().startswith("mock") else OllamaProvider(default_model=model_name)


def _candidate_is_a(sample_id: str) -> bool:
    """Gán A/B ổn định nhưng không xuất mapping cho evaluator."""
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return digest[0] % 2 == 0


def _quality_flags(issues: Iterable[Mapping[str, Any]]) -> Dict[str, bool]:
    codes = {str(issue.get("code", "")).upper() for issue in issues}
    return {
        "semantic_critical_error": bool(codes.intersection(CRITICAL_CODES)),
        "hard_glossary_error": "GLOSSARY_MISMATCH" in codes,
        "deterministic_error": bool(any(issue.get("severity") == "ERROR" for issue in issues)),
    }


def _naturalness(provider: Any, source: str, target: str, domain: str) -> Dict[str, Any]:
    started = time.perf_counter()
    result = VietnameseNaturalnessCritic.review(
        provider=provider,
        source_text=source,
        translated_text=target,
        document_type=domain,
        register="",
        sentence_style="MODERATE",
        previous_context=[],
        glossary_terms={},
        entity_context={},
        model="mock" if isinstance(provider, MockProvider) else getattr(provider, "default_model", ""),
    )
    return {
        "score": result.score,
        "naturalness": round((result.score or 0.0) * 5, 3),
        "naturalness_critic_calls": result.critic_calls,
        "naturalness_latency_ms": result.critic_latency_ms or (time.perf_counter() - started) * 1000,
        "naturalness_status": result.status,
    }


def _baseline_row(provider: Any, row: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    source = str(row.get("source", ""))
    started = time.perf_counter()
    target = provider.translate_single(source, "Translate completely and return Vietnamese plain text.", row.get("glossary", {}), model=model_name)
    gate = TranslationQualityGate().validate(source, target, row.get("glossary", {}))
    natural = _naturalness(provider, source, target, str(row.get("domain", "GENERAL")))
    flags = _quality_flags(gate.issues)
    return {
        "sample_id": row["sample_id"], "domain": row["domain"], "source": source,
        "baseline_translation": target, "candidate_translation": "",
        "preferred_output": row.get("preferred_output", ""), **flags,
        "provider_calls": 1, "latency_ms": (time.perf_counter() - started) * 1000,
        "provider_calls_per_node": 1,
        "request_tokens": len(source.split()), "few_shot_count": 0, "style_memory_hits": 0,
        "naturalness": natural["naturalness"], "naturalness_critic_calls": natural["naturalness_critic_calls"],
        "editorial_rewrite_count": 0, "semantic_repairs": 0,
    }


def _candidate_row(engine: ContextualTranslationEngine, chapter: ChapterModel, node: NodeModel, row: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    result = engine.translate_node(chapter, engine.canonical_node(node))
    natural = _naturalness(engine.provider, node.content, result.translated_text, str(row.get("domain", "GENERAL")))
    flags = _quality_flags(result.quality.issues)
    telemetry = result.telemetry
    return {
        "sample_id": row["sample_id"], "domain": row["domain"], "source": node.content,
        "baseline_translation": "", "candidate_translation": result.translated_text,
        "preferred_output": row.get("preferred_output", ""), **flags,
        "provider_calls": telemetry.provider_calls, "provider_calls_per_node": telemetry.provider_calls,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "request_tokens": telemetry.total_tokens, "few_shot_count": telemetry.few_shot_count,
        "style_memory_hits": telemetry.style_memory_hits,
        "naturalness": natural["naturalness"], "naturalness_critic_calls": natural["naturalness_critic_calls"],
        "editorial_rewrite_count": 0, "semantic_repairs": telemetry.retries,
        "deterministic_error": not result.quality.passed,
    }


def run(
    model_name: str = "mock-qwen2.5:7b",
    dataset: Optional[Iterable[Mapping[str, Any]]] = None,
    human_csv: Optional[Path] = None,
    blind_ab_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = [dict(row) for row in (dataset or load_official_dataset())]
    errors = validate_official_dataset(rows)
    if errors:
        raise ValueError("; ".join(errors[:5]))
    provider = _provider(model_name)
    if not provider.health_check():
        raise RuntimeError(f"Model local chưa sẵn sàng: {model_name}")

    original_projects_dir = settings.PROJECTS_DIR
    baseline_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="translation-p1-p2-eval-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        db_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(db_engine)
        db = sessionmaker(bind=db_engine)()
        try:
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[str(row["domain"]).upper()].append(row)
            for domain, domain_rows in sorted(grouped.items()):
                project = ProjectModel(
                    id=f"p1_p2_eval_{domain.lower()}", title=f"P1/P2 {domain}",
                    selected_model=model_name, document_type=domain, translation_mode="NATURAL",
                )
                chapter = ChapterModel(
                    id=f"p1_p2_chapter_{domain.lower()}", project_id=project.id,
                    title=f"{domain} benchmark", order_index=0,
                )
                db.add_all([project, chapter])
                db.commit()
                nodes = []
                domain_glossary: Dict[str, str] = {}
                for index, row in enumerate(domain_rows):
                    node = NodeModel(
                        id=f"{domain.lower()}_{index:04d}", project_id=project.id, chapter_id=chapter.id,
                        node_type=row.get("node_type", "paragraph"), content=row["source"],
                        status="PENDING", order_index=index,
                    )
                    db.add(node)
                    nodes.append(node)
                    domain_glossary.update(row.get("glossary", {}))
                db.commit()
                config = TranslationConfig.from_project(project, model_override=model_name)
                engine = ContextualTranslationEngine(db, project, provider, config, domain_glossary)
                for row, node in zip(domain_rows, nodes):
                    baseline_rows.append(_baseline_row(provider, row, model_name))
                    candidate_rows.append(_candidate_row(engine, chapter, node, row))
        finally:
            db.close()
            settings.PROJECTS_DIR = original_projects_dir

    baseline = summarize_evaluations(baseline_rows)
    candidate = summarize_evaluations(candidate_rows)
    human_evaluation = {
        "status": "PENDING_HUMAN",
        "columns": list(HUMAN_EVAL_COLUMNS),
        "blind_ab": True,
        "blind_ab_columns": ["sample_id", "domain", "source", "translation_a", "translation_b", "preference", "notes"],
    }
    report = {
        "dataset_version": EVAL_DATASET_VERSION,
        "samples": len(rows),
        "domains": {domain: sum(1 for row in rows if row["domain"] == domain) for domain in sorted(set(row["domain"] for row in rows))},
        "baseline": baseline,
        "candidate": candidate,
        "regression": compare_regression(baseline, candidate),
        "human_evaluation": human_evaluation,
    }
    if human_csv:
        human_csv.parent.mkdir(parents=True, exist_ok=True)
        with human_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(HUMAN_EVAL_COLUMNS))
            writer.writeheader()
            for baseline_row, candidate_row, source_row in zip(baseline_rows, candidate_rows, rows):
                writer.writerow({
                    "sample_id": source_row["sample_id"], "domain": source_row["domain"], "source": source_row["source"],
                    "baseline_translation": baseline_row.get("baseline_translation", ""),
                    "candidate_translation": candidate_row.get("candidate_translation", ""),
                    "preferred_output": source_row.get("preferred_output", ""), "notes": "",
                })
    if blind_ab_csv:
        blind_ab_csv.parent.mkdir(parents=True, exist_ok=True)
        with blind_ab_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "sample_id", "domain", "source", "translation_a", "translation_b", "preference", "notes",
            ])
            writer.writeheader()
            for baseline_row, candidate_row, source_row in zip(baseline_rows, candidate_rows, rows):
                candidate_is_a = _candidate_is_a(source_row["sample_id"])
                writer.writerow({
                    "sample_id": source_row["sample_id"], "domain": source_row["domain"], "source": source_row["source"],
                    "translation_a": candidate_row["candidate_translation"] if candidate_is_a else baseline_row["baseline_translation"],
                    "translation_b": baseline_row["baseline_translation"] if candidate_is_a else candidate_row["candidate_translation"],
                    "preference": "", "notes": "",
                })
        human_evaluation["blind_ab_csv"] = str(blind_ab_csv)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chất lượng dịch P1/P2 offline.")
    parser.add_argument("--model", default="mock-qwen2.5:7b")
    parser.add_argument("--human-csv", type=Path)
    parser.add_argument("--blind-ab-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    report = run(args.model, human_csv=args.human_csv, blind_ab_csv=args.blind_ab_csv)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
