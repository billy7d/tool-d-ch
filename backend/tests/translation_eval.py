"""Chạy: python -m tests.translation_eval [đường-dẫn-jsonl]."""

import json
import time
import argparse
from pathlib import Path

from app.models.canonical import DocumentNode, NodeType, TranslationMode
from app.services.translation.context_assembler import ContextAssembler
from app.services.translation.context_assembler import estimate_tokens
from app.services.translation.context_memory import ChapterMemory
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.mock_provider import MockProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.prompt_profiles import select_few_shots
from app.services.translation.quality_gate import TranslationQualityGate


def build_phase1_baseline_prompt(node: DocumentNode, glossary: dict) -> str:
    return (
        "You are a master English-to-Vietnamese translator and publishing editor.\n"
        "Produce natural, grammatically correct Vietnamese suitable for printed books. "
        "Preserve proper nouns, trademarks, acronyms, URLs, code, identifiers, formatting, numbers, dates, formulas and units. "
        "Restructure passive English clauses into clear Vietnamese, follow the locked glossary, do not summarize, omit or hallucinate, and return only the translation.\n"
        "Example: The decision was made by the board with respect to expansion. -> Hội đồng quản trị quyết định mở rộng hoạt động.\n"
        "Example: It is crucial that risk management practices be implemented. -> Cần triển khai các quy trình quản trị rủi ro.\n"
        f"Locked glossary: {json.dumps(glossary, ensure_ascii=False)}\n"
        f"Source: {node.content}"
    )


def run(dataset_path: Path, model_name: str = "mock-qwen2.5:7b") -> dict:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = MockProvider() if model_name.lower().startswith("mock") else OllamaProvider(default_model=model_name)
    if not provider.health_check():
        raise RuntimeError(f"Mô hình {model_name} chưa sẵn sàng qua provider local.")
    capabilities = provider.get_model_capabilities(model_name)
    metrics = {"model": model_name, "baseline": {"latency_ms": 0.0, "passed": 0, "prompt_tokens": 0}, "phase2": {"latency_ms": 0.0, "passed": 0, "prompt_tokens": 0}, "samples": len(rows), "human_naturalness": None}
    baseline_human = []
    phase2_human = []
    for index, row in enumerate(rows):
        node = DocumentNode(id=f"eval_{index}", type=NodeType(row.get("node_type", "paragraph")), content=row["source"])
        glossary = row.get("glossary", {})
        started = time.perf_counter()
        baseline_prompt = build_phase1_baseline_prompt(node, glossary)
        baseline = provider.translate_single(node.content, baseline_prompt, glossary, model=model_name)
        metrics["baseline"]["latency_ms"] += (time.perf_counter() - started) * 1000
        metrics["baseline"]["prompt_tokens"] += estimate_tokens(baseline_prompt + node.content)
        metrics["baseline"]["passed"] += int(TranslationQualityGate().validate(node.content, baseline, glossary).passed)
        started = time.perf_counter()
        profile = DocumentProfiler.fallback_profile(row["domain"], row["source"])
        context = ContextAssembler.assemble_context(
            [node], profile, ChapterMemory(summary=row.get("chapter_context", "")), [], glossary,
            capabilities, select_few_shots(row["domain"], "NATURAL", node.type.value),
        )
        system = PromptBuilder.build_system_prompt(row["domain"], TranslationMode.NATURAL)
        user = PromptBuilder.build_user_prompt([node], translation_context=context, document_type=row["domain"])
        phase2 = provider.translate_single(node.content, f"{system}\n\n{user}", glossary, model=model_name)
        metrics["phase2"]["latency_ms"] += (time.perf_counter() - started) * 1000
        metrics["phase2"]["prompt_tokens"] += estimate_tokens(system + user + node.content)
        metrics["phase2"]["passed"] += int(TranslationQualityGate().validate(node.content, phase2, glossary).passed)
        if row.get("baseline_naturalness") is not None and row.get("phase2_naturalness") is not None:
            baseline_human.append(float(row["baseline_naturalness"]))
            phase2_human.append(float(row["phase2_naturalness"]))
    for group in ("baseline", "phase2"):
        metrics[group]["validation_pass_rate"] = metrics[group]["passed"] / max(1, len(rows))
        metrics[group]["avg_latency_ms"] = metrics[group]["latency_ms"] / max(1, len(rows))
        metrics[group]["avg_prompt_tokens"] = metrics[group]["prompt_tokens"] / max(1, len(rows))
    if baseline_human:
        baseline_avg = sum(baseline_human) / len(baseline_human)
        phase2_avg = sum(phase2_human) / len(phase2_human)
        metrics["human_naturalness"] = {"baseline": baseline_avg, "phase2": phase2_avg, "improvement": phase2_avg - baseline_avg, "rated_samples": len(baseline_human)}
    return metrics


if __name__ == "__main__":
    default = Path(__file__).parent / "fixtures" / "translation_eval" / "phase2_eval.jsonl"
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=default)
    parser.add_argument("--model", default="mock-qwen2.5:7b")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.model), ensure_ascii=False, indent=2))
