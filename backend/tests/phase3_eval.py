import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from app.services.qa.semantic_critic import SemanticCritic
from app.services.translation.mock_provider import MockProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.semantic_risk import SemanticRiskScorer


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "translation_eval" / "phase3_semantic_eval.jsonl"
ARTIFACTS = ROOT / "artifacts"


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def run(model: str = "mock") -> dict:
    samples = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = MockProvider() if model.lower().startswith("mock") else OllamaProvider(default_model=model)
    if not provider.health_check():
        raise RuntimeError(f"Model local chưa sẵn sàng: {model}")

    tp = fp = fn = critic_errors = critic_passes = repair_attempts = repair_successes = 0
    critic_calls = critic_tokens = 0
    critic_latencies = []
    levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    rows = []
    for sample in samples:
        risk = SemanticRiskScorer.score(
            sample["source"], sample["candidate"], document_type=sample["domain"],
        )
        levels[risk.level] += 1
        status = "NOT_REQUIRED"
        issue = ""
        if risk.requires_critic:
            candidate = sample["candidate"]
            if isinstance(provider, MockProvider):
                candidate += " " + sample.get("mock_marker", "")
            review = SemanticCritic.review(provider, sample["source"], candidate, {}, {}, sample["domain"], model)
            critic_calls += 1
            critic_tokens += review.critic_request_tokens
            critic_latencies.append(review.critic_latency_ms)
            status = review.status
            issue = review.errors[0]["type"] if review.errors else ""
            critic_errors += review.status == "ERROR"
            critic_passes += review.status == "PASS"
            if sample.get("exercise_repair") and review.status == "FAIL":
                repair_attempts += 1
                repaired = sample["repair_candidate"]
                repaired_review = SemanticCritic.review(provider, sample["source"], repaired, {}, {}, sample["domain"], model)
                critic_calls += 1
                critic_tokens += repaired_review.critic_request_tokens
                critic_latencies.append(repaired_review.critic_latency_ms)
                repair_successes += repaired_review.status == "PASS"
        expected_fail = sample["expected"] == "FAIL"
        predicted_fail = status in {"FAIL", "ERROR"}
        tp += expected_fail and predicted_fail
        fp += (not expected_fail) and predicted_fail
        fn += expected_fail and not predicted_fail
        rows.append({
            "sample_id": sample["id"], "domain": sample["domain"], "source": sample["source"],
            "translation": sample["candidate"], "risk_score": risk.score,
            "critic_status": status, "critic_issue": issue,
            "human_correct": "", "human_issue": "", "naturalness_score": "", "notes": "",
        })

    expected_errors = sum(sample["expected"] == "FAIL" for sample in samples)
    initial_critic_calls = sum(row["critic_status"] != "NOT_REQUIRED" for row in rows)
    # Mỗi repair đại diện thêm một lần sinh candidate và một lần re-review.
    provider_calls = len(samples) + initial_critic_calls + repair_attempts + repair_attempts
    metrics = {
        "samples": len(samples),
        "semantic_error_recall": _safe_rate(tp, expected_errors),
        "semantic_error_precision": _safe_rate(tp, tp + fp),
        "false_positive_rate": _safe_rate(fp, len(samples) - expected_errors),
        "critic_review_rate": _safe_rate(initial_critic_calls, len(samples)),
        "high_risk_recall": _safe_rate(sum(row["critic_status"] != "NOT_REQUIRED" and sample["expected"] == "FAIL" for row, sample in zip(rows, samples)), expected_errors),
        "low_risk_false_negative_rate": _safe_rate(fn, expected_errors),
        "critic_pass_rate": _safe_rate(critic_passes, initial_critic_calls),
        "critic_error_rate": _safe_rate(critic_errors, initial_critic_calls),
        "repair_success_rate": _safe_rate(repair_successes, repair_attempts),
        "avg_critic_calls_per_node": round(critic_calls / len(samples), 4),
        "avg_semantic_repairs_per_node": round(repair_attempts / len(samples), 4),
        "provider_calls_per_node": round(provider_calls / len(samples), 4),
        "avg_critic_latency_ms": round(mean(critic_latencies), 3) if critic_latencies else 0.0,
        "avg_critic_tokens": round(critic_tokens / critic_calls, 2) if critic_calls else 0.0,
        "context_overflow": 0,
        "risk_distribution": {**levels, "critic_required": initial_critic_calls},
        "model": model,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "phase3_semantic_benchmark.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (ARTIFACTS / "phase3_human_eval.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 semantic benchmark cho mock CI hoặc model local.")
    parser.add_argument("--model", default="mock", help="mock hoặc tên model Ollama, ví dụ qwen2.5:7b")
    args = parser.parse_args()
    print(json.dumps(run(args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
