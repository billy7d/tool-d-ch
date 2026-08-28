from tests.phase3_eval import run


def test_phase3_mock_benchmark_meets_selectivity_and_recall_targets():
    metrics = run("mock", write_artifacts=False)
    assert metrics["samples"] == 80
    assert metrics["critic_review_rate"] <= 0.40
    assert metrics["semantic_error_recall"] >= 0.90
    assert metrics["semantic_error_precision"] >= 0.90
    assert metrics["provider_calls_per_node"] <= 1.60
    assert metrics["context_overflow"] == 0
