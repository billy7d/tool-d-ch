import json
from pathlib import Path

from app.services.qa.vietnamese_naturalness_critic import VietnameseNaturalnessCritic
from app.services.translation.mock_provider import MockProvider


FIXTURE = Path(__file__).parent / "fixtures" / "translation_eval" / "vietnamese_naturalness_eval.jsonl"


def test_naturalness_fixture_has_required_p0_domain_coverage():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1

    assert len(rows) >= 60
    assert counts == {
        "GENERAL": 10,
        "BUSINESS": 10,
        "FINANCE": 10,
        "SELF_HELP": 10,
        "TECHNICAL": 8,
        "ACADEMIC": 6,
        "LEGAL": 3,
        "LITERATURE": 3,
    }


def test_mock_critic_ranks_natural_candidate_over_literal_candidate():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    for row in rows[:30]:
        literal = VietnameseNaturalnessCritic.review(
            MockProvider(), row["source"], row["literal_candidate"],
            document_type=row["domain"], register="NEUTRAL", sentence_style="MODERATE", model="mock",
        )
        natural = VietnameseNaturalnessCritic.review(
            MockProvider(), row["source"], row["natural_candidate"],
            document_type=row["domain"], register="NEUTRAL", sentence_style="MODERATE", model="mock",
        )
        correct += literal.score is not None and natural.score is not None and literal.score < natural.score

    assert correct / 30 >= 0.90
