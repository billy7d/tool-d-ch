from app.db.engine import get_project_db
from app.db.models import ChapterModel, GlossaryModel, NodeModel, ProjectModel
from app.services.translation.term_consistency import (
    attested, choose_terms, ensure_auto_terms, examples_for, mine_candidates, valid_rendering,
)

PARAGRAPHS = (
    [f"Through grace we are helped in case {i}. Grace is available to everyone." for i in range(9)]
    + [f"The ego boundaries collapse when people fall in love, number {i}." for i in range(7)]
    + [f"Marcia said that Peter would never come, for reason {i}." for i in range(9)]
    + ["An unrelated paragraph about the weather."]
)


class FakeProvider:
    def __init__(self, renderings):
        self.renderings = renderings
        self.calls = 0

    def json_task(self, system_message, user_message, schema, model=None, num_predict=400, temperature=0.1):
        self.calls += 1
        if "terms" in schema["properties"]:
            return {"terms": ["grace", "ego boundaries", "not a candidate"]}
        term = user_message.splitlines()[0].removeprefix("TERM: ")
        return self.renderings[term]


def test_mining_counts_paragraphs_and_skips_names_and_common_words():
    candidates = dict(mine_candidates(PARAGRAPHS))
    assert candidates["grace"] == 9
    assert candidates["ego boundaries"] == 7
    assert "marcia" not in candidates and "peter" not in candidates
    assert "people" not in candidates and "never" not in candidates


def test_examples_are_sentences_containing_the_term():
    examples = examples_for("grace", PARAGRAPHS)
    assert len(examples) == 3 and all("grace" in example.lower() for example in examples)


def test_rendering_must_be_short_vietnamese():
    assert valid_rendering("grace", "ân sủng")
    assert valid_rendering("evil", "cái ác")
    assert not valid_rendering("psychotherapy", "therapia tâm lý")
    assert not valid_rendering("grace", "grace")
    assert not valid_rendering("grace", "ân sủng (ơn của Chúa)")
    assert not valid_rendering("cathexis", "Цетес")
    assert not valid_rendering("grace", "một khái niệm rất khó dịch sang tiếng Việt")


def test_rendering_must_appear_in_the_example_translations():
    assert attested("ân sủng", ["Nhờ ân sủng ta được giúp.", "Ân sủng dành cho mọi người.", "Một câu khác."])
    assert not attested("tiểu thuyết", ["Trị liệu là một quá trình dài.", "Trị liệu cần can đảm."])
    assert not attested("ân sủng", ["Nhờ ân sủng ta được giúp."])


def test_choose_terms_drops_unattested_and_unknown_terms():
    provider = FakeProvider({
        "grace": {"translations": ["Nhờ ân sủng.", "Ân sủng có sẵn.", "Qua ân sủng."], "rendering": "ân sủng"},
        "ego boundaries": {"translations": ["Ranh giới sụp đổ.", "Giới hạn tan."], "rendering": "bội giới tự ngã"},
    })
    assert choose_terms(provider, "m", PARAGRAPHS) == [{"source_term": "grace", "target_term": "ân sủng"}]


def test_auto_terms_are_opt_in_soft_and_run_once(tmp_path, monkeypatch):
    project_id = "auto-terms"
    db = get_project_db(project_id)
    try:
        db.add(ProjectModel(id=project_id, title="Book"))
        db.add(ChapterModel(id="c1", project_id=project_id, title="One", order_index=0))
        for index, text in enumerate(PARAGRAPHS):
            db.add(NodeModel(id=f"n{index}", project_id=project_id, chapter_id="c1", node_type="paragraph",
                             content=text, status="PENDING", order_index=index))
        db.add(GlossaryModel(id="user1", project_id=project_id, source_term="Ego Boundaries", target_term="ranh giới bản ngã"))
        db.commit()
        provider = FakeProvider({
            "grace": {"translations": ["Nhờ ân sủng.", "Ân sủng có sẵn.", "Qua ân sủng."], "rendering": "ân sủng"},
            "ego boundaries": {"translations": ["ranh giới tôi", "ranh giới tôi", "ranh giới tôi"], "rendering": "ranh giới tôi"},
        })

        monkeypatch.delenv("TOOL_DICH_AUTO_TERMS", raising=False)
        assert ensure_auto_terms(db, project_id, provider, "m", tmp_path) == 0 and provider.calls == 0

        monkeypatch.setenv("TOOL_DICH_AUTO_TERMS", "1")
        assert ensure_auto_terms(db, project_id, provider, "m", tmp_path) == 1
        added = db.query(GlossaryModel).filter(GlossaryModel.category == "AUTO").all()
        assert [(row.source_term, row.target_term, row.lock_level, row.locked) for row in added] == [("grace", "ân sủng", "SOFT", False)]
        assert db.query(GlossaryModel).filter(GlossaryModel.id == "user1").one().target_term == "ranh giới bản ngã"

        calls = provider.calls
        db.delete(added[0])
        db.commit()
        assert ensure_auto_terms(db, project_id, provider, "m", tmp_path) == 0 and provider.calls == calls
    finally:
        db.close()
