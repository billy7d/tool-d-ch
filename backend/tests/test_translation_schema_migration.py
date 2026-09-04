from sqlalchemy import create_engine, inspect, text

from app.db.engine import _apply_additive_project_migrations
from app.db.models import Base


def test_fresh_schema_contains_contextual_glossary_and_style_memory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _apply_additive_project_migrations(engine)
    inspector = inspect(engine)

    glossary_columns = {column["name"] for column in inspector.get_columns("glossary")}
    assert {"preferred_target", "allowed_variants", "sense_hint", "domain", "lock_level"} <= glossary_columns
    assert "style_memory" in inspector.get_table_names()


def test_existing_legacy_glossary_is_upgraded_additively_without_losing_rows():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE glossary ("
            "id VARCHAR(64) PRIMARY KEY, project_id VARCHAR(64) NOT NULL, "
            "source_term VARCHAR(255) NOT NULL, target_term VARCHAR(255) NOT NULL, "
            "category VARCHAR(64), notes TEXT, locked BOOLEAN)"
        ))
        connection.execute(text(
            "INSERT INTO glossary (id, project_id, source_term, target_term, category, locked) "
            "VALUES ('legacy', 'p', 'interest', 'sự quan tâm', 'GENERAL', 0)"
        ))

    _apply_additive_project_migrations(engine)
    with engine.connect() as connection:
        row = connection.execute(text(
            "SELECT source_term, target_term, locked, preferred_target, lock_level FROM glossary WHERE id='legacy'"
        )).one()
    columns = {column["name"] for column in inspect(engine).get_columns("glossary")}

    assert row[0:3] == ("interest", "sự quan tâm", 0)
    assert {"preferred_target", "allowed_variants", "sense_hint", "domain", "lock_level"} <= columns
