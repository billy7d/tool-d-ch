import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.db.models import Base
from sqlalchemy import text

# Global database for system settings and project index
GLOBAL_DB_PATH = settings.DATA_DIR / "global.db"
global_engine = create_engine(
    f"sqlite:///{GLOBAL_DB_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
    echo=False
)

# Enable WAL mode and foreign keys on SQLite connections
@event.listens_for(global_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

GlobalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=global_engine)


def init_global_db():
    Base.metadata.create_all(bind=global_engine)
    _apply_additive_project_migrations(global_engine)


def get_global_db() -> Generator[Session, None, None]:
    db = GlobalSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Cache of engine connections per project
_project_engines = {}
_project_sessions = {}


def get_project_db_path(project_id: str) -> Path:
    return settings.PROJECTS_DIR / project_id / "project.db"


def get_project_engine(project_id: str):
    if project_id not in _project_engines:
        db_path = get_project_db_path(project_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
            echo=False
        )
        
        @event.listens_for(engine, "connect")
        def set_project_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            
        Base.metadata.create_all(bind=engine)
        _apply_additive_project_migrations(engine)
        _project_engines[project_id] = engine
        _project_sessions[project_id] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
    return _project_engines[project_id]


def _apply_additive_project_migrations(engine) -> None:
    """Bổ sung schema additive cho DB cũ mà không xóa hay ghi đè dữ liệu."""
    with engine.begin() as connection:
        tables = {row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        if "semantic_reviews" in tables:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(semantic_reviews)"))}
            if "is_stale" not in columns:
                connection.execute(text("ALTER TABLE semantic_reviews ADD COLUMN is_stale BOOLEAN NOT NULL DEFAULT 0"))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_semantic_reviews_is_stale ON semantic_reviews (is_stale)"
            ))
        if "glossary" in tables:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(glossary)"))}
            additive_columns = {
                "preferred_target": "VARCHAR(255)",
                "allowed_variants": "JSON",
                "sense_hint": "TEXT",
                "domain": "VARCHAR(64) DEFAULT 'GENERAL'",
                "part_of_speech": "VARCHAR(64)",
                "preserve_original": "BOOLEAN NOT NULL DEFAULT 0",
                "lock_level": "VARCHAR(16) DEFAULT 'HARD'",
            }
            for column_name, column_type in additive_columns.items():
                if column_name not in columns:
                    connection.execute(text(f"ALTER TABLE glossary ADD COLUMN {column_name} {column_type}"))


def get_project_db(project_id: str) -> Session:
    if project_id not in _project_sessions:
        get_project_engine(project_id)
    return _project_sessions[project_id]()
