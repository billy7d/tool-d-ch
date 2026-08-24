import hashlib
import uuid
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.db.models import TranslationMemoryModel


class TranslationMemoryService:
    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def lookup(
        cls,
        db: Session,
        source_text: str,
        style_hash: str = "",
        glossary_hash: str = ""
    ) -> Optional[str]:
        source_hash = cls.compute_hash(source_text)
        query = db.query(TranslationMemoryModel).filter(
            TranslationMemoryModel.source_hash == source_hash
        )
        if style_hash:
            query = query.filter(TranslationMemoryModel.style_hash == style_hash)
        if glossary_hash:
            query = query.filter(TranslationMemoryModel.glossary_hash == glossary_hash)

        match = query.first()
        return match.translated_text if match else None

    @classmethod
    def store(
        cls,
        db: Session,
        source_text: str,
        translated_text: str,
        style_hash: str = "",
        glossary_hash: str = "",
        model_name: str = "",
        prompt_version: str = "v1"
    ):
        source_hash = cls.compute_hash(source_text)
        existing = db.query(TranslationMemoryModel).filter(
            TranslationMemoryModel.source_hash == source_hash,
            TranslationMemoryModel.style_hash == style_hash,
            TranslationMemoryModel.glossary_hash == glossary_hash
        ).first()

        if existing:
            existing.translated_text = translated_text
            existing.model_name = model_name
        else:
            tm_entry = TranslationMemoryModel(
                id=str(uuid.uuid4()),
                source_hash=source_hash,
                source_text=source_text,
                translated_text=translated_text,
                style_hash=style_hash,
                glossary_hash=glossary_hash,
                model_name=model_name,
                prompt_version=prompt_version,
            )
            db.add(tm_entry)
        db.commit()
