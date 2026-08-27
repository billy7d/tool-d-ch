import hashlib
import uuid
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models import TranslationMemoryModel
from app.services.translation.quality_gate import TranslationQualityGate


class TranslationMemoryService:
    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def lookup(
        cls,
        db: Session,
        source_text: str,
        style_hash: str,
        glossary_hash: str,
        prompt_version: str,
    ) -> Optional[str]:
        # Production chỉ tái sử dụng bản ghi có đầy đủ chữ ký Phase 1.
        if not style_hash or not glossary_hash or not prompt_version:
            return None
        source_hash = cls.compute_hash(source_text)
        query = db.query(TranslationMemoryModel).filter(
            TranslationMemoryModel.source_hash == source_hash,
            TranslationMemoryModel.style_hash == style_hash,
            TranslationMemoryModel.glossary_hash == glossary_hash,
            TranslationMemoryModel.prompt_version == prompt_version,
        )

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
        prompt_version: str = "phase1-v2"
    ):
        if not style_hash or not glossary_hash or not prompt_version:
            raise ValueError("TM Phase 1 yêu cầu đầy đủ style_hash, glossary_hash và prompt_version")
        if not translated_text or not translated_text.strip():
            raise ValueError("Không được lưu bản dịch rỗng vào TM")
        quality = TranslationQualityGate().validate(source_text, translated_text, {})
        if not quality.passed:
            codes = ", ".join(issue["code"] for issue in quality.issues if issue["severity"] == "ERROR")
            raise ValueError(f"Không được lưu candidate không qua Quality Gate vào TM: {codes}")
        source_hash = cls.compute_hash(source_text)
        existing = db.query(TranslationMemoryModel).filter(
            TranslationMemoryModel.source_hash == source_hash,
            TranslationMemoryModel.style_hash == style_hash,
            TranslationMemoryModel.glossary_hash == glossary_hash,
            TranslationMemoryModel.prompt_version == prompt_version,
        ).first()

        if existing:
            existing.translated_text = translated_text
            existing.model_name = model_name
            existing.prompt_version = prompt_version
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
