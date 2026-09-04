import csv
import io
import json
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.db.models import GlossaryModel
from app.db.repository import GlossaryRepository


class GlossaryService:
    @staticmethod
    def _lock_level(term: GlossaryModel) -> str:
        requested = str(getattr(term, "lock_level", "") or "").upper()
        # DB cũ chỉ có locked; cờ cũ false luôn phải tiếp tục có nghĩa là SOFT.
        if not bool(getattr(term, "locked", True)):
            return "SOFT"
        return requested if requested in {"HARD", "SOFT"} else "HARD"

    @staticmethod
    def preferred_target(term: GlossaryModel) -> str:
        return str(getattr(term, "preferred_target", None) or term.target_term or "").strip()

    @classmethod
    def to_context_entry(cls, term: GlossaryModel) -> Dict[str, Any]:
        """Chuẩn hóa row legacy và row mới thành context record ổn định."""
        return {
            "id": term.id,
            "source_term": term.source_term,
            "target_term": term.target_term,
            "preferred_target": cls.preferred_target(term),
            "allowed_variants": list(getattr(term, "allowed_variants", None) or []),
            "sense_hint": getattr(term, "sense_hint", None) or "",
            "domain": str(getattr(term, "domain", None) or term.category or "GENERAL").upper(),
            "part_of_speech": getattr(term, "part_of_speech", None) or "",
            "preserve_original": bool(getattr(term, "preserve_original", False)),
            "lock_level": cls._lock_level(term),
            "locked": cls._lock_level(term) == "HARD",
            "notes": term.notes or "",
        }

    @classmethod
    def get_contextual_glossary(
        cls,
        db: Session,
        project_id: str,
        domain: str = "",
    ) -> List[Dict[str, Any]]:
        terms = db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).order_by(GlossaryModel.source_term).all()
        normalized_domain = str(domain or "").upper()
        if normalized_domain:
            exact_sources = {
                str(term.source_term).casefold() for term in terms
                if str(getattr(term, "domain", None) or term.category or "GENERAL").upper() == normalized_domain
            }
            terms = [
                term for term in terms
                if str(getattr(term, "domain", None) or term.category or "GENERAL").upper() == normalized_domain
                or (
                    str(getattr(term, "domain", None) or term.category or "GENERAL").upper() == "GENERAL"
                    and str(term.source_term).casefold() not in exact_sources
                )
            ]
        return [cls.to_context_entry(term) for term in terms]

    @classmethod
    def get_soft_glossary_map(cls, db: Session, project_id: str, domain: str = "") -> Dict[str, str]:
        return {
            item["source_term"]: item["preferred_target"]
            for item in cls.get_contextual_glossary(db, project_id, domain=domain)
            if item["lock_level"] == "SOFT" and item["preferred_target"]
        }

    @staticmethod
    def get_locked_glossary_map(db: Session, project_id: str) -> Dict[str, str]:
        """Trả map HARD legacy để các provider và chữ ký P0 vẫn tương thích."""
        terms = db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).all()
        return {
            item["source_term"]: item["preferred_target"]
            for item in (GlossaryService.to_context_entry(term) for term in terms)
            if item["lock_level"] == "HARD" and item["preferred_target"]
        }

    @classmethod
    def export_csv(cls, db: Session, project_id: str) -> str:
        """Exports project glossary to CSV format."""
        terms = db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "source_term", "target_term", "category", "notes", "locked",
            "preferred_target", "allowed_variants", "sense_hint", "domain",
            "part_of_speech", "preserve_original", "lock_level",
        ])
        for t in terms:
            writer.writerow([
                t.source_term, t.target_term, t.category, t.notes or "", "true" if t.locked else "false",
                cls.preferred_target(t), json.dumps(getattr(t, "allowed_variants", None) or [], ensure_ascii=False),
                getattr(t, "sense_hint", None) or "", getattr(t, "domain", None) or t.category or "GENERAL",
                getattr(t, "part_of_speech", None) or "", "true" if getattr(t, "preserve_original", False) else "false",
                cls._lock_level(t),
            ])
        return output.getvalue()

    @staticmethod
    def import_csv(db: Session, project_id: str, csv_content: str) -> int:
        """Imports glossary terms from CSV content and returns count of imported terms."""
        f = io.StringIO(csv_content)
        reader = csv.DictReader(f)
        repo = GlossaryRepository(db)
        count = 0
        for row in reader:
            src = row.get("source_term", "").strip()
            tgt = row.get("target_term", "").strip()
            if not src or not tgt:
                continue
            cat = row.get("category", "GENERAL").strip()
            notes = row.get("notes", "").strip()
            locked_str = str(row.get("locked", "true")).lower()
            locked = locked_str in ["true", "1", "yes"]
            variants_value = row.get("allowed_variants", "")
            try:
                variants = json.loads(variants_value) if variants_value else []
            except json.JSONDecodeError:
                variants = [value.strip() for value in variants_value.split("|") if value.strip()]
            preserve_original = str(row.get("preserve_original", "false")).lower() in {"true", "1", "yes"}

            repo.add_term(
                project_id=project_id,
                source_term=src,
                target_term=tgt,
                category=cat,
                notes=notes,
                locked=locked,
                preferred_target=row.get("preferred_target") or tgt,
                allowed_variants=variants if isinstance(variants, list) else [],
                sense_hint=(row.get("sense_hint") or "").strip(),
                domain=(row.get("domain") or cat).strip(),
                part_of_speech=(row.get("part_of_speech") or "").strip() or None,
                preserve_original=preserve_original,
                lock_level=(row.get("lock_level") or ("HARD" if locked else "SOFT")).strip().upper(),
            )
            count += 1
        return count
