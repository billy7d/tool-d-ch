import csv
import io
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.db.models import GlossaryModel
from app.db.repository import GlossaryRepository


class GlossaryService:
    @staticmethod
    def get_locked_glossary_map(db: Session, project_id: str) -> Dict[str, str]:
        """Returns a dict of locked {source_term: target_term}."""
        terms = db.query(GlossaryModel).filter(
            GlossaryModel.project_id == project_id,
            GlossaryModel.locked == True
        ).all()
        return {t.source_term: t.target_term for t in terms}

    @staticmethod
    def export_csv(db: Session, project_id: str) -> str:
        """Exports project glossary to CSV format."""
        terms = db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["source_term", "target_term", "category", "notes", "locked"])
        for t in terms:
            writer.writerow([t.source_term, t.target_term, t.category, t.notes or "", "true" if t.locked else "false"])
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

            repo.add_term(
                project_id=project_id,
                source_term=src,
                target_term=tgt,
                category=cat,
                notes=notes,
                locked=locked,
            )
            count += 1
        return count
