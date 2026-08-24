from typing import List, Dict, Any
from app.models.canonical import CanonicalDocument, Chapter, NodeType


class TOCGenerator:
    @staticmethod
    def generate_toc(doc: CanonicalDocument) -> List[Dict[str, Any]]:
        """
        Generates a hierarchical Table of Contents from translated chapters and headings.
        """
        toc_tree = []
        for ch_idx, ch in enumerate(doc.chapters):
            ch_title = ch.translated_title or ch.title
            ch_num = f"Chương {ch.number}: " if ch.number else ""
            
            subheadings = []
            for n in ch.nodes:
                if n.type == NodeType.HEADING:
                    h_title = n.translated_content or n.content
                    h_level = n.metadata.heading_level or 2
                    subheadings.append({
                        "id": n.id,
                        "title": h_title,
                        "level": h_level,
                    })

            toc_tree.append({
                "id": ch.id,
                "title": ch_num + ch_title,
                "level": 1,
                "subheadings": subheadings
            })

        return toc_tree
