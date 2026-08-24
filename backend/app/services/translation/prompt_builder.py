import re
import json
from typing import List, Dict, Any, Optional
from app.models.canonical import TranslationMode, DocumentType, DocumentNode


class PromptBuilder:
    MODE_INSTRUCTIONS = {
        TranslationMode.NATURAL: (
            "TRANSLATION MODE: Natural Publishing Vietnamese (Văn phong xuất bản tự nhiên).\n"
            "Priority: High fluency, idiomatic Vietnamese sentence structures, active voice, avoiding rigid word-by-word translation."
        ),
        TranslationMode.BALANCED: (
            "TRANSLATION MODE: Balanced (Cân bằng).\n"
            "Priority: Balance between Vietnamese naturalness and faithful structural alignment with the source text."
        ),
        TranslationMode.FAITHFUL: (
            "TRANSLATION MODE: Faithful (Bám sát nguyên tác).\n"
            "Priority: Close fidelity to original nuances, sentence clauses, and strict semantic accuracy."
        ),
        TranslationMode.ACADEMIC: (
            "TRANSLATION MODE: Academic (Học thuật, Chuyên khảo).\n"
            "Priority: Formal scholarly tone, rigorous specialist terminology, scholarly precision and objective syntax."
        ),
        TranslationMode.TECHNICAL: (
            "TRANSLATION MODE: Technical (Kỹ thuật & Công nghệ).\n"
            "Priority: Strict engineering precision. Keep standard technical terms, variables, equations, and identifiers intact."
        ),
        TranslationMode.CUSTOM: (
            "TRANSLATION MODE: Custom Instructions.\n"
        ),
    }

    FEW_SHOT_GUIDANCE = (
        "\nFEW-SHOT QUALITY STANDARDS (Learn the natural Vietnamese transformation style):\n"
        "Example 1:\n"
        "Source: \"The decision was made by the board of directors with respect to the expansion of operations.\"\n"
        "Natural Vietnamese: \"Hội đồng quản trị đã quyết định mở rộng quy mô hoạt động.\"\n"
        "Example 2:\n"
        "Source: \"It is crucial that risk management practices be implemented across all departments.\"\n"
        "Natural Vietnamese: \"Việc triển khai các quy trình quản trị rủi ro trên toàn bộ các phòng ban là điều tối quan trọng.\"\n"
    )

    @classmethod
    def filter_relevant_glossary(
        cls,
        nodes: List[DocumentNode],
        glossary_terms: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Token & focus optimization: Filters the global glossary to only include terms
        that actually appear in the current batch of nodes.
        """
        if not glossary_terms:
            return {}

        combined_text = " ".join([n.content for n in nodes]).lower()
        relevant: Dict[str, str] = {}
        for src, tgt in glossary_terms.items():
            if src.lower() in combined_text:
                relevant[src] = tgt

        # If batch is small and filtered list is empty, include up to 10 top terms if glossary is small
        if not relevant and len(glossary_terms) <= 10:
            return glossary_terms

        return relevant

    @classmethod
    def build_system_prompt(
        cls,
        document_type: str = "GENERAL",
        translation_mode: TranslationMode = TranslationMode.NATURAL,
        style_guide: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        base_core = (
            "You are a master English-to-Vietnamese translator and publishing editor.\n\n"
            "CORE TRANSLATION PRINCIPLES:\n"
            "1. STRICT 100% VIETNAMESE LANGUAGE MANDATE: All translations MUST be completely in Vietnamese. "
            "Under NO circumstances should any Chinese characters (中文字符 / Hán tự), Pinyin, or foreign scripts be included.\n"
            "2. Produce highly natural, grammatically flawless Vietnamese suitable for printed books.\n"
            "3. Restructure passive/clunky English clauses into active, clear Vietnamese sentences.\n"
            "4. Strictly adhere to mandatory locked glossary terms whenever corresponding concepts appear.\n"
            "5. Preserve formatting, markdown marks (**bold**, *italic*, # headings), numbers, dates, formulas, and units exactly ($100, 45%, 2026, Fig 1).\n"
            "6. Do NOT summarize. Do NOT omit sentences. Do NOT hallucinate. Do NOT add conversational chatter or preambles.\n"
            "7. Output MUST be strictly valid JSON matching the schema.\n"
        )

        doc_type_guide = f"\nDOCUMENT DOMAIN: {document_type}\n"
        mode_guide = f"\n{cls.MODE_INSTRUCTIONS.get(translation_mode, '')}\n"

        if translation_mode == TranslationMode.CUSTOM and custom_instructions:
            mode_guide += f"USER INSTRUCTIONS: {custom_instructions}\n"

        style_str = ""
        if style_guide:
            style_str = f"\nGLOBAL STYLE GUIDE:\n{json.dumps(style_guide, ensure_ascii=False, indent=2)}\n"

        return f"{base_core}{doc_type_guide}{mode_guide}{cls.FEW_SHOT_GUIDANCE}{style_str}"

    @classmethod
    def build_user_prompt(
        cls,
        nodes: List[DocumentNode],
        chapter_title: str = "",
        chapter_summary: str = "",
        previous_context: str = "",
        glossary_terms: Optional[Dict[str, str]] = None
    ) -> str:
        parts = []

        if chapter_title:
            parts.append(f"CURRENT CHAPTER: {chapter_title}")

        if chapter_summary:
            parts.append(f"CHAPTER CONTEXT & MEMORY:\n{chapter_summary}")

        if previous_context:
            parts.append(f"IMMEDIATELY PRECEDING TEXT (FOR COHERENCE REFERENCE ONLY - DO NOT TRANSLATE):\n{previous_context}")

        relevant_glossary = cls.filter_relevant_glossary(nodes, glossary_terms)
        if relevant_glossary:
            glossary_list = [f"- \"{k}\" -> \"{v}\"" for k, v in relevant_glossary.items()]
            parts.append("MANDATORY LOCKED GLOSSARY (You must use these exact Vietnamese terms):\n" + "\n".join(glossary_list))

        blocks_json = []
        for n in nodes:
            blocks_json.append({
                "node_id": n.id,
                "type": n.type.value,
                "text": n.content
            })

        parts.append(
            "BLOCKS TO TRANSLATE:\n"
            f"{json.dumps(blocks_json, ensure_ascii=False, indent=2)}\n\n"
            "CRITICAL: Translate 100% into Vietnamese. Absolutely NO Chinese characters (中文字符) or foreign text allowed.\n\n"
            "REQUIRED JSON RESPONSE FORMAT:\n"
            "{\n"
            "  \"translations\": [\n"
            "    {\"node_id\": \"...\", \"text\": \"...\"}\n"
            "  ]\n"
            "}"
        )

        return "\n\n".join(parts)


