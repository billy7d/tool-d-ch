import json
import re
from typing import Any, Dict, List, Optional

from app.models.canonical import DocumentNode, TranslationMode
from app.services.translation.context_assembler import TranslationContext
from app.services.translation.prompt_profiles import STYLE_PACK_VERSION, get_style_pack, select_few_shots


PROMPT_VERSION = "translation-v3-contextual"


class PromptBuilder:
    MODE_INSTRUCTIONS = {
        TranslationMode.NATURAL: "Use Natural Vietnamese suitable for publishing and moderately restructure English syntax.",
        TranslationMode.BALANCED: "Balance natural Vietnamese with close semantic alignment.",
        TranslationMode.FAITHFUL: "Stay close to the source structure while keeping grammatical Vietnamese.",
        TranslationMode.ACADEMIC: "Use formal scholarly Vietnamese with precise terminology.",
        TranslationMode.TECHNICAL: "Prioritize technical accuracy and preserve identifiers.",
        TranslationMode.CUSTOM: "Apply valid user preferences after all correctness rules.",
    }
    CONFLICTING_INSTRUCTION_PATTERNS = (
        r"(?i)\btóm tắt\b", r"(?i)\brút gọn\b", r"(?i)\bbỏ (?:qua|bớt)\b",
        r"(?i)\bsummar(?:y|ize)\b", r"(?i)\bom(?:it|ission)\b",
    )

    @classmethod
    def validate_custom_instructions(cls, custom_instructions: Optional[str]) -> None:
        if custom_instructions and any(re.search(pattern, custom_instructions) for pattern in cls.CONFLICTING_INSTRUCTION_PATTERNS):
            raise ValueError("Chỉ dẫn tùy chỉnh xung đột với yêu cầu dịch đầy đủ, không tóm tắt hoặc bỏ ý.")

    @classmethod
    def filter_relevant_glossary(cls, nodes: List[DocumentNode], glossary_terms: Optional[Dict[str, str]]) -> Dict[str, str]:
        combined = " ".join(node.content for node in nodes)
        return {
            source: target for source, target in (glossary_terms or {}).items()
            if re.search(rf"(?i)(?<!\w){re.escape(source)}(?:s|es|ed|ing)?(?!\w)", combined)
        }

    @classmethod
    def build_system_prompt(
        cls,
        document_type: str = "GENERAL",
        translation_mode: TranslationMode = TranslationMode.NATURAL,
        style_guide: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[str] = None,
        register: Optional[str] = None,
        sentence_style: Optional[str] = None,
    ) -> str:
        cls.validate_custom_instructions(custom_instructions)
        style_pack = get_style_pack(document_type)
        layers = [
            f"PROMPT VERSION: {PROMPT_VERSION}",
            (
                "SYSTEM CORE\nYou are a professional English-to-Vietnamese translator and Vietnamese publishing editor.\n"
                "PRIORITY 1 - SEMANTIC FIDELITY: Preserve every proposition, condition, negation, causal relation, number, entity, reference and degree of certainty.\n"
                "PRIORITY 2 - COMPLETENESS: Do not summarize, skip, compress, explain or add information.\n"
                "PRIORITY 3 - TERMINOLOGY: Follow the locked glossary and established entity naming exactly.\n"
                "PRIORITY 4 - NATURAL VIETNAMESE: Rewrite English syntax into idiomatic Vietnamese only after priorities 1-3 are satisfied.\n"
                "PRIORITY 5 - STYLE: Maintain the document tone and approved preceding translation.\n"
                "You may move adverbials, convert passive to active, reduce nominalization, split overly long sentences, merge short sentences when no meaning changes, and remove unnecessary possessives or pronouns.\n"
                "Do not change facts, polarity, scope, modality or certainty. Preserve formatting, numbers, dates, formulas, units, URLs, code and references.\n"
                "Vietnamese prose must be Vietnamese. Preserve proper nouns, company/product/trademark names, acronyms, programming identifiers, commands, paths, code, formulas and intentionally retained technical terms.\n"
                "Avoid word-for-word calques such as 'with respect to', 'in terms of', 'in order to', 'there is/are', 'the fact that', 'make a decision', 'take action' and 'provide assistance' when Vietnamese has a direct natural construction."
            ),
            "TRANSLATION MODE\n" + cls.MODE_INSTRUCTIONS.get(translation_mode, cls.MODE_INSTRUCTIONS[TranslationMode.NATURAL]),
            (
                f"DOCUMENT DOMAIN\n{style_pack.domain}\nSTYLE PACK {STYLE_PACK_VERSION}\n"
                f"Register: {register or style_pack.register}\nRules: {style_pack.instructions}\n"
                f"Forbidden: {style_pack.forbidden}\nSentence restructuring: {sentence_style or 'moderate'}"
            ),
        ]
        if style_guide:
            layers.append("DOCUMENT/USER STYLE SETUP\n" + json.dumps(style_guide, ensure_ascii=False, indent=2))
        if custom_instructions:
            layers.append("USER CUSTOM INSTRUCTION (lower priority than correctness and glossary)\n" + custom_instructions.strip())
        layers.append("OUTPUT CONTRACT\nReturn strictly valid JSON matching the requested node mapping. Return every requested node exactly once and no unknown node IDs.")
        return "\n\n".join(layers)

    @classmethod
    def build_user_prompt(
        cls,
        nodes: List[DocumentNode],
        chapter_title: str = "",
        chapter_summary: str = "",
        previous_context: Any = "",
        glossary_terms: Optional[Dict[str, str]] = None,
        translation_context: Optional[TranslationContext] = None,
        document_type: str = "GENERAL",
        translation_mode: TranslationMode = TranslationMode.NATURAL,
    ) -> str:
        parts: List[str] = []
        if chapter_title:
            parts.append("CURRENT CHAPTER\n" + chapter_title)
        if translation_context:
            if translation_context.document_memory:
                parts.append("DOCUMENT MEMORY\n" + translation_context.document_memory)
            if translation_context.chapter_memory:
                parts.append("CHAPTER MEMORY\n" + translation_context.chapter_memory)
            if translation_context.few_shots:
                parts.append("DOMAIN EXAMPLES (style reference only)\n" + "\n".join(
                    f"Source: {item['source']}\nNatural Vietnamese: {item['target']}" for item in translation_context.few_shots
                ))
            if translation_context.previous_context:
                previous = "\n\n".join(
                    f"SOURCE PREVIOUS:\n{item.source}\nAPPROVED VIETNAMESE:\n{item.translation}"
                    for item in translation_context.previous_context
                )
                parts.append("PREVIOUS BILINGUAL CONTEXT - REFERENCE ONLY; DO NOT TRANSLATE IT AGAIN.\nUse only for terminology, pronouns, naming, tone, rhythm and discourse continuity.\n" + previous)
            relevant_glossary = translation_context.glossary
        else:
            if chapter_summary:
                parts.append("CHAPTER MEMORY\n" + chapter_summary)
            if previous_context:
                parts.append("PREVIOUS CONTEXT - REFERENCE ONLY; DO NOT TRANSLATE AGAIN\n" + str(previous_context))
            relevant_glossary = cls.filter_relevant_glossary(nodes, glossary_terms)
            shots = select_few_shots(document_type, translation_mode.value, nodes[0].type.value if nodes else "paragraph")
            if shots:
                parts.append("DOMAIN EXAMPLE\n" + "\n".join(f"Source: {item['source']}\nNatural Vietnamese: {item['target']}" for item in shots))
        if relevant_glossary:
            parts.append("MANDATORY LOCKED GLOSSARY\n" + "\n".join(f'- "{source}" -> "{target}"' for source, target in relevant_glossary.items()))
        blocks = [{"node_id": node.id, "type": node.type.value, "text": node.content} for node in nodes]
        parts.append("CURRENT SOURCE BLOCKS\n" + json.dumps(blocks, ensure_ascii=False, indent=2))
        parts.append("Translate CURRENT SOURCE BLOCKS only. Preserve all meaning and return every node exactly once.\nREQUIRED JSON RESPONSE FORMAT\n" + '{"translations":[{"node_id":"...","text":"..."}]}')
        return "\n\n".join(parts)
