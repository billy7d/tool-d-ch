import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from app.models.canonical import DocumentNode, TranslationMode
from app.services.translation.context_assembler import TranslationContext
from app.services.translation.prompt_profiles import STYLE_PACK_VERSION, get_style_pack, select_few_shots
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import PROMPT_VERSION
from app.services.translation.term_matcher import TermMatcher


class PromptBuilder:
    MODE_INSTRUCTIONS = {
        "NATURAL": (
            "Use Natural, fluent Vietnamese suitable for publishing. Preserve meaning, not English wording or grammar. "
            "Read and understand the complete sentence or paragraph before translating; do not map words one-to-one "
            "and do not preserve English word order merely because it is grammatical. Recreate each sentence with "
            "natural Vietnamese syntax and idiomatic Vietnamese collocations. You may move adverbials, change active "
            "and passive voice when the meaning stays identical, turn nominalizations into verbs, remove needless "
            "filler, and split or combine sentences when every proposition, reference and relationship is preserved. "
            "Avoid these anti-patterns: word-for-word translation, English word order, excessive passive voice, "
            "unnecessary nominalization, awkward literal collocations, redundant phrases, and unsuitable pronoun or "
            "reference choices. Avoid needless filler such as 'thực hiện việc', 'tiến hành việc' or 'một cách' when "
            "natural Vietnamese does not need it. Do not invent Sino-Vietnamese or academic terminology when the "
            "source does not require it. Preserve facts, negation, conditions, modality, scope, causality, entities, "
            "locked terminology and formatting exactly; naturalness never outranks semantic correctness."
        ),
        "BALANCED": "Balance natural Vietnamese with close semantic alignment.",
        "FAITHFUL": "Stay close to the source structure while keeping grammatical Vietnamese.",
        "ACADEMIC": "Use formal scholarly Vietnamese with precise terminology.",
        "TECHNICAL": "Prioritize technical accuracy and preserve identifiers.",
        "CUSTOM": "Apply valid user preferences after all correctness rules.",
    }
    CONFLICTING_INSTRUCTION_PATTERNS = (
        r"(?i)\btóm tắt\b", r"(?i)\brút gọn\b", r"(?i)\bbỏ (?:qua|bớt)\b",
        r"(?i)\bsummar(?:y|ize)\b", r"(?i)\bom(?:it|ission)\b",
    )
    BATCH_OUTPUT_CONTRACT = (
        "Return strictly valid JSON only, using this schema:\n"
        '{"translations":[{"node_id":"...","text":"..."}]}\n'
        "Return every requested node exactly once and no unknown node IDs."
    )
    SINGLE_OUTPUT_CONTRACT = "Return plain text only. Do not return structured objects, notes, labels, preambles or markdown fences."
    SEGMENT_OUTPUT_CONTRACT = "Translate only CURRENT SEGMENT. Return plain translated text only. Do not repeat surrounding context."

    @classmethod
    def validate_custom_instructions(cls, custom_instructions: Optional[str]) -> None:
        if custom_instructions and any(re.search(pattern, custom_instructions) for pattern in cls.CONFLICTING_INSTRUCTION_PATTERNS):
            raise ValueError("Chỉ dẫn tùy chỉnh xung đột với yêu cầu dịch đầy đủ, không tóm tắt hoặc bỏ ý.")

    @staticmethod
    def _mode_value(value: Any) -> str:
        return str(getattr(value, "value", value or "NATURAL")).upper()

    @classmethod
    def _legacy_config(
        cls,
        document_type: str,
        translation_mode: Any,
        style_guide: Optional[Dict[str, Any]],
        custom_instructions: Optional[str],
        register: Optional[str],
        sentence_style: Optional[str],
    ) -> Any:
        guide = dict(style_guide or {})
        style_pack = get_style_pack(document_type)
        return SimpleNamespace(
            document_type=(document_type or "GENERAL").upper(),
            translation_mode=cls._mode_value(translation_mode),
            style_guide=guide,
            custom_instructions=custom_instructions or "",
            register=str(register or guide.get("register") or style_pack.register).upper(),
            sentence_style=str(sentence_style or guide.get("sentence_style") or "MODERATE").upper(),
        )

    @classmethod
    def build_system_prompt(
        cls,
        document_type: Any = "GENERAL",
        translation_mode: Any = TranslationMode.NATURAL,
        style_guide: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[str] = None,
        register: Optional[str] = None,
        sentence_style: Optional[str] = None,
        include_optional_style_notes: bool = True,
    ) -> str:
        config = document_type if isinstance(document_type, TranslationConfig) else cls._legacy_config(
            str(document_type), translation_mode, style_guide, custom_instructions, register, sentence_style,
        )
        cls.validate_custom_instructions(config.custom_instructions)
        style_pack = get_style_pack(config.document_type)
        layers = [
            f"PROMPT VERSION: {PROMPT_VERSION}",
            (
                "SYSTEM CORE\nYou are a professional English-to-Vietnamese translator and Vietnamese publishing editor.\n"
                "PRIORITY 1 - SEMANTIC FIDELITY: Preserve every proposition, condition, negation, causal relation, number, entity, reference and degree of certainty.\n"
                "PRIORITY 2 - COMPLETENESS: Do not summarize, skip, compress, explain or add information.\n"
                "PRIORITY 3 - TERMINOLOGY: Follow the locked glossary and established entity naming exactly.\n"
                "PRIORITY 4 - NATURAL VIETNAMESE: Rewrite English syntax into idiomatic Vietnamese only after priorities 1-3 are satisfied.\n"
                "PRIORITY 5 - STYLE: Maintain the document tone and approved preceding translation.\n"
                "You may move adverbials, convert passive to active, reduce nominalization and split overly long sentences only when no meaning changes.\n"
                "Do not change facts, polarity, scope, modality or certainty. Preserve formatting, numbers, dates, formulas, units, URLs, code and references.\n"
                "Vietnamese prose must be Vietnamese. Preserve proper nouns, company/product/trademark names, acronyms, programming identifiers, commands, paths, code, formulas and intentionally retained technical terms."
            ),
            "TRANSLATION MODE\n" + cls.MODE_INSTRUCTIONS.get(config.translation_mode, cls.MODE_INSTRUCTIONS["NATURAL"]),
            (
                f"DOCUMENT DOMAIN\n{style_pack.domain}\nSTYLE PACK {STYLE_PACK_VERSION}\n"
                f"Register: {config.register}\nRules: {style_pack.instructions}\n"
                f"Forbidden: {style_pack.forbidden}\nSentence restructuring: {config.sentence_style}"
            ),
        ]
        optional_style = {
            key: value for key, value in (config.style_guide or {}).items()
            if key not in {"register", "sentence_style"}
        }
        if optional_style and include_optional_style_notes:
            layers.append("OPTIONAL STYLE NOTES\n" + json.dumps(optional_style, ensure_ascii=False, indent=2))
        if config.custom_instructions:
            layers.append("USER CUSTOM INSTRUCTION (lower priority than correctness and glossary)\n" + config.custom_instructions)
        return "\n\n".join(layers)

    @classmethod
    def filter_relevant_glossary(cls, nodes: List[DocumentNode], glossary_terms: Optional[Dict[str, str]]) -> Dict[str, str]:
        combined = " ".join(node.content for node in nodes)
        return {
            source: target for source, target in (glossary_terms or {}).items()
            if TermMatcher.contains(combined, source)
        }

    @staticmethod
    def _context_parts(context: Optional[TranslationContext], chapter_title: str = "") -> List[str]:
        parts: List[str] = []
        if chapter_title:
            parts.append("CURRENT CHAPTER\n" + chapter_title)
        if not context:
            return parts
        if context.document_memory:
            parts.append(
                "SOURCE DOCUMENT CHARACTERISTICS - DESCRIPTIVE ONLY\n"
                "These observations describe the source. They must not override TARGET register, sentence style, "
                "translation mode, locked glossary or user instructions.\n" + context.document_memory
            )
        if context.chapter_memory:
            parts.append("CHAPTER MEMORY\n" + context.chapter_memory)
        if context.few_shots:
            parts.append("DOMAIN EXAMPLES - STYLE REFERENCE ONLY\n" + "\n".join(
                f"Source: {item['source']}\nNatural Vietnamese: {item['target']}" for item in context.few_shots
            ))
        if context.previous_context:
            previous = "\n\n".join(
                f"SOURCE PREVIOUS:\n{item.source}\nAPPROVED VIETNAMESE:\n{item.translation}"
                for item in context.previous_context
            )
            parts.append(
                "PREVIOUS BILINGUAL CONTEXT - REFERENCE ONLY; DO NOT TRANSLATE IT AGAIN.\n"
                "Use only for terminology, pronouns, naming, tone, rhythm and continuity.\n" + previous
            )
        if context.glossary:
            parts.append("MANDATORY RELEVANT GLOSSARY\n" + "\n".join(
                f'- "{source}" -> "{target}"' for source, target in context.glossary.items()
            ))
        if context.entity_decisions:
            parts.append("CURRENT RELEVANT ENTITY DECISIONS\n" + "\n".join(
                f'- "{source}" -> "{target}"' for source, target in context.entity_decisions.items()
            ))
        return parts

    @classmethod
    def build_batch_prompt(cls, nodes: List[DocumentNode], context: TranslationContext, chapter_title: str = "") -> str:
        parts = cls._context_parts(context, chapter_title)
        blocks = [{"node_id": node.id, "type": node.type.value, "text": node.content} for node in nodes]
        parts.append("CURRENT SOURCE BLOCKS\n" + json.dumps(blocks, ensure_ascii=False, indent=2))
        parts.append("Translate CURRENT SOURCE BLOCKS only. Preserve every proposition.\n" + cls.BATCH_OUTPUT_CONTRACT)
        return "\n\n".join(parts)

    @classmethod
    def build_single_prompt(cls, node: DocumentNode, context: TranslationContext, chapter_title: str = "") -> str:
        parts = cls._context_parts(context, chapter_title)
        parts.append("CURRENT SOURCE\n" + node.content)
        parts.append("Translate CURRENT SOURCE completely.\n" + cls.SINGLE_OUTPUT_CONTRACT)
        return "\n\n".join(parts)

    @classmethod
    def build_repair_prompt(
        cls,
        node: DocumentNode,
        context: TranslationContext,
        issues: List[Dict[str, Any]],
        chapter_title: str = "",
    ) -> str:
        parts = cls._context_parts(context, chapter_title)
        issue_lines = "\n".join(f"- {issue.get('code', 'QA')}: {issue.get('message', '')}" for issue in issues)
        parts.extend([
            "ORIGINAL SOURCE\n" + node.content,
            "VALIDATION OR QA ISSUES\n" + (issue_lines or "- The previous candidate did not pass validation."),
            "Translate the ORIGINAL SOURCE again and correct the issues without omitting or adding meaning.\n" + cls.SINGLE_OUTPUT_CONTRACT,
        ])
        return "\n\n".join(parts)

    @classmethod
    def build_segment_prompt(
        cls,
        segment_text: str,
        context: TranslationContext,
        chapter_title: str = "",
        segment_index: int = 1,
        segment_count: int = 1,
    ) -> str:
        parts = cls._context_parts(context, chapter_title)
        parts.append(f"SEGMENT {segment_index}/{segment_count}\nCURRENT SEGMENT\n{segment_text}")
        parts.append(cls.SEGMENT_OUTPUT_CONTRACT)
        return "\n\n".join(parts)

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
        context = translation_context or TranslationContext(
            chapter_memory=chapter_summary,
            glossary=cls.filter_relevant_glossary(nodes, glossary_terms),
        )
        if previous_context and not context.previous_context:
            context.chapter_memory = "\n".join(filter(None, [context.chapter_memory, str(previous_context)]))
        if not context.few_shots and nodes:
            context.few_shots = select_few_shots(document_type, cls._mode_value(translation_mode), nodes[0].type.value)
        return cls.build_batch_prompt(nodes, context, chapter_title)
