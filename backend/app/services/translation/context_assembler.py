import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.translation.context_memory import BilingualContextItem, ChapterMemory
from app.services.translation.document_profiler import DocumentTranslationProfile
from app.services.translation.model_capabilities import ModelCapabilities


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    punctuation = sum(1 for char in text if char in ".,;:!?()[]{}\"'")
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return max(1, int(words * 1.22 + punctuation * 0.22 + non_ascii * 0.08 + 8))


@dataclass
class ContextTokenBudget:
    context_window: int
    reserved_output: int
    safety_margin: int
    fixed_prompt_tokens: int
    context_tokens: int
    source_budget: int
    total_estimated_input: int = 0


@dataclass
class TranslationContext:
    document_memory: str = ""
    chapter_memory: str = ""
    previous_context: List[BilingualContextItem] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    few_shots: List[dict] = field(default_factory=list)
    token_budget: Optional[ContextTokenBudget] = None


class ContextAssembler:
    FIXED_PROMPT_ESTIMATE = 900
    STRUCTURED_OUTPUT_OVERHEAD = 180

    @staticmethod
    def filter_relevant_glossary(source_text: str, glossary: Dict[str, str]) -> Dict[str, str]:
        import re
        relevant: Dict[str, str] = {}
        for source, target in (glossary or {}).items():
            pattern = rf"(?i)(?<!\w){re.escape(source)}(?:s|es|ed|ing)?(?!\w)"
            if re.search(pattern, source_text):
                relevant[source] = target
        return relevant

    @classmethod
    def assemble_context(
        cls,
        nodes: List[Any],
        document_profile: DocumentTranslationProfile,
        chapter_memory: ChapterMemory,
        rolling_context: List[BilingualContextItem],
        glossary: Dict[str, str],
        model_capabilities: ModelCapabilities,
        few_shots: Optional[List[dict]] = None,
    ) -> TranslationContext:
        source_text = "\n".join(str(getattr(node, "content", "") or "") for node in nodes)
        document_memory = json.dumps(document_profile.to_dict(), ensure_ascii=False, separators=(",", ":"))
        chapter_text = json.dumps(chapter_memory.to_dict(), ensure_ascii=False, separators=(",", ":"))
        relevant_glossary = cls.filter_relevant_glossary(source_text, glossary)
        selected_shots = list(few_shots or [])
        previous = list(rolling_context)

        context_limit = min(model_capabilities.context_window, model_capabilities.recommended_context_window)
        reserved_output = max(700, min(1400, int(context_limit * 0.29)))
        safety_margin = max(220, int(context_limit * 0.07))

        def cost() -> int:
            return (
                estimate_tokens(document_memory)
                + estimate_tokens(chapter_text)
                + estimate_tokens(json.dumps(relevant_glossary, ensure_ascii=False))
                + estimate_tokens(json.dumps([item.__dict__ for item in previous], ensure_ascii=False))
                + estimate_tokens(json.dumps(selected_shots, ensure_ascii=False))
            )

        max_context = max(300, context_limit - reserved_output - safety_margin - cls.FIXED_PROMPT_ESTIMATE - cls.STRUCTURED_OUTPUT_OVERHEAD - 500)
        while cost() > max_context and selected_shots:
            selected_shots.pop()
        while cost() > max_context and len(previous) > 1:
            previous.pop(0)
        if cost() > max_context:
            chapter_text = json.dumps({
                "summary": chapter_memory.summary[:1200],
                "entities": chapter_memory.entities[:10],
                "terminology": chapter_memory.terminology[:10],
                "pronoun_notes": chapter_memory.pronoun_notes[:8],
                "tone": chapter_memory.tone[:300],
            }, ensure_ascii=False, separators=(",", ":"))
        if cost() > max_context:
            document_memory = json.dumps({
                "document_type": document_profile.document_type,
                "domain": document_profile.domain,
                "audience": document_profile.intended_audience,
                "tone": document_profile.tone,
                "register": document_profile.register,
                "preferences": document_profile.translation_preferences,
                "style_notes": document_profile.style_notes[:5],
                "summary": document_profile.summary[:900],
            }, ensure_ascii=False, separators=(",", ":"))
        if cost() > max_context:
            chapter_text = json.dumps({
                "summary": chapter_memory.summary[:500],
                "entities": chapter_memory.entities[:5],
                "terminology": chapter_memory.terminology[:5],
            }, ensure_ascii=False, separators=(",", ":"))
            document_memory = json.dumps({
                "domain": document_profile.domain,
                "tone": document_profile.tone,
                "register": document_profile.register,
                "summary": document_profile.summary[:400],
            }, ensure_ascii=False, separators=(",", ":"))

        context_tokens = cost()
        source_budget = max(
            240,
            min(
                model_capabilities.recommended_source_tokens,
                context_limit - reserved_output - safety_margin - cls.FIXED_PROMPT_ESTIMATE - cls.STRUCTURED_OUTPUT_OVERHEAD - context_tokens,
            ),
        )
        budget = ContextTokenBudget(
            context_window=context_limit,
            reserved_output=reserved_output,
            safety_margin=safety_margin,
            fixed_prompt_tokens=cls.FIXED_PROMPT_ESTIMATE + cls.STRUCTURED_OUTPUT_OVERHEAD,
            context_tokens=context_tokens,
            source_budget=source_budget,
            total_estimated_input=context_tokens + estimate_tokens(source_text) + cls.FIXED_PROMPT_ESTIMATE + cls.STRUCTURED_OUTPUT_OVERHEAD,
        )
        return TranslationContext(document_memory, chapter_text, previous, relevant_glossary, selected_shots, budget)
