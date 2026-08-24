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


class ContextBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextTokenBudget:
    model_context_limit: int
    system_tokens: int
    document_tokens: int
    chapter_tokens: int
    rolling_tokens: int
    glossary_tokens: int
    few_shot_tokens: int
    contract_tokens: int
    context_tokens: int
    source_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    total_estimated_tokens: int
    available_source_tokens: int
    fits: bool

    @property
    def context_window(self) -> int:
        return self.model_context_limit

    @property
    def reserved_output(self) -> int:
        return self.reserved_output_tokens

    @property
    def safety_margin(self) -> int:
        return self.safety_margin_tokens

    @property
    def fixed_prompt_tokens(self) -> int:
        return self.system_tokens + self.contract_tokens

    @property
    def source_budget(self) -> int:
        return self.available_source_tokens

    @property
    def total_estimated_input(self) -> int:
        return self.system_tokens + self.context_tokens + self.source_tokens + self.contract_tokens

    def assert_within_budget(self) -> None:
        if not self.fits:
            raise ContextBudgetExceeded(
                f"Ngữ cảnh ước tính {self.total_estimated_tokens} token vượt giới hạn {self.model_context_limit}."
            )


@dataclass
class TranslationContext:
    document_memory: str = ""
    chapter_memory: str = ""
    previous_context: List[BilingualContextItem] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    few_shots: List[dict] = field(default_factory=list)
    token_budget: Optional[ContextTokenBudget] = None
    system_prompt: str = ""
    trim_steps: List[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return bool(self.token_budget and self.token_budget.fits)

    def assert_within_budget(self) -> None:
        if not self.token_budget:
            raise ContextBudgetExceeded("Thiếu token budget cho request dịch.")
        self.token_budget.assert_within_budget()


class ContextAssembler:
    MIN_SOURCE_BUDGET = 64
    SAFETY_MARGIN_MIN = 220
    CONTEXT_LABEL_OVERHEAD = 72

    @staticmethod
    def filter_relevant_glossary(source_text: str, glossary: Dict[str, str]) -> Dict[str, str]:
        import re
        relevant: Dict[str, str] = {}
        for source, target in (glossary or {}).items():
            pattern = rf"(?i)(?<!\w){re.escape(source)}(?:s|es|ed|ing)?(?!\w)"
            if re.search(pattern, source_text):
                relevant[source] = target
        return relevant

    @staticmethod
    def _source_payload(nodes: List[Any], prompt_kind: str) -> str:
        if prompt_kind == "batch":
            blocks = []
            for node in nodes:
                node_type = getattr(getattr(node, "type", None), "value", getattr(node, "node_type", "paragraph"))
                blocks.append({"node_id": getattr(node, "id", ""), "type": node_type, "text": getattr(node, "content", "") or ""})
            return json.dumps(blocks, ensure_ascii=False, indent=2)
        return "\n".join(str(getattr(node, "content", "") or "") for node in nodes)

    @staticmethod
    def _compact_chapter(memory: ChapterMemory, level: int) -> str:
        if level == 1:
            value = {
                "summary": memory.summary[:1200], "entities": memory.entities[:10],
                "terminology": memory.terminology[:10], "pronoun_notes": memory.pronoun_notes[:8],
                "tone": memory.tone[:300], "style_notes": memory.style_notes[:5],
            }
        else:
            value = {
                "summary": memory.summary[:500], "entities": memory.entities[:5],
                "terminology": memory.terminology[:5], "pronoun_notes": memory.pronoun_notes[:3],
            }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compact_document(profile: DocumentTranslationProfile, level: int) -> str:
        if level == 1:
            value = {
                "document_type": profile.document_type, "domain": profile.domain,
                "audience": profile.intended_audience, "tone": profile.tone,
                "register": profile.register, "preferences": profile.translation_preferences,
                "style_notes": profile.style_notes[:5], "summary": profile.summary[:900],
            }
        else:
            value = {
                "domain": profile.domain, "tone": profile.tone,
                "register": profile.register, "summary": profile.summary[:400],
            }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

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
        system_prompt: str = "",
        output_contract: str = "",
        prompt_kind: str = "batch",
        compact_system_prompt: Optional[str] = None,
    ) -> TranslationContext:
        source_text = "\n".join(str(getattr(node, "content", "") or "") for node in nodes)
        source_payload = cls._source_payload(nodes, prompt_kind)
        document_memory = json.dumps(document_profile.to_dict(), ensure_ascii=False, separators=(",", ":"))
        chapter_text = json.dumps(chapter_memory.to_dict(), ensure_ascii=False, separators=(",", ":"))
        relevant_glossary = cls.filter_relevant_glossary(source_text, glossary)
        selected_shots = list(few_shots or [])
        previous = list(rolling_context)
        selected_system = system_prompt
        trim_steps: List[str] = []
        context_limit = min(model_capabilities.context_window, model_capabilities.recommended_context_window)
        source_tokens = estimate_tokens(source_payload)
        reserved_output = max(320, min(1400, int(source_tokens * 1.35) + 128))
        safety_margin = max(cls.SAFETY_MARGIN_MIN, int(context_limit * 0.06))

        def components() -> Dict[str, int]:
            return {
                "system": estimate_tokens(selected_system),
                "document": estimate_tokens(document_memory),
                "chapter": estimate_tokens(chapter_text),
                "rolling": estimate_tokens(json.dumps([item.__dict__ for item in previous], ensure_ascii=False)),
                "glossary": estimate_tokens(json.dumps(relevant_glossary, ensure_ascii=False)),
                "few_shot": estimate_tokens(json.dumps(selected_shots, ensure_ascii=False)),
                "contract": estimate_tokens(output_contract) + cls.CONTEXT_LABEL_OVERHEAD,
            }

        def total() -> int:
            values = components()
            return sum(values.values()) + source_tokens + reserved_output + safety_margin

        while total() > context_limit and selected_shots:
            selected_shots.pop()
            if "few_shot" not in trim_steps:
                trim_steps.append("few_shot")
        while total() > context_limit and previous:
            previous.pop(0)
            if "rolling_context" not in trim_steps:
                trim_steps.append("rolling_context")
        if total() > context_limit:
            chapter_text = cls._compact_chapter(chapter_memory, 1)
            trim_steps.append("chapter_memory_compact_1")
        if total() > context_limit:
            chapter_text = cls._compact_chapter(chapter_memory, 2)
            trim_steps.append("chapter_memory_compact_2")
        if total() > context_limit:
            document_memory = cls._compact_document(document_profile, 1)
            trim_steps.append("document_memory_compact_1")
        if total() > context_limit:
            document_memory = cls._compact_document(document_profile, 2)
            trim_steps.append("document_memory_compact_2")
        if total() > context_limit and compact_system_prompt and compact_system_prompt != selected_system:
            selected_system = compact_system_prompt
            trim_steps.append("optional_style_notes")

        values = components()
        context_tokens = values["document"] + values["chapter"] + values["rolling"] + values["glossary"] + values["few_shot"]
        non_source = values["system"] + context_tokens + values["contract"] + reserved_output + safety_margin
        available_source = min(model_capabilities.recommended_source_tokens, max(0, context_limit - non_source))
        total_estimated = non_source + source_tokens
        budget = ContextTokenBudget(
            model_context_limit=context_limit,
            system_tokens=values["system"],
            document_tokens=values["document"],
            chapter_tokens=values["chapter"],
            rolling_tokens=values["rolling"],
            glossary_tokens=values["glossary"],
            few_shot_tokens=values["few_shot"],
            contract_tokens=values["contract"],
            context_tokens=context_tokens,
            source_tokens=source_tokens,
            reserved_output_tokens=reserved_output,
            safety_margin_tokens=safety_margin,
            total_estimated_tokens=total_estimated,
            available_source_tokens=available_source,
            fits=total_estimated <= context_limit,
        )
        return TranslationContext(
            document_memory=document_memory,
            chapter_memory=chapter_text,
            previous_context=previous,
            glossary=relevant_glossary,
            few_shots=selected_shots,
            token_budget=budget,
            system_prompt=selected_system,
            trim_steps=trim_steps,
        )
