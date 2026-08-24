import re
from typing import Dict, List, Optional

from app.models.canonical import DocumentNode, NodeType
from app.services.translation.context_assembler import estimate_tokens


class SemanticChunk:
    def __init__(
        self,
        chunk_id: str,
        nodes: List[DocumentNode],
        estimated_tokens: int = 0,
        oversized_segments: Optional[Dict[str, List[str]]] = None,
    ):
        self.chunk_id = chunk_id
        self.nodes = nodes
        self.estimated_tokens = estimated_tokens
        self.oversized_segments = oversized_segments or {}


class AdaptiveSemanticChunker:
    NON_TRANSLATABLE = {NodeType.IMAGE, NodeType.HORIZONTAL_RULE, NodeType.PAGE_BREAK_HINT, NodeType.CODE_BLOCK}
    ISOLATED = {NodeType.TABLE, NodeType.FOOTNOTE}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return estimate_tokens(text)

    @classmethod
    def split_sentences(cls, text: str, max_tokens: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
        parts: List[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and cls.estimate_tokens(candidate) > max_tokens:
                parts.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            parts.append(current)
        if len(parts) == 1 and cls.estimate_tokens(parts[0]) > max_tokens:
            words = parts[0].split()
            parts = []
            current_words: List[str] = []
            for word in words:
                if current_words and cls.estimate_tokens(" ".join(current_words + [word])) > max_tokens:
                    parts.append(" ".join(current_words))
                    current_words = [word]
                else:
                    current_words.append(word)
            if current_words:
                parts.append(" ".join(current_words))
        return parts

    @classmethod
    def chunk_nodes(
        cls,
        nodes: List[DocumentNode],
        source_budget: int = 1200,
        hard_limit: Optional[int] = None,
        **legacy: int,
    ) -> List[SemanticChunk]:
        if "target_tokens" in legacy:
            source_budget = int(legacy["target_tokens"])
        hard_limit = max(64, hard_limit or source_budget)
        target = min(source_budget, hard_limit)
        chunks: List[SemanticChunk] = []
        current: List[DocumentNode] = []
        current_tokens = 0
        chunk_index = 1

        def flush() -> None:
            nonlocal current, current_tokens, chunk_index
            if current:
                chunks.append(SemanticChunk(f"chunk_{chunk_index:05d}", current, current_tokens))
                chunk_index += 1
                current = []
                current_tokens = 0

        for node in nodes:
            if node.type in cls.NON_TRANSLATABLE:
                continue
            node_tokens = cls.estimate_tokens(node.content)
            if node_tokens > hard_limit:
                flush()
                segments = cls.split_sentences(node.content, hard_limit)
                chunks.append(SemanticChunk(
                    f"chunk_{chunk_index:05d}",
                    [node],
                    max(cls.estimate_tokens(part) for part in segments),
                    {node.id: segments},
                ))
                chunk_index += 1
                continue
            if node.type in cls.ISOLATED:
                flush()
                chunks.append(SemanticChunk(f"chunk_{chunk_index:05d}", [node], node_tokens))
                chunk_index += 1
                continue
            if node.type == NodeType.HEADING and current:
                flush()
            if current and current_tokens + node_tokens > target:
                flush()
            current.append(node)
            current_tokens += node_tokens
        flush()
        return chunks


SemanticChunker = AdaptiveSemanticChunker
